#!/usr/bin/env python3
""" Training for F3Tennis """
import os
import argparse
from contextlib import nullcontext
import random
import numpy as np
import torch
torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import (
    ChainedScheduler, LinearLR, CosineAnnealingLR)
from torch.utils.data import DataLoader
import torchvision
from itertools import groupby
import timm
from tqdm import tqdm

from model.common import step, BaseRGBModel
from model.shift import make_temporal_shift
from model.slowfast import ResNet3dSlowFast
from model.modules import *
from dataset.frame import ActionSeqDataset, ActionSeqVideoDataset
from util.eval import edit_score, non_maximum_suppression
from util.io import load_json, store_json, clear_files
from util.dataset import DATASETS, load_classes
import warnings
warnings.filterwarnings("ignore")

EPOCH_NUM_FRAMES = 500000
BASE_NUM_WORKERS = 4
BASE_NUM_VAL_EPOCHS = 20
INFERENCE_BATCH_SIZE = 4
HIDDEN_DIM = 768

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, choices=DATASETS)
    parser.add_argument('frame_dir', type=str, help='Path to extracted frames')

    parser.add_argument(
        '-m', '--feature_arch', type=str, required=True, choices=[
            # From torchvision
            'rn50_gsm',
            'rny002_gsm',
            'rny002_tsm',
            'rny008_gsm',
            'rny008_tsm',
            'slowfast'
        ], help='architecture for feature extraction')
    parser.add_argument(
        '-t', '--temporal_arch', type=str, default='gru',
        choices=['gru', 'deeper_gru', 'mstcn', 'asformer', 'actionformer', 'gcn', 'tcn', 'fc'])

    parser.add_argument('--clip_len', type=int, default=96)
    parser.add_argument('--crop_dim', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--stride', type=int, default=2)
    parser.add_argument('-ag', '--acc_grad_iter', type=int, default=1,
                        help='Use gradient accumulation')

    parser.add_argument('--warm_up_epochs', type=int, default=3)
    parser.add_argument('--num_epochs', type=int, default=50)

    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
    parser.add_argument('-s', '--save_dir', type=str, required=True,
                        help='Dir to save checkpoints and predictions')

    parser.add_argument('--resume', action='store_true',
                        help='Resume training from checkpoint in <save_dir>')

    parser.add_argument('--start_val_epoch', type=int, default=30)
    parser.add_argument('--criterion', choices=['edit', 'loss'], default='edit')

    parser.add_argument('-j', '--num_workers', type=int,
                        help='Base number of dataloader workers')

    parser.add_argument('-mgpu', '--gpu_parallel', action='store_true')
    return parser.parse_args()

class F3Tennis(BaseRGBModel):

    class Impl(nn.Module):

        def __init__(self, num_classes, feature_arch, temporal_arch, clip_len, step=1, device='cuda'):
            super().__init__()
            self._device = device
            self._num_classes = num_classes

            # video encoder
            if 'rn50' in feature_arch:
                resnet_name = feature_arch.split('_')[0].replace('rn', 'resnet')
                glb_feat = getattr(torchvision.models, resnet_name)(pretrained=True)
                glb_feat_dim = glb_feat.fc.in_features
                glb_feat.fc = nn.Identity()

            elif feature_arch.startswith(('rny002', 'rny008')):
                glb_feat = timm.create_model({
                    'rny002': 'regnety_002',
                    'rny008': 'regnety_008',
                }[feature_arch.rsplit('_', 1)[0]], pretrained=True)
                glb_feat_dim = glb_feat.head.fc.in_features
                glb_feat.head.fc = nn.Identity()

            elif 'slowfast' in feature_arch:
                glb_feat = ResNet3dSlowFast(None, slow_upsample=8)
                glb_feat.load_pretrained_weight()
                glb_feat_dim = 2304

            else:
                raise NotImplementedError(feature_arch)

            # Add Temporal Shift Modules
            self._require_clip_len = clip_len
            if feature_arch.endswith('_tsm'):
                make_temporal_shift(glb_feat, clip_len, is_gsm=False, step=step)
                self._require_clip_len = clip_len

            self._glb_feat = glb_feat  # global feature extractor
            self._feat_dim = glb_feat_dim
            self._is_3d = 'slowfast' in feature_arch

            # head modules
            d_model = min(HIDDEN_DIM, self._feat_dim)
            if temporal_arch == 'gru':  # single layer GRU
                self._pred_fine = GRUPrediction(self._feat_dim, num_classes, d_model, num_layers=1)
            elif temporal_arch == 'deeper_gru':  # deeper GRU
                self._pred_fine = GRUPrediction(self._feat_dim, num_classes, d_model, num_layers=3)
            elif temporal_arch == 'tcn':  # single TCN
                self._pred_fine = TCNPrediction(self._feat_dim, num_classes, 1)
            elif temporal_arch == 'mstcn':  # multi-stage TCN
                self._pred_fine = TCNPrediction(self._feat_dim, num_classes, 3)
            elif temporal_arch == 'asformer':  # ASFormer
                self._pred_fine = ASFormerPrediction(self._feat_dim, num_classes, 3)
            elif temporal_arch == 'gcn':  # G-TAD
                self._pred_fine = GCNPrediction(self._feat_dim, num_classes, hidden_dim=d_model)
            elif temporal_arch == 'actionformer':  # Actionformer
                self._pred_fine = ActionFormerPrediction(self._feat_dim, num_classes, hidden_dim=d_model)
            elif temporal_arch == 'fc':  # Simple Fully-Connected Layer
                self._pred_fine = FCPrediction(self._feat_dim, num_classes)
            else:
                raise NotImplementedError(temporal_arch)

        def forward(self, frame):
            batch_size, true_clip_len, channels, height, width = frame.shape

            clip_len = true_clip_len
            if self._require_clip_len > 0:
                # TSM module requires clip len to be known
                assert true_clip_len <= self._require_clip_len, \
                    'Expected {}, got {}'.format(
                        self._require_clip_len, true_clip_len)
                if true_clip_len < self._require_clip_len:
                    frame = F.pad(
                        frame, (0,) * 7 + (self._require_clip_len - true_clip_len,))
                    clip_len = self._require_clip_len

            # global visual embedding
            if self._is_3d:
                im_feat = self._glb_feat(frame.transpose(1, 2)).transpose(1, 2)
            else:
                im_feat = self._glb_feat(
                    frame.view(-1, channels, height, width)
                ).reshape(batch_size, clip_len, -1)

            return self._pred_fine(im_feat)

    def __init__(self, num_classes, feature_arch, temporal_arch, clip_len, step=1, device='cuda', multi_gpu=False):
        self._device = device
        self._multi_gpu = multi_gpu
        self._model = F3Tennis.Impl(num_classes, feature_arch, temporal_arch, clip_len, step=step)

        if multi_gpu:
            self._model = nn.DataParallel(self._model)

        self._model.to(device)
        self._num_classes = num_classes

    def epoch(self, loader, optimizer=None, scaler=None, lr_scheduler=None, acc_grad_iter=1, fg_weight=5):
        if optimizer is None:
            self._model.eval()
        else:
            optimizer.zero_grad()
            self._model.train()

        ce_kwargs = {}
        if fg_weight != 1:
            ce_kwargs['weight'] = torch.FloatTensor(
                [1] + [fg_weight] * (self._num_classes - 1)).to(self._device)

        epoch_loss = 0.
        with (torch.no_grad() if optimizer is None else nullcontext()):
            for batch_idx, batch in enumerate(tqdm(loader)):
                frame = loader.dataset.load_frame_gpu(batch, self._device)
                frame_full_label = batch['frame_full_label'].to(self._device)

                with torch.cuda.amp.autocast():
                    pred = self._model(frame)

                    if len(pred.shape) == 3:
                        pred = pred.unsqueeze(0)

                    # per-frame loss
                    loss = 0.
                    for i in range(pred.shape[0]):
                        loss += F.cross_entropy(
                            pred[i].reshape(-1, self._num_classes), frame_full_label.flatten(), **ce_kwargs)
                        
                if optimizer is not None:
                    step(optimizer, scaler, loss / acc_grad_iter,
                         lr_scheduler=lr_scheduler,
                         backward_only=(batch_idx + 1) % acc_grad_iter != 0)

                epoch_loss += loss.detach().item()
        return epoch_loss / len(loader)     # Avg loss

    def predict(self, seq, use_amp=True):
        if not isinstance(seq, torch.Tensor):
            seq = torch.FloatTensor(seq)
        if len(seq.shape) == 4:  # (L, C, H, W)
            seq = seq.unsqueeze(0)
        if seq.device != self._device:
            seq = seq.to(self._device)

        self._model.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast() if use_amp else nullcontext():
                pred = self._model(seq)
            if isinstance(pred, tuple):
                pred = pred[0]
            if len(pred.shape) > 3:
                pred = pred[-1]
            pred = torch.softmax(pred, axis=2)
            pred = non_maximum_suppression(pred, 5)
            pred_cls = torch.argmax(pred, axis=2)
            return pred_cls.cpu().numpy(), pred.cpu().numpy()


def evaluate(model, dataset, classes, dataset_name='f3tennis'):
    pred_dict = {}
    for video, video_len, _ in dataset.videos:
        pred_dict[video] = (
            np.zeros((video_len, len(classes) + 1), np.float32),
            np.zeros(video_len, np.int32))

    classes_inv = {v: k for k, v in classes.items()}
    classes_inv[0] = 'NA'

    per_label = load_classes(os.path.join('data', dataset_name, 'elements.txt'))

    # Do not up the batch size if the dataset augments
    batch_size = 1 if dataset.augment else INFERENCE_BATCH_SIZE
    for clip in tqdm(DataLoader(
            dataset, num_workers=BASE_NUM_WORKERS * 2, pin_memory=True,
            batch_size=batch_size
    )):

        if batch_size > 1:
            # Batched by dataloader
            _, batch_pred_scores = model.predict(clip['frame'])
            for i in range(clip['frame'].shape[0]):
                video = clip['video'][i]
                scores, support = pred_dict[video]
                pred_scores = batch_pred_scores[i]
                start = clip['start'][i].item()
                if start < 0:
                    pred_scores = pred_scores[-start:, :]
                    start = 0
                end = start + pred_scores.shape[0]
                if end >= scores.shape[0]:
                    end = scores.shape[0]
                    pred_scores = pred_scores[:end - start, :]
                scores[start:end, :] += pred_scores
                support[start:end] += 1

    sets = []
    for delta in [0, 1]:
        f = open('error_sequences.txt', 'w')
        edit_scores_high, edit_scores_mid, edit_scores_low = [], [], []
        y_true, y_pred = None, None
        acc, sr, num_samples, num_actions = 0, 0, 0, 0
        len_not_match = 0
        eval_matrix = np.zeros((len(classes), 3), int)
        f1 = np.array([0, 0, 0])
        for video, (coarse_scores, fine_scores, support) in sorted(pred_dict.items()):
            coarse_label, fine_label = dataset.get_labels(video)
            coarse_scores /= support[:, None]
            fine_scores /= support[:, None]

            coarse_scores = non_maximum_suppression_np(coarse_scores, 5)

            # argmax pred
            coarse_pred = np.argmax(coarse_scores, axis=1)
            fine_pred = np.zeros_like(fine_scores, int)
            for i in range(len(fine_scores)):
                for start, end in [[0, 2], [2, 5], [5, 8], [8, 10], [10, 16], [16, 24], [24, 26], [26, 30]]:
                    max_idx = np.argmax(fine_scores[i, start:end])
                    fine_pred[i, start + max_idx] = 1

            for i in range(len(fine_pred)):
                if coarse_label[i] == 1:
                    pred_i = i
                    if sum(coarse_pred[(i - delta):(i + delta + 1)]) >= 1:
                        for i_ in range(i - delta, i + delta + 1):
                            if coarse_pred[i_] == 1:
                                # pred_i = i_
                                temp_pred = fine_pred[i_]
                                break
                    else:
                        temp_pred = np.zeros(len(fine_pred[0]))

                    for j in range(len(fine_pred[0])):
                        if temp_pred[j] == 1 and fine_label[i, j] == 1:
                            eval_matrix[j, 0] += 1  # tp
                        if temp_pred[j] == 1 and fine_label[i, j] == 0:
                            eval_matrix[j, 1] += 1  # fp
                        if temp_pred[j] == 0 and fine_label[i, j] == 1:
                            eval_matrix[j, 2] += 1  # fn

            print_preds, print_gts = [], []
            # tp, fp, fn
            for i in range(len(fine_pred)):
                if coarse_label[i] == 1:
                    print_gt = []
                    for j in range(len(fine_pred[0])):
                        if fine_label[i, j] == 1:
                            print_gt.append(classes_inv[j + 1])
                    print_gts.append('_'.join(print_gt))
                if coarse_pred[i] == 1:
                    print_pred = []
                    for j in range(len(fine_pred[0])):
                        if fine_pred[i, j] == 1:
                            print_pred.append(classes_inv[j + 1])
                    print_preds.append('_'.join(print_pred))

            fine_labels_full = fine_label
            fine_preds_full = fine_pred
            fine_labels_mid = fine_labels_full[:, :24]
            fine_preds_mid = fine_preds_full[:, :24]
            fine_labels_low = fine_labels_full[:, [0, 1, 5, 6, 7, 8, 9, 26, 27, 28, 29]]
            fine_preds_low = fine_preds_full[:, [0, 1, 5, 6, 7, 8, 9, 26, 27, 28, 29]]

            fine_labels_full = [int(''.join(str(x) for x in row), 2) for row in fine_labels_full]
            fine_preds_full = [int(''.join(str(x) for x in row), 2) for row in fine_preds_full]
            fine_labels_mid = [int(''.join(str(x) for x in row), 2) for row in fine_labels_mid]
            fine_preds_mid = [int(''.join(str(x) for x in row), 2) for row in fine_preds_mid]
            fine_labels_low = [int(''.join(str(x) for x in row), 2) for row in fine_labels_low]
            fine_preds_low = [int(''.join(str(x) for x in row), 2) for row in fine_preds_low]

            fine_preds_full = coarse_pred * fine_preds_full
            fine_preds_mid = coarse_pred * fine_preds_mid
            fine_preds_low = coarse_pred * fine_preds_low

            gt = [k for k, g in groupby(fine_labels_full) if k != 0]
            pred = [k for k, g in groupby(fine_preds_full) if k != 0]
            gt_mid = [k for k, g in groupby(fine_labels_mid) if k != 0]
            pred_mid = [k for k, g in groupby(fine_preds_mid) if k != 0]
            gt_low = [k for k, g in groupby(fine_labels_low) if k != 0]
            pred_low = [k for k, g in groupby(fine_preds_low) if k != 0]

            # sucess rate
            if len(pred) == len(gt):
                success = True
                for j in range(len(pred)):
                    if pred[j] != gt[j] and {pred[j], gt[j]} not in sets:
                        success = False
                        f.write(video + '\n')
                        f.write('->'.join(print_preds) + '\n')
                        f.write('->'.join(print_gts) + '\n')
                        f.write('\n')
                        break
                if success:
                    sr += 1
            else:
                f.write(video + '\n')
                f.write('->'.join(print_preds) + '\n')
                f.write('->'.join(print_gts) + '\n')
                f.write('\n')
                len_not_match += 1
            # edit_scores.append(edit_score(pred, gt, sets))
            edit_scores_high.append(edit_score(pred, gt, sets))
            edit_scores_mid.append(edit_score(pred_mid, gt_mid, sets))
            edit_scores_low.append(edit_score(pred_low, gt_low, sets))
            max_length = max(len(pred), len(gt))
            min_length = min(len(pred), len(gt))
            pred = np.pad(pred, (0, max_length - len(pred)), 'constant')
            gt = np.pad(gt, (0, max_length - len(gt)), 'constant')

        f.close()
        precision = eval_matrix[:, 0] / (eval_matrix[:, 0] + eval_matrix[:, 1] + 1e-10)
        recall = eval_matrix[:, 0] / (eval_matrix[:, 0] + eval_matrix[:, 2] + 1e-10)
        f1_high = 2 * precision * recall / (precision + recall + 1e-10)
        f1_mid = f1_high[:24]
        f1_low = f1_high[[0, 1, 5, 6, 7, 8, 9, 26, 27, 28, 29]]

        print('F1:', f1_high)
        print('Mean F1 high:', np.mean(f1_high))
        print('Mean F1 mid:', np.mean(f1_mid))
        print('Mean F1 low:', np.mean(f1_low))

    print()
    print('Edit high:', sum(edit_scores_high) / len(edit_scores_high))
    print('Edit mid:', sum(edit_scores_mid) / len(edit_scores_mid))
    print('Edit low:', sum(edit_scores_low) / len(edit_scores_low))
    return sum(edit_scores_high) / len(edit_scores_high)


def get_last_epoch(save_dir):
    max_epoch = -1
    for file_name in os.listdir(save_dir):
        if not file_name.startswith('optim_'):
            continue
        epoch = int(os.path.splitext(file_name)[0].split('optim_')[1])
        if epoch > max_epoch:
            max_epoch = epoch
    return max_epoch


def get_best_epoch_and_history(save_dir, criterion):
    data = load_json(os.path.join(save_dir, 'loss.json'))
    if criterion == 'edit':
        key = 'val_edit'
        best = max(data, key=lambda x: x[key])
    else:
        key = 'val'
        best = min(data, key=lambda x: x[key])
    return data, best['epoch'], best[key]


def get_datasets(args):
    classes = load_classes(os.path.join('data', args.dataset, 'events.txt'))

    dataset_len = EPOCH_NUM_FRAMES // (args.clip_len * args.stride)
    dataset_kwargs = {
        'crop_dim': args.crop_dim, 'stride': args.stride
    }

    print('Dataset size:', dataset_len)
    train_data = ActionSeqDataset(
        classes, os.path.join('data', args.dataset, 'train.json'),
        args.frame_dir, args.clip_len, dataset_len, is_eval=False, **dataset_kwargs)
    train_data.print_info()
    val_data = ActionSeqDataset(
        classes, os.path.join('data', args.dataset, 'val.json'),
        args.frame_dir, args.clip_len, dataset_len // 4, **dataset_kwargs)
    val_data.print_info()

    val_data_frames = None
    if args.criterion == 'edit':
        # Only perform edit evaluation during training if criterion is edit
        val_data_frames = ActionSeqVideoDataset(
            classes, os.path.join('data', args.dataset, 'val.json'),
            args.frame_dir, args.clip_len, crop_dim=args.crop_dim, stride=args.stride, overlap_len=0)

    return classes, train_data, val_data, None, val_data_frames


def load_from_save(
        args, model, optimizer, scaler, lr_scheduler
):
    assert args.save_dir is not None
    epoch = get_last_epoch(args.save_dir)

    print('Loading from epoch {}'.format(epoch))
    model.load(torch.load(os.path.join(
        args.save_dir, 'checkpoint_{:03d}.pt'.format(epoch))))

    if args.resume:
        opt_data = torch.load(os.path.join(
            args.save_dir, 'optim_{:03d}.pt'.format(epoch)))
        optimizer.load_state_dict(opt_data['optimizer_state_dict'])
        scaler.load_state_dict(opt_data['scaler_state_dict'])
        lr_scheduler.load_state_dict(opt_data['lr_state_dict'])

    losses, best_epoch, best_criterion = get_best_epoch_and_history(
        args.save_dir, args.criterion)
    return epoch, losses, best_epoch, best_criterion


def store_config(file_path, args, num_epochs, classes):
    config = {
        'dataset': args.dataset,
        'num_classes': len(classes),
        'feature_arch': args.feature_arch,
        'temporal_arch': args.temporal_arch,
        'clip_len': args.clip_len,
        'batch_size': args.batch_size,
        'crop_dim': args.crop_dim,
        'stride': args.stride,
        'num_epochs': num_epochs,
        'warm_up_epochs': args.warm_up_epochs,
        'learning_rate': args.learning_rate,
        'start_val_epoch': args.start_val_epoch,
        'gpu_parallel': args.gpu_parallel,
        'epoch_num_frames': EPOCH_NUM_FRAMES
    }
    store_json(file_path, config, pretty=True)


def get_num_train_workers(args):
    n = BASE_NUM_WORKERS * 2
    return min(os.cpu_count(), n)


def get_lr_scheduler(args, optimizer, num_steps_per_epoch):
    cosine_epochs = args.num_epochs - args.warm_up_epochs
    print('Using Linear Warmup ({}) + Cosine Annealing LR ({})'.format(
        args.warm_up_epochs, cosine_epochs))
    return args.num_epochs, ChainedScheduler([
        LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                 total_iters=args.warm_up_epochs * num_steps_per_epoch),
        CosineAnnealingLR(optimizer,
            num_steps_per_epoch * cosine_epochs)])


def main(args):
    if args.num_workers is not None:
        global BASE_NUM_WORKERS
        BASE_NUM_WORKERS = args.num_workers

    assert args.batch_size % args.acc_grad_iter == 0
    if args.start_val_epoch is None:
        args.start_val_epoch = args.num_epochs - BASE_NUM_VAL_EPOCHS
    if args.crop_dim <= 0:
        args.crop_dim = None

    classes, train_data, val_data, train_data_frames, val_data_frames = get_datasets(args)

    def worker_init_fn(id):
        random.seed(id + epoch * 100)
    loader_batch_size = args.batch_size // args.acc_grad_iter
    train_loader = DataLoader(
        train_data, shuffle=False, batch_size=loader_batch_size,
        pin_memory=True, num_workers=get_num_train_workers(args),
        prefetch_factor=1, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(
        val_data, shuffle=False, batch_size=loader_batch_size,
        pin_memory=True, num_workers=BASE_NUM_WORKERS,
        worker_init_fn=worker_init_fn)

    model = F3Tennis(len(classes) + 1, args.feature_arch, args.temporal_arch, clip_len=args.clip_len, step=args.stride,
                  multi_gpu=args.gpu_parallel)
    optimizer, scaler = model.get_optimizer({'lr': args.learning_rate})

    # Warmup schedule
    num_steps_per_epoch = len(train_loader) // args.acc_grad_iter
    num_epochs, lr_scheduler = get_lr_scheduler(
        args, optimizer, num_steps_per_epoch)

    losses = []
    best_epoch = None
    best_criterion = 0 if args.criterion == 'edit' else float('inf')
    best_loss, stop_criterion = float('inf'), 0

    epoch = 0
    if args.resume:
        epoch, losses, best_epoch, best_criterion = load_from_save(
            args, model, optimizer, scaler, lr_scheduler)
        epoch += 1

    # Write it to console
    store_config('/dev/stdout', args, num_epochs, classes)

    for epoch in range(epoch, num_epochs):
        train_loss = model.epoch(
            train_loader, optimizer, scaler, lr_scheduler=lr_scheduler, acc_grad_iter=args.acc_grad_iter)
        val_loss = model.epoch(val_loader, acc_grad_iter=args.acc_grad_iter)
        print('[Epoch {}] Train loss: {:0.5f} Val loss: {:0.5f}'.format(
            epoch, train_loss, val_loss))

        val_edit = 0
        if args.criterion == 'loss':
            if val_loss < best_criterion:
                best_criterion = val_loss
                best_epoch = epoch
                print('New best epoch!')
        elif args.criterion == 'edit':
            if epoch >= args.start_val_epoch:
                val_edit = evaluate(model, val_data_frames, classes, dataset_name=args.dataset)
                if args.criterion == 'edit' and val_edit > best_criterion:
                    best_criterion = val_edit
                    best_epoch = epoch
                    print('New best epoch!')
        else:
            print('Unknown criterion:', args.criterion)

        losses.append({
            'epoch': epoch, 'train': train_loss, 'val': val_loss, 'val_edit': val_edit})
        if args.save_dir is not None:
            os.makedirs(args.save_dir, exist_ok=True)
            store_json(os.path.join(args.save_dir, 'loss.json'), losses,
                        pretty=True)
            torch.save(
                model.state_dict(),
                os.path.join(args.save_dir,
                    'checkpoint_{:03d}.pt'.format(epoch)))
            clear_files(args.save_dir, r'optim_\d+\.pt')
            torch.save(
                {'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'lr_state_dict': lr_scheduler.state_dict()},
                os.path.join(args.save_dir,
                                'optim_{:03d}.pt'.format(epoch)))
            store_config(os.path.join(args.save_dir, 'config.json'),
                            args, num_epochs, classes)

    print('Best epoch: {}\n'.format(best_epoch))

    if args.save_dir is not None:
        model.load(torch.load(os.path.join(
            args.save_dir, 'checkpoint_{:03d}.pt'.format(best_epoch))))

        # Evaluate on hold out splits
        eval_splits = ['test']
        for split in eval_splits:
            split_path = os.path.join(
                'data', args.dataset, '{}.json'.format(split))
            if os.path.exists(split_path):
                split_data = ActionSeqVideoDataset(classes, split_path, args.frame_dir, args.clip_len, overlap_len=0,
                                                   crop_dim=args.crop_dim, stride=args.stride)
                split_data.print_info()
                evaluate(model, split_data, classes, dataset_name=args.dataset)


if __name__ == '__main__':
    main(get_args())
