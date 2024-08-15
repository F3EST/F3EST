# Analyzing Fast, Frequent, and Fine-grained Events for Sports Strategy Analytics
## Overview
Analyzing Fast, Frequent, and Fine-grained ($F^3$) event sequences in sports videos is crucial for advanced sports analytics. The substantial labor required for labeling these videos constitutes a significant scalability barrier. Towards addressing this, we have built $FineTennis$, a comprehensive fine-grained tennis video dataset designed to facilitate deep, strategic analysis. This dataset, featuring over 1,000 event types and multi-level granularity, enables automated analysis of complex interactions and strategies required in the literature. However,  existing video understanding methods showed poor performance on $FineTennis$, highlighting a gap in current methods' ability to handle such detailed data. In response, we introduce $F^3EST$, an end-to-end model specifically designed to locate and recognize $F^3$ event sequences. This model integrates visual features with strategic causality across events. Our evaluations demonstrate that $F^3EST$ significantly outperforms existing approaches. Various applications have demonstrated the model's capability to automate the analysis of actions and tactics with appropriate levels of granularity. 

## Environment
The code is tested in Linux (Ubuntu 22.04) with the dependency versions in requirements.txt.

## Dataset
Refer to the READMEs in the [data](https://github.com/F3EST/F3EST/tree/main/data) directory for pre-processing and setup instructions.

## Basic usage
To train a model, use `python3 train_f3est.py <dataset_name> <frame_dir> -s <save_dir> -m <model_arch>`.

* `<dataset_name>`: supports finetennis, badmintonDB, finediving, finegym
* `<frame_dir>`: path to the extracted frames
* `<save_dir>`: path to save logs, checkpoints, and inference results
* `<model_arch>`: feature extractor architecture (e.g., rny002_gsm)

Training will produce checkpoints, predictions for the `val` split, and predictions for the `test` split on the best validation epoch.

### Trained models
Models and configurations can be found in [f3est-model](https://github.com/F3EST/F3EST/tree/main/f3est-model). Place the checkpoint file and config.json file in the same directory.

To perform inference with an already trained model, use `python3 test_f3est.py <model_dir> <frame_dir> -s <split> --save`. This will output results for 2 evaluation metrics (F1 score and edit score).

## Data format
Each dataset has plaintext files that contain the list of event types `events.txt` and sub-class elements: `elements.txt`

This is a list of the event names, one per line: `{split}.json`

This file contains entries for each video and its contained events.
```
[
    {
        "video": VIDEO_ID,
        "num_frames": 518,                 // Video length
        "events": [
            {
                "frame": 100,               // Frame
                "label": EVENT_NAME,        // Event type
            },
            ...
        ],
        "fps": 25,
        "width": 1280,      // Metadata about the source video
        "height": 720
    },
    ...
]
```
**Frame directory**

We assume pre-extracted frames, that have been resized to 224 pixels high or similar. The organization of the frames is expected to be <frame_dir>/<video_id>/<frame_number>.jpg. For example,
```
video1/
├─ 000000.jpg
├─ 000001.jpg
├─ 000002.jpg
├─ ...
video2/
├─ 000000.jpg
├─ ...
```
Similar format applies to the frames containing objects of interest.








