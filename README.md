# Analyzing Fast, Frequent, and Fine-grained Event Sequences from Videos
## Overview
Analyzing Fast, Frequent, and Fine-grained ($F^3$) events presents a significant challenge in video analytics and multi-modal LLMs. Current methods, despite their effectiveness on public benchmarks, fail to accurately detect and identify $F^3$ events due to issues like motion blur and subtle visual discrepancies. To address this, we focus on $F^3Tennis$, a dataset built on tennis video for $F^3$ event detection. $F^3Tennis$ is distinguished by its large-scale and detailed information, consisting of over 1,000 event types and multi-level granularity. However, existing approaches perform poorly on $F^3Tennis$, highlighting a gap in current methods' ability to handle such detailed data. Consequently, we propose $F^3EST$, an end-to-end model that efficiently locates and recognizes $F^3$ event sequences by integrating visual features with event causality, thereby improving the interpretation of rapid, fine-grained events. Our evaluations show that $F^3EST$ outperforms existing models. Moreover, using tennis as a case study, we demonstrate the potential of $F^3$ event sequences for advanced automated strategic analytics.

## Environment
The code is tested in Linux (Ubuntu 22.04) with the dependency versions in requirements.txt.

## Dataset
Refer to the READMEs in the [data](https://github.com/F3EST/F3EST/tree/main/data) directory for pre-processing and setup instructions.

## Basic usage
To train baseline models, use `python3 train_baselines.py <dataset_name> <frame_dir> -s <save_dir> -m <model_arch> -t <head_arch>`.
To train the $F^3EST$ model, use `python3 train_f3est.py <dataset_name> <frame_dir> -s <save_dir> -m <model_arch>`.

* `<dataset_name>`: supports f3tennis, shuttleset, finediving, finegym
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








