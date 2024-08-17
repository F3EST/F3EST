# F3Tennis Dataset

Obtain the videos from Youtube, following the ids and naming conventions in data/finetennis/videos.csv. Make sure to download the correct frame rate and quality settings. We use 720P for all of the videos.

Run `python3 download_videos.py` to download match videos from Youtube. However, some videos cannot by downloaded through the python script. You can manually download those videos.

Run `python3 extract_frames.py` to extract frames from videos and save to a directory named `vid_frames_224/`.
