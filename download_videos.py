from pytube import YouTube
import pandas as pd
import os

videos = pd.read_csv('data/f3tennis/videos.csv')
links = list(videos.yt_id)
file_names = list(videos.video_name)

# Note that there are some videos cannot be downloaded directly through the python code
# You can manually download those...
for i in range(len(links)):
    name = file_names[i] + '.mp4'
    try:
        yt = YouTube('https://www.youtube.com/watch?v=%s' % links[i], use_oauth=False, allow_oauth_cache=True)

        # Showing details
        print('Video number:', i)
        print('Title: ', yt.title)
        print('Match_id: ', file_names[i])
        print('Number of views: ', yt.views)
        print('Length of video: ', yt.length)
        print('Rating of video: ', yt.rating)
        # Getting the highest resolution possible
        ys = yt.streams.get_highest_resolution()

        # Starting download
        print('Downloading...')
        ys.download(output_path='./videos/', filename=name, )
        print('Download completed!!')
        print()
    except:
        try:
            yt = YouTube('https://youtu.be/%s' % links[i], use_oauth=False, allow_oauth_cache=True)

            # Showing details
            print('Video number:', i)
            print('Title: ', yt.title)
            print('Match_id: ', file_names[i])
            print('Number of views: ', yt.views)
            print('Length of video: ', yt.length)
            print('Rating of video: ', yt.rating)
            # Getting the highest resolution possible
            ys = yt.streams.get_highest_resolution()

            # Starting download
            print('Downloading...')
            ys.download(output_path='./videos/', filename=name, )
            print('Download completed!!')
            print()
        except:
            print('FAIL...')