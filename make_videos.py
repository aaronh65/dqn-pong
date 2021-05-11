import cv2
import os, argparse
from pathlib import Path

def main(args):

    runs = sorted(list(Path(args.target_dir).glob('*')))
    for run in runs:
        videos = run / 'videos'
        videos.mkdir(exist_ok=True)
        episodes = (run / 'images').glob('*')

        width, height = None, None
        for episode in sorted(list(episodes)):
                
                episode_name = episode.stem


                save_path = str(videos / f'{episode_name}.mp4')

                if width is None or height is None:
                    fname = sorted(list(episode.glob('*')))[0]
                    im = cv2.imread(str(fname))
                    height, width, channels = im.shape
                    print(height,width,channels)

                if width % 2 == 1 or height % 2 == 1:
                    cmd = f'ffmpeg -y -r {args.fps} -s {width}x{height} -f image2 -i {episode}/%06d.png -pix_fmt yuv420p -vf \"pad=ceil(iw/2)*2:ceil(ih/2)*2\" {save_path}'
                else:
                    cmd = f'ffmpeg -y -r {args.fps} -s {width}x{height} -f image2 -i {episode}/%06d.png -pix_fmt yuv420p {save_path}'
                print(cmd)

                # make video
                #cmd = f'ffmpeg -y -r 2 -s 1627x256 -f image2 -i {input_dir}/%06d.png -pix_fmt yuv420p -vf \"pad=ceil(iw/2)*2:ceil(ih/2)*2\" {save_path}'
                os.system(cmd)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_dir', type=str, required=True)
    parser.add_argument('--fps', type=int, default=60)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)
