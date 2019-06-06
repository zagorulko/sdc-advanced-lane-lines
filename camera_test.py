#!/usr/bin/env python3
import argparse
import json

import cv2
import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--camera', default='camera.json')
    parser.add_argument('-p', '--plot-output')
    parser.add_argument('image')
    args = parser.parse_args()

    with open(args.camera, 'r') as f:
        camera = json.load(f)
        mtx = np.array(camera['mtx'], np.float64)
        dist = np.array(camera['dist'], np.float64)

    img = mpimg.imread(args.image)
    undistorted_img = cv2.undistort(img, mtx, dist, None, mtx)

    if args.plot_output:
        matplotlib.rcParams.update({'font.size': 5})

    fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True)
    ax[0].imshow(img)
    ax[0].set_title('Original image')
    ax[1].imshow(undistorted_img)
    ax[1].set_title('Undistorted image')

    if args.plot_output:
        fig.savefig(args.save_plot, bbox_inches='tight', dpi=450)
    else:
        plt.show()

if __name__ == '__main__':
    main()
