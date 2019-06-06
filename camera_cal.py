#!/usr/bin/env python3
import argparse
import glob
import json

import cv2
import numpy as np

def calibrate_camera(images, nx, ny):
    shape = None
    objpoints = []
    imgpoints = []
    objp = np.zeros((nx*ny, 3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
    for path in images:
        print(path)
        img = cv2.imread(path)
        if shape is None:
            shape = img.shape[1::-1]
        elif img.shape[::-1] != shape:
            img = cv2.resize(img, shape)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(img, (nx, ny), None)
        if found:
            objpoints.append(objp)
            imgpoints.append(corners)
        else:
            print('NOT FOUND')
    _, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, shape,
                                             None, None)
    return mtx, dist

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--cal-images', default='camera_cal')
    parser.add_argument('-x', '--nx', type=int, default=9)
    parser.add_argument('-y', '--ny', type=int, default=6)
    parser.add_argument('output_file', nargs='?', default='camera.json')
    args = parser.parse_args()

    images = glob.glob(args.cal_images+'/*')
    mtx, dist = calibrate_camera(images, args.nx, args.ny)
    with open(args.output_file, 'w') as f:
        camera = {'mtx': mtx.tolist(), 'dist': dist.tolist()}
        json.dump(camera, f)

if __name__ == '__main__':
    main()
