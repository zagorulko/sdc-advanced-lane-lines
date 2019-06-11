#!/usr/bin/env python3
import argparse
import glob
import json
import os
import queue
import threading
import time
from collections import defaultdict
from copy import deepcopy

import cv2
import numpy as np

class Camera:
    """Contains logic dependent on camera and its position, i.e. undistortion
    and perspective transformations."""

    def __init__(self):
        self.mtx = None
        self.dist = None
        self.xm_per_pix = 3.7/700
        self.ym_per_pix = 30/720

    def load_calibration(self, filename):
        """Loads camera calibration data from JSON file."""
        with open(filename, 'r') as f:
            data = json.load(f)
        self.mtx = np.array(data['mtx'], np.float64)
        self.dist = np.array(data['dist'], np.float64)

    def undistort(self, img):
        """Transforms an image to compensate for lens distortion."""
        norm_size = (1280, 720)
        if (img.shape[1], img.shape[0]) != norm_size:
            img = cv2.resize(img, norm_size)
        if self.mtx is None:
            return img
        else:
            return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)

    def _get_warp_matrix(self, width, height):
        src = np.float32([
            [width/2-55,   height/2+100],
            [width/6-10,   height],
            [width*5/6+60, height],
            [width/2+55,   height/2+100]
        ])
        dst = np.float32([
            [width/4,   0],
            [width/4,   height],
            [width*3/4, height],
            [width*3/4, 0]
        ])
        M = cv2.getPerspectiveTransform(src, dst)
        return M

    def warp(self, img):
        """Warps an image to bird's eye perspective."""
        M = self._get_warp_matrix(img.shape[1], img.shape[0])
        return cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]),
                                   flags=cv2.INTER_NEAREST)

    def unwarp(self, img):
        """Warps an image back from bird's eye perspective."""
        M = self._get_warp_matrix(img.shape[1], img.shape[0])
        M = np.linalg.inv(M)
        return cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]),
                                   flags=cv2.INTER_NEAREST)

def grad_thresh(img, orient='x', ksize=3, thresh=(0, 255)):
    """Simple Sobel threshold."""
    orient = (1, 0) if orient == 'y' else (0, 1)
    sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, *orient))
    return cv2.inRange(sobel, *thresh)

def mag_thresh(img, ksize=3, thresh=(0, 255)):
    """Gradient magnitutde threshold."""
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    mag = np.sqrt(sobelx**2 + sobely**2)
    mag = np.uint8(255*mag/np.max(mag))
    return cv2.inRange(mag, *thresh)

def dir_thresh(img, ksize=3, thresh=(0, np.pi/2)):
    """Gradient direction threshold."""
    sobelx = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0))
    sobely = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1))
    grad = np.arctan2(sobelx, sobely)
    return cv2.inRange(grad, *thresh)

def find_lane_pixels(mask_warped, left_x_base=None, right_x_base=None):
    """Finds left and right lane points on `mask_warped` using sliding window
    starting at `left_x_base` and `right_x_base` if set, and histogram peaks
    otherwise.

    Returns (left_pt, right_pt, lane_img).
    """
    nwindows = 28
    margin = 50
    # Minimum amount of hot pixels before window is considered non-empty
    minpix = 60
    # Search will stop if window gets this far from mask's horizontal border
    border = -10
    # Amplifier of direction calculated from average of x coordinates
    dir_gain = 2.0
    # Maximum number of consecutive empty windows before the search is stopped
    max_emptiness = 9
    # Minimum distance between line bases
    min_spacing = 220

    if left_x_base is None or right_x_base is None:
        hist = np.sum(mask_warped[mask_warped.shape[0] // 3 * 2:, :], axis=0)
        midpoint = np.int(hist.shape[0] // 2)
    if left_x_base is None:
        left_x_base = np.argmax(hist[:midpoint])
    if right_x_base is None:
        right_x_base = np.argmax(hist[midpoint:]) + midpoint

    window_height = np.int(mask_warped.shape[0] // nwindows)
    nonzero = mask_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    lane_img = np.dstack((mask_warped,) * 3)

    def find_single(x_base):
        x_current = x_base
        lane_inds = []
        emptiness = 0
        for window in range(nwindows):
            win_y_low = mask_warped.shape[0] - (window+1)*window_height
            win_y_high = mask_warped.shape[0] - window*window_height
            win_x_low = x_current - margin
            win_x_high = x_current + margin
            cv2.rectangle(lane_img, (win_x_low, win_y_low),
                          (win_x_high, win_y_high), (0, 255, 0), 2)
            good_inds = \
                ((nonzerox >= win_x_low) & (nonzerox < win_x_high) &
                 (nonzeroy >= win_y_low) & (nonzeroy < win_y_high)).nonzero()[0]
            if good_inds.size < minpix:
                emptiness += 1
            else:
                emptiness = 0
                # Note that we add indices only from non-empty windows. This
                # helps preserve line form if we see only part of it.
                lane_inds.append(good_inds)
                # Calculate next widow direction
                x_mean = int(np.mean(nonzerox[good_inds]))
                delta = int((x_mean - x_current) * dir_gain)
                x_current += delta
            # Stop if we reached horizontal border
            if win_x_low < border or win_x_high > mask_warped.shape[1]-border:
                break
            # Stop if we had too many consecutive empty windows
            if emptiness >= max_emptiness:
                break
        if not lane_inds:
            return
        lane_inds = np.concatenate(lane_inds)
        pt = np.concatenate([[nonzerox[lane_inds]], [nonzeroy[lane_inds]]]).T
        return pt

    # If line bases are too close, inspect only one line
    left_pt = find_single(left_x_base)
    if right_x_base - left_x_base >= min_spacing:
        right_pt = find_single(right_x_base)
    else:
        right_pt = None
    return left_pt, right_pt, lane_img

def search_around_poly(mask_warped, left_coef, right_coef, recheck=False):
    """Looks for lane line points in the vicinity of polynomials defined by
    `left_coef` and `right_coef`."""
    margin = 60
    recheck_thresh = 120000

    nonzero = mask_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_x = np.polyval(left_coef, nonzeroy)
    right_x = np.polyval(right_coef, nonzeroy)
    left_idx = (nonzerox > left_x-margin) & (nonzerox < left_x+margin)
    right_idx = (nonzerox > right_x-margin) & (nonzerox < right_x+margin)

    # If there are not too many points, consider them all. This approach gives
    # smooth results when there isn't much noise.
    if (not recheck and
            np.mean([left_idx.shape[0], right_idx.shape[0]]) < recheck_thresh):
        left_pt = np.concatenate([[nonzerox[left_idx]], [nonzeroy[left_idx]]]).T
        right_pt = np.concatenate([[nonzerox[right_idx]], [nonzeroy[right_idx]]]).T
        return left_pt, right_pt, None

    # Mask out any other pixels
    search_mask = np.zeros_like(mask_warped)
    search_mask[nonzeroy[left_idx], nonzerox[left_idx]] = 255
    search_mask[nonzeroy[right_idx], nonzerox[right_idx]] = 255
    search_mask = mask_warped & search_mask

    # Perform sliding window search around the polynomial mask. This approach
    # gives better results with noisy masks.
    left_pt, right_pt, lane_img = find_lane_pixels(search_mask)
    return left_pt, right_pt, lane_img

class Line:
    """Single lane line in the context of an image stream."""

    def __init__(self, camera, window=14, max_errors=6, new_fit_weight=0.,
                 max_base_reuse=5):
        """
        `new_fit_weight` is additional weight for newly fit line coefficients
        in the average with recent fits. It can be used to let the line adjust
        faster while retaining a bigger window.
        `max_base_reuse` is the maximum number of times get_x_base() can return
        result before next good fit.
        """
        self.camera = camera
        self.window = window
        self.max_errors = max_errors
        self.new_fit_weight = new_fit_weight
        self.max_base_reuse = max_base_reuse

        # These properties persist through frame drops
        self.R_curve = 0
        self.x_base = None
        self.x_base_age = 0

        self.reset()

    def reset(self):
        """Resets line state."""
        self.err_count = 0
        self.recent_coef = np.full((self.window, 3), np.nan)
        self.diffs = np.array([np.inf, np.inf, np.inf], np.float32)
        self.best_coef = None
        self.best_x = None
        self.plot_y = None
        self.pts = None
        self.mask_size = None
        self.center_dist = np.nan
        self.has_fallback = False

    def empty(self):
        """Indicates whether the line has coordinates."""
        return self.best_coef is None

    def fit(self, pts, plot_y, mask_size):
        """Adjusts line with given points."""
        if pts is None or not pts.size:
            self._add_error()
            return

        new_coef = np.polyfit(pts[:, 1], pts[:, 0], 2)

        # Discard fit if it deviates a lot from the best
        if self.best_coef is not None:
            diffs = np.absolute(new_coef - self.best_coef)
            if diffs[0] > 1 or diffs[1] > 80:
                self._add_error()
                return
            self.diffs = diffs

        # Discard fit if it is too parabolic
        if abs(new_coef[0]) > 0.03 or abs(new_coef[1]) > 20:
            self._add_error()
            return

        self.pts = pts
        self.plot_y = plot_y
        self.mask_size = mask_size

        self.recent_coef = np.roll(self.recent_coef, 1, axis=0)
        self.recent_coef[0] = new_coef
        self._update_best()

    def _update_best(self):
        """Updates properties derived from recent_coef."""
        cur_coef = self.recent_coef[0]
        mean_coef = np.nanmean(self.recent_coef, axis=0)
        self.best_coef = np.average([cur_coef, mean_coef],
                                    weights=[self.new_fit_weight, 1.0],
                                    axis=0)
        self.best_x = np.polyval(self.best_coef, self.plot_y)

        # Calculate radius of curvature
        mx, my = self.camera.xm_per_pix, self.camera.ym_per_pix
        A = self.best_coef[0] * (mx / (my ** 2))
        B = self.best_coef[1] * (mx / my)
        y = np.max(self.plot_y) * my
        R_curve = (1 + (2*A*y+B)**2)**(3/2) / np.absolute(2*A)
        if 40 <= R_curve <= 9000:
            self.R_curve = R_curve

        # Calculate distance from the line to the vehicle center
        center_x = self.mask_size[0] // 2
        self.center_dist = np.absolute(self.best_x[-1] - center_x) * mx

    def _add_error(self):
        """Either takes error into account or resets the state if there are too
        many errors.

        Returns True if the line is not empty.
        """
        if not self.empty():
            self.err_count += 1
            if self.err_count < self.max_errors:
                return True
            else:
                self.reset()

    def drop_current_frame(self):
        """Replaces the latest fit with a copy of the previous or resets line
        state."""
        if self._add_error():
            if not self.has_fallback:
                self.reset()
            else:
                # Note that we do not shift history here
                self.recent_coef[0] = self.recent_coef[1]
                self.diffs = np.array([0, 0, 0], np.float32)
                self._update_best()

    def commit(self):
        """Hint that current fit is a good one."""
        assert not self.empty()
        self.err_count = 0
        self.has_fallback = True
        if not np.isnan(self.best_x[-1]):
            self.x_base = int(self.best_x[-1])
            self.x_base_age = 0

    def get_x_base(self):
        """Gives x coordinate of the last good line base."""
        if self.x_base is not None and self.x_base_age < self.max_base_reuse:
            self.x_base_age += 1
            return self.x_base

def lines_are_valid(left_line, right_line):
    """Cross-line sanity check."""
    if left_line.empty() or right_line.empty():
        return
    # Minimum distance between lines
    dist_min = 20.0
    if np.min(right_line.best_x - left_line.best_x) < dist_min:
        return
    # Lane width
    lane_width_min_m = 2
    lane_width = np.absolute(right_line.best_x[-1] - left_line.best_x[-1])
    if lane_width * left_line.camera.xm_per_pix < lane_width_min_m:
        return
    # Are lines roughly parallel?
    max_a_diff = 0.02
    max_b_diff = 3
    if ((abs(right_line.best_coef[0] - left_line.best_coef[0]) > max_a_diff) or
            (abs(right_line.best_coef[1] - left_line.best_coef[1]) > max_b_diff)):
        return
    return True

class LaneFinder:
    """Supposed to locate lane lines on sequences of images."""

    def __init__(self, camera, report=None, trace=False, is_curvy=False,
                 show_bad_lines=False):
        self.camera = camera
        self.report = report
        self.trace = trace
        self.is_curvy = is_curvy
        self.show_bad_lines = show_bad_lines
        line_kwargs = {}
        if is_curvy:
            line_kwargs['new_fit_weight'] = 0.5
            line_kwargs['window'] = 7
            line_kwargs['max_errors'] = 4
            line_kwargs['max_base_reuse'] = 10
        self.left_line = Line(camera, **line_kwargs)
        self.right_line = Line(camera, **line_kwargs)
        if not show_bad_lines:
            # These lines do not take part in any calculations. They are copies
            # of the latest good pair of lines for visualization purposes.
            self.good_left_line = deepcopy(self.left_line)
            self.good_right_line = deepcopy(self.right_line)
        self.R_curve = 0
        self.v_offset = 0

    def _select_pixels(self, bgr):
        """Creates a mask of pixels suspected in being part of lane lines."""
        mask = np.zeros(bgr.shape[:2], np.uint8)

        # Bilateral filter reduces noise while keeping edges reasonably sharp
        bgr = cv2.bilateralFilter(bgr, 5, 10, 50)

        # RGB
        _, _, R = cv2.split(bgr)

        # HLS
        hls = cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS)
        H, L, S = cv2.split(hls)
        S_eq = cv2.equalizeHist(S)

        # CIELAB
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        lab_l, lab_a, lab_b = cv2.split(lab)
        lab_b_eq = cv2.equalizeHist(lab_b)

        # Calculate histogram of HLS-L channel
        hist_gray = L
        hist_gray = hist_gray[hist_gray.shape[0]//3*2:]
        hist_sm = cv2.calcHist([hist_gray], [0], None, [5], [0, 256])
        hist_sm /= hist_gray.shape[0] * hist_gray.shape[1]
        hist_big = cv2.calcHist([hist_gray], [0], None, [10], [0, 256])
        hist_big /= hist_gray.shape[0] * hist_gray.shape[1]

        # Detect overexposed images
        too_bright = np.argmax(hist_sm) == hist_sm.shape[0]-1

        # Estimate the overall lightness
        light_level = np.sum(np.multiply(hist_big.reshape(-1),
                                         np.exp(np.arange(hist_big.shape[0]))))
        much_light = light_level >= 400

        # White (HLS)
        if not too_bright and not much_light:
            m = cv2.inRange(L, 200, 255)
            mask = cv2.bitwise_or(mask, m)

        # White (Lab)
        if not too_bright and much_light:
            m = cv2.inRange(lab_l, 230, 255)
            mask = cv2.bitwise_or(mask, m)

        # Yellow (HLS)
        m = cv2.inRange(H, 20, 40) & (cv2.inRange(S_eq, 120, 255) |
                                      cv2.inRange(L, 10, 80))
        mask = cv2.bitwise_or(mask, m)

        # Yellow (Lab)
        m = cv2.inRange(lab_b_eq, 240, 255)
        mask = cv2.bitwise_or(mask, m)

        # Red (RGB)
        if not too_bright and not much_light:
            m = cv2.inRange(R, 230, 255)
            mask = cv2.bitwise_or(mask, m)

        # Saturation (HLS)
        if not too_bright:
            m = cv2.inRange(S_eq, 250, 255)
            mask = cv2.bitwise_or(mask, m)

        # Gradient
        if too_bright:
            mag_mask = mag_thresh(S_eq, ksize=9, thresh=(20, 255))
        else:
            mag_mask = mag_thresh(S_eq, ksize=9, thresh=(32, 255))
        dir_mask = dir_thresh(S_eq, ksize=15, thresh=(0.7, 1.3))
        mask = cv2.bitwise_or(mask, mag_mask & dir_mask)

        return mask

    def _fit_lines(self, left_pt, right_pt, plot_y, mask_size):
        """Adjusts lines with given points."""
        self.left_line.fit(left_pt, plot_y, mask_size)
        self.right_line.fit(right_pt, plot_y, mask_size)

    def _validate_lines(self):
        """Checks if lines are valid and adjust accordingly."""
        if lines_are_valid(self.left_line, self.right_line):
            self.left_line.commit()
            self.right_line.commit()
            if not self.show_bad_lines:
                self.good_left_line = deepcopy(self.left_line)
                self.good_right_line = deepcopy(self.right_line)
            return True

        left_diffs = abs(self.left_line.diffs[0] * self.left_line.diffs[1])
        right_diffs = abs(self.right_line.diffs[0] * self.right_line.diffs[2])

        if left_diffs >= right_diffs:
            self.left_line.drop_current_frame()
        if right_diffs >= left_diffs:
            self.right_line.drop_current_frame()

    def _prepare_job(self, img):
        """Non-stateful part of the image processing pipeline.

        `img` is expected to be BGR.
        Thread-safe. See preprocess_video() for more on that.
        """
        # Account for lens distortions
        img = self.camera.undistort(img)
        if self.report:
            self.report('undistorted', img)

        # Compute lane pixel mask
        mask = self._select_pixels(img)
        if self.report:
            self.report('mask', mask)

        # Warp the mask to bird's-eye view
        mask_warped = self.camera.warp(mask)
        if self.report:
            self.report('mask_warped', mask_warped)

        return img, mask, mask_warped

    def _process_job(self, job):
        """Stateful part of the image processing pipeline.

        `job` should be the result of _prepare_job().
        Returns a BGR image.
        """
        img, mask, mask_warped = job

        mask_size = (mask_warped.shape[1], mask_warped.shape[0])
        plot_y = np.linspace(0, mask_warped.shape[0]-1, mask_warped.shape[0])
        lane_img = None

        # If we have good lines from before, look for new ones nearby
        if self._validate_lines():
            left_pt, right_pt, lane_img = search_around_poly(
                mask_warped, self.left_line.best_coef, self.right_line.best_coef,
                recheck=self.is_curvy)
            self._fit_lines(left_pt, right_pt, plot_y, mask_size)

        # Resort to sliding window search if needed
        if not self._validate_lines():
            left_x_base = self.left_line.get_x_base()
            right_x_base = self.right_line.get_x_base()
            left_pt, right_pt, lane_img = find_lane_pixels(mask_warped,
                                                           left_x_base,
                                                           right_x_base)
            self._fit_lines(left_pt, right_pt, plot_y, mask_size)

        # Estimate vehicle position
        self.v_offset = 0
        if not self.left_line.empty() and not self.right_line.empty():
            self.v_offset = (self.left_line.center_dist -
                             self.right_line.center_dist)

        return self._render_result(img, mask_warped, lane_img=lane_img)

    def _render_result(self, img, mask_warped, lane_img=None):
        """Visualizes lane lines."""
        if self.report or self.trace:
            img_warped = self.camera.warp(img)
        if self.report:
            self.report('img_warped', img_warped)

        # Highlight lane pixels and polynomials
        plot_y = np.linspace(0, mask_warped.shape[0]-1, mask_warped.shape[0])
        if self.report or self.trace:
            if lane_img is None:
                lane_img = np.dstack((mask_warped,)*3)
            poly_img = np.dstack((mask_warped,)*3)
            if not self.left_line.empty():
                pts = self.left_line.pts
                lane_img[pts[:, 1], pts[:, 0]] = (0, 0, 255)
                pts = np.dstack([self.left_line.best_x, plot_y]).astype(np.int32)
                cv2.polylines(poly_img, pts, False, (0, 255, 255), 8)
            if not self.right_line.empty():
                pts = self.right_line.pts
                lane_img[pts[:, 1], pts[:, 0]] = (255, 0, 0)
                pts = np.dstack([self.right_line.best_x, plot_y]).astype(np.int32)
                cv2.polylines(poly_img, pts, False, (0, 255, 255), 8)
        if self.report:
            self.report('lane_pixels', lane_img)

        if (self.show_bad_lines
                or (self.good_left_line.empty())
                or (not self.left_line.empty() and self.right_line.empty())):
            left_line = self.left_line
            right_line = self.right_line
        else:
            left_line = self.good_left_line
            right_line = self.good_right_line

        # Draw area
        area_img = np.dstack((np.zeros_like(mask_warped, np.uint8),) * 3)
        polygon_pts = []
        if not left_line.empty():
            polygon_pts += [[np.vstack([left_line.best_x, plot_y]).T]]
        if not right_line.empty():
            polygon_pts += [[np.flipud(np.vstack([right_line.best_x, plot_y]).T)]]
        if right_line.empty():
            polygon_pts += [[np.array([[area_img.shape[1], area_img.shape[0]]])]]
        if left_line.empty():
            polygon_pts += [[np.array([[0, 0]])]]
        polygon_pts = np.hstack(polygon_pts)
        if self.show_bad_lines and not lines_are_valid(left_line, right_line):
            area_color = (0, 0, 255)
        else:
            area_color = (0, 255, 0)
        cv2.fillPoly(area_img, np.int_([polygon_pts]), area_color)
        fg = self.camera.unwarp(area_img)
        result = cv2.addWeighted(img, 1, fg, 0.3, 0)

        # Draw text
        text = [
            'Radius of curvature (L): {:.2f} m'.format(left_line.R_curve),
            'Radius of curvature (R): {:.2f} m'.format(right_line.R_curve),
            'Vehicle is {:.2f} m {} of center'.format(abs(self.v_offset),
                'left' if self.v_offset <= 0 else 'right'),
        ]
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        thickness = 2
        top_offset = 50
        for line in text:
            _, height = cv2.getTextSize(line, font_face, font_scale, thickness)
            top_offset += height
            cv2.putText(result, line, (40, top_offset), font_face,
                        font_scale, (0, 0, 0), thickness+3)
            cv2.putText(result, line, (40, top_offset), font_face,
                        font_scale, (255, 255, 255), thickness)
            top_offset += 40

        # Draw extra windows on top
        if self.trace:
            sub_h, sub_w = img.shape[0] // 3, img.shape[1] // 3
            sub_s = (sub_w, sub_h)
            mix = np.zeros((img.shape[0] + sub_h,)+img.shape[1:], np.uint8)
            mix[:sub_h, :sub_w] = cv2.resize(img_warped, sub_s)
            mix[:sub_h:, sub_w:2*sub_w] = cv2.resize(lane_img, sub_s)
            mix[:sub_h:, 2*sub_w:3*sub_w] = cv2.resize(poly_img, sub_s)
            mix[sub_h:] = result
            result = mix

        return result

    def process_image(self, img):
        """The pipeline.

        `img` is expected to be BGR.
        Returns BGR image.
        """
        return self._process_job(self._prepare_job(img))

    def process_video(self, input_file, output_file, nthreads=2,
                      start=0, end=None):
        """Processes video utilizing multiple threads for preprocessing frames.

        Implementation is somewhat hacky, but gives a performance of about
        1.5X-2X compared to single thread flow, thus offsetting image gradient
        computation overhead.
        """
        from moviepy.editor import VideoClip, VideoFileClip

        input_queue = queue.Queue(maxsize=250)
        job_events = defaultdict(threading.Event)
        job_data = {}

        input_clip = VideoFileClip(input_file)
        input_clip = input_clip.subclip(start, end)
        fps, duration = input_clip.fps, input_clip.duration
        nframes = int(fps * duration)

        def input_worker():
            while True:
                i, img = input_queue.get()
                if img is None:
                    input_queue.put((None, None))
                    break
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                job_data[i] = self._prepare_job(img)
                job_events[i].set()

        def output_worker():
            i = 0
            def make_frame(img):
                nonlocal i
                if i >= nframes:
                    i = nframes - 1
                job_events[i].wait()
                img = self._process_job(job_data[i])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                i += 1
                return img
            clip = VideoFileClip(input_file)
            clip = clip.subclip(start, end)
            output_clip = clip.fl_image(make_frame)
            output_clip.write_videofile(output_file, audio=False)
            output_clip.close()

        threads = []
        for i in range(nthreads):
            t = threading.Thread(target=input_worker)
            t.start()
            threads.append(t)
        t = threading.Thread(target=output_worker)
        t.start()
        threads.append(t)

        try:
            for i, frame in enumerate(input_clip.iter_frames()):
                input_queue.put((i, frame))
            input_queue.put((None, None))
            for t in threads:
                t.join()
        except KeyboardInterrupt:
            os._exit(0)

def write_report(output_dir):
    def f(label, img):
        path = os.path.join(output_dir, label+'.jpg')
        print('step: '+path)
        cv2.imwrite(path, img)
    return f

def iter_files(input_files, output):
    """Yields tuples (input_file, output_file, is_static_image)."""
    for path in input_files:
        if os.path.isdir(path):
            yield from iter_files(glob.glob(path+'/*'), output)
            continue
        if output and os.path.isdir(output):
            output_path = os.path.join(output, os.path.basename(path))
        else:
            output_path = output
        is_static_image = os.path.splitext(path)[1] in {'.jpg'}
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        yield path, output_path, is_static_image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--camera', default='camera.json')
    parser.add_argument('-o', '--output')
    parser.add_argument('-t', '--trace', action='store_true')
    parser.add_argument('-s', '--video-start', type=int, default=0)
    parser.add_argument('-e', '--video-end', type=int)
    parser.add_argument('-r', '--report-dir')
    parser.add_argument('-C', '--curvy-road', action='store_true',
                        help='Use hyperparameters better tuned for curvy roads')
    parser.add_argument('input', nargs='+')
    args = parser.parse_args()

    camera = Camera()
    if args.camera:
        camera.load_calibration(args.camera)

    report = write_report(args.report_dir) if args.report_dir else None

    for input_file, output_file, is_image in iter_files(args.input, args.output):
        print('-> '+input_file)
        if output_file:
            print('output: '+output_file)
        if is_image:
            lane_finder = LaneFinder(camera, report=report, trace=args.trace,
                                     is_curvy=args.curvy_road)
            img = cv2.imread(input_file)
            output_img = lane_finder.process_image(img)
            if output_file:
                cv2.imwrite(output_file, output_img)
            report = None
        else:
            lane_finder = LaneFinder(camera, trace=args.trace,
                                     is_curvy=args.curvy_road)
            started = time.time()
            lane_finder.process_video(input_file, output_file, nthreads=2,
                                      start=args.video_start, end=args.video_end)
            elapsed = time.time() - started
            print('Elapsed: {:.2f} s'.format(elapsed))

if __name__ == '__main__':
    main()
