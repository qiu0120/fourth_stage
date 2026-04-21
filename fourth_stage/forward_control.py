#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


@dataclass
class Detection:
    det_type: str
    center_img: Tuple[int, int]
    bbox_img: Tuple[int, int, int, int]
    score: float
    extra: Dict[str, Any]


class BaseDetector:
    def __init__(self, cfg: Dict[str, Any]):
        self.roi_x_ratio_min = cfg['roi_x_ratio_min']
        self.roi_x_ratio_max = cfg['roi_x_ratio_max']
        self.roi_y_ratio_min = cfg['roi_y_ratio_min']
        self.roi_y_ratio_max = cfg['roi_y_ratio_max']

    def _roi(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        x1 = int(w * self.roi_x_ratio_min)
        x2 = int(w * self.roi_x_ratio_max)
        y1 = int(h * self.roi_y_ratio_min)
        y2 = int(h * self.roi_y_ratio_max)
        return (x1, y1, x2, y2), frame_bgr[y1:y2, x1:x2].copy()


class BarColorDetector(BaseDetector):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg)
        self.lower_bar = np.array([cfg['h_min'], cfg['s_min'], cfg['v_min']], dtype=np.uint8)
        self.upper_bar = np.array([cfg['h_max'], cfg['s_max'], cfg['v_max']], dtype=np.uint8)
        self.kernel_open = np.ones((cfg['open_kernel'], cfg['open_kernel']), np.uint8)
        self.kernel_close = np.ones((cfg['close_kernel_h'], cfg['close_kernel_w']), np.uint8)
        self.min_area = cfg['min_area']
        self.min_width = cfg['min_width']
        self.max_height = cfg['max_height']
        self.min_aspect_ratio = cfg['min_aspect_ratio']
        self.max_aspect_ratio = cfg['max_aspect_ratio']
        self.max_center_y_ratio_in_roi = cfg['max_center_y_ratio_in_roi']
        self.center_weight_base = cfg['center_weight_base']
        self.center_weight_gain = cfg['center_weight_gain']

    def detect(self, frame_bgr) -> Optional[Detection]:
        (x1, y1, x2, y2), roi = self._roi(frame_bgr)
        roi_h, roi_w = roi.shape[:2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_bar, self.upper_bar)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_close)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        roi_center_x = roi_w / 2.0
        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            rx, ry, rw, rh = cv2.boundingRect(cnt)
            if rw <= 0 or rh <= 0:
                continue
            aspect_ratio = rw / float(rh)
            center_y_ratio = (ry + rh * 0.5) / float(max(roi_h, 1))
            center_x = rx + rw / 2.0
            x_dist_norm = abs(center_x - roi_center_x) / max(roi_w / 2.0, 1.0)
            center_bonus = 1.0 - x_dist_norm
            if area < self.min_area:
                continue
            if rw < self.min_width:
                continue
            if rh > self.max_height:
                continue
            if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
                continue
            if center_y_ratio > self.max_center_y_ratio_in_roi:
                continue
            score = area * min(aspect_ratio, 10.0) * (self.center_weight_base + self.center_weight_gain * center_bonus)
            candidates.append((score, rx, ry, rw, rh, aspect_ratio))
        if not candidates:
            return None
        score, rx, ry, rw, rh, aspect_ratio = max(candidates, key=lambda x: x[0])
        bx1, by1 = x1 + rx, y1 + ry
        bx2, by2 = bx1 + rw, by1 + rh
        cx = bx1 + rw // 2
        cy = by1 + rh // 2
        return Detection('bar', (cx, cy), (bx1, by1, bx2, by2), float(score), {'aspect_ratio': float(aspect_ratio)})


class BallDetector(BaseDetector):
    def __init__(self, cfg: Dict[str, Any], det_type: str):
        super().__init__(cfg)
        self.det_type = det_type
        self.lower = np.array([cfg['h_min'], cfg['s_min'], cfg['v_min']], dtype=np.uint8)
        self.upper = np.array([cfg['h_max'], cfg['s_max'], cfg['v_max']], dtype=np.uint8)
        self.kernel_open = np.ones((cfg['open_kernel'], cfg['open_kernel']), np.uint8)
        self.kernel_close = np.ones((cfg['close_kernel'], cfg['close_kernel']), np.uint8)
        self.min_area = cfg['min_area']
        self.max_area = cfg['max_area']
        self.min_radius = cfg['min_radius']
        self.max_radius = cfg['max_radius']
        self.min_circularity = cfg['min_circularity']
        self.min_wh_ratio = cfg['min_wh_ratio']
        self.max_wh_ratio = cfg['max_wh_ratio']
        self.max_center_y_ratio_in_roi = cfg['max_center_y_ratio_in_roi']
        self.center_weight_base = cfg['center_weight_base']
        self.center_weight_gain = cfg['center_weight_gain']
        self.radius_score_gain = cfg['radius_score_gain']

    def detect(self, frame_bgr) -> Optional[Detection]:
        (x1, y1, x2, y2), roi = self._roi(frame_bgr)
        roi_h, roi_w = roi.shape[:2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower, self.upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_close)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        roi_center_x = roi_w / 2.0
        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw <= 0 or bh <= 0:
                continue
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            perimeter = cv2.arcLength(cnt, True)
            circularity = (4.0 * math.pi * area) / (perimeter * perimeter) if perimeter > 1e-6 else 0.0
            wh_ratio = bw / float(bh)
            center_y_ratio = cy / float(max(roi_h, 1))
            x_dist_norm = abs(cx - roi_center_x) / max(roi_w / 2.0, 1.0)
            center_bonus = 1.0 - x_dist_norm
            if area < self.min_area or area > self.max_area:
                continue
            if radius < self.min_radius or radius > self.max_radius:
                continue
            if circularity < self.min_circularity:
                continue
            if wh_ratio < self.min_wh_ratio or wh_ratio > self.max_wh_ratio:
                continue
            if center_y_ratio > self.max_center_y_ratio_in_roi:
                continue
            score = (radius * self.radius_score_gain) * max(circularity, 0.0) * (self.center_weight_base + self.center_weight_gain * center_bonus)
            candidates.append((score, x, y, bw, bh, cx, cy, radius, circularity))
        if not candidates:
            return None
        score, x, y, bw, bh, cx, cy, radius, circularity = max(candidates, key=lambda c: c[0])
        bx1, by1 = x1 + x, y1 + y
        bx2, by2 = bx1 + bw, by1 + bh
        cx_img = x1 + int(round(cx))
        cy_img = y1 + int(round(cy))
        return Detection(self.det_type, (cx_img, cy_img), (bx1, by1, bx2, by2), float(score), {
            'radius': float(radius),
            'circularity': float(circularity),
        })


class ColaDetector(BaseDetector):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg)
        self.lower = np.array([cfg['h_min'], cfg['s_min'], cfg['v_min']], dtype=np.uint8)
        self.upper = np.array([cfg['h_max'], cfg['s_max'], cfg['v_max']], dtype=np.uint8)
        self.kernel_open = np.ones((cfg['open_kernel'], cfg['open_kernel']), np.uint8)
        self.kernel_close = np.ones((cfg['close_kernel'], cfg['close_kernel']), np.uint8)
        self.min_area = cfg['min_area']
        self.max_area = cfg['max_area']
        self.min_width = cfg['min_width']
        self.max_width = cfg['max_width']
        self.min_height = cfg['min_height']
        self.max_height = cfg['max_height']
        self.min_hw_ratio = cfg['min_hw_ratio']
        self.max_hw_ratio = cfg['max_hw_ratio']
        self.max_center_y_ratio_in_roi = cfg['max_center_y_ratio_in_roi']
        self.center_weight_base = cfg['center_weight_base']
        self.center_weight_gain = cfg['center_weight_gain']

    def detect(self, frame_bgr) -> Optional[Detection]:
        (x1, y1, x2, y2), roi = self._roi(frame_bgr)
        roi_h, roi_w = roi.shape[:2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower, self.upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_close)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        roi_center_x = roi_w / 2.0
        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw <= 0 or bh <= 0:
                continue
            hw_ratio = bh / float(bw)
            center_y_ratio = (y + 0.5 * bh) / float(max(roi_h, 1))
            center_x = x + 0.5 * bw
            x_dist_norm = abs(center_x - roi_center_x) / max(roi_w / 2.0, 1.0)
            center_bonus = 1.0 - x_dist_norm
            if area < self.min_area or area > self.max_area:
                continue
            if bw < self.min_width or bw > self.max_width:
                continue
            if bh < self.min_height or bh > self.max_height:
                continue
            if hw_ratio < self.min_hw_ratio or hw_ratio > self.max_hw_ratio:
                continue
            if center_y_ratio > self.max_center_y_ratio_in_roi:
                continue
            score = area * min(hw_ratio, 8.0) * (self.center_weight_base + self.center_weight_gain * center_bonus)
            candidates.append((score, x, y, bw, bh, hw_ratio))
        if not candidates:
            return None
        score, x, y, bw, bh, hw_ratio = max(candidates, key=lambda c: c[0])
        bx1, by1 = x1 + x, y1 + y
        bx2, by2 = bx1 + bw, by1 + bh
        cx = bx1 + bw // 2
        cy = by1 + bh // 2
        return Detection('cola', (cx, cy), (bx1, by1, bx2, by2), float(score), {'hw_ratio': float(hw_ratio)})


class UnderBarTargetTaskNode(Node):
    SEARCH_BAR_AND_FORWARD = 'SEARCH_BAR_AND_FORWARD'
    APPROACH_BAR_PREPARE_LOWER = 'APPROACH_BAR_PREPARE_LOWER'
    PASS_BAR_WITH_LOW_BODY = 'PASS_BAR_WITH_LOW_BODY'
    SEARCH_TARGET_AFTER_BAR = 'SEARCH_TARGET_AFTER_BAR'
    APPROACH_AND_ALIGN_TARGET = 'APPROACH_AND_ALIGN_TARGET'
    HIT_TARGET = 'HIT_TARGET'
    TASK_DONE = 'TASK_DONE'

    def __init__(self):
        super().__init__('under_bar_target_task_node')
        self.bridge = CvBridge()

        # =========================
        # 话题与控制频率
        # =========================
        self.declare_parameter('rgb_topic', '/rgb_camera/rgb_camera/image_raw')
        self.declare_parameter('depth_topic', '/d435/depth/d435_depth/depth/image_raw')
        self.declare_parameter('control_hz', 30.0)

        # =========================
        # 限高杆参数
        # =========================
        self._declare_bar_params()

        # =========================
        # 蓝球参数
        # =========================
        self._declare_ball_params('blue_ball', defaults={
            'h_min': 90, 'h_max': 135, 's_min': 80, 's_max': 255, 'v_min': 40, 'v_max': 255,
            'roi_x_ratio_min': 0.20, 'roi_x_ratio_max': 0.80, 'roi_y_ratio_min': 0.15, 'roi_y_ratio_max': 0.95,
            'open_kernel': 3, 'close_kernel': 5,
            'min_area': 80, 'max_area': 50000,
            'min_radius': 5.0, 'max_radius': 120.0,
            'min_circularity': 0.50,
            'min_wh_ratio': 0.50, 'max_wh_ratio': 2.0,
            'max_center_y_ratio_in_roi': 1.0,
            'center_weight_base': 0.3, 'center_weight_gain': 0.7,
            'radius_score_gain': 10.0,
        })

        # =========================
        # 白球参数
        # =========================
        self._declare_ball_params('white_ball', defaults={
            'h_min': 0, 'h_max': 20, 's_min': 0, 's_max': 20, 'v_min': 95, 'v_max': 255,
            'roi_x_ratio_min': 0.20, 'roi_x_ratio_max': 0.80, 'roi_y_ratio_min': 0.15, 'roi_y_ratio_max': 0.95,
            'open_kernel': 3, 'close_kernel': 5,
            'min_area': 350, 'max_area': 10000,
            'min_radius': 10.0, 'max_radius': 100.0,
            'min_circularity': 0.55,
            'min_wh_ratio': 0.6, 'max_wh_ratio': 1.40,
            'max_center_y_ratio_in_roi': 1.0,
            'center_weight_base': 0.3, 'center_weight_gain': 0.7,
            'radius_score_gain': 10.0,
        })

        # =========================
        # 可乐参数
        # =========================
        self._declare_cola_params()

        # =========================
        # 主流程参数
        # =========================
        self.declare_parameter('bar_search_forward_speed', 0.15)
        self.declare_parameter('bar_approach_speed', 0.15)
        self.declare_parameter('bar_pass_speed', 0.10)
        self.declare_parameter('bar_trigger_distance_m', 0.80)
        self.declare_parameter('bar_pass_enter_distance_m', 0.50)
        self.declare_parameter('bar_pass_distance_m', 0.55)
        self.declare_parameter('low_body_z', 0.18)
        self.declare_parameter('normal_body_z', 0.24)

        self.declare_parameter('target_search_forward_speed', 0.08)
        self.declare_parameter('align_forward_speed_far', 0.10)
        self.declare_parameter('align_forward_speed_near', 0.06)
        self.declare_parameter('align_vy_k', 0.35)
        self.declare_parameter('align_vy_max', 0.15)
        self.declare_parameter('align_vy_min', 0.04)
        self.declare_parameter('target_stable_frames', 3)
        self.declare_parameter('hit_trigger_distance_m', 0.20)
        self.declare_parameter('center_px_deadband', 15)

        self.declare_parameter('hit_blue_ball_speed', 0.18)
        self.declare_parameter('hit_blue_ball_distance', 0.18)
        self.declare_parameter('hit_white_ball_speed', 0.18)
        self.declare_parameter('hit_white_ball_distance', 0.18)
        self.declare_parameter('hit_cola_speed', 0.12)
        self.declare_parameter('hit_cola_distance', 0.25)

        # =========================
        # 读取参数
        # =========================
        self.rgb_topic = self.get_parameter('rgb_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.control_hz = float(self.get_parameter('control_hz').value)

        self.bar_search_forward_speed = float(self.get_parameter('bar_search_forward_speed').value)
        self.bar_approach_speed = float(self.get_parameter('bar_approach_speed').value)
        self.bar_pass_speed = float(self.get_parameter('bar_pass_speed').value)
        self.bar_trigger_distance_m = float(self.get_parameter('bar_trigger_distance_m').value)
        self.bar_pass_enter_distance_m = float(self.get_parameter('bar_pass_enter_distance_m').value)
        self.bar_pass_distance_m = float(self.get_parameter('bar_pass_distance_m').value)
        self.low_body_z = float(self.get_parameter('low_body_z').value)
        self.normal_body_z = float(self.get_parameter('normal_body_z').value)

        self.target_search_forward_speed = float(self.get_parameter('target_search_forward_speed').value)
        self.align_forward_speed_far = float(self.get_parameter('align_forward_speed_far').value)
        self.align_forward_speed_near = float(self.get_parameter('align_forward_speed_near').value)
        self.align_vy_k = float(self.get_parameter('align_vy_k').value)
        self.align_vy_max = float(self.get_parameter('align_vy_max').value)
        self.align_vy_min = float(self.get_parameter('align_vy_min').value)
        self.target_stable_frames = int(self.get_parameter('target_stable_frames').value)
        self.hit_trigger_distance_m = float(self.get_parameter('hit_trigger_distance_m').value)
        self.center_px_deadband = int(self.get_parameter('center_px_deadband').value)

        self.hit_params = {
            'blue_ball': {'speed': float(self.get_parameter('hit_blue_ball_speed').value), 'distance': float(self.get_parameter('hit_blue_ball_distance').value)},
            'white_ball': {'speed': float(self.get_parameter('hit_white_ball_speed').value), 'distance': float(self.get_parameter('hit_white_ball_distance').value)},
            'cola': {'speed': float(self.get_parameter('hit_cola_speed').value), 'distance': float(self.get_parameter('hit_cola_distance').value)},
        }

        # =========================
        # 构造检测器
        # =========================
        self.bar_detector = BarColorDetector(self._read_bar_cfg())
        self.blue_ball_detector = BallDetector(self._read_ball_cfg('blue_ball'), 'blue_ball')
        self.white_ball_detector = BallDetector(self._read_ball_cfg('white_ball'), 'white_ball')
        self.cola_detector = ColaDetector(self._read_cola_cfg())

        # 订阅
        self.rgb_sub = self.create_subscription(Image, self.rgb_topic, self.rgb_callback, qos_profile_sensor_data)
        self.depth_sub = self.create_subscription(Image, self.depth_topic, self.depth_callback, qos_profile_sensor_data)

        self.latest_bgr = None
        self.latest_depth = None
        self.latest_target: Optional[Detection] = None
        self.locked_target: Optional[Detection] = None

        self.state = self.SEARCH_BAR_AND_FORWARD
        self.state_enter_time = self.now_s()
        self.target_stable_count = 0
        self.stable_target_type = None
        self.last_log_time = self.now_s()
        self.body_height_cmd = self.normal_body_z
        self.motion_cmd = (0.0, 0.0, 0.0)

        self.timer = self.create_timer(1.0 / self.control_hz, self.control_loop)
        self.get_logger().info('under_bar_target_task_node_param_style started')

    # ---------- 参数声明 / 读取 ----------
    def _declare_bar_params(self):
        p = self.declare_parameter
        p('bar.h_min', 85); p('bar.h_max', 100)
        p('bar.s_min', 15); p('bar.s_max', 45)
        p('bar.v_min', 35); p('bar.v_max', 80)
        p('bar.roi_x_ratio_min', 0.30); p('bar.roi_x_ratio_max', 0.70)
        p('bar.roi_y_ratio_min', 0.20); p('bar.roi_y_ratio_max', 0.75)
        p('bar.open_kernel', 3)
        p('bar.close_kernel_h', 7); p('bar.close_kernel_w', 11)
        p('bar.min_area', 500)
        p('bar.min_width', 15)
        p('bar.max_height', 1000)
        p('bar.min_aspect_ratio', 1.5)
        p('bar.max_aspect_ratio', 50.0)
        p('bar.max_center_y_ratio_in_roi', 0.95)
        p('bar.center_weight_base', 0.3)
        p('bar.center_weight_gain', 0.7)

    def _read_bar_cfg(self):
        gp = self.get_parameter
        return {
            'h_min': int(gp('bar.h_min').value), 'h_max': int(gp('bar.h_max').value),
            's_min': int(gp('bar.s_min').value), 's_max': int(gp('bar.s_max').value),
            'v_min': int(gp('bar.v_min').value), 'v_max': int(gp('bar.v_max').value),
            'roi_x_ratio_min': float(gp('bar.roi_x_ratio_min').value), 'roi_x_ratio_max': float(gp('bar.roi_x_ratio_max').value),
            'roi_y_ratio_min': float(gp('bar.roi_y_ratio_min').value), 'roi_y_ratio_max': float(gp('bar.roi_y_ratio_max').value),
            'open_kernel': int(gp('bar.open_kernel').value),
            'close_kernel_h': int(gp('bar.close_kernel_h').value), 'close_kernel_w': int(gp('bar.close_kernel_w').value),
            'min_area': int(gp('bar.min_area').value), 'min_width': int(gp('bar.min_width').value),
            'max_height': int(gp('bar.max_height').value),
            'min_aspect_ratio': float(gp('bar.min_aspect_ratio').value), 'max_aspect_ratio': float(gp('bar.max_aspect_ratio').value),
            'max_center_y_ratio_in_roi': float(gp('bar.max_center_y_ratio_in_roi').value),
            'center_weight_base': float(gp('bar.center_weight_base').value), 'center_weight_gain': float(gp('bar.center_weight_gain').value),
        }

    def _declare_ball_params(self, prefix: str, defaults: Dict[str, Any]):
        for k, v in defaults.items():
            self.declare_parameter(f'{prefix}.{k}', v)

    def _read_ball_cfg(self, prefix: str):
        keys = [
            'h_min','h_max','s_min','s_max','v_min','v_max',
            'roi_x_ratio_min','roi_x_ratio_max','roi_y_ratio_min','roi_y_ratio_max',
            'open_kernel','close_kernel','min_area','max_area','min_radius','max_radius',
            'min_circularity','min_wh_ratio','max_wh_ratio','max_center_y_ratio_in_roi',
            'center_weight_base','center_weight_gain','radius_score_gain'
        ]
        cfg = {}
        for k in keys:
            val = self.get_parameter(f'{prefix}.{k}').value
            cfg[k] = float(val) if isinstance(val, float) else int(val) if isinstance(val, int) else val
        return cfg

    def _declare_cola_params(self):
        p = self.declare_parameter
        p('cola.h_min', 0); p('cola.h_max', 20)
        p('cola.s_min', 0); p('cola.s_max', 20)
        p('cola.v_min', 0); p('cola.v_max', 20)
        p('cola.roi_x_ratio_min', 0.20); p('cola.roi_x_ratio_max', 0.80)
        p('cola.roi_y_ratio_min', 0.10); p('cola.roi_y_ratio_max', 0.95)
        p('cola.open_kernel', 3); p('cola.close_kernel', 5)
        p('cola.min_area', 250); p('cola.max_area', 80000)
        p('cola.min_width', 8); p('cola.max_width', 5000)
        p('cola.min_height', 20); p('cola.max_height', 10000)
        p('cola.min_hw_ratio', 1.5); p('cola.max_hw_ratio', 20.0)
        p('cola.max_center_y_ratio_in_roi', 1.0)
        p('cola.center_weight_base', 0.3); p('cola.center_weight_gain', 0.7)

    def _read_cola_cfg(self):
        gp = self.get_parameter
        return {
            'h_min': int(gp('cola.h_min').value), 'h_max': int(gp('cola.h_max').value),
            's_min': int(gp('cola.s_min').value), 's_max': int(gp('cola.s_max').value),
            'v_min': int(gp('cola.v_min').value), 'v_max': int(gp('cola.v_max').value),
            'roi_x_ratio_min': float(gp('cola.roi_x_ratio_min').value), 'roi_x_ratio_max': float(gp('cola.roi_x_ratio_max').value),
            'roi_y_ratio_min': float(gp('cola.roi_y_ratio_min').value), 'roi_y_ratio_max': float(gp('cola.roi_y_ratio_max').value),
            'open_kernel': int(gp('cola.open_kernel').value), 'close_kernel': int(gp('cola.close_kernel').value),
            'min_area': int(gp('cola.min_area').value), 'max_area': int(gp('cola.max_area').value),
            'min_width': int(gp('cola.min_width').value), 'max_width': int(gp('cola.max_width').value),
            'min_height': int(gp('cola.min_height').value), 'max_height': int(gp('cola.max_height').value),
            'min_hw_ratio': float(gp('cola.min_hw_ratio').value), 'max_hw_ratio': float(gp('cola.max_hw_ratio').value),
            'max_center_y_ratio_in_roi': float(gp('cola.max_center_y_ratio_in_roi').value),
            'center_weight_base': float(gp('cola.center_weight_base').value), 'center_weight_gain': float(gp('cola.center_weight_gain').value),
        }

    # ---------- ROS回调与基础工具 ----------
    def now_s(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def rgb_callback(self, msg: Image):
        try:
            self.latest_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'RGB convert failed: {e}')

    def depth_callback(self, msg: Image):
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f'DEPTH convert failed: {e}')

    def depth_to_meters(self, depth_img):
        if depth_img is None:
            return None
        if depth_img.dtype == np.float32:
            depth_m = depth_img.copy()
        elif depth_img.dtype == np.uint16:
            depth_m = depth_img.astype(np.float32) / 1000.0
        else:
            depth_m = depth_img.astype(np.float32)
        depth_m[~np.isfinite(depth_m)] = 0.0
        return depth_m

    def estimate_depth_at_center(self, center_img: Tuple[int, int]) -> Optional[float]:
        if self.latest_depth is None or self.latest_bgr is None:
            return None
        depth_m = self.depth_to_meters(self.latest_depth)
        if depth_m is None:
            return None
        dh, dw = depth_m.shape[:2]
        ih, iw = self.latest_bgr.shape[:2]
        cx, cy = center_img
        dx = int(cx * dw / max(iw, 1))
        dy = int(cy * dh / max(ih, 1))
        half = 3
        x1 = max(0, dx - half); x2 = min(dw, dx + half + 1)
        y1 = max(0, dy - half); y2 = min(dh, dy + half + 1)
        patch = depth_m[y1:y2, x1:x2]
        valid = patch[np.isfinite(patch)]
        valid = valid[(valid > 0.05) & (valid < 10.0)]
        if valid.size == 0:
            return None
        return float(np.percentile(valid, 20))

    def set_body_low(self):
        self.body_height_cmd = self.low_body_z

    def set_body_normal(self):
        self.body_height_cmd = self.normal_body_z

    def send_motion_cmd(self, vx: float, vy: float, wz: float):
        # TODO: 替换成你的 Robot_Ctrl / LCM 发送函数
        self.motion_cmd = (vx, vy, wz)

    def stop(self):
        self.send_motion_cmd(0.0, 0.0, 0.0)

    def enter_state(self, new_state: str):
        self.state = new_state
        self.state_enter_time = self.now_s()
        self.get_logger().info(f'ENTER STATE -> {new_state}')

    # ---------- 目标选择 ----------
    def choose_most_centered_target(self, frame_bgr) -> Optional[Detection]:
        candidates = [
            self.blue_ball_detector.detect(frame_bgr),
            self.white_ball_detector.detect(frame_bgr),
            self.cola_detector.detect(frame_bgr),
        ]
        candidates = [c for c in candidates if c is not None]
        if not candidates:
            return None
        img_center_x = frame_bgr.shape[1] / 2.0
        return min(candidates, key=lambda c: abs(c.center_img[0] - img_center_x))

    def compute_align_cmd(self, target: Detection) -> Tuple[float, float]:
        img_center_x = self.latest_bgr.shape[1] / 2.0
        err_px = target.center_img[0] - img_center_x
        depth_m = self.estimate_depth_at_center(target.center_img)
        vx = self.align_forward_speed_near if (depth_m is not None and depth_m < 0.35) else self.align_forward_speed_far
        if abs(err_px) <= self.center_px_deadband:
            vy = 0.0
        else:
            err_norm = err_px / max(self.latest_bgr.shape[1] / 2.0, 1.0)
            vy = self.align_vy_k * err_norm
            vy = float(np.clip(vy, -self.align_vy_max, self.align_vy_max))
            if 0 < abs(vy) < self.align_vy_min:
                vy = math.copysign(self.align_vy_min, vy)
        return vx, vy

    # ---------- 主循环 ----------
    def control_loop(self):
        if self.latest_bgr is None:
            return
        now = self.now_s()

        if self.state == self.SEARCH_BAR_AND_FORWARD:
            self.set_body_normal()
            bar = self.bar_detector.detect(self.latest_bgr)
            self.send_motion_cmd(self.bar_search_forward_speed, 0.0, 0.0)
            if bar is not None:
                d = self.estimate_depth_at_center(bar.center_img)
                if d is not None and d < self.bar_trigger_distance_m:
                    self.enter_state(self.APPROACH_BAR_PREPARE_LOWER)

        elif self.state == self.APPROACH_BAR_PREPARE_LOWER:
            self.set_body_low()
            bar = self.bar_detector.detect(self.latest_bgr)
            self.send_motion_cmd(self.bar_approach_speed, 0.0, 0.0)
            if bar is not None:
                d = self.estimate_depth_at_center(bar.center_img)
                if d is not None and d < self.bar_pass_enter_distance_m:
                    self.enter_state(self.PASS_BAR_WITH_LOW_BODY)
            else:
                if now - self.state_enter_time > 0.8:
                    self.enter_state(self.PASS_BAR_WITH_LOW_BODY)

        elif self.state == self.PASS_BAR_WITH_LOW_BODY:
            self.set_body_low()
            self.send_motion_cmd(self.bar_pass_speed, 0.0, 0.0)
            pass_duration = self.bar_pass_distance_m / max(self.bar_pass_speed, 1e-3)
            if now - self.state_enter_time >= pass_duration:
                self.target_stable_count = 0
                self.stable_target_type = None
                self.enter_state(self.SEARCH_TARGET_AFTER_BAR)

        elif self.state == self.SEARCH_TARGET_AFTER_BAR:
            self.set_body_low()
            self.send_motion_cmd(self.target_search_forward_speed, 0.0, 0.0)
            target = self.choose_most_centered_target(self.latest_bgr)
            if target is not None:
                if self.stable_target_type == target.det_type:
                    self.target_stable_count += 1
                else:
                    self.stable_target_type = target.det_type
                    self.target_stable_count = 1
                self.latest_target = target
                if self.target_stable_count >= self.target_stable_frames:
                    self.locked_target = target
                    self.enter_state(self.APPROACH_AND_ALIGN_TARGET)
            else:
                self.stable_target_type = None
                self.target_stable_count = 0

        elif self.state == self.APPROACH_AND_ALIGN_TARGET:
            self.set_body_low()
            target = self.choose_most_centered_target(self.latest_bgr)
            if target is None:
                self.locked_target = None
                self.enter_state(self.SEARCH_TARGET_AFTER_BAR)
                return
            self.locked_target = target
            vx, vy = self.compute_align_cmd(target)
            self.send_motion_cmd(vx, vy, 0.0)
            d = self.estimate_depth_at_center(target.center_img)
            if d is not None and d < self.hit_trigger_distance_m:
                self.enter_state(self.HIT_TARGET)

        elif self.state == self.HIT_TARGET:
            self.set_body_low()
            if self.locked_target is None:
                self.enter_state(self.SEARCH_TARGET_AFTER_BAR)
                return
            params = self.hit_params.get(self.locked_target.det_type, {'speed': 0.15, 'distance': 0.18})
            self.send_motion_cmd(params['speed'], 0.0, 0.0)
            hit_duration = params['distance'] / max(params['speed'], 1e-3)
            if now - self.state_enter_time >= hit_duration:
                self.stop()
                self.enter_state(self.TASK_DONE)

        elif self.state == self.TASK_DONE:
            self.stop()

        if now - self.last_log_time > 0.25:
            self.last_log_time = now
            depth_text = 'None'
            target_text = 'None'
            if self.locked_target is not None:
                d = self.estimate_depth_at_center(self.locked_target.center_img)
                depth_text = 'None' if d is None else f'{d:.3f}'
                target_text = f'{self.locked_target.det_type}@{self.locked_target.center_img}'
            vx, vy, wz = self.motion_cmd
            self.get_logger().info(
                f'state={self.state} body_z={self.body_height_cmd:.3f} '
                f'cmd=({vx:.3f},{vy:.3f},{wz:.3f}) target={target_text} depth={depth_text}'
            )


def main(args=None):
    rclpy.init(args=args)
    node = UnderBarTargetTaskNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
