#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List

import cv2
import numpy as np
from threading import Thread, Lock
import lcm
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException

from fourth_stage.robot_control_cmd_lcmt import robot_control_cmd_lcmt
from fourth_stage.robot_control_response_lcmt import robot_control_response_lcmt

from cyberdog_msg.msg import YamlParam, ApplyForce

@dataclass
class Detection:
    det_type: str
    center_img: Tuple[int, int]
    bbox_img: Tuple[int, int, int, int]
    score: float
    extra: Dict[str, Any]

class ControlParameterValueKind:
    kDOUBLE = 1
    kS64 = 2
    kVEC_X_DOUBLE = 3
    kMAT_X_DOUBLE = 4

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

            center_score = max(center_bonus, 0.0)
            area_score = math.sqrt(max(area, 1.0))
            shape_score = min(aspect_ratio, 10.0)

            score = center_score

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
    INITIAL_LATERAL_MOVE = 'INITIAL_LATERAL_MOVE'
    LATERAL_SEARCH_BAR = 'LATERAL_SEARCH_BAR'
    CENTER_BAR_BEFORE_FORWARD = 'CENTER_BAR_BEFORE_FORWARD'
    LATERAL_MOVE_TO_NEXT_BAR = 'LATERAL_MOVE_TO_NEXT_BAR'
    SEARCH_BAR_AND_FORWARD = 'SEARCH_BAR_AND_FORWARD'
    SEARCH_TARGET_AFTER_BAR = 'SEARCH_TARGET_AFTER_BAR'
    APPROACH_AND_ALIGN_TARGET = 'APPROACH_AND_ALIGN_TARGET'
    HIT_TARGET = 'HIT_TARGET'
    BACKOFF_TO_BAR_AFTER_HIT = 'BACKOFF_TO_BAR_AFTER_HIT'
    TASK_DONE = 'TASK_DONE'

    def __init__(self):
        super().__init__('under_bar_target_task_node')
        self.bridge = CvBridge()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # =========================
        # 话题与控制频率
        # =========================
        self.declare_parameter('rgb_topic', '/rgb_camera/rgb_camera/image_raw')
        self.declare_parameter('depth_topic', '/d435/depth/d435_depth/depth/image_raw')
        self.declare_parameter('control_hz', 30.0)
        self.declare_parameter('tf_parent_frame', 'vodom')
        self.declare_parameter('tf_child_frame', 'base_link')
        self.yaml_node = yaml_pub()
        self.Ctrl = Robot_Ctrl()
        self.Ctrl.run()
        self.msg = robot_control_cmd_lcmt()
        if not hasattr(self.msg, 'life_count'):
            self.msg.life_count = 0

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
            'min_area': 80, 'max_area': 5000000,
            'min_radius': 5.0, 'max_radius': 200.0,
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
            'min_area': 80, 'max_area': 50000,
            'min_radius': 10.0, 'max_radius': 150.0,
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
        self.declare_parameter('bar_search_forward_speed', 0.30)
        self.declare_parameter('bar_trigger_distance_m', 0.40)

        # 限高杆居中对齐参数：识别到限高杆后，边前进边横移，让限高杆中心靠近图像中心
        self.declare_parameter('bar_align_vy_k', 0.30)
        self.declare_parameter('bar_align_vy_max', 0.15)
        self.declare_parameter('bar_align_vy_min', 0.10)
        self.declare_parameter('bar_center_px_deadband', 7)

        self.declare_parameter('target_search_forward_speed', 0.2)
        self.declare_parameter('align_forward_speed_far', 0.3)
        self.declare_parameter('align_forward_speed_near', 0.30)
        self.declare_parameter('align_vy_k', 0.35)
        self.declare_parameter('align_vy_max', 0.15)
        self.declare_parameter('align_vy_min', 0.10)
        self.declare_parameter('target_stable_frames', 3)
        self.declare_parameter('hit_trigger_distance_m', 0.20)
        self.declare_parameter('center_px_deadband', 7)
        self.declare_parameter('show_debug_vis', True)
        self.declare_parameter('debug_window_name', 'under_bar_target_debug')

        self.declare_parameter('hit_blue_ball_speed', 0.30)
        self.declare_parameter('hit_blue_ball_distance', 0.25)
        self.declare_parameter('hit_white_ball_speed', 0.40)
        self.declare_parameter('hit_white_ball_distance', 0.30)
        self.declare_parameter('hit_cola_speed', 0.30)
        self.declare_parameter('hit_cola_distance', 0.25)

        # 撞击完成后后退回第一次看到限高杆的位置：后退时继续识别限高杆并横向居中
        self.declare_parameter('backoff_after_hit_speed', 0.30)          # 正数，实际发送时会变成负 vx
        self.declare_parameter('backoff_bar_depth_tolerance_m', 0.05)    # 回到记录杆距的允许误差
        self.declare_parameter('backoff_min_time_s', 0.30)               # 防止刚进后退状态立刻误停

        # 启动预横移参数：启动后先向左移动一段距离，再开始识别限高杆
        self.declare_parameter('initial_lateral_shift_distance_m', 0.30)  # 启动后先横移多少米
        self.declare_parameter('initial_lateral_shift_vy', 0.20)          # 启动预横移速度

        # 两轮限高杆任务参数
        self.declare_parameter('total_bar_count', 2)                     # 一共识别/完成几个限高杆任务
        self.declare_parameter('lateral_search_vy', 0.20)                # 初始向左横移搜索限高杆
        self.declare_parameter('bar_center_stable_frames', 3)            # 限高杆居中连续稳定帧数
        self.declare_parameter('next_bar_lateral_shift_distance_m', 1.0) # 完成一轮后先横移多少米，避免重复识别旧杆
        self.declare_parameter('next_bar_lateral_shift_vy', 0.20)        # 换下一个限高杆时的横移速度

        # =========================
        # 读取参数
        # =========================
        self.rgb_topic = self.get_parameter('rgb_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.control_hz = float(self.get_parameter('control_hz').value)
        self.tf_parent_frame = self.get_parameter('tf_parent_frame').value
        self.tf_child_frame = self.get_parameter('tf_child_frame').value

        self.bar_search_forward_speed = float(self.get_parameter('bar_search_forward_speed').value)
        self.bar_trigger_distance_m = float(self.get_parameter('bar_trigger_distance_m').value)
        self.bar_align_vy_k = float(self.get_parameter('bar_align_vy_k').value)
        self.bar_align_vy_max = float(self.get_parameter('bar_align_vy_max').value)
        self.bar_align_vy_min = float(self.get_parameter('bar_align_vy_min').value)
        self.bar_center_px_deadband = int(self.get_parameter('bar_center_px_deadband').value)

        self.target_search_forward_speed = float(self.get_parameter('target_search_forward_speed').value)
        self.align_forward_speed_far = float(self.get_parameter('align_forward_speed_far').value)
        self.align_forward_speed_near = float(self.get_parameter('align_forward_speed_near').value)
        self.align_vy_k = float(self.get_parameter('align_vy_k').value)
        self.align_vy_max = float(self.get_parameter('align_vy_max').value)
        self.align_vy_min = float(self.get_parameter('align_vy_min').value)
        self.target_stable_frames = int(self.get_parameter('target_stable_frames').value)
        self.hit_trigger_distance_m = float(self.get_parameter('hit_trigger_distance_m').value)
        self.center_px_deadband = int(self.get_parameter('center_px_deadband').value)
        self.show_debug_vis = bool(self.get_parameter('show_debug_vis').value)
        self.debug_window_name = str(self.get_parameter('debug_window_name').value)

        self.hit_params = {
            'blue_ball': {'speed': float(self.get_parameter('hit_blue_ball_speed').value), 'distance': float(self.get_parameter('hit_blue_ball_distance').value)},
            'white_ball': {'speed': float(self.get_parameter('hit_white_ball_speed').value), 'distance': float(self.get_parameter('hit_white_ball_distance').value)},
            'cola': {'speed': float(self.get_parameter('hit_cola_speed').value), 'distance': float(self.get_parameter('hit_cola_distance').value)},
        }

        self.backoff_after_hit_speed = abs(float(self.get_parameter('backoff_after_hit_speed').value))
        self.backoff_bar_depth_tolerance_m = float(self.get_parameter('backoff_bar_depth_tolerance_m').value)
        self.backoff_min_time_s = float(self.get_parameter('backoff_min_time_s').value)

        self.initial_lateral_shift_distance_m = float(self.get_parameter('initial_lateral_shift_distance_m').value)
        self.initial_lateral_shift_vy = float(self.get_parameter('initial_lateral_shift_vy').value)

        self.total_bar_count = int(self.get_parameter('total_bar_count').value)
        self.lateral_search_vy = float(self.get_parameter('lateral_search_vy').value)
        self.bar_center_stable_frames = int(self.get_parameter('bar_center_stable_frames').value)
        self.next_bar_lateral_shift_distance_m = float(self.get_parameter('next_bar_lateral_shift_distance_m').value)
        self.next_bar_lateral_shift_vy = float(self.get_parameter('next_bar_lateral_shift_vy').value)

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
        self.hit_start_xy: Optional[Tuple[float, float]] = None
        self.backoff_start_xy: Optional[Tuple[float, float]] = None
        self.bar_return_target_depth_m: Optional[float] = None
        self.initial_lateral_shift_start_xy: Optional[Tuple[float, float]] = None
        self.lateral_shift_start_xy: Optional[Tuple[float, float]] = None
        self.completed_bar_count = 0
        self.bar_center_stable_count = 0
        self.task_done_stop_sent = False

        # 启动先向左横移一段固定距离，然后再搜索第一个限高杆
        self.state = self.INITIAL_LATERAL_MOVE
        self.state_enter_time = self.now_s()
        self.target_stable_count = 0
        self.stable_target_type = None
        self.last_log_time = self.now_s()
        self.body_height_cmd = 0.25
        self.motion_cmd = (0.0, 0.0, 0.0)
        self._vis_warned = False

        # 启动时直接调低机身，后续全程保持低身步态，不再在状态切换中重复改高度
        self.set_body_low()

        self.timer = self.create_timer(1.0 / self.control_hz, self.control_loop)
        self.get_logger().info('under_bar_target_task_node_param_style started')

    # ---------- 参数声明 / 读取 ----------
    def _declare_bar_params(self):
        p = self.declare_parameter
        p('bar.h_min', 85); p('bar.h_max', 100)
        p('bar.s_min', 15); p('bar.s_max', 45)
        p('bar.v_min', 35); p('bar.v_max', 80)
        p('bar.roi_x_ratio_min', 0.20); p('bar.roi_x_ratio_max', 0.80)
        p('bar.roi_y_ratio_min', 0.10); p('bar.roi_y_ratio_max', 0.90)
        p('bar.open_kernel', 3)
        p('bar.close_kernel_h', 7); p('bar.close_kernel_w', 11)
        p('bar.min_area', 300)
        p('bar.min_width', 15)
        p('bar.max_height', 1000)
        p('bar.min_aspect_ratio', 1.5)
        p('bar.max_aspect_ratio', 50.0)
        p('bar.max_center_y_ratio_in_roi', 1.0)
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
        p('cola.roi_y_ratio_min', 0.0); p('cola.roi_y_ratio_max', 1.0)
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

    def estimate_bar_depth(self, bar_det: Detection) -> Optional[float]:
        if self.latest_depth is None or self.latest_bgr is None or bar_det is None:
            return None

        depth_m = self.depth_to_meters(self.latest_depth)
        if depth_m is None:
            return None

        ih, iw = self.latest_bgr.shape[:2]
        dh, dw = depth_m.shape[:2]

        x1, y1, x2, y2 = bar_det.bbox_img
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)

        # 只取限高杆框上部的一条横向带状区域，避免取到外接框中心的空区域/背景
        sx1 = x1 + int(0.15 * bw)
        sx2 = x2 - int(0.15 * bw)
        sy1 = y1 + int(0.05 * bh)
        sy2 = y1 + int(0.15 * bh)

        sx1 = max(0, min(iw - 1, sx1))
        sx2 = max(sx1 + 1, min(iw, sx2))
        sy1 = max(0, min(ih - 1, sy1))
        sy2 = max(sy1 + 1, min(ih, sy2))

        dx1 = int(sx1 * dw / max(iw, 1))
        dx2 = int(sx2 * dw / max(iw, 1))
        dy1 = int(sy1 * dh / max(ih, 1))
        dy2 = int(sy2 * dh / max(ih, 1))

        dx1 = max(0, min(dw - 1, dx1))
        dx2 = max(dx1 + 1, min(dw, dx2))
        dy1 = max(0, min(dh - 1, dy1))
        dy2 = max(dy1 + 1, min(dh, dy2))

        patch = depth_m[dy1:dy2, dx1:dx2]
        valid = patch[np.isfinite(patch)]
        valid = valid[(valid > 0.05) & (valid < 10.0)]
        if valid.size == 0:
            return None

        # 取较近分位数，尽量更像杆本体而不是后方背景
        return float(np.percentile(valid, 20))

    def set_body_low(self):
        self.body_height_cmd = 0.17
        values = [0.0] * 12
        values[0] = 0.0   # roll
        values[2] = 0.17  # height
        self.yaml_node.publish_yaml_vecxd("des_roll_pitch_height_motion", values, is_user=1)
        self.yaml_node.publish_yaml_vecxd("des_roll_pitch_height", values, is_user=1)
        self.stop()

    def set_body_normal(self):
        self.body_height_cmd = 0.25
        values = [0.0] * 12
        values[0] = 0.0   # roll
        values[2] = 0.25  # height
        self.yaml_node.publish_yaml_vecxd("des_roll_pitch_height_motion", values, is_user=1)
        self.yaml_node.publish_yaml_vecxd("des_roll_pitch_height", values, is_user=1)
        self.stop()

    def _inc_life_count(self):
        self.msg.life_count += 1
        if self.msg.life_count > 127:
            self.msg.life_count = 0

    def send_motion_cmd(self, vx: float, vy: float, wz: float):
        self.motion_cmd = (vx, vy, wz)
        self.msg.mode = 11
        self.msg.gait_id = 3
        self._inc_life_count()
        self.msg.vel_des = [vx, vy, wz]
        self.msg.step_height = [0.02, 0.02]
        self.msg.rpy_des = [0.0, 0.0, 0.0]
        self.Ctrl.Send_cmd(self.msg)

    def stop(self):
        self.msg.mode = 12
        self.msg.gait_id = 0
        self._inc_life_count()
        self.Ctrl.Send_cmd(self.msg)
        self.get_logger().info('[CMD] STOP', throttle_duration_sec=1.0)
        self.Ctrl.Wait_finish(12, 0)

    def enter_state(self, new_state: str):
        self.state = new_state
        self.state_enter_time = self.now_s()
        self.get_logger().info(f'ENTER STATE -> {new_state}')

        if new_state == self.INITIAL_LATERAL_MOVE:
            self.initial_lateral_shift_start_xy = self.get_current_xy_from_tf()
            self.get_logger().info(
                f'INITIAL_LATERAL_MOVE start xy={self.initial_lateral_shift_start_xy}, '
                f'target_shift={self.initial_lateral_shift_distance_m:.3f} m'
            )

        if new_state == self.LATERAL_SEARCH_BAR:
            # 每一轮找新限高杆时清空本轮记录，避免用上一根杆的深度。
            self.hit_start_xy = None
            self.backoff_start_xy = None
            self.lateral_shift_start_xy = None
            self.locked_target = None
            self.latest_target = None
            self.target_stable_count = 0
            self.stable_target_type = None
            self.bar_center_stable_count = 0
            self.bar_return_target_depth_m = None

        if new_state == self.CENTER_BAR_BEFORE_FORWARD:
            self.bar_center_stable_count = 0

        if new_state == self.LATERAL_MOVE_TO_NEXT_BAR:
            # 完成一轮后，先横移一段距离再开始识别下一根杆，避免又识别到刚刚那根杆。
            self.lateral_shift_start_xy = self.get_current_xy_from_tf()
            self.bar_return_target_depth_m = None
            self.bar_center_stable_count = 0
            self.get_logger().info(
                f'LATERAL_MOVE_TO_NEXT_BAR start xy={self.lateral_shift_start_xy}, '
                f'target_shift={self.next_bar_lateral_shift_distance_m:.3f} m'
            )

        if new_state == self.SEARCH_BAR_AND_FORWARD:
            self.hit_start_xy = None
            self.target_stable_count = 0
            self.stable_target_type = None
            # 注意：这里不要清空 bar_return_target_depth_m。
            # 它是在 CENTER_BAR_BEFORE_FORWARD 居中稳定、准备前进时记录的，后退时还要用。

        if new_state == self.SEARCH_TARGET_AFTER_BAR:
            # 启动时已经 set_body_low()，后续保持低身步态；这里不再重复 set_body_low()，避免状态切换时多一次 STOP。
            pass

        if new_state == self.HIT_TARGET:
            self.hit_start_xy = self.get_current_xy_from_tf()
            self.get_logger().info(f'HIT start xy={self.hit_start_xy}')

        if new_state == self.BACKOFF_TO_BAR_AFTER_HIT:
            self.backoff_start_xy = self.get_current_xy_from_tf()
            self.get_logger().info(
                f'BACKOFF_TO_BAR_AFTER_HIT start xy={self.backoff_start_xy}, '
                f'return_target_bar_depth={self.bar_return_target_depth_m}'
            )

        if new_state == self.TASK_DONE:
            self.task_done_stop_sent = False

    def get_current_xy_from_tf(self) -> Optional[Tuple[float, float]]:
        try:
            trans = self.tf_buffer.lookup_transform(
                self.tf_parent_frame,
                self.tf_child_frame,
                rclpy.time.Time()
            )
            x = trans.transform.translation.x
            y = trans.transform.translation.y
            return (float(x), float(y))
        except (LookupException, ConnectivityException, ExtrapolationException):
            return None
        except Exception as e:
            self.get_logger().warn(f'TF lookup failed: {e}', throttle_duration_sec=1.0)
            return None

    def planar_distance(self, p0: Tuple[float, float], p1: Tuple[float, float]) -> float:
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        return math.sqrt(dx * dx + dy * dy)

    # ---------- 目标选择 ----------
    def choose_most_centered_target(self, frame_bgr) -> Optional[Detection]:
        candidates = self.detect_all_targets(frame_bgr)
        if not candidates:
            return None
        img_center_x = frame_bgr.shape[1] / 2.0
        return min(candidates, key=lambda c: abs(c.center_img[0] - img_center_x))

    def detect_all_targets(self, frame_bgr) -> List[Detection]:
        candidates = [
            self.blue_ball_detector.detect(frame_bgr),
            self.white_ball_detector.detect(frame_bgr),
            self.cola_detector.detect(frame_bgr),
        ]
        return [c for c in candidates if c is not None]

    def _target_color(self, det_type: str):
        if det_type == 'blue_ball':
            return (255, 0, 0)
        if det_type == 'white_ball':
            return (220, 220, 220)
        if det_type == 'cola':
            return (0, 0, 255)
        if det_type == 'bar':
            return (0, 255, 255)
        return (0, 255, 0)

    def draw_detection(self, vis, det: Optional[Detection], color, label_prefix: str = '', thickness: int = 2):
        if det is None:
            return
        x1, y1, x2, y2 = det.bbox_img
        cx, cy = det.center_img
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
        cv2.circle(vis, (cx, cy), 4, color, -1)
        label = f'{label_prefix}{det.det_type} ({cx},{cy}) s={det.score:.1f}'
        text_y = y1 - 8 if y1 > 20 else y2 + 20
        cv2.putText(vis, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def update_debug_visualization(self, bar_det: Optional[Detection], target_candidates: List[Detection], chosen_target: Optional[Detection]):
        if not self.show_debug_vis or self.latest_bgr is None:
            return
        try:
            vis = self.latest_bgr.copy()
            h, w = vis.shape[:2]

            cv2.line(vis, (w // 2, 0), (w // 2, h - 1), (0, 255, 0), 1)
            cv2.putText(vis, f'state: {self.state}', (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.putText(vis, f'body_z: {self.body_height_cmd:.3f}', (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            vx, vy, wz = self.motion_cmd
            cv2.putText(vis, f'cmd: ({vx:.2f}, {vy:.2f}, {wz:.2f})', (10, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if bar_det is not None:
                self.draw_detection(vis, bar_det, self._target_color('bar'), label_prefix='BAR: ', thickness=2)
                d = self.estimate_bar_depth(bar_det)
                if d is not None:
                    cv2.putText(vis, f'bar_depth={d:.3f}m', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            for det in target_candidates:
                color = self._target_color(det.det_type)
                self.draw_detection(vis, det, color, label_prefix='DET: ', thickness=2)

            if chosen_target is not None:
                color = self._target_color(chosen_target.det_type)
                self.draw_detection(vis, chosen_target, color, label_prefix='SELECT: ', thickness=3)
                d = self.estimate_depth_at_center(chosen_target.center_img)
                depth_text = 'None' if d is None else f'{d:.3f}m'
                cv2.putText(vis, f'target_depth={depth_text}', (10, 126), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.imshow(self.debug_window_name, vis)
            cv2.waitKey(1)
        except Exception as e:
            if not self._vis_warned:
                self._vis_warned = True
                self.get_logger().warn(f'debug visualization failed: {e}')

    def is_bar_centered(self, bar: Detection) -> bool:
        """判断限高杆中心是否已经在图像中心死区内。"""
        if self.latest_bgr is None or bar is None:
            return False
        img_center_x = self.latest_bgr.shape[1] / 2.0
        err_px = bar.center_img[0] - img_center_x
        return abs(err_px) <= self.bar_center_px_deadband

    def compute_bar_align_vy(self, bar: Detection) -> float:
        """
        根据限高杆中心点和图像中心的偏差，计算横向速度 vy。
        目标：在 SEARCH_BAR_AND_FORWARD 阶段边前进边横移，让限高杆保持在图像中心附近。

        当前符号沿用目标对齐 compute_align_cmd() 的方向：
        err_px = 目标中心x - 图像中心x
        vy = -k * err_norm

        如果实测发现限高杆在右侧时机器人反而越偏越远，
        把下面的负号改成正号即可。
        """
        if self.latest_bgr is None or bar is None:
            return 0.0

        img_center_x = self.latest_bgr.shape[1] / 2.0
        err_px = bar.center_img[0] - img_center_x

        if abs(err_px) <= self.bar_center_px_deadband:
            return 0.0

        err_norm = err_px / max(self.latest_bgr.shape[1] / 2.0, 1.0)
        vy = -self.bar_align_vy_k * err_norm
        vy = float(np.clip(vy, -self.bar_align_vy_max, self.bar_align_vy_max))

        if 0 < abs(vy) < self.bar_align_vy_min:
            vy = math.copysign(self.bar_align_vy_min, vy)

        return vy

    def compute_align_cmd(self, target: Detection) -> Tuple[float, float]:
        img_center_x = self.latest_bgr.shape[1] / 2.0
        err_px = target.center_img[0] - img_center_x
        depth_m = self.estimate_depth_at_center(target.center_img)
        vx = self.align_forward_speed_near if (depth_m is not None and depth_m < 0.35) else self.align_forward_speed_far
        if abs(err_px) <= self.center_px_deadband:
            vy = 0.0
        else:
            err_norm = err_px / max(self.latest_bgr.shape[1] / 2.0, 1.0)
            vy = -self.align_vy_k * err_norm
            vy = float(np.clip(vy, -self.align_vy_max, self.align_vy_max))
            if 0 < abs(vy) < self.align_vy_min:
                vy = math.copysign(self.align_vy_min, vy)
        return vx, vy

    # ---------- 主循环 ----------
    def control_loop(self):
        if self.latest_bgr is None:
            return
        now = self.now_s()
        bar_for_vis: Optional[Detection] = None
        target_candidates_for_vis: List[Detection] = []
        chosen_target_for_vis: Optional[Detection] = None

        if self.state == self.INITIAL_LATERAL_MOVE:
            # 启动后先向左横移一段固定距离，这一段不识别限高杆。
            self.send_motion_cmd(0.0, self.initial_lateral_shift_vy, 0.0)

            if self.initial_lateral_shift_start_xy is None:
                self.initial_lateral_shift_start_xy = self.get_current_xy_from_tf()

            current_xy = self.get_current_xy_from_tf()
            if self.initial_lateral_shift_start_xy is not None and current_xy is not None:
                moved = self.planar_distance(self.initial_lateral_shift_start_xy, current_xy)
                self.get_logger().info(
                    f'[INITIAL_SHIFT] moved={moved:.3f} / target={self.initial_lateral_shift_distance_m:.3f}',
                    throttle_duration_sec=0.2
                )
                if moved >= self.initial_lateral_shift_distance_m:
                    self.get_logger().info('[INITIAL_SHIFT] finished, start searching first bar')
                    self.enter_state(self.LATERAL_SEARCH_BAR)

        elif self.state == self.LATERAL_SEARCH_BAR:
            bar = self.bar_detector.detect(self.latest_bgr)
            bar_for_vis = bar

            if bar is None:
                self.send_motion_cmd(0.0, self.lateral_search_vy, 0.0)
            else:
                # 这里只负责发现限高杆，不记录距离。
                # 距离等限高杆居中稳定、准备向前走时再记录。
                self.enter_state(self.CENTER_BAR_BEFORE_FORWARD)

        elif self.state == self.CENTER_BAR_BEFORE_FORWARD:
            bar = self.bar_detector.detect(self.latest_bgr)
            bar_for_vis = bar

            if bar is None:
                self.bar_center_stable_count = 0
                self.send_motion_cmd(0.0, self.lateral_search_vy, 0.0)
            else:
                bar_vy = self.compute_bar_align_vy(bar)
                self.send_motion_cmd(0.0, bar_vy, 0.0)

                if self.is_bar_centered(bar):
                    self.bar_center_stable_count += 1
                else:
                    self.bar_center_stable_count = 0

                if self.bar_center_stable_count >= self.bar_center_stable_frames:
                    # 限高杆居中稳定，准备向前走时，记录当前杆距离。
                    # 这个距离用于撞击后后退回到本轮起始位置。
                    d = self.estimate_bar_depth(bar)
                    if d is not None:
                        self.bar_return_target_depth_m = d
                        self.get_logger().info(
                            f'[BAR] centered for {self.bar_center_stable_count} frames, '
                            f'record return target depth={d:.3f} m, start forward under bar '
                            f'({self.completed_bar_count + 1}/{self.total_bar_count})'
                        )
                    else:
                        self.get_logger().warn(
                            '[BAR] centered but depth=None, start forward without return target depth'
                        )

                    self.enter_state(self.SEARCH_BAR_AND_FORWARD)

        elif self.state == self.LATERAL_MOVE_TO_NEXT_BAR:
            # 完成一轮后，先横移一段距离，不检测限高杆，避免重复识别刚刚那根杆。
            self.send_motion_cmd(0.0, self.next_bar_lateral_shift_vy, 0.0)

            if self.lateral_shift_start_xy is None:
                self.lateral_shift_start_xy = self.get_current_xy_from_tf()

            current_xy = self.get_current_xy_from_tf()
            if self.lateral_shift_start_xy is not None and current_xy is not None:
                moved = self.planar_distance(self.lateral_shift_start_xy, current_xy)
                self.get_logger().info(
                    f'[NEXT_BAR_SHIFT] moved={moved:.3f} / target={self.next_bar_lateral_shift_distance_m:.3f}',
                    throttle_duration_sec=0.2
                )
                if moved >= self.next_bar_lateral_shift_distance_m:
                    self.get_logger().info('[NEXT_BAR_SHIFT] finished, start searching next bar')
                    self.enter_state(self.LATERAL_SEARCH_BAR)

        elif self.state == self.SEARCH_BAR_AND_FORWARD:
            bar = self.bar_detector.detect(self.latest_bgr)
            bar_for_vis = bar

            bar_vy = self.compute_bar_align_vy(bar) if bar is not None else 0.0
            self.send_motion_cmd(self.bar_search_forward_speed, bar_vy, 0.0)

            if bar is not None:
                d = self.estimate_bar_depth(bar)

                # 正常情况下本轮杆距离已经在 CENTER_BAR_BEFORE_FORWARD 居中稳定、准备前进时记录。
                # 这里仅做兜底：如果前面居中时没有有效深度，则第一次有效深度在这里记录。
                if d is not None and self.bar_return_target_depth_m is None:
                    self.bar_return_target_depth_m = d
                    self.get_logger().info(
                        f'[BAR] fallback first valid depth, record return target depth={d:.3f} m'
                    )

                if d is not None and d < self.bar_trigger_distance_m:
                    self.target_stable_count = 0
                    self.stable_target_type = None
                    self.enter_state(self.SEARCH_TARGET_AFTER_BAR)

        elif self.state == self.SEARCH_TARGET_AFTER_BAR:
            self.send_motion_cmd(self.target_search_forward_speed, 0.0, 0.0)
            target_candidates_for_vis = self.detect_all_targets(self.latest_bgr)
            if target_candidates_for_vis:
                img_center_x = self.latest_bgr.shape[1] / 2.0
                target = min(target_candidates_for_vis, key=lambda c: abs(c.center_img[0] - img_center_x))
                chosen_target_for_vis = target
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
            target_candidates_for_vis = self.detect_all_targets(self.latest_bgr)
            if not target_candidates_for_vis:
                self.locked_target = None
                self.enter_state(self.SEARCH_TARGET_AFTER_BAR)
                self.update_debug_visualization(bar_for_vis, target_candidates_for_vis, chosen_target_for_vis)
                return
            img_center_x = self.latest_bgr.shape[1] / 2.0
            target = min(target_candidates_for_vis, key=lambda c: abs(c.center_img[0] - img_center_x))
            chosen_target_for_vis = target
            self.locked_target = target
            vx, vy = self.compute_align_cmd(target)
            self.send_motion_cmd(vx, vy, 0.0)
            d = self.estimate_depth_at_center(target.center_img)
            if d is not None and d < self.hit_trigger_distance_m:
                self.enter_state(self.HIT_TARGET)

        elif self.state == self.HIT_TARGET:
            if self.locked_target is not None:
                chosen_target_for_vis = self.locked_target
                target_candidates_for_vis = [self.locked_target]
            else:
                self.enter_state(self.SEARCH_TARGET_AFTER_BAR)
                self.update_debug_visualization(bar_for_vis, target_candidates_for_vis, chosen_target_for_vis)
                return

            params = self.hit_params.get(self.locked_target.det_type, {'speed': 0.15, 'distance': 0.18})
            self.send_motion_cmd(params['speed'], 0.0, 0.0)

            if self.hit_start_xy is None:
                self.hit_start_xy = self.get_current_xy_from_tf()

            current_xy = self.get_current_xy_from_tf()
            if self.hit_start_xy is not None and current_xy is not None:
                hit_moved_dist = self.planar_distance(self.hit_start_xy, current_xy)
                self.get_logger().info(
                    f'[HIT] moved_dist={hit_moved_dist:.3f} m / target={params["distance"]:.3f} m',
                    throttle_duration_sec=0.2
                )

                if hit_moved_dist >= params['distance']:
                    # 撞击距离完成后，不直接 TASK_DONE，而是进入后退找限高杆状态。
                    self.enter_state(self.BACKOFF_TO_BAR_AFTER_HIT)
                    self.update_debug_visualization(bar_for_vis, target_candidates_for_vis, chosen_target_for_vis)
                    return

            max_hit_time = 10.0
            if now - self.state_enter_time >= max_hit_time:
                self.get_logger().warn('[HIT] timeout reached, enter BACKOFF_TO_BAR_AFTER_HIT')
                self.enter_state(self.BACKOFF_TO_BAR_AFTER_HIT)

        elif self.state == self.BACKOFF_TO_BAR_AFTER_HIT:
            bar = self.bar_detector.detect(self.latest_bgr)
            bar_for_vis = bar

            if bar is not None:
                bar_vy = self.compute_bar_align_vy(bar)
            else:
                bar_vy = 0.0

            # 后退：vx 为负；如果识别到限高杆，同时用 vy 做居中。
            self.send_motion_cmd(-self.backoff_after_hit_speed, bar_vy, 0.0)

            if bar is None:
                self.get_logger().info(
                    '[BACKOFF] bar=None, keep backing',
                    throttle_duration_sec=0.5
                )
            else:
                d = self.estimate_bar_depth(bar)
                target_d = self.bar_return_target_depth_m
                if d is not None and target_d is not None:
                    depth_err = d - target_d
                    self.get_logger().info(
                        f'[BACKOFF] bar_depth={d:.3f} m / target={target_d:.3f} m '
                        f'/ err={depth_err:.3f} m / vy={bar_vy:.3f}',
                        throttle_duration_sec=0.2
                    )

                    if (
                        now - self.state_enter_time >= self.backoff_min_time_s
                        and abs(depth_err) <= self.backoff_bar_depth_tolerance_m
                    ):
                        self.completed_bar_count += 1
                        self.get_logger().info(
                            f'[BACKOFF] reached recorded bar distance, '
                            f'completed_bar_count={self.completed_bar_count}/{self.total_bar_count}'
                        )

                        if self.completed_bar_count >= self.total_bar_count:
                            self.enter_state(self.TASK_DONE)
                        else:
                            self.enter_state(self.LATERAL_MOVE_TO_NEXT_BAR)

                        self.update_debug_visualization(bar_for_vis, target_candidates_for_vis, chosen_target_for_vis)
                        return
                elif d is None:
                    self.get_logger().info(
                        '[BACKOFF] bar detected but depth=None, keep backing and aligning',
                        throttle_duration_sec=0.5
                    )
                else:
                    self.get_logger().warn(
                        '[BACKOFF] bar_return_target_depth_m is None, cannot use depth stop condition',
                        throttle_duration_sec=1.0
                    )

        elif self.state == self.TASK_DONE:
            if self.locked_target is not None:
                chosen_target_for_vis = self.locked_target
                target_candidates_for_vis = [self.locked_target]
            # TASK_DONE 只发一次 STOP，避免每个 timer 周期反复 STOP。
            if not self.task_done_stop_sent:
                self.stop()
                self.task_done_stop_sent = True

        self.update_debug_visualization(bar_for_vis, target_candidates_for_vis, chosen_target_for_vis)

        if now - self.last_log_time > 0.25:
            self.last_log_time = now
            depth_text = 'None'
            target_text = 'None'
            log_target = chosen_target_for_vis if chosen_target_for_vis is not None else self.locked_target
            if log_target is not None:
                d = self.estimate_depth_at_center(log_target.center_img)
                depth_text = 'None' if d is None else f'{d:.3f}'
                target_text = f'{log_target.det_type}@{log_target.center_img}'
            vx, vy, wz = self.motion_cmd
            self.get_logger().info(
                f'state={self.state} body_z={self.body_height_cmd:.3f} '
                f'cmd=({vx:.3f},{vy:.3f},{wz:.3f}) target={target_text} depth={depth_text} '
                f'bar_count={self.completed_bar_count}/{self.total_bar_count}'
            )


class Robot_Ctrl(object):
    def __init__(self):
        self.rec_thread = Thread(target=self.rec_responce)
        self.send_thread = Thread(target=self.send_publish)
        self.lc_r = lcm.LCM("udpm://239.255.76.67:7670?ttl=255")
        self.lc_s = lcm.LCM("udpm://239.255.76.67:7671?ttl=255")
        self.cmd_msg = robot_control_cmd_lcmt()
        self.rec_msg = robot_control_response_lcmt()
        self.send_lock = Lock()
        self.delay_cnt = 0
        self.mode_ok = 0
        self.gait_ok = 0
        self.runing = 1

    def run(self):
        self.lc_r.subscribe("robot_control_response", self.msg_handler)
        self.send_thread.start()
        self.rec_thread.start()

    def msg_handler(self, channel, data):
        self.rec_msg = robot_control_response_lcmt().decode(data)
        if(self.rec_msg.order_process_bar >= 95):
            self.mode_ok = self.rec_msg.mode
            self.gait_ok = self.rec_msg.gait_id
        else:
            self.mode_ok = 0          
            self.gait_ok = 0

    def rec_responce(self):
        while self.runing:
            self.lc_r.handle()
            time.sleep( 0.002 )

    def Wait_finish(self, mode, gait_id):
        count = 0
        while self.runing and count < 2000: #10s
            if self.mode_ok == mode and self.gait_ok == gait_id:
                return True
            else:
                time.sleep(0.005)
                count += 1

    def send_publish(self):
        while self.runing:
            self.send_lock.acquire()
            if self.delay_cnt > 20: # Heartbeat signal 10HZ, It is used to maintain the heartbeat when life count is not updated
                self.lc_s.publish("robot_control_cmd",self.cmd_msg.encode())
                self.delay_cnt = 0
            self.delay_cnt += 1
            self.send_lock.release()
            time.sleep( 0.005 )

    def Send_cmd(self, msg):
        self.send_lock.acquire()
        self.delay_cnt = 50
        self.cmd_msg = msg
        self.send_lock.release()

    def quit(self):
        self.runing = 0
        self.rec_thread.join()
        self.send_thread.join()

class yaml_pub(Node):
    def __init__(self):
        super().__init__("cyberdogmsg_node")

        self.para_pub = self.create_publisher(YamlParam, "yaml_parameter", 10)
        self.force_pub = self.create_publisher(ApplyForce, "apply_force", 10)

    def publish_yaml_kDOUBLE(self, name: str, value: float, is_user: int = 0):
        msg = YamlParam()
        msg.name = name
        msg.kind = ControlParameterValueKind.kDOUBLE
        msg.s64_value = float(value)
        msg.is_user = int(is_user)
        self.para_pub.publish(msg)

    def publish_yaml_s64(self, name: str, value: int, is_user: int = 0):
        msg = YamlParam()
        msg.name = name
        msg.kind = ControlParameterValueKind.kS64
        msg.s64_value = int(value)
        msg.is_user = int(is_user)
        self.para_pub.publish(msg)

    def publish_yaml_vecxd(self, name: str, values, is_user: int = 1):
        msg = YamlParam()
        msg.name = name
        msg.kind = ControlParameterValueKind.kVEC_X_DOUBLE

        vec = [0.0] * 12
        for i, v in enumerate(values):
            if i < 12:
                vec[i] = float(v)

        msg.vecxd_value = vec
        msg.is_user = int(is_user)
        self.para_pub.publish(msg)

    def publish_apply_force(self, link_name: str, rel_pos, force, duration: float):
        msg = ApplyForce()
        msg.link_name = link_name
        msg.rel_pos = [float(x) for x in rel_pos]
        msg.force = [float(x) for x in force]
        msg.time = float(duration)
        self.force_pub.publish(msg)
    def publish_apply_force(self, link_name: str, rel_pos, force, duration: float):
        msg = ApplyForce()
        msg.link_name = link_name
        msg.rel_pos = [float(x) for x in rel_pos]
        msg.force = [float(x) for x in force]
        msg.time = float(duration)
        self.force_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = UnderBarTargetTaskNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
