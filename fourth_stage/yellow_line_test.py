#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Optional

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


class YellowDashedLineDetector:
    """
    只检测黄色竖直虚线。

    流程：
    1. 从 RGB 图像中截取 ROI。
    2. HSV 提取黄色 mask。
    3. 找所有黄色块 blobs。
    4. 过滤面积、宽高、形状不合理的黄色块。
    5. 按 y 从上到下排序。
    6. 每个黄色块都当一次起点。
    7. 往下找和上一段 x 接近、y 间隔合理的黄色块。
    8. 得到多组虚线候选。
    9. 去重。
    10. 按竖直跨度 total_span_y 排序。
    11. 返回最长的 max_dashed_lines 条。
    """

    def __init__(self, cfg: Dict[str, Any]):
        # =========================
        # ROI 参数，只看地面区域
        # =========================
        self.roi_x_ratio_min = cfg['roi_x_ratio_min']
        self.roi_x_ratio_max = cfg['roi_x_ratio_max']
        self.roi_y_ratio_min = cfg['roi_y_ratio_min']
        self.roi_y_ratio_max = cfg['roi_y_ratio_max']

        # =========================
        # 黄色 HSV 参数
        # =========================
        self.lower_yellow = np.array(
            [cfg['h_min'], cfg['s_min'], cfg['v_min']],
            dtype=np.uint8
        )
        self.upper_yellow = np.array(
            [cfg['h_max'], cfg['s_max'], cfg['v_max']],
            dtype=np.uint8
        )

        # =========================
        # 形态学参数
        # =========================
        self.open_kernel = np.ones(
            (cfg['open_kernel'], cfg['open_kernel']),
            np.uint8
        )

        # 虚线不能 close 太强，尤其 close_kernel_h 不能太大。
        # 如果太大，上下虚线段会被粘成一整条实线。
        self.close_kernel = np.ones(
            (cfg['dash_close_kernel_h'], cfg['dash_close_kernel_w']),
            np.uint8
        )

        # =========================
        # 基础黄色块过滤
        # =========================
        self.min_area = cfg['min_area']
        self.max_area = cfg['max_area']
        self.min_width = cfg['min_width']
        self.min_height = cfg['min_height']

        # =========================
        # 虚线组合参数
        # =========================
        self.dash_min_segments = cfg['dash_min_segments']
        self.dash_min_total_span_y = cfg['dash_min_total_span_y']
        self.dash_max_adjacent_x_diff = cfg['dash_max_adjacent_x_diff']
        self.dash_max_gap_y = cfg['dash_max_gap_y']
        self.dash_min_gap_y = cfg['dash_min_gap_y']
        self.dash_max_total_x_range = cfg['dash_max_total_x_range']

        # 避免把很长的黄色块当作虚线短段
        self.dash_segment_max_aspect_ratio = cfg['dash_segment_max_aspect_ratio']
        self.dash_segment_max_long_side = cfg['dash_segment_max_long_side']

        # 多条虚线去重参数
        self.dash_duplicate_iou_thresh = cfg['dash_duplicate_iou_thresh']
        self.dash_duplicate_center_x_thresh = cfg['dash_duplicate_center_x_thresh']

        # 最多显示几条虚线
        self.max_dashed_lines = cfg['max_dashed_lines']

    # ---------- ROI / mask / blob ----------
    def _roi(self, frame_bgr):
        h, w = frame_bgr.shape[:2]

        x1 = int(w * self.roi_x_ratio_min)
        x2 = int(w * self.roi_x_ratio_max)
        y1 = int(h * self.roi_y_ratio_min)
        y2 = int(h * self.roi_y_ratio_max)

        x1 = max(0, min(w - 1, x1))
        x2 = max(x1 + 1, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(y1 + 1, min(h, y2))

        roi = frame_bgr[y1:y2, x1:x2].copy()
        return (x1, y1, x2, y2), roi

    def _make_mask(self, roi_bgr):
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(
            hsv,
            self.lower_yellow,
            self.upper_yellow
        )

        # 开运算：去掉小噪声
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.open_kernel)

        # 闭运算：连接单个黄色块内部的小断裂
        # 注意：竖直虚线场景下 close_kernel_h 不要太大。
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.close_kernel)

        return mask

    def _is_valid_dash_segment_blob(self, b) -> bool:
        """
        判断这个黄色块能不能作为虚线的一段。

        太长、太细的黄色块更像实线或者其他长条，
        不参与虚线组合。
        """
        if b['long_side'] > self.dash_segment_max_long_side:
            return False

        if b['aspect_ratio'] > self.dash_segment_max_aspect_ratio:
            return False

        return True

    def _get_all_yellow_blobs(self, mask):
        """
        返回所有通过基础面积/宽高过滤的黄色块。
        这里还没有过滤是否适合作为虚线段。
        """
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        blobs = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area or area > self.max_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            if w < self.min_width or h < self.min_height:
                continue

            rect = cv2.minAreaRect(cnt)
            (cx, cy), (rw, rh), angle = rect

            long_side = max(rw, rh)
            short_side = max(1.0, min(rw, rh))
            aspect_ratio = long_side / short_side

            blob = {
                'cnt': cnt,
                'area': float(area),
                'x': int(x),
                'y': int(y),
                'w': int(w),
                'h': int(h),
                'cx': float(cx),
                'cy': float(cy),
                'long_side': float(long_side),
                'short_side': float(short_side),
                'aspect_ratio': float(aspect_ratio),
                'angle': float(angle),
                'valid_dash_segment': False,
            }

            blob['valid_dash_segment'] = self._is_valid_dash_segment_blob(blob)
            blobs.append(blob)

        return blobs

    def _get_dash_blobs(self, mask):
        """
        只返回能作为虚线短段的黄色块。
        """
        all_blobs = self._get_all_yellow_blobs(mask)
        return [b for b in all_blobs if b['valid_dash_segment']]

    # ---------- 虚线组合 ----------
    def _build_group_from_start(self, start_idx: int, blobs_sorted: List[Dict[str, Any]]):
        """
        从某个黄色块开始，向下找同一条竖直虚线的后续块。

        注意：
        这里不是和起点 base 比较 x，而是和上一段 last 比较 x。
        这样虚线比较长或者稍微倾斜时也能连起来。
        """
        base = blobs_sorted[start_idx]
        group = [base]
        last = base

        for j in range(start_idx + 1, len(blobs_sorted)):
            b = blobs_sorted[j]

            # 当前块和上一块的横向偏差
            x_diff = abs(b['cx'] - last['cx'])

            # 当前块顶部到上一块底部的间隔
            gap_y = b['y'] - (last['y'] + last['h'])

            # gap_y 过小表示重叠太多，一般不是正常虚线间隔
            if gap_y < self.dash_min_gap_y:
                continue

            if x_diff <= self.dash_max_adjacent_x_diff and gap_y <= self.dash_max_gap_y:
                group.append(b)
                last = b

        return group

    def _group_to_detection(self, group, rx1: int, ry1: int, roi_h: int) -> Optional[Detection]:
        if len(group) < self.dash_min_segments:
            return None

        min_x = min(b['x'] for b in group)
        min_y = min(b['y'] for b in group)
        max_x = max(b['x'] + b['w'] for b in group)
        max_y = max(b['y'] + b['h'] for b in group)

        total_span_y = max_y - min_y
        total_x_range = max(b['cx'] for b in group) - min(b['cx'] for b in group)
        total_area = sum(b['area'] for b in group)

        if total_span_y < self.dash_min_total_span_y:
            return None

        if total_x_range > self.dash_max_total_x_range:
            return None

        x1 = rx1 + min_x
        y1 = ry1 + min_y
        x2 = rx1 + max_x
        y2 = ry1 + max_y

        cx = rx1 + int((min_x + max_x) / 2)
        cy = ry1 + int((min_y + max_y) / 2)

        bottom_ratio = max_y / float(max(roi_h, 1))

        score = (
            300.0 * len(group)
            + 2.0 * total_span_y
            + 100.0 * bottom_ratio
            + 0.01 * total_area
            - 0.5 * total_x_range
        )

        return Detection(
            det_type='yellow_vertical_dashed_line',
            center_img=(cx, cy),
            bbox_img=(x1, y1, x2, y2),
            score=float(score),
            extra={
                'segments': len(group),
                'total_span_y': float(total_span_y),
                'total_x_range': float(total_x_range),
                'total_area': float(total_area),
                'bottom_ratio': float(bottom_ratio),
                'group_centers': [
                    (float(rx1 + b['cx']), float(ry1 + b['cy']))
                    for b in group
                ],
            }
        )

    def _bbox_iou(self, box_a, box_b) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(1, (bx2 - bx1) * (by2 - by1))

        return inter_area / float(area_a + area_b - inter_area + 1e-6)

    def _remove_duplicate_dashed(self, detections: List[Detection]) -> List[Detection]:
        """
        去掉重复虚线。

        因为每个黄色块都会当起点，所以同一条虚线可能被识别出多次：
        A+B+C+D
        B+C+D
        C+D

        这些要合并，只保留最长、最完整的那条。
        """
        if not detections:
            return []

        detections = sorted(
            detections,
            key=lambda d: (
                d.extra.get('total_span_y', 0.0),
                d.extra.get('segments', 0),
                d.score
            ),
            reverse=True
        )

        kept = []

        for det in detections:
            keep = True
            cx = det.center_img[0]

            for old in kept:
                old_cx = old.center_img[0]
                iou = self._bbox_iou(det.bbox_img, old.bbox_img)
                center_x_close = abs(cx - old_cx) <= self.dash_duplicate_center_x_thresh

                if iou >= self.dash_duplicate_iou_thresh or center_x_close:
                    keep = False
                    break

            if keep:
                kept.append(det)

        return kept

    def detect_dashed_lines(self, frame_bgr) -> List[Detection]:
        """
        返回所有满足条件的竖直虚线，按长度从长到短排序。
        """
        (rx1, ry1, rx2, ry2), roi = self._roi(frame_bgr)
        roi_h = ry2 - ry1

        mask = self._make_mask(roi)
        blobs = self._get_dash_blobs(mask)

        if len(blobs) < self.dash_min_segments:
            return []

        blobs_sorted = sorted(blobs, key=lambda b: b['cy'])

        raw_detections = []

        # 每个黄色块都当一次起点
        for i in range(len(blobs_sorted)):
            group = self._build_group_from_start(i, blobs_sorted)
            det = self._group_to_detection(group, rx1, ry1, roi_h)

            if det is not None:
                raw_detections.append(det)

        if not raw_detections:
            return []

        detections = self._remove_duplicate_dashed(raw_detections)

        detections.sort(
            key=lambda d: (
                d.extra.get('total_span_y', 0.0),
                d.extra.get('segments', 0),
                d.score
            ),
            reverse=True
        )

        return detections

    def detect_top_dashed_lines(self, frame_bgr) -> List[Detection]:
        """
        返回最长的 max_dashed_lines 条虚线。
        """
        dashed = self.detect_dashed_lines(frame_bgr)
        return dashed[:self.max_dashed_lines]

    def get_debug_blobs(self, frame_bgr):
        """
        调试用：
        返回 ROI、mask、所有通过基础过滤的黄色块 blobs。
        注意：这里包括所有黄色块，不只是 valid_dash_segment=True 的块。
        """
        (rx1, ry1, rx2, ry2), roi = self._roi(frame_bgr)
        mask = self._make_mask(roi)
        blobs = self._get_all_yellow_blobs(mask)

        return (rx1, ry1, rx2, ry2), roi, mask, blobs


class YellowDashedLineDebugNode(Node):
    def __init__(self):
        super().__init__('yellow_dashed_line_debug_node')

        self.bridge = CvBridge()

        # =========================
        # 基础参数
        # =========================
        self.declare_parameter('rgb_topic', '/rgb_camera/rgb_camera/image_raw')
        self.declare_parameter('show_debug_vis', True)
        self.declare_parameter('debug_window_name', 'yellow_dashed_line_debug')

        # 黄色块调试窗口
        self.declare_parameter('show_blob_debug_window', True)
        self.declare_parameter('blob_debug_window_name', 'yellow_blobs_debug')

        # 默认关闭，不再每帧输出所有黄色块
        self.declare_parameter('print_blob_info', True)

        # 默认关闭，不再每帧输出虚线检测结果
        self.declare_parameter('print_dashed_info', False)

        # =========================
        # 黄色 HSV 参数
        # =========================
        self.declare_parameter('yellow.h_min', 18)
        self.declare_parameter('yellow.h_max', 45)
        self.declare_parameter('yellow.s_min', 70)
        self.declare_parameter('yellow.s_max', 255)
        self.declare_parameter('yellow.v_min', 70)
        self.declare_parameter('yellow.v_max', 255)

        # =========================
        # ROI 参数
        # =========================
        self.declare_parameter('yellow.roi_x_ratio_min', 0.00)
        self.declare_parameter('yellow.roi_x_ratio_max', 1.0)
        self.declare_parameter('yellow.roi_y_ratio_min', 0.60)
        self.declare_parameter('yellow.roi_y_ratio_max', 1.00)

        # =========================
        # 基础黄色块过滤
        # =========================
        self.declare_parameter('yellow.open_kernel', 3)
        self.declare_parameter('yellow.min_area', 50)
        self.declare_parameter('yellow.max_area', 4000)
        self.declare_parameter('yellow.min_width', 3)
        self.declare_parameter('yellow.min_height', 5)

        # =========================
        # 虚线形态学参数
        # =========================
        self.declare_parameter('yellow.dash_close_kernel_h', 3)
        self.declare_parameter('yellow.dash_close_kernel_w', 5)

        # =========================
        # 虚线组合参数
        # =========================
        self.declare_parameter('yellow.dash_min_segments', 2)
        self.declare_parameter('yellow.dash_min_total_span_y', 20)
        self.declare_parameter('yellow.dash_max_adjacent_x_diff', 110)
        self.declare_parameter('yellow.dash_max_gap_y', 3000)
        self.declare_parameter('yellow.dash_min_gap_y', -10)
        self.declare_parameter('yellow.dash_max_total_x_range', 5000)

        # 避免长条参与虚线组合
        self.declare_parameter('yellow.dash_segment_max_aspect_ratio', 10.0)
        self.declare_parameter('yellow.dash_segment_max_long_side', 200)

        # 多条虚线去重参数
        self.declare_parameter('yellow.dash_duplicate_iou_thresh', 0.35)
        self.declare_parameter('yellow.dash_duplicate_center_x_thresh', 30)

        # 最多显示最长几条虚线
        self.declare_parameter('yellow.max_dashed_lines', 2)

        self.rgb_topic = self.get_parameter('rgb_topic').value
        self.show_debug_vis = bool(self.get_parameter('show_debug_vis').value)
        self.debug_window_name = str(self.get_parameter('debug_window_name').value)

        self.show_blob_debug_window = bool(self.get_parameter('show_blob_debug_window').value)
        self.blob_debug_window_name = str(self.get_parameter('blob_debug_window_name').value)

        self.print_blob_info = bool(self.get_parameter('print_blob_info').value)
        self.print_dashed_info = bool(self.get_parameter('print_dashed_info').value)

        self.detector = YellowDashedLineDetector(self._read_cfg())

        # 鼠标点击黄色块调试用
        self.latest_blob_click_records = []
        self.selected_blob_index = None
        self.blob_window_ready = False

        self.sub = self.create_subscription(
            Image,
            self.rgb_topic,
            self.rgb_callback,
            qos_profile_sensor_data
        )

        self.get_logger().info(
            f'yellow_dashed_line_debug_node started, rgb_topic={self.rgb_topic}'
        )
        self.get_logger().info(
            f'Click a yellow blob in window [{self.blob_debug_window_name}] to print its info once.'
        )

    def _read_cfg(self):
        gp = self.get_parameter

        return {
            'h_min': int(gp('yellow.h_min').value),
            'h_max': int(gp('yellow.h_max').value),
            's_min': int(gp('yellow.s_min').value),
            's_max': int(gp('yellow.s_max').value),
            'v_min': int(gp('yellow.v_min').value),
            'v_max': int(gp('yellow.v_max').value),

            'roi_x_ratio_min': float(gp('yellow.roi_x_ratio_min').value),
            'roi_x_ratio_max': float(gp('yellow.roi_x_ratio_max').value),
            'roi_y_ratio_min': float(gp('yellow.roi_y_ratio_min').value),
            'roi_y_ratio_max': float(gp('yellow.roi_y_ratio_max').value),

            'open_kernel': int(gp('yellow.open_kernel').value),
            'min_area': int(gp('yellow.min_area').value),
            'max_area': int(gp('yellow.max_area').value),
            'min_width': int(gp('yellow.min_width').value),
            'min_height': int(gp('yellow.min_height').value),

            'dash_close_kernel_h': int(gp('yellow.dash_close_kernel_h').value),
            'dash_close_kernel_w': int(gp('yellow.dash_close_kernel_w').value),

            'dash_min_segments': int(gp('yellow.dash_min_segments').value),
            'dash_min_total_span_y': int(gp('yellow.dash_min_total_span_y').value),
            'dash_max_adjacent_x_diff': int(gp('yellow.dash_max_adjacent_x_diff').value),
            'dash_max_gap_y': int(gp('yellow.dash_max_gap_y').value),
            'dash_min_gap_y': int(gp('yellow.dash_min_gap_y').value),
            'dash_max_total_x_range': int(gp('yellow.dash_max_total_x_range').value),

            'dash_segment_max_aspect_ratio': float(gp('yellow.dash_segment_max_aspect_ratio').value),
            'dash_segment_max_long_side': int(gp('yellow.dash_segment_max_long_side').value),

            'dash_duplicate_iou_thresh': float(gp('yellow.dash_duplicate_iou_thresh').value),
            'dash_duplicate_center_x_thresh': int(gp('yellow.dash_duplicate_center_x_thresh').value),

            'max_dashed_lines': int(gp('yellow.max_dashed_lines').value),
        }

    def rgb_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'RGB convert failed: {e}')
            return

        dashed_lines = self.detector.detect_top_dashed_lines(frame)

        # 黄色块调试信息：只用于窗口显示和鼠标点击查询
        roi_info, roi, mask, blobs = self.detector.get_debug_blobs(frame)

        # 默认不输出；只有用户显式设置 print_blob_info=True 才输出所有 blob
        if self.print_blob_info:
            self.print_blobs_info(blobs)

        # 默认不输出；只有用户显式设置 print_dashed_info=True 才输出虚线结果
        if self.print_dashed_info:
            self.print_dashed_lines_info(dashed_lines)

        if self.show_debug_vis:
            self.draw_debug(frame, dashed_lines)

        if self.show_blob_debug_window:
            self.draw_blob_debug(frame, roi_info, blobs, mask)

    def print_dashed_lines_info(self, dashed_lines: List[Detection]):
        if not dashed_lines:
            self.get_logger().info(
                '[YELLOW_DASHED] None',
                throttle_duration_sec=0.3
            )
            return

        parts = []
        for i, d in enumerate(dashed_lines):
            parts.append(
                f'#{i + 1}: center={d.center_img}, '
                f'bbox={d.bbox_img}, '
                f'span_y={d.extra.get("total_span_y", 0):.1f}, '
                f'seg={d.extra.get("segments", 0)}, '
                f'x_range={d.extra.get("total_x_range", 0):.1f}'
            )

        self.get_logger().info(
            '[YELLOW_DASHED] ' + ' | '.join(parts),
            throttle_duration_sec=0.3
        )

    def print_blobs_info(self, blobs: List[Dict[str, Any]]):
        """
        保留原来的整体输出函数。
        默认不调用；除非 print_blob_info=True。
        """
        if not blobs:
            self.get_logger().info(
                '[YELLOW_BLOBS] count=0',
                throttle_duration_sec=0.5
            )
            return

        lines = [f'[YELLOW_BLOBS] count={len(blobs)}']

        blobs_sorted = sorted(blobs, key=lambda b: b['cy'])

        for i, b in enumerate(blobs_sorted):
            valid = 'Y' if b.get('valid_dash_segment', False) else 'N'
            lines.append(
                f'  blob#{i + 1}: '
                f'valid_dash={valid}, '
                f'area={b["area"]:.1f}, '
                f'roi_xywh=({b["x"]},{b["y"]},{b["w"]},{b["h"]}), '
                f'center_roi=({b["cx"]:.1f},{b["cy"]:.1f}), '
                f'long={b["long_side"]:.1f}, '
                f'short={b["short_side"]:.1f}, '
                f'aspect={b["aspect_ratio"]:.2f}, '
                f'angle={b["angle"]:.1f}'
            )

        self.get_logger().info(
            '\n'.join(lines),
            throttle_duration_sec=0.5
        )

    # ---------- 鼠标点击黄色块输出信息 ----------
    def on_blob_mouse(self, event, x, y, flags, param):
        """
        鼠标点击 yellow_blobs_debug 窗口时触发。

        注意：
        yellow_blobs_debug 窗口左边是原图 vis，右边是 mask。
        只有点击左边原图区域时，才判断黄色块。

        输出只发生在鼠标左键按下这一瞬间，不会后续持续输出。
        """
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if not self.latest_blob_click_records:
            self.get_logger().info('[BLOB_CLICK] 当前没有黄色块可选')
            return

        # 右边是 mask 拼接图，点击右边不处理
        frame_w = self.latest_blob_click_records[0].get('frame_w', None)
        if frame_w is not None and x >= frame_w:
            self.get_logger().info('[BLOB_CLICK] 你点击的是右侧 mask 区域，不是左侧原图区域')
            return

        hit_items = []

        for item in self.latest_blob_click_records:
            idx = item['index']
            b = item['blob']
            x1, y1, x2, y2 = item['bbox_img']

            if x1 <= x <= x2 and y1 <= y <= y2:
                area_box = max(1, (x2 - x1) * (y2 - y1))
                hit_items.append((area_box, idx, b, item))

        if not hit_items:
            self.selected_blob_index = None
            self.get_logger().info(
                f'[BLOB_CLICK] 点击位置=({x},{y})，没有点中任何黄色块'
            )
            return

        # 如果点中了多个重叠框，选面积最小的那个，通常更符合点击目标
        hit_items.sort(key=lambda t: t[0])
        _, idx, b, item = hit_items[0]

        self.selected_blob_index = idx
        self.print_one_blob_info(idx, b, item)

    def print_one_blob_info(self, idx: int, b: Dict[str, Any], item: Dict[str, Any]):
        valid = 'Y' if b.get('valid_dash_segment', False) else 'N'

        x1, y1, x2, y2 = item['bbox_img']
        cx_img, cy_img = item['center_img']

        msg = (
            f'\n========== SELECTED YELLOW BLOB #{idx + 1} ==========\n'
            f'valid_dash_segment: {valid}\n'
            f'area: {b["area"]:.1f}\n'
            f'bbox_roi_xywh: ({b["x"]}, {b["y"]}, {b["w"]}, {b["h"]})\n'
            f'bbox_img_xyxy: ({x1}, {y1}, {x2}, {y2})\n'
            f'center_roi: ({b["cx"]:.1f}, {b["cy"]:.1f})\n'
            f'center_img: ({cx_img}, {cy_img})\n'
            f'long_side: {b["long_side"]:.1f}\n'
            f'short_side: {b["short_side"]:.1f}\n'
            f'aspect_ratio: {b["aspect_ratio"]:.2f}\n'
            f'angle: {b["angle"]:.1f}\n'
            f'============================================='
        )

        self.get_logger().info(msg)

    # ---------- 主窗口：显示最长虚线 ----------
    def draw_debug(self, frame, dashed_lines: List[Detection]):
        vis = frame.copy()
        h, w = vis.shape[:2]

        # 画 ROI
        cfg = self.detector
        rx1 = int(w * cfg.roi_x_ratio_min)
        rx2 = int(w * cfg.roi_x_ratio_max)
        ry1 = int(h * cfg.roi_y_ratio_min)
        ry2 = int(h * cfg.roi_y_ratio_max)

        cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), (255, 255, 0), 1)

        # 图像中心线
        cv2.line(vis, (w // 2, 0), (w // 2, h - 1), (0, 255, 0), 1)

        colors = [
            (0, 165, 255),  # 橙色 DASH#1
            (0, 0, 255),    # 红色 DASH#2
        ]

        for i, det in enumerate(dashed_lines):
            color = colors[i % len(colors)]
            self.draw_detection(
                vis,
                det,
                color=color,
                label=f'DASH#{i + 1}',
                thickness=3
            )

            centers = det.extra.get('group_centers', [])

            for k, p in enumerate(centers):
                px = int(round(p[0]))
                py = int(round(p[1]))

                cv2.circle(vis, (px, py), 4, color, -1)

                if k > 0:
                    qx = int(round(centers[k - 1][0]))
                    qy = int(round(centers[k - 1][1]))
                    cv2.line(vis, (qx, qy), (px, py), color, 2)

        cv2.putText(
            vis,
            f'dashed_show={len(dashed_lines)}',
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2
        )

        if dashed_lines:
            for i, det in enumerate(dashed_lines):
                color = colors[i % len(colors)]
                text = (
                    f'DASH#{i + 1}: span_y={det.extra.get("total_span_y", 0):.1f}, '
                    f'seg={det.extra.get("segments", 0)}, '
                    f'x_range={det.extra.get("total_x_range", 0):.1f}'
                )
                cv2.putText(
                    vis,
                    text,
                    (10, 58 + 26 * i),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.60,
                    color,
                    2
                )
        else:
            cv2.putText(
                vis,
                'DASH: None',
                (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2
            )

        cv2.imshow(self.debug_window_name, vis)
        cv2.waitKey(1)

    def draw_detection(self, vis, det: Detection, color, label: str, thickness: int = 2):
        x1, y1, x2, y2 = det.bbox_img
        cx, cy = det.center_img

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
        cv2.circle(vis, (cx, cy), 5, color, -1)

        info = (
            f'{label} seg={det.extra.get("segments", 0)} '
            f'span={det.extra.get("total_span_y", 0):.0f}'
        )

        text_y = y1 - 8 if y1 > 22 else y2 + 20

        cv2.putText(
            vis,
            info,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2
        )

    # ---------- 新窗口：显示所有黄色块 ----------
    def draw_blob_debug(self, frame, roi_info, blobs: List[Dict[str, Any]], mask):
        """
        单独开一个窗口显示所有黄色块。

        左边：原图 + ROI + 黄色块框 + blob 编号 + 面积
        右边：黄色 mask

        点击左边原图中的某个黄色块，只输出该黄色块的信息一次。
        """
        rx1, ry1, rx2, ry2 = roi_info

        vis = frame.copy()
        frame_h, frame_w = vis.shape[:2]

        # 画 ROI 大框
        cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), (255, 255, 0), 1)

        blobs_sorted = sorted(blobs, key=lambda b: b['cy'])

        # 更新当前帧可点击 blob 信息
        click_records = []

        for i, b in enumerate(blobs_sorted):
            x1 = rx1 + b['x']
            y1 = ry1 + b['y']
            x2 = x1 + b['w']
            y2 = y1 + b['h']

            cx = rx1 + int(round(b['cx']))
            cy = ry1 + int(round(b['cy']))

            # 能参与虚线组合的块：黄色
            # 不能参与虚线组合的块：灰色
            if b.get('valid_dash_segment', False):
                color = (0, 255, 255)
            else:
                color = (160, 160, 160)

            thickness = 2

            # 被选中的 blob 用绿色粗框高亮
            if self.selected_blob_index == i:
                color = (0, 255, 0)
                thickness = 4

            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
            cv2.circle(vis, (cx, cy), 4, color, -1)

            label = f'B{i + 1} A={b["area"]:.0f}'
            text_y = y1 - 6 if y1 > 20 else y2 + 18

            cv2.putText(
                vis,
                label,
                (x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1
            )

            click_records.append({
                'index': i,
                'blob': b,
                'bbox_img': (x1, y1, x2, y2),
                'center_img': (cx, cy),
                'frame_w': frame_w,
                'frame_h': frame_h,
            })

        self.latest_blob_click_records = click_records

        cv2.putText(
            vis,
            f'yellow blobs count={len(blobs)}',
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2
        )

        cv2.putText(
            vis,
            'click a blob to print info once',
            (10, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (0, 255, 0),
            2
        )

        cv2.putText(
            vis,
            'yellow=valid dash segment, gray=filtered by shape',
            (10, 82),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 255, 255),
            2
        )

        # mask 转成 BGR，方便拼接显示
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        mh, mw = mask_bgr.shape[:2]
        vh, vw = vis.shape[:2]

        scale = vh / max(mh, 1)
        new_w = max(1, int(mw * scale))
        mask_resized = cv2.resize(mask_bgr, (new_w, vh))

        cv2.putText(
            mask_resized,
            'yellow mask',
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2
        )

        combined = np.hstack([vis, mask_resized])

        if not self.blob_window_ready:
            cv2.namedWindow(self.blob_debug_window_name, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(self.blob_debug_window_name, self.on_blob_mouse)
            self.blob_window_ready = True

        cv2.imshow(self.blob_debug_window_name, combined)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = YellowDashedLineDebugNode()

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