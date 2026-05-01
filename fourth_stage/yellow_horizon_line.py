#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional, List

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


class YellowHorizontalLineDetector:
    """
    单独版横向黄线检测器。

    逻辑对应你主代码里的 YellowHorizontalLineDetector：
    1. 取 ROI
    2. HSV 阈值提取黄色
    3. 开运算 + 闭运算
    4. 找轮廓
    5. 过滤面积、宽高、宽高比、宽度比例、中心偏移、倾斜角
    6. 根据 bottom_y、面积、宽度比例、倾斜角打分
    7. 返回分数最高的横向黄线
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.roi_x_ratio_min = cfg['roi_x_ratio_min']
        self.roi_x_ratio_max = cfg['roi_x_ratio_max']
        self.roi_y_ratio_min = cfg['roi_y_ratio_min']
        self.roi_y_ratio_max = cfg['roi_y_ratio_max']

        self.lower_yellow = np.array(
            [cfg['h_min'], cfg['s_min'], cfg['v_min']],
            dtype=np.uint8
        )
        self.upper_yellow = np.array(
            [cfg['h_max'], cfg['s_max'], cfg['v_max']],
            dtype=np.uint8
        )

        self.open_kernel = np.ones(
            (cfg['open_kernel'], cfg['open_kernel']),
            np.uint8
        )
        self.close_kernel = np.ones(
            (cfg['close_kernel_h'], cfg['close_kernel_w']),
            np.uint8
        )

        self.min_area = cfg['min_area']
        self.min_width = cfg['min_width']
        self.min_height = cfg['min_height']
        self.min_width_ratio = cfg['min_width_ratio']
        self.min_wh_ratio = cfg['min_wh_ratio']
        self.max_tilt_deg = cfg['max_tilt_deg']
        self.center_tolerance_ratio = cfg['center_tolerance_ratio']

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

        return (x1, y1, x2, y2), frame_bgr[y1:y2, x1:x2].copy()

    def _signed_line_angle_deg(self, cnt) -> float:
        """
        用 cv2.fitLine 估计横线倾斜角。
        0 度表示水平线。
        正负号可以用于后面 wz 修正。
        """
        if cnt is None or len(cnt) < 2:
            return 0.0

        vx, vy, _, _ = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
        vx = float(vx)
        vy = float(vy)

        angle = math.degrees(math.atan2(vy, vx))

        while angle > 90.0:
            angle -= 180.0
        while angle < -90.0:
            angle += 180.0

        return float(angle)

    def detect(self, frame_bgr) -> Tuple[Optional[Detection], np.ndarray, Tuple[int, int, int, int], List[Dict[str, Any]]]:
        """
        返回：
        det: 检测到的最佳横向黄线，没有则 None
        mask: 当前 ROI 里的黄色 mask
        roi_box: ROI 在原图中的位置
        debug_infos: 每一个黄色候选块的检测信息，包括不满足条件的原因
        """
        h, w = frame_bgr.shape[:2]
        (rx1, ry1, rx2, ry2), roi = self._roi(frame_bgr)
        roi_h, roi_w = roi.shape[:2]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.open_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.close_kernel)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        candidates = []
        debug_infos = []

        for idx, cnt in enumerate(contours):
            reasons = []

            area = cv2.contourArea(cnt)
            x, y, bw, bh = cv2.boundingRect(cnt)

            wh_ratio = bw / float(max(bh, 1))
            width_ratio = bw / float(max(roi_w, 1))

            cx_roi = x + bw / 2.0
            roi_cx = roi_w / 2.0
            center_offset_ratio = abs(cx_roi - roi_cx) / float(max(roi_w, 1))

            angle_deg = self._signed_line_angle_deg(cnt)
            abs_tilt_deg = abs(angle_deg)

            bx1 = rx1 + x
            by1 = ry1 + y
            bx2 = bx1 + bw
            by2 = by1 + bh

            cx = bx1 + bw // 2
            cy = by1 + bh // 2

            bottom_y = by2
            bottom_ratio = bottom_y / float(max(h, 1))

            # ========= 逐项记录失败原因，不再直接 continue =========
            if area < self.min_area:
                reasons.append(f'area<{self.min_area}')

            if bw < self.min_width:
                reasons.append(f'w<{self.min_width}')

            if bh < self.min_height:
                reasons.append(f'h<{self.min_height}')

            if wh_ratio < self.min_wh_ratio:
                reasons.append(f'wh<{self.min_wh_ratio:.1f}')

            if width_ratio < self.min_width_ratio:
                reasons.append(f'wr<{self.min_width_ratio:.2f}')

            if center_offset_ratio > self.center_tolerance_ratio:
                reasons.append(f'center>{self.center_tolerance_ratio:.2f}')

            if abs_tilt_deg > self.max_tilt_deg:
                reasons.append(f'tilt>{self.max_tilt_deg:.0f}')

            passed = len(reasons) == 0

            info = {
                'idx': idx,
                'bbox_roi': (int(x), int(y), int(bw), int(bh)),
                'bbox_img': (int(bx1), int(by1), int(bx2), int(by2)),
                'center_img': (int(cx), int(cy)),
                'area': float(area),
                'width': int(bw),
                'height': int(bh),
                'wh_ratio': float(wh_ratio),
                'width_ratio': float(width_ratio),
                'center_offset_ratio': float(center_offset_ratio),
                'angle_deg': float(angle_deg),
                'abs_tilt_deg': float(abs_tilt_deg),
                'bottom_y': int(bottom_y),
                'bottom_ratio': float(bottom_ratio),
                'passed': passed,
                'reasons': reasons,
            }
            debug_infos.append(info)

            if not passed:
                continue

            # 和主代码一致：优先选择更靠近图像底部的横线
            score = (
                3.0 * bottom_y
                + 0.02 * area
                + 100.0 * width_ratio
                - 2.0 * abs_tilt_deg
            )

            det = Detection(
                det_type='yellow_horizontal_line',
                center_img=(int(cx), int(cy)),
                bbox_img=(int(bx1), int(by1), int(bx2), int(by2)),
                score=float(score),
                extra={
                    'area': float(area),
                    'angle_deg': float(angle_deg),
                    'abs_tilt_deg': float(abs_tilt_deg),
                    'width_ratio': float(width_ratio),
                    'wh_ratio': float(wh_ratio),
                    'center_offset_ratio': float(center_offset_ratio),
                    'bottom_y': int(bottom_y),
                    'bottom_ratio': float(bottom_ratio),
                }
            )

            candidates.append(det)

        if not candidates:
            return None, mask, (rx1, ry1, rx2, ry2), debug_infos

        best = max(candidates, key=lambda d: d.score)
        return best, mask, (rx1, ry1, rx2, ry2), debug_infos

class TestYellowHorizontalLineNode(Node):
    def __init__(self):
        super().__init__('test_yellow_horizontal_line_node')

        self.bridge = CvBridge()

        # 图像话题
        self.declare_parameter('rgb_topic', '/rgb_camera/rgb_camera/image_raw')

        # 是否显示 OpenCV 窗口
        self.declare_parameter('show_debug_vis', True)

        # 是否显示黄色 mask 窗口
        self.declare_parameter('show_mask', True)

        # 失败候选日志输出间隔。会把当前帧里每一个 FAIL 候选都编号输出。
        # 设小一点输出更频繁，设大一点减少刷屏。
        self.declare_parameter('fail_log_interval_s', 0.5)

        # 判断黄线是否到达图像下方的阈值，只用于调试显示，不控制机器人
        self.declare_parameter('stop_line_y_ratio', 0.82)

        # =========================
        # 横向黄线参数：默认按你现在主代码 final_yellow 参数
        # =========================
        self.declare_parameter('final_yellow.h_min', 18)
        self.declare_parameter('final_yellow.h_max', 45)
        self.declare_parameter('final_yellow.s_min', 70)
        self.declare_parameter('final_yellow.s_max', 255)
        self.declare_parameter('final_yellow.v_min', 70)
        self.declare_parameter('final_yellow.v_max', 255)

        # 看图像中下区域
        self.declare_parameter('final_yellow.roi_x_ratio_min', 0.30)
        self.declare_parameter('final_yellow.roi_x_ratio_max', 0.70)
        self.declare_parameter('final_yellow.roi_y_ratio_min', 0.50)
        self.declare_parameter('final_yellow.roi_y_ratio_max', 1.00)

        self.declare_parameter('final_yellow.open_kernel', 3)
        self.declare_parameter('final_yellow.close_kernel_h', 5)
        self.declare_parameter('final_yellow.close_kernel_w', 11)

        self.declare_parameter('final_yellow.min_area', 1000)
        self.declare_parameter('final_yellow.min_width', 20)
        self.declare_parameter('final_yellow.min_height', 3)
        self.declare_parameter('final_yellow.min_width_ratio', 0.75)
        self.declare_parameter('final_yellow.min_wh_ratio', 1.5)
        self.declare_parameter('final_yellow.max_tilt_deg', 35.0)
        self.declare_parameter('final_yellow.center_tolerance_ratio', 0.60)

        self.rgb_topic = self.get_parameter('rgb_topic').value
        self.show_debug_vis = bool(self.get_parameter('show_debug_vis').value)
        self.show_mask = bool(self.get_parameter('show_mask').value)
        self.fail_log_interval_s = float(self.get_parameter('fail_log_interval_s').value)
        self.stop_line_y_ratio = float(self.get_parameter('stop_line_y_ratio').value)
        self.last_fail_log_time_s = -1e9

        cfg = self._read_final_yellow_cfg()
        self.detector = YellowHorizontalLineDetector(cfg)

        self.sub = self.create_subscription(
            Image,
            self.rgb_topic,
            self.image_callback,
            qos_profile_sensor_data
        )

        self.get_logger().info(
            f'[INIT] test yellow horizontal line node started, rgb_topic={self.rgb_topic}'
        )

    def _read_final_yellow_cfg(self):
        gp = self.get_parameter

        return {
            'h_min': int(gp('final_yellow.h_min').value),
            'h_max': int(gp('final_yellow.h_max').value),
            's_min': int(gp('final_yellow.s_min').value),
            's_max': int(gp('final_yellow.s_max').value),
            'v_min': int(gp('final_yellow.v_min').value),
            'v_max': int(gp('final_yellow.v_max').value),

            'roi_x_ratio_min': float(gp('final_yellow.roi_x_ratio_min').value),
            'roi_x_ratio_max': float(gp('final_yellow.roi_x_ratio_max').value),
            'roi_y_ratio_min': float(gp('final_yellow.roi_y_ratio_min').value),
            'roi_y_ratio_max': float(gp('final_yellow.roi_y_ratio_max').value),

            'open_kernel': int(gp('final_yellow.open_kernel').value),
            'close_kernel_h': int(gp('final_yellow.close_kernel_h').value),
            'close_kernel_w': int(gp('final_yellow.close_kernel_w').value),

            'min_area': int(gp('final_yellow.min_area').value),
            'min_width': int(gp('final_yellow.min_width').value),
            'min_height': int(gp('final_yellow.min_height').value),
            'min_width_ratio': float(gp('final_yellow.min_width_ratio').value),
            'min_wh_ratio': float(gp('final_yellow.min_wh_ratio').value),
            'max_tilt_deg': float(gp('final_yellow.max_tilt_deg').value),
            'center_tolerance_ratio': float(gp('final_yellow.center_tolerance_ratio').value),
        }

    def _short_reason_text(self, reasons, max_items=3):
        if not reasons:
            return 'PASS'
        show = reasons[:max_items]
        text = ','.join(show)
        if len(reasons) > max_items:
            text += ',...'
        return text

    def image_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge convert failed: {e}')
            return

        det, mask, roi_box, debug_infos = self.detector.detect(frame)

        vis = frame.copy()
        h, w = vis.shape[:2]

        rx1, ry1, rx2, ry2 = roi_box

        # 画 ROI
        cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), (255, 255, 0), 2)
        cv2.putText(
            vis,
            'ROI',
            (rx1, max(20, ry1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2
        )

        # 画 stop_line_y_ratio 对应的水平线
        stop_y = int(h * self.stop_line_y_ratio)
        cv2.line(vis, (0, stop_y), (w - 1, stop_y), (0, 0, 255), 2)
        cv2.putText(
            vis,
            f'stop_ratio={self.stop_line_y_ratio:.2f}',
            (10, max(20, stop_y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

        # ============================================================
        # 新增调试显示：
        # 对每一个 HSV 黄色候选块都画框。
        # 通过条件：绿色框 PASS
        # 不通过条件：红色框 FAIL，并显示不满足的条件
        # ============================================================
        fail_count = 0
        pass_count = 0

        for cand_idx, info in enumerate(debug_infos):
            bx1, by1, bx2, by2 = info['bbox_img']
            reasons = info.get('reasons', [])
            passed = bool(info.get('passed', False))

            if passed:
                pass_count += 1
                color = (0, 255, 0)
                label = (
                    f"P{cand_idx} wr={info['width_ratio']:.2f} "
                    f"wh={info['wh_ratio']:.1f} "
                    f"tilt={info['abs_tilt_deg']:.1f}"
                )
            else:
                fail_count += 1
                color = (0, 0, 255)
                label = f'F{cand_idx} ' + self._short_reason_text(reasons)

            cv2.rectangle(vis, (bx1, by1), (bx2, by2), color, 1)

            text_y = max(15, by1 - 5)
            cv2.putText(
                vis,
                label,
                (bx1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1
            )

        cv2.putText(
            vis,
            f'yellow candidates: pass={pass_count}, fail={fail_count}',
            (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        if det is None:
            self.get_logger().info(
                f'[YELLOW_H] not detected, yellow_blobs={len(debug_infos)}, fail={fail_count}',
                throttle_duration_sec=0.3
            )

            cv2.putText(
                vis,
                'NO HORIZONTAL YELLOW LINE',
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

            # 日志里把每一个失败候选都输出出来。
            # 注意：这里不用 throttle_duration_sec，因为同一行循环打印会共享 throttle，
            # 容易导致只看到 FAIL_0。改成手动按时间间隔整批输出。
            now_s = self.get_clock().now().nanoseconds * 1e-9
            if now_s - self.last_fail_log_time_s >= self.fail_log_interval_s:
                self.last_fail_log_time_s = now_s

                fail_infos = [
                    (idx, x) for idx, x in enumerate(debug_infos)
                    if not x.get('passed', False)
                ]
                fail_infos = sorted(
                    fail_infos,
                    key=lambda item: item[1].get('area', 0.0),
                    reverse=True
                )

                self.get_logger().info(
                    f'[YELLOW_H_FAIL_SUMMARY] total_yellow_blobs={len(debug_infos)}, '
                    f'pass={pass_count}, fail={fail_count}'
                )

                for rank, (orig_idx, info) in enumerate(fail_infos):
                    self.get_logger().info(
                        f"[YELLOW_H_FAIL_{rank}] orig_id=F{orig_idx}, "
                        f"reason={self._short_reason_text(info.get('reasons', []), max_items=10)} | "
                        f"bbox={info['bbox_img']}, "
                        f"area={info['area']:.0f}/{self.detector.min_area}, "
                        f"w={info['width']}/{self.detector.min_width}, "
                        f"h={info['height']}/{self.detector.min_height}, "
                        f"wr={info['width_ratio']:.2f}/{self.detector.min_width_ratio:.2f}, "
                        f"wh={info['wh_ratio']:.2f}/{self.detector.min_wh_ratio:.2f}, "
                        f"center={info['center_offset_ratio']:.2f}/{self.detector.center_tolerance_ratio:.2f}, "
                        f"tilt={info['abs_tilt_deg']:.1f}/{self.detector.max_tilt_deg:.1f}"
                    )

        else:
            bx1, by1, bx2, by2 = det.bbox_img
            cx, cy = det.center_img

            bottom_y = int(det.extra.get('bottom_y', 0))
            bottom_ratio = float(det.extra.get('bottom_ratio', 0.0))
            angle_deg = float(det.extra.get('angle_deg', 0.0))
            abs_tilt_deg = float(det.extra.get('abs_tilt_deg', 0.0))
            width_ratio = float(det.extra.get('width_ratio', 0.0))
            wh_ratio = float(det.extra.get('wh_ratio', 0.0))
            area = float(det.extra.get('area', 0.0))
            center_offset_ratio = float(det.extra.get('center_offset_ratio', 0.0))

            reached = bottom_ratio >= self.stop_line_y_ratio

            # 最佳检测框：用更粗的黄色框覆盖一遍
            cv2.rectangle(vis, (bx1, by1), (bx2, by2), (0, 255, 255), 3)
            cv2.circle(vis, (cx, cy), 5, (0, 255, 255), -1)

            # bottom_y 线
            cv2.line(vis, (0, bottom_y), (w - 1, bottom_y), (0, 255, 255), 1)

            text_color = (0, 255, 0) if reached else (0, 255, 255)

            cv2.putText(
                vis,
                f'YELLOW_H detected reached={reached}',
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                text_color,
                2
            )
            cv2.putText(
                vis,
                f'bottom={bottom_y}, ratio={bottom_ratio:.3f}/{self.stop_line_y_ratio:.3f}',
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                text_color,
                2
            )
            cv2.putText(
                vis,
                f'angle={angle_deg:.1f} deg, abs={abs_tilt_deg:.1f}',
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                text_color,
                2
            )
            cv2.putText(
                vis,
                f'area={area:.0f}, width_ratio={width_ratio:.2f}, wh={wh_ratio:.2f}, center_off={center_offset_ratio:.2f}',
                (20, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                text_color,
                2
            )

            self.get_logger().info(
                f'[YELLOW_H] detected: '
                f'bottom={bottom_y}, ratio={bottom_ratio:.3f}/{self.stop_line_y_ratio:.3f}, '
                f'angle={angle_deg:.1f}deg, '
                f'area={area:.0f}, width_ratio={width_ratio:.2f}, '
                f'wh={wh_ratio:.2f}, center_offset={center_offset_ratio:.2f}, '
                f'reached={reached}, pass={pass_count}, fail={fail_count}',
                throttle_duration_sec=0.2
            )

        if self.show_debug_vis:
            cv2.imshow('yellow_horizontal_debug', vis)

            if self.show_mask:
                cv2.imshow('yellow_horizontal_mask', mask)

            cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = TestYellowHorizontalLineNode()

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