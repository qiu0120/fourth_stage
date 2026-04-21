#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.qos import qos_profile_sensor_data


class BarColorDetector:
    def __init__(self):
        # ========= 颜色阈值（先按你当前仿真测出来的灰杆范围）=========
        self.lower_bar = np.array([85, 15, 35], dtype=np.uint8)
        self.upper_bar = np.array([100, 45, 80], dtype=np.uint8)

        # ========= ROI =========
        self.roi_x_ratio_min = 0.3
        self.roi_x_ratio_max = 0.7
        self.roi_y_ratio_min = 0.20
        self.roi_y_ratio_max = 0.75

        # ========= 形态学 =========
        self.kernel_open = np.ones((3, 3), np.uint8)
        self.kernel_close = np.ones((7, 11), np.uint8)  # 横向更强一些，适合连横杆

        # ========= 轮廓筛选参数 =========
        self.min_area = 500
        self.min_width = 15
        self.max_height = 1000
        self.min_aspect_ratio = 1.5
        self.max_aspect_ratio = 50.0

        # 候选位置约束：尽量不要太靠图像底部
        self.max_center_y_ratio_in_roi = 0.95

    def detect(self, frame_bgr):
        h, w = frame_bgr.shape[:2]

        # 1) 裁剪 ROI
        x1 = int(w * self.roi_x_ratio_min)
        x2 = int(w * self.roi_x_ratio_max)
        y1 = int(h * self.roi_y_ratio_min)
        y2 = int(h * self.roi_y_ratio_max)

        roi = frame_bgr[y1:y2, x1:x2].copy()
        roi_h, roi_w = roi.shape[:2]

        # 2) HSV
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # 3) 颜色分割
        mask_raw = cv2.inRange(hsv, self.lower_bar, self.upper_bar)

        # 4) 形态学处理
        mask_morph = cv2.morphologyEx(mask_raw, cv2.MORPH_OPEN, self.kernel_open)
        mask_morph = cv2.morphologyEx(mask_morph, cv2.MORPH_CLOSE, self.kernel_close)

        # 5) 找轮廓
        contours, _ = cv2.findContours(
            mask_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # 可视化图
        frame_vis = frame_bgr.copy()
        cv2.rectangle(frame_vis, (x1, y1), (x2, y2), (255, 0, 0), 2)

        contour_vis = cv2.cvtColor(mask_morph, cv2.COLOR_GRAY2BGR)
        roi_vis = roi.copy()

        candidates = []
        debug_infos = []

        roi_center_x = roi_w / 2.0

        for idx, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            rx, ry, rw, rh = cv2.boundingRect(cnt)

            if rw <= 0 or rh <= 0:
                debug_infos.append({
                    "idx": idx,
                    "area": float(area),
                    "bbox_roi": (rx, ry, rw, rh),
                    "aspect_ratio": 0.0,
                    "center_y_ratio": 0.0,
                    "center_bonus": 0.0,
                    "score": -1.0,
                    "passed": False,
                    "reasons": ["invalid_bbox"]
                })
                continue

            aspect_ratio = rw / float(rh)
            center_y_ratio = (ry + rh * 0.5) / float(max(roi_h, 1))

            # ===== 新增：更靠图像中间更优先 =====
            center_x = rx + rw / 2.0
            x_dist_norm = abs(center_x - roi_center_x) / max(roi_w / 2.0, 1.0)
            center_bonus = 1.0 - x_dist_norm  # 越靠中间越接近1，越靠边越接近0

            reasons = []

            if area < self.min_area:
                reasons.append(f"area<{self.min_area}")
            if rw < self.min_width:
                reasons.append(f"width<{self.min_width}")
            if rh > self.max_height:
                reasons.append(f"height>{self.max_height}")
            if aspect_ratio < self.min_aspect_ratio:
                reasons.append(f"aspect<{self.min_aspect_ratio}")
            if aspect_ratio > self.max_aspect_ratio:
                reasons.append(f"aspect>{self.max_aspect_ratio}")
            if center_y_ratio > self.max_center_y_ratio_in_roi:
                reasons.append(f"center_y_ratio>{self.max_center_y_ratio_in_roi:.2f}")

            passed = len(reasons) == 0

            # ===== 新增：位置优先打分 =====
            score = area * min(aspect_ratio, 10.0) * (0.3 + 0.7 * center_bonus)

            debug_infos.append({
                "idx": idx,
                "area": float(area),
                "bbox_roi": (rx, ry, rw, rh),
                "aspect_ratio": float(aspect_ratio),
                "center_y_ratio": float(center_y_ratio),
                "center_bonus": float(center_bonus),
                "score": float(score),
                "passed": passed,
                "reasons": reasons
            })

            # 所有轮廓都画灰框，方便观察
            cv2.rectangle(contour_vis, (rx, ry), (rx + rw, ry + rh), (120, 120, 120), 1)

            if passed:
                candidates.append({
                    "bbox_roi": (rx, ry, rw, rh),
                    "area": area,
                    "aspect_ratio": aspect_ratio,
                    "center_bonus": center_bonus,
                    "score": score
                })

                # 候选画黄框
                cv2.rectangle(contour_vis, (rx, ry), (rx + rw, ry + rh), (0, 255, 255), 2)
                cv2.rectangle(roi_vis, (rx, ry), (rx + rw, ry + rh), (0, 255, 255), 2)
                cv2.putText(
                    roi_vis,
                    f"PASS a={int(area)} ar={aspect_ratio:.1f} cb={center_bonus:.2f}",
                    (rx, max(ry - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 255),
                    1
                )
            else:
                # 失败画红框
                cv2.rectangle(roi_vis, (rx, ry), (rx + rw, ry + rh), (0, 0, 255), 1)
                short_reason = reasons[0] if len(reasons) > 0 else "unknown"
                cv2.putText(
                    roi_vis,
                    short_reason[:18],
                    (rx, min(ry + rh + 12, roi_h - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.40,
                    (0, 0, 255),
                    1
                )

        # 6) 选最佳候选
        if len(candidates) > 0:
            best_candidate = max(candidates, key=lambda c: c["score"])
            rx, ry, rw, rh = best_candidate["bbox_roi"]

            # ROI 坐标 -> 原图坐标
            bx1 = x1 + rx
            by1 = y1 + ry
            bx2 = bx1 + rw
            by2 = by1 + rh

            # 画最佳检测框
            cv2.rectangle(frame_vis, (bx1, by1), (bx2, by2), (0, 255, 0), 3)
            cv2.putText(
                frame_vis,
                f"BAR area={int(best_candidate['area'])} ar={best_candidate['aspect_ratio']:.2f} cb={best_candidate['center_bonus']:.2f}",
                (bx1, max(by1 - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            # ROI 上也高亮
            cv2.rectangle(roi_vis, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 3)

            detected = True
            bbox = (bx1, by1, bx2, by2)
        else:
            detected = False
            bbox = None
            cv2.putText(
                frame_vis,
                "BAR NOT FOUND",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        result = {
            "detected": detected,
            "bbox": bbox,
            "num_contours": len(contours),
            "num_candidates": len(candidates),
            "frame_vis": frame_vis,
            "roi_vis": roi_vis,
            "mask_raw": mask_raw,
            "mask_morph": mask_morph,
            "contour_vis": contour_vis,
            "debug_infos": debug_infos,
        }
        return result


class BarColorDebugNode(Node):
    def __init__(self):
        super().__init__('bar_color_debug_node')

        self.bridge = CvBridge()
        self.detector = BarColorDetector()

        self.sub = self.create_subscription(
            Image,
            '/rgb_camera/rgb_camera/image_raw',
            self.image_callback,
            qos_profile_sensor_data
        )

        self.last_log_ns = 0
        self.get_logger().info('bar_color_debug_node started')
        self.get_logger().info('Subscribed topic: /rgb_camera/rgb_camera/image_raw')
        self.get_logger().info('Press q or ESC in image window to quit.')

    def image_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge convert failed: {e}')
            return

        result = self.detector.detect(frame)

        cv2.imshow('bar_frame_vis', result['frame_vis'])
        cv2.imshow('bar_roi_vis', result['roi_vis'])
        cv2.imshow('bar_mask_raw', result['mask_raw'])
        cv2.imshow('bar_mask_morph', result['mask_morph'])
        cv2.imshow('bar_contour_vis', result['contour_vis'])

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            self.get_logger().info('Exit requested by keyboard.')
            cv2.destroyAllWindows()
            rclpy.shutdown()
            return

        # 控制日志频率，避免刷屏
        now_ns = self.get_clock().now().nanoseconds
        if now_ns - self.last_log_ns > int(0.6 * 1e9):
            self.last_log_ns = now_ns

            self.get_logger().info("========== CONTOUR DEBUG BEGIN ==========")
            self.get_logger().info(
                f"total_contours={result['num_contours']} total_candidates={result['num_candidates']}"
            )

            for info in result["debug_infos"]:
                idx = info["idx"]
                rx, ry, rw, rh = info["bbox_roi"]
                area = info["area"]
                ar = info["aspect_ratio"]
                cy = info["center_y_ratio"]
                cb = info["center_bonus"]
                score = info["score"]

                if info["passed"]:
                    self.get_logger().info(
                        f"[Contour {idx}] PASS | "
                        f"area={area:.1f}, bbox=({rx},{ry},{rw},{rh}), "
                        f"aspect={ar:.2f}, cy_ratio={cy:.2f}, "
                        f"center_bonus={cb:.2f}, score={score:.2f}"
                    )
                else:
                    reason_text = ", ".join(info["reasons"])
                    self.get_logger().info(
                        f"[Contour {idx}] FAIL | "
                        f"area={area:.1f}, bbox=({rx},{ry},{rw},{rh}), "
                        f"aspect={ar:.2f}, cy_ratio={cy:.2f}, "
                        f"center_bonus={cb:.2f}, score={score:.2f} | reasons: {reason_text}"
                    )

            if result['detected']:
                bx1, by1, bx2, by2 = result['bbox']
                self.get_logger().info(
                    f"BAR detected | bbox=({bx1},{by1},{bx2},{by2})"
                )
            else:
                self.get_logger().info("BAR not found")

            self.get_logger().info("========== CONTOUR DEBUG END ==========")


def main(args=None):
    rclpy.init(args=args)
    node = BarColorDebugNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()