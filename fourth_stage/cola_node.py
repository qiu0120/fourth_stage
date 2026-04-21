#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.qos import qos_profile_sensor_data


class ColaDetector:
    def __init__(self):
        # ========= 可乐 HSV 范围 =========
        # 这里先给一个占位范围，你后面按实际测到的 HSV 改
        # 例如如果可乐偏深色/灰色/棕色，需要你替换
        self.lower_cola = np.array([0, 0, 0], dtype=np.uint8)
        self.upper_cola = np.array([20, 20, 20], dtype=np.uint8)

        # ========= ROI =========
        self.roi_x_ratio_min = 0.20
        self.roi_x_ratio_max = 0.80
        self.roi_y_ratio_min = 0.10
        self.roi_y_ratio_max = 0.95

        # ========= 形态学 =========
        self.kernel_open = np.ones((3, 3), np.uint8)
        self.kernel_close = np.ones((5, 5), np.uint8)

        # ========= 圆柱体候选筛选参数 =========
        self.min_area = 250
        self.max_area = 80000

        self.min_width = 8
        self.max_width = 5000

        self.min_height = 20
        self.max_height = 10000

        # 高宽比 h / w，越大越像竖直圆柱
        self.min_hw_ratio = 1.5
        self.max_hw_ratio = 20.0

        # 候选位置约束
        self.max_center_y_ratio_in_roi = 1.00

    def detect(self, frame_bgr):
        h, w = frame_bgr.shape[:2]

        # 1) 裁 ROI
        x1 = int(w * self.roi_x_ratio_min)
        x2 = int(w * self.roi_x_ratio_max)
        y1 = int(h * self.roi_y_ratio_min)
        y2 = int(h * self.roi_y_ratio_max)

        roi = frame_bgr[y1:y2, x1:x2].copy()
        roi_h, roi_w = roi.shape[:2]

        frame_vis = frame_bgr.copy()
        cv2.rectangle(frame_vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
        roi_vis = roi.copy()

        # 2) HSV 提取可乐颜色
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask_raw = cv2.inRange(hsv, self.lower_cola, self.upper_cola)

        # 3) 形态学
        mask_morph = cv2.morphologyEx(mask_raw, cv2.MORPH_OPEN, self.kernel_open)
        mask_morph = cv2.morphologyEx(mask_morph, cv2.MORPH_CLOSE, self.kernel_close)

        # 4) 找轮廓
        contours, _ = cv2.findContours(
            mask_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        contour_vis = cv2.cvtColor(mask_morph, cv2.COLOR_GRAY2BGR)

        candidates = []
        debug_infos = []

        roi_center_x = roi_w / 2.0

        for idx, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            x, y, bw, bh = cv2.boundingRect(cnt)

            if bw <= 0 or bh <= 0:
                debug_infos.append({
                    "idx": idx,
                    "bbox_roi": (x, y, bw, bh),
                    "area": float(area),
                    "hw_ratio": 0.0,
                    "center_y_ratio": 0.0,
                    "center_bonus": 0.0,
                    "score": -1.0,
                    "passed": False,
                    "reasons": ["invalid_bbox"],
                })
                continue

            hw_ratio = bh / float(bw)
            center_y_ratio = (y + 0.5 * bh) / float(max(roi_h, 1))

            # 越靠中间越优先
            center_x = x + 0.5 * bw
            x_dist_norm = abs(center_x - roi_center_x) / max(roi_w / 2.0, 1.0)
            center_bonus = 1.0 - x_dist_norm

            reasons = []

            if area < self.min_area:
                reasons.append(f"area<{self.min_area}")
            if area > self.max_area:
                reasons.append(f"area>{self.max_area}")
            if bw < self.min_width:
                reasons.append(f"width<{self.min_width}")
            if bw > self.max_width:
                reasons.append(f"width>{self.max_width}")
            if bh < self.min_height:
                reasons.append(f"height<{self.min_height}")
            if bh > self.max_height:
                reasons.append(f"height>{self.max_height}")
            if hw_ratio < self.min_hw_ratio:
                reasons.append(f"hw_ratio<{self.min_hw_ratio:.2f}")
            if hw_ratio > self.max_hw_ratio:
                reasons.append(f"hw_ratio>{self.max_hw_ratio:.2f}")
            if center_y_ratio > self.max_center_y_ratio_in_roi:
                reasons.append(f"center_y_ratio>{self.max_center_y_ratio_in_roi:.2f}")

            passed = len(reasons) == 0

            # 打分：面积大 + 更高更细长 + 更居中
            score = area * min(hw_ratio, 8.0) * (0.3 + 0.7 * center_bonus)

            debug_infos.append({
                "idx": idx,
                "bbox_roi": (x, y, bw, bh),
                "area": float(area),
                "hw_ratio": float(hw_ratio),
                "center_y_ratio": float(center_y_ratio),
                "center_bonus": float(center_bonus),
                "score": float(score),
                "passed": passed,
                "reasons": reasons,
            })

            # 所有轮廓先画灰框
            cv2.rectangle(contour_vis, (x, y), (x + bw, y + bh), (120, 120, 120), 1)

            if passed:
                candidates.append({
                    "bbox_roi": (x, y, bw, bh),
                    "area": float(area),
                    "hw_ratio": float(hw_ratio),
                    "center_bonus": float(center_bonus),
                    "score": float(score),
                })

                # contour_vis 里通过候选画黄框
                cv2.rectangle(contour_vis, (x, y), (x + bw, y + bh), (0, 255, 255), 2)
            else:
                # roi_vis 里失败候选画红框并标失败原因
                cv2.rectangle(roi_vis, (x, y), (x + bw, y + bh), (0, 0, 255), 1)
                short_reason = reasons[0] if len(reasons) > 0 else "unknown"
                cv2.putText(
                    roi_vis,
                    short_reason[:18],
                    (x, min(y + bh + 12, roi_h - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.40,
                    (0, 0, 255),
                    1
                )

        # 5) 只选一个最佳可乐
        best_cola = None
        if len(candidates) > 0:
            best_cola = max(candidates, key=lambda c: c["score"])

            bx, by, bw, bh = best_cola["bbox_roi"]

            # 原图坐标
            bx1 = x1 + bx
            by1 = y1 + by
            bx2 = bx1 + bw
            by2 = by1 + bh

            cx_img = bx1 + bw // 2
            cy_img = by1 + bh // 2

            # frame_vis 只画最佳可乐
            cv2.rectangle(frame_vis, (bx1, by1), (bx2, by2), (0, 0, 255), 3)
            cv2.circle(frame_vis, (cx_img, cy_img), 4, (0, 255, 255), -1)
            cv2.putText(
                frame_vis,
                f"BEST COLA hw={best_cola['hw_ratio']:.2f}",
                (bx1, max(by1 - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

            # roi_vis 只画最佳可乐
            cv2.rectangle(roi_vis, (bx, by), (bx + bw, by + bh), (0, 255, 255), 3)
            cv2.circle(roi_vis, (bx + bw // 2, by + bh // 2), 4, (0, 255, 255), -1)
            cv2.putText(
                roi_vis,
                f"BEST hw={best_cola['hw_ratio']:.2f}",
                (bx, max(by - 5, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1
            )

            detected = True
        else:
            detected = False
            cv2.putText(
                frame_vis,
                "COLA NOT FOUND",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        result = {
            "detected": detected,
            "best_cola": best_cola,
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


class ColaDebugNode(Node):
    def __init__(self):
        super().__init__('cola_debug_node')

        self.bridge = CvBridge()
        self.detector = ColaDetector()

        self.sub = self.create_subscription(
            Image,
            '/rgb_camera/rgb_camera/image_raw',
            self.image_callback,
            qos_profile_sensor_data
        )

        self.last_log_ns = 0
        self.get_logger().info('cola_debug_node started')
        self.get_logger().info('Subscribed topic: /rgb_camera/rgb_camera/image_raw')
        self.get_logger().info('Press q or ESC in image window to quit.')

    def image_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge convert failed: {e}')
            return

        result = self.detector.detect(frame)

        cv2.imshow('cola_frame_vis', result['frame_vis'])
        cv2.imshow('cola_roi_vis', result['roi_vis'])
        cv2.imshow('cola_mask_raw', result['mask_raw'])
        cv2.imshow('cola_mask_morph', result['mask_morph'])
        cv2.imshow('cola_contour_vis', result['contour_vis'])

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            self.get_logger().info('Exit requested by keyboard.')
            cv2.destroyAllWindows()
            rclpy.shutdown()
            return

        now_ns = self.get_clock().now().nanoseconds
        if now_ns - self.last_log_ns > int(0.6 * 1e9):
            self.last_log_ns = now_ns

            self.get_logger().info("========== COLA DEBUG BEGIN ==========")
            self.get_logger().info(
                f"total_contours={result['num_contours']} total_candidates={result['num_candidates']}"
            )

            for info in result["debug_infos"]:
                idx = info["idx"]
                x, y, bw, bh = info["bbox_roi"]
                area = info["area"]
                hw_ratio = info["hw_ratio"]
                cy = info["center_y_ratio"]
                cb = info["center_bonus"]
                score = info["score"]

                if info["passed"]:
                    self.get_logger().info(
                        f"[Contour {idx}] PASS | "
                        f"area={area:.1f}, bbox=({x},{y},{bw},{bh}), "
                        f"hw_ratio={hw_ratio:.2f}, cy_ratio={cy:.2f}, "
                        f"center_bonus={cb:.2f}, score={score:.2f}"
                    )
                else:
                    reason_text = ", ".join(info["reasons"])
                    self.get_logger().info(
                        f"[Contour {idx}] FAIL | "
                        f"area={area:.1f}, bbox=({x},{y},{bw},{bh}), "
                        f"hw_ratio={hw_ratio:.2f}, cy_ratio={cy:.2f}, "
                        f"center_bonus={cb:.2f}, score={score:.2f} | reasons: {reason_text}"
                    )

            if result["best_cola"] is not None:
                best = result["best_cola"]
                x, y, bw, bh = best["bbox_roi"]
                self.get_logger().info(
                    f"BEST COLA | bbox_roi=({x},{y},{bw},{bh}), "
                    f"area={best['area']:.1f}, hw_ratio={best['hw_ratio']:.2f}, "
                    f"center_bonus={best['center_bonus']:.2f}, score={best['score']:.2f}"
                )
            else:
                self.get_logger().info("COLA not found")

            self.get_logger().info("========== COLA DEBUG END ==========")


def main(args=None):
    rclpy.init(args=args)
    node = ColaDebugNode()
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