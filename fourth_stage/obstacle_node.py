#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.qos import qos_profile_sensor_data


class ObstacleBlueDepthDetector:
    def __init__(self):
        # ========= ROI =========
        self.roi_x_ratio_min = 0.15
        self.roi_x_ratio_max = 0.85
        self.roi_y_ratio_min = 0.20
        self.roi_y_ratio_max = 0.95

        # ========= 蓝色 HSV 范围 =========
        # 先给一个比较宽的初值，后面你可以按仿真测量值再调
        self.lower_blue = np.array([90, 60, 40], dtype=np.uint8)
        self.upper_blue = np.array([140, 255, 255], dtype=np.uint8)

        # ========= 深度阈值（米）=========
        self.depth_min_m = 0.05
        self.depth_max_m = 1.20

        # ========= 形态学 =========
        self.kernel_open = np.ones((3, 3), np.uint8)
        self.kernel_close = np.ones((5, 5), np.uint8)

        # ========= 蓝色轮廓筛选 =========
        self.min_area = 150
        self.min_width = 10
        self.min_height = 10
        self.max_aspect_ratio = 4.5   # 太细长的蓝色区域不像方块障碍
        self.min_bottom_y_ratio_in_roi = 0.70  # 太靠上不要

        # ========= 深度筛选 =========
        self.min_valid_depth_ratio = 0.20   # 候选框里至少有多少比例是有效深度
        self.min_near_depth_ratio = 0.50    # 候选框里至少有多少比例落在近距离范围内

    def _depth_to_meters(self, depth_img):
        if depth_img.dtype == np.float32:
            depth_m = depth_img.copy()
        elif depth_img.dtype == np.uint16:
            depth_m = depth_img.astype(np.float32) / 1000.0
        else:
            depth_m = depth_img.astype(np.float32)

        depth_m[~np.isfinite(depth_m)] = 0.0
        return depth_m

    def _normalize_depth_for_vis(self, depth_roi_m):
        vis = depth_roi_m.copy()
        valid = np.isfinite(vis) & (vis > 0.0)
        if not np.any(valid):
            return np.zeros(vis.shape, dtype=np.uint8)

        v = vis[valid]
        dmin = np.percentile(v, 5)
        dmax = np.percentile(v, 95)
        if dmax - dmin < 1e-6:
            dmax = dmin + 1e-6

        vis = np.clip(vis, dmin, dmax)
        vis = (vis - dmin) / (dmax - dmin)
        vis = (255.0 * (1.0 - vis)).astype(np.uint8)
        vis[~valid] = 0
        return vis

    def detect(self, frame_bgr, depth_img):
        h, w = frame_bgr.shape[:2]
        depth_m = self._depth_to_meters(depth_img)

        # 1) 裁 ROI
        x1 = int(w * self.roi_x_ratio_min)
        x2 = int(w * self.roi_x_ratio_max)
        y1 = int(h * self.roi_y_ratio_min)
        y2 = int(h * self.roi_y_ratio_max)

        roi_bgr = frame_bgr[y1:y2, x1:x2].copy()
        roi_depth_m = depth_m[y1:y2, x1:x2].copy()
        roi_h, roi_w = roi_bgr.shape[:2]

        frame_vis = frame_bgr.copy()
        cv2.rectangle(frame_vis, (x1, y1), (x2, y2), (255, 0, 0), 2)

        roi_vis = roi_bgr.copy()

        # 2) 深度可视化
        depth_vis_gray = self._normalize_depth_for_vis(roi_depth_m)
        depth_vis = cv2.applyColorMap(depth_vis_gray, cv2.COLORMAP_JET)

        # 3) 先提蓝色
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        blue_mask_raw = cv2.inRange(hsv, self.lower_blue, self.upper_blue)

        # 4) 形态学处理蓝色 mask
        blue_mask_morph = cv2.morphologyEx(blue_mask_raw, cv2.MORPH_OPEN, self.kernel_open)
        blue_mask_morph = cv2.morphologyEx(blue_mask_morph, cv2.MORPH_CLOSE, self.kernel_close)

        # 5) 在蓝色 mask 上找轮廓
        contours, _ = cv2.findContours(
            blue_mask_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        candidates = []
        debug_infos = []

        for idx, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            rx, ry, rw, rh = cv2.boundingRect(cnt)

            if rw <= 0 or rh <= 0:
                debug_infos.append({
                    "idx": idx,
                    "bbox_roi": (rx, ry, rw, rh),
                    "area": float(area),
                    "aspect_ratio": 0.0,
                    "bottom_y_ratio": 0.0,
                    "valid_depth_ratio": 0.0,
                    "near_depth_ratio": 0.0,
                    "median_depth": -1.0,
                    "passed": False,
                    "reasons": ["invalid_bbox"],
                })
                continue

            aspect_ratio = rw / float(rh)
            bottom_y_ratio = (ry + rh) / float(max(roi_h, 1))

            reasons = []

            # 蓝色几何筛选
            if area < self.min_area:
                reasons.append(f"area<{self.min_area}")
            if rw < self.min_width:
                reasons.append(f"width<{self.min_width}")
            if rh < self.min_height:
                reasons.append(f"height<{self.min_height}")
            if aspect_ratio > self.max_aspect_ratio:
                reasons.append(f"aspect>{self.max_aspect_ratio}")
            if bottom_y_ratio < self.min_bottom_y_ratio_in_roi:
                reasons.append(f"bottom_y_ratio<{self.min_bottom_y_ratio_in_roi:.2f}")

            # 6) 对应深度区域做筛选
            depth_patch = roi_depth_m[ry:ry + rh, rx:rx + rw]
            valid_mask = np.isfinite(depth_patch) & (depth_patch > 0.0)
            near_mask = valid_mask & (depth_patch >= self.depth_min_m) & (depth_patch <= self.depth_max_m)

            total_pixels = max(rw * rh, 1)
            valid_depth_ratio = float(np.count_nonzero(valid_mask)) / float(total_pixels)
            near_depth_ratio = float(np.count_nonzero(near_mask)) / float(total_pixels)

            if np.any(near_mask):
                median_depth = float(np.median(depth_patch[near_mask]))
            else:
                median_depth = -1.0

            if valid_depth_ratio < self.min_valid_depth_ratio:
                reasons.append(f"valid_depth_ratio<{self.min_valid_depth_ratio:.2f}")
            if near_depth_ratio < self.min_near_depth_ratio:
                reasons.append(f"near_depth_ratio<{self.min_near_depth_ratio:.2f}")

            passed = len(reasons) == 0

            debug_infos.append({
                "idx": idx,
                "bbox_roi": (rx, ry, rw, rh),
                "area": float(area),
                "aspect_ratio": float(aspect_ratio),
                "bottom_y_ratio": float(bottom_y_ratio),
                "valid_depth_ratio": float(valid_depth_ratio),
                "near_depth_ratio": float(near_depth_ratio),
                "median_depth": float(median_depth),
                "passed": passed,
                "reasons": reasons,
            })

            # 可视化：所有轮廓先画灰框
            cv2.rectangle(depth_vis, (rx, ry), (rx + rw, ry + rh), (120, 120, 120), 1)

            if passed:
                candidates.append({
                    "bbox_roi": (rx, ry, rw, rh),
                    "area": area,
                    "aspect_ratio": aspect_ratio,
                    "bottom_y_ratio": bottom_y_ratio,
                    "valid_depth_ratio": valid_depth_ratio,
                    "near_depth_ratio": near_depth_ratio,
                    "median_depth": median_depth,
                })

                # 通过的候选：黄框
                cv2.rectangle(roi_vis, (rx, ry), (rx + rw, ry + rh), (0, 255, 255), 2)
                cv2.rectangle(depth_vis, (rx, ry), (rx + rw, ry + rh), (0, 255, 255), 2)
                cv2.putText(
                    roi_vis,
                    f"PASS d={median_depth:.2f}",
                    (rx, max(ry - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 255),
                    1
                )
            else:
                # 不通过：红框
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

        # 7) 原图上把所有通过候选都画出来，不只留一个
        if len(candidates) > 0:
            for i, cand in enumerate(candidates):
                rx, ry, rw, rh = cand["bbox_roi"]
                bx1 = x1 + rx
                by1 = y1 + ry
                bx2 = bx1 + rw
                by2 = by1 + rh

                cv2.rectangle(frame_vis, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                cv2.putText(
                    frame_vis,
                    f"OBS{i} d={cand['median_depth']:.2f}",
                    (bx1, max(by1 - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

            detected = True
        else:
            detected = False
            cv2.putText(
                frame_vis,
                "OBSTACLE NOT FOUND",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        result = {
            "detected": detected,
            "num_contours": len(contours),
            "num_candidates": len(candidates),
            "candidates": candidates,   # 保留全部通过候选
            "frame_vis": frame_vis,
            "roi_vis": roi_vis,
            "depth_vis": depth_vis,
            "blue_mask_raw": blue_mask_raw,
            "blue_mask_morph": blue_mask_morph,
            "debug_infos": debug_infos,
        }
        return result


class ObstacleRGBBlueDepthFilterNode(Node):
    def __init__(self):
        super().__init__('obstacle_rgb_blue_depth_filter_node')

        self.bridge = CvBridge()
        self.detector = ObstacleBlueDepthDetector()

        self.latest_bgr = None
        self.latest_depth = None

        self.rgb_sub = self.create_subscription(
            Image,
            '/rgb_camera/rgb_camera/image_raw',
            self.rgb_callback,
            qos_profile_sensor_data
        )

        self.depth_sub = self.create_subscription(
            Image,
            '/d435/depth/d435_depth/depth/image_raw',
            self.depth_callback,
            qos_profile_sensor_data
        )

        self.last_log_ns = 0
        self.get_logger().info('obstacle_rgb_blue_depth_filter_node started')
        self.get_logger().info('Subscribed RGB: /rgb_camera/rgb_camera/image_raw')
        self.get_logger().info('Subscribed DEPTH: /d435/depth/d435_depth/depth/image_raw')
        self.get_logger().info('Press q or ESC in image window to quit.')

    def rgb_callback(self, msg: Image):
        try:
            self.latest_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'RGB cv_bridge convert failed: {e}')
            return
        self.try_process()

    def depth_callback(self, msg: Image):
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f'DEPTH cv_bridge convert failed: {e}')
            return
        self.try_process()

    def try_process(self):
        if self.latest_bgr is None or self.latest_depth is None:
            return

        if self.latest_bgr.shape[:2] != self.latest_depth.shape[:2]:
            self.get_logger().warn(
                f"RGB size {self.latest_bgr.shape[:2]} != DEPTH size {self.latest_depth.shape[:2]}, skip"
            )
            return

        result = self.detector.detect(self.latest_bgr, self.latest_depth)

        cv2.imshow('obstacle_frame_vis', result['frame_vis'])
        cv2.imshow('obstacle_roi_vis', result['roi_vis'])
        cv2.imshow('obstacle_depth_vis', result['depth_vis'])
        cv2.imshow('obstacle_blue_mask_raw', result['blue_mask_raw'])
        cv2.imshow('obstacle_blue_mask_morph', result['blue_mask_morph'])

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            self.get_logger().info('Exit requested by keyboard.')
            cv2.destroyAllWindows()
            rclpy.shutdown()
            return

        now_ns = self.get_clock().now().nanoseconds
        if now_ns - self.last_log_ns > int(0.6 * 1e9):
            self.last_log_ns = now_ns

            self.get_logger().info("========== OBSTACLE DEBUG BEGIN ==========")
            self.get_logger().info(
                f"total_contours={result['num_contours']} total_candidates={result['num_candidates']}"
            )

            for info in result["debug_infos"]:
                idx = info["idx"]
                rx, ry, rw, rh = info["bbox_roi"]
                area = info["area"]
                ar = info["aspect_ratio"]
                by = info["bottom_y_ratio"]
                vdr = info["valid_depth_ratio"]
                ndr = info["near_depth_ratio"]
                md = info["median_depth"]

                if info["passed"]:
                    self.get_logger().info(
                        f"[Contour {idx}] PASS | "
                        f"area={area:.1f}, bbox=({rx},{ry},{rw},{rh}), "
                        f"aspect={ar:.2f}, bottom_ratio={by:.2f}, "
                        f"valid_depth_ratio={vdr:.2f}, near_depth_ratio={ndr:.2f}, "
                        f"median_depth={md:.2f}"
                    )
                else:
                    reason_text = ", ".join(info["reasons"])
                    self.get_logger().info(
                        f"[Contour {idx}] FAIL | "
                        f"area={area:.1f}, bbox=({rx},{ry},{rw},{rh}), "
                        f"aspect={ar:.2f}, bottom_ratio={by:.2f}, "
                        f"valid_depth_ratio={vdr:.2f}, near_depth_ratio={ndr:.2f}, "
                        f"median_depth={md:.2f} | reasons: {reason_text}"
                    )

            if result["num_candidates"] > 0:
                for i, cand in enumerate(result["candidates"]):
                    rx, ry, rw, rh = cand["bbox_roi"]
                    self.get_logger().info(
                        f"[Candidate {i}] bbox_roi=({rx},{ry},{rw},{rh}), "
                        f"area={cand['area']:.1f}, aspect={cand['aspect_ratio']:.2f}, "
                        f"bottom_ratio={cand['bottom_y_ratio']:.2f}, "
                        f"valid_depth_ratio={cand['valid_depth_ratio']:.2f}, "
                        f"near_depth_ratio={cand['near_depth_ratio']:.2f}, "
                        f"median_depth={cand['median_depth']:.2f}"
                    )
            else:
                self.get_logger().info("OBSTACLE not found")

            self.get_logger().info("========== OBSTACLE DEBUG END ==========")


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleRGBBlueDepthFilterNode()
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