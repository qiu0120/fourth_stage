#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.qos import qos_profile_sensor_data


class YellowLineDebugger:
    def __init__(self):
        # HSV 黄色阈值，可根据你的仿真画面再调
        self.lower_yellow = np.array([18, 100, 100], dtype=np.uint8)
        self.upper_yellow = np.array([38, 255, 255], dtype=np.uint8)

        # 形态学参数
        self.kernel_open = np.ones((3, 3), np.uint8)
        self.kernel_close = np.ones((9, 3), np.uint8)   # 竖向更强，适合竖线

        # ROI 比例：只看下方地面区域
        self.roi_y_ratio_min = 0.55
        self.roi_y_ratio_max = 1.0
        self.roi_x_ratio_min = 0
        self.roi_x_ratio_max = 1.0

        # 峰值/间距阈值
        self.peak_thresh = 20.0
        self.min_gap_ratio = 0.18
        self.max_gap_ratio = 0.90

        # 是否显示列投影
        self.show_projection = True

    def detect(self, frame):
        h, w = frame.shape[:2]

        # 1) 裁 ROI
        y1 = int(h * self.roi_y_ratio_min)
        y2 = int(h * self.roi_y_ratio_max)
        x1 = int(w * self.roi_x_ratio_min)
        x2 = int(w * self.roi_x_ratio_max)

        roi = frame[y1:y2, x1:x2].copy()
        roi_h, roi_w = roi.shape[:2]

        # 2) HSV 提黄
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask_raw = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)

        # 3) 形态学
        mask_morph = cv2.morphologyEx(mask_raw, cv2.MORPH_OPEN, self.kernel_open)
        mask_morph = cv2.morphologyEx(mask_morph, cv2.MORPH_CLOSE, self.kernel_close)

        # 4) 列投影
        col_sum = np.sum(mask_morph > 0, axis=0).astype(np.float32)

        # 平滑
        if roi_w >= 21:
            col_sum_smooth = cv2.GaussianBlur(
                col_sum.reshape(1, -1), (1, 21), 0
            ).flatten()
        else:
            col_sum_smooth = col_sum.copy()

        # 5) 左右半区找峰
        mid = roi_w // 2
        x_left_local = int(np.argmax(col_sum_smooth[:mid]))
        x_right_local = int(np.argmax(col_sum_smooth[mid:]) + mid)

        left_strength = float(col_sum_smooth[x_left_local])
        right_strength = float(col_sum_smooth[x_right_local])

        gap = x_right_local - x_left_local
        min_gap = int(roi_w * self.min_gap_ratio)
        max_gap = int(roi_w * self.max_gap_ratio)

        found = (
            left_strength > self.peak_thresh and
            right_strength > self.peak_thresh and
            min_gap < gap < max_gap
        )

        frame_vis = frame.copy()
        cv2.rectangle(frame_vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
        roi_vis = roi.copy()

        result = {
            "found": False,
            "frame_vis": frame_vis,
            "roi_vis": roi_vis,
            "mask_raw": mask_raw,
            "mask_morph": mask_morph,
            "projection_img": self.make_projection_image(
                col_sum_smooth, x_left_local, x_right_local
            ) if self.show_projection else None,
            "x_left": None,
            "x_right": None,
            "x_mid": None,
            "gap": None,
            "left_strength": left_strength,
            "right_strength": right_strength,
        }

        if found:
            # ROI 坐标 -> 原图坐标
            x_left = x1 + x_left_local
            x_right = x1 + x_right_local
            x_mid = (x_left + x_right) // 2

            result["found"] = True
            result["x_left"] = x_left
            result["x_right"] = x_right
            result["x_mid"] = x_mid
            result["gap"] = gap

            # ROI 上画线
            cv2.line(roi_vis, (x_left_local, 0), (x_left_local, roi_h - 1), (0, 255, 0), 2)
            cv2.line(roi_vis, (x_right_local, 0), (x_right_local, roi_h - 1), (0, 255, 0), 2)
            roi_mid = (x_left_local + x_right_local) // 2
            cv2.line(roi_vis, (roi_mid, 0), (roi_mid, roi_h - 1), (0, 0, 255), 2)

            # 原图上画线
            cv2.line(frame_vis, (x_left, y1), (x_left, y2), (0, 255, 0), 2)
            cv2.line(frame_vis, (x_right, y1), (x_right, y2), (0, 255, 0), 2)
            cv2.line(frame_vis, (x_mid, y1), (x_mid, y2), (0, 0, 255), 2)

            # 图像中心
            cx = w // 2
            cv2.line(frame_vis, (cx, y1), (cx, y2), (255, 255, 255), 2)

            error = x_mid - cx
            text = f"left={x_left}, right={x_right}, mid={x_mid}, err={error}, gap={gap}"
            cv2.putText(
                frame_vis, text, (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
            )

            result["frame_vis"] = frame_vis
            result["roi_vis"] = roi_vis
        else:
            cv2.putText(
                frame_vis, "LINES NOT FOUND", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
            )
            result["frame_vis"] = frame_vis
            result["roi_vis"] = roi_vis

        return result

    def make_projection_image(self, col_sum_smooth, x_left_local, x_right_local):
        w = len(col_sum_smooth)
        h = 300
        img = np.zeros((h, w, 3), dtype=np.uint8)

        max_val = np.max(col_sum_smooth)
        if max_val < 1e-5:
            return img

        pts = []
        for x in range(w):
            y = int(h - 1 - (col_sum_smooth[x] / max_val) * (h - 20))
            pts.append((x, y))

        for i in range(1, len(pts)):
            cv2.line(img, pts[i - 1], pts[i], (0, 255, 255), 1)

        cv2.line(img, (x_left_local, 0), (x_left_local, h - 1), (0, 255, 0), 2)
        cv2.line(img, (x_right_local, 0), (x_right_local, h - 1), (0, 255, 0), 2)

        return img


class YellowLineDebugNode(Node):
    def __init__(self):
        super().__init__('yellow_line_debug_node')

        self.bridge = CvBridge()
        self.detector = YellowLineDebugger()

        self.sub = self.create_subscription(
            Image,
            '/rgb_camera/rgb_camera/image_raw',
            self.image_callback,
            qos_profile_sensor_data
        )

        self.last_log_time = self.get_clock().now()
        self.get_logger().info('yellow_line_debug_node started, subscribing /rgb_camera/rgb_camera/image_raw')

    def image_callback(self, msg: Image):
        try:
            # 常见写法：bgr8
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge convert failed: {e}')
            return

        result = self.detector.detect(frame)

        cv2.imshow("frame_vis", result["frame_vis"])
        cv2.imshow("roi_vis", result["roi_vis"])
        cv2.imshow("mask_raw", result["mask_raw"])
        cv2.imshow("mask_morph", result["mask_morph"])

        if result["projection_img"] is not None:
            cv2.imshow("projection", result["projection_img"])

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            self.get_logger().info("Pressed q/ESC, shutting down...")
            cv2.destroyAllWindows()
            rclpy.shutdown()
            return

        # 降低日志频率，别每帧都刷屏
        now = self.get_clock().now()
        dt = (now - self.last_log_time).nanoseconds / 1e9
        if dt > 0.3:
            self.last_log_time = now
            if result["found"]:
                self.get_logger().info(
                    f"found | left={result['x_left']} "
                    f"right={result['x_right']} "
                    f"mid={result['x_mid']} "
                    f"gap={result['gap']} "
                    f"L={result['left_strength']:.1f} "
                    f"R={result['right_strength']:.1f}"
                )
            else:
                self.get_logger().info(
                    f"not found | "
                    f"L={result['left_strength']:.1f} "
                    f"R={result['right_strength']:.1f}"
                )


def main(args=None):
    rclpy.init(args=args)
    node = YellowLineDebugNode()
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