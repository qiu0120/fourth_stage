#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.qos import qos_profile_sensor_data


class HSVPickerNode(Node):
    def __init__(self):
        super().__init__('hsv_picker_node')

        self.bridge = CvBridge()
        self.window_name = 'HSV Picker'
        self.image_topic = '/rgb_camera/rgb_camera/image_raw'

        self.frame_bgr = None
        self.frame_hsv = None

        self.clicked_point = None
        self.last_info_text = 'Left click to inspect HSV'

        self.neighborhood_half_size = 4   # 9x9 邻域
        self.need_create_window = True

        self.sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            qos_profile_sensor_data
        )

        self.get_logger().info(f'Subscribing: {self.image_topic}')
        self.get_logger().info('Left click image to inspect pixel BGR/HSV.')
        self.get_logger().info('Press q or ESC in image window to quit.')

    def create_window_once(self):
        if self.need_create_window:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(self.window_name, self.on_mouse)
            self.need_create_window = False

    def image_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge convert failed: {e}')
            return

        self.frame_bgr = frame
        self.frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        self.create_window_once()

        vis = self.draw_overlay(frame.copy())

        cv2.imshow(self.window_name, vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            self.get_logger().info('Exit requested by keyboard.')
            cv2.destroyAllWindows()
            rclpy.shutdown()

    def on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if self.frame_bgr is None or self.frame_hsv is None:
            return

        h, w = self.frame_bgr.shape[:2]
        if x < 0 or x >= w or y < 0 or y >= h:
            return

        self.clicked_point = (x, y)

        # 单点像素
        bgr = self.frame_bgr[y, x].astype(np.int32)
        hsv = self.frame_hsv[y, x].astype(np.int32)

        # 邻域统计
        x1 = max(0, x - self.neighborhood_half_size)
        x2 = min(w, x + self.neighborhood_half_size + 1)
        y1 = max(0, y - self.neighborhood_half_size)
        y2 = min(h, y + self.neighborhood_half_size + 1)

        patch_bgr = self.frame_bgr[y1:y2, x1:x2]
        patch_hsv = self.frame_hsv[y1:y2, x1:x2]

        mean_bgr = np.mean(patch_bgr.reshape(-1, 3), axis=0)
        mean_hsv = np.mean(patch_hsv.reshape(-1, 3), axis=0)

        min_hsv = np.min(patch_hsv.reshape(-1, 3), axis=0)
        max_hsv = np.max(patch_hsv.reshape(-1, 3), axis=0)

        self.last_info_text = (
            f'P({x},{y}) '
            f'BGR=({int(bgr[0])},{int(bgr[1])},{int(bgr[2])}) '
            f'HSV=({int(hsv[0])},{int(hsv[1])},{int(hsv[2])})'
        )

        self.get_logger().info('---------------- HSV PICK ----------------')
        self.get_logger().info(f'Point: ({x}, {y})')
        self.get_logger().info(
            f'Single Pixel BGR: ({int(bgr[0])}, {int(bgr[1])}, {int(bgr[2])})'
        )
        self.get_logger().info(
            f'Single Pixel HSV: ({int(hsv[0])}, {int(hsv[1])}, {int(hsv[2])})'
        )
        self.get_logger().info(
            'Patch Range: '
            f'x=[{x1},{x2-1}], y=[{y1},{y2-1}], size={patch_bgr.shape[1]}x{patch_bgr.shape[0]}'
        )
        self.get_logger().info(
            'Patch Mean BGR: '
            f'({mean_bgr[0]:.2f}, {mean_bgr[1]:.2f}, {mean_bgr[2]:.2f})'
        )
        self.get_logger().info(
            'Patch Mean HSV: '
            f'({mean_hsv[0]:.2f}, {mean_hsv[1]:.2f}, {mean_hsv[2]:.2f})'
        )
        self.get_logger().info(
            f'Patch Min HSV: ({int(min_hsv[0])}, {int(min_hsv[1])}, {int(min_hsv[2])})'
        )
        self.get_logger().info(
            f'Patch Max HSV: ({int(max_hsv[0])}, {int(max_hsv[1])}, {int(max_hsv[2])})'
        )

    def draw_overlay(self, frame):
        h, w = frame.shape[:2]

        # 顶部提示条
        cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 0), -1)
        cv2.putText(
            frame,
            self.last_info_text[:120],
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

        if self.clicked_point is not None:
            x, y = self.clicked_point

            # 十字线
            cv2.line(frame, (x - 15, y), (x + 15, y), (0, 255, 0), 2)
            cv2.line(frame, (x, y - 15), (x, y + 15), (0, 255, 0), 2)
            cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

            # 邻域框
            x1 = max(0, x - self.neighborhood_half_size)
            x2 = min(w - 1, x + self.neighborhood_half_size)
            y1 = max(0, y - self.neighborhood_half_size)
            y2 = min(h - 1, y + self.neighborhood_half_size)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 1)

            # 坐标文本
            cv2.putText(
                frame,
                f'({x},{y})',
                (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )

        return frame


def main(args=None):
    rclpy.init(args=args)
    node = HSVPickerNode()

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