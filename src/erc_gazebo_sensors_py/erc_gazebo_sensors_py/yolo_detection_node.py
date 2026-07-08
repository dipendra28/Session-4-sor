import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2
import numpy as np
import threading
import time

from geometry_msgs.msg import Twist
from ultralytics import YOLO


class YoloDetectorNode(Node):

    def __init__(self):
        super().__init__('yolo_detector')

        self.model = YOLO("yolov8s.pt")
        self.get_logger().info("YOLO model loaded")

        self.target_class = input("Enter target object: ").strip().lower()
        self.get_logger().info(f"Searching for: {self.target_class}")
        print(f"Searching for: {self.target_class}")

        self.bridge = CvBridge()

        self.subscription = self.create_subscription(
            Image,
            'camera/image',
            self.image_callback,
            1
        )

        self.depth_subscription = self.create_subscription(
            Image,
            'camera/depth_image',
            self.depth_callback,
            1
        )

        self.latest_frame = None
        self.latest_depth = None

        self.frame_lock = threading.Lock()
        self.depth_lock = threading.Lock()

        self.running = True
        self.prev_time = time.time()

        self.spin_thread = threading.Thread(
            target=self.spin_thread_func,
            daemon=True
        )
        self.spin_thread.start()
        
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.last_search_print = 0.0
    
    def reset_for_next_mission(self):
        self.publish_velocity(0.0, 0.0)
    
        print("\nMission Completed!")
        print("Target Reached Successfully\n")
    
        self.target_class = input("Enter target object: ").strip().lower()
    
        print(f"Searching for: {self.target_class}")
        self.get_logger().info(f"Searching for: {self.target_class}")
    
        self.last_search_print = 0.0
    
    def spin_thread_func(self):
        while rclpy.ok() and self.running:
            rclpy.spin_once(self, timeout_sec=0.05)
            
    def publish_velocity(self, linear_x=0.0, angular_z=0.0):
        msg = Twist()
        msg.linear.x = linear_x
        msg.angular.z = angular_z
        self.cmd_pub.publish(msg)

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        with self.frame_lock:
            self.latest_frame = frame

    def depth_callback(self, msg):
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            with self.depth_lock:
                self.latest_depth = depth
        except Exception as e:
            self.get_logger().error(f"Depth conversion error: {e}")

    def get_distance_at_center(self, cx, cy, patch_radius=3):
        with self.depth_lock:
            if self.latest_depth is None:
                return None
            depth = self.latest_depth.copy()

        h, w = depth.shape[:2]

        cx = max(0, min(cx, w - 1))
        cy = max(0, min(cy, h - 1))

        x1 = max(0, cx - patch_radius)
        x2 = min(w, cx + patch_radius + 1)
        y1 = max(0, cy - patch_radius)
        y2 = min(h, cy + patch_radius + 1)

        patch = depth[y1:y2, x1:x2].astype(np.float32)

        if depth.dtype == np.uint16:
            patch = patch / 1000.0

        valid = patch[np.isfinite(patch)]
        valid = valid[valid > 0.0]

        if valid.size == 0:
            return None

        return float(np.median(valid))

    def display_image(self):
        cv2.namedWindow("YOLO Detection", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow("YOLO Detection", 1600, 900)

        while rclpy.ok() and self.running:
            with self.frame_lock:
                frame = None if self.latest_frame is None else self.latest_frame.copy()

            if frame is not None:
                result = self.run_yolo(frame)
                cv2.imshow("YOLO Detection", result)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                self.running = False
                break

        cv2.destroyAllWindows()

    def run_yolo(self, frame):
        CONF_THRESHOLD = 0.35
        
        STOP_DISTANCE = 0.70
        CENTER_TOLERANCE = 35
        FORWARD_SPEED = 0.18
        TURN_SPEED = 0.25

        results = self.model(
            frame,
            conf=CONF_THRESHOLD,
            imgsz=640,
            verbose=False
        )

        detections = []
        target_found = False
        target_distance = None
        target_cx = None

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = self.model.names[class_id]

                if class_name.lower() != self.target_class:
                    continue

                target_found = True

                color = self.class_color(class_id)

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                target_cx = cx

                distance = self.get_distance_at_center(cx, cy)

                if distance is not None:
                    target_distance = distance
                    print("Target Found")
                    print(f"Distance to {self.target_class}: {distance:.2f} m")
                    detections.append(
                        f"{class_name} {confidence:.2f} | {distance:.2f} m"
                    )
                else:
                    detections.append(
                        f"{class_name} {confidence:.2f} | depth unavailable"
                    )

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                label = f"{class_name} {confidence:.2f}"
                if distance is not None:
                    label += f" {distance:.2f}m"

                (tw, th), baseline = cv2.getTextSize(
                    label,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    2
                )

                text_y = max(y1 - 10, th + 10)

                cv2.rectangle(
                    frame,
                    (x1, text_y - th - baseline),
                    (x1 + tw + 10, text_y + baseline),
                    color,
                    -1
                )

                cv2.putText(
                    frame,
                    label,
                    (x1 + 5, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

                cv2.circle(frame, (cx, cy), 5, color, -1)

                if distance is not None:
                    cv2.putText(
                        frame,
                        f"Distance: {distance:.2f} m",
                        (x1, min(y2 + 25, frame.shape[0] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2
                    )

        target_mode = "SEARCH"

        if target_found and target_distance is not None:
            image_center = frame.shape[1] / 2
            error_x = target_cx - image_center

            print("Target Locked")
            print(f"Distance: {target_distance:.2f} m")

            if target_distance <= STOP_DISTANCE:
                target_mode = "COMPLETE"
                self.reset_for_next_mission()

            elif abs(error_x) > CENTER_TOLERANCE:
                target_mode = "TRACKING"

                if error_x < 0:
                    self.publish_velocity(0.0, TURN_SPEED)
                else:
                    self.publish_velocity(0.0, -TURN_SPEED)

            else:
                target_mode = "APPROACH"
                self.publish_velocity(FORWARD_SPEED, 0.0)

        elif target_found:
            target_mode = "TARGET FOUND"
            self.publish_velocity(0.0, 0.0)

        else:
            target_mode = "SEARCH"
            self.publish_velocity(0.0, TURN_SPEED)

            now = time.time()

            if now - self.last_search_print > 1.0:
                print("Searching...")
                self.last_search_print = now

        current_time = time.time()
        fps = 1.0 / max(current_time - self.prev_time, 1e-6)
        self.prev_time = current_time

        dashboard_width = 360
        dashboard = np.zeros((frame.shape[0], dashboard_width, 3), dtype=np.uint8)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale_title = 0.65
        scale_text = 0.45
        thick_title = 2
        thick_text = 1
        
        cv2.putText(dashboard, "Object Hunt", (15, 30),
                    font, scale_title, (0, 255, 255), thick_title)
        
        cv2.putText(dashboard, f"TARGET: {self.target_class}", (15, 65),
                    font, scale_text, (255, 255, 255), thick_text)
        
        status = "FOUND" if target_found else "SEARCHING"
        
        cv2.putText(dashboard, f"STATUS: {status}", (15, 95),
                    font, scale_text, (0, 255, 0), thick_text)
        
        cv2.putText(dashboard, f"MODE: {target_mode}", (15, 125),
                    font, scale_text, (0, 255, 255), thick_text)
        
        if target_distance is not None:
            cv2.putText(dashboard, f"DIST: {target_distance:.2f} m", (15, 155),
                        font, scale_text, (0, 255, 255), thick_text)
        else:
            cv2.putText(dashboard, "DIST: --", (15, 155),
                        font, scale_text, (0, 255, 255), thick_text)

        y = 190
        for det in detections[:2]:
            cv2.putText(dashboard, det[:32], (15, y),
                        font, 0.38, (255, 255, 255), 1)
            y += 25

        return np.hstack((frame, dashboard))

    def class_color(self, class_id):
        np.random.seed(class_id)
        return tuple(int(c) for c in np.random.randint(100, 255, 3))

    def stop(self):
        self.running = False
        if self.spin_thread.is_alive():
            self.spin_thread.join(timeout=1)


def main(args=None):
    print("OpenCV Version:", cv2.__version__)

    rclpy.init(args=args)
    node = YoloDetectorNode()

    try:
        node.display_image()
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
