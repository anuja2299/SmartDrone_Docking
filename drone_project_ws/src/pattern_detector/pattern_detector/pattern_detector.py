#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv_bridge
import cv2
import numpy as np
import tensorflow as tf
from geometry_msgs.msg import Point

class PatternDetector(Node):
    def __init__(self):
        super().__init__('pattern_detector')
        self.position_pub = self.create_publisher(Point, '/pattern_position', 10)
        self.image_sub = self.create_subscription(Image, '/camera', self.image_callback, 10)
        self.bridge = cv_bridge.CvBridge()
        self.model = self.create_cnn_model()
        self.get_logger().info("Pattern Detector Node Started")

    def create_cnn_model(self):
        model = tf.keras.models.load_model('/home/anuja/drone_project_ws/src/pattern_detector/qr_pattern_detector.h5')
        return model

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')  # Shape: (480, 640, 3)
        cv2.imwrite('/tmp/raw_image.jpg', cv_image)
        
        processed_image = cv2.resize(cv_image, (160, 120))  # Shape: (120, 160, 3)
        processed_image = processed_image / 255.0  # Normalize to [0, 1]
        processed_image = np.expand_dims(processed_image, axis=0)  # Shape: (1, 120, 160, 3)
        
        cv2.imwrite('/tmp/processed_image.jpg', (processed_image[0] * 255).astype(np.uint8))
        
        prediction = self.model.predict(processed_image)
        self.get_logger().info(f'Raw prediction: {prediction}')
        
        position = Point()
        position.x = float(prediction[0][0])
        position.y = float(prediction[0][1])
        position.z = 0.0
        
        self.position_pub.publish(position)
        self.get_logger().info(f'Published pattern position: [x: {position.x}, y: {position.y}, z: {position.z}]')

def main(args=None):
    rclpy.init(args=args)
    pattern_detector = PatternDetector()
    rclpy.spin(pattern_detector)
    pattern_detector.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
