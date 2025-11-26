import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, TransformStamped
import tf2_ros
import numpy as np
from rclpy.time import Time

class SimulatedOdomNode(Node):
    def __init__(self):
        super().__init__('simulated_odom_node')
        self.publisher_ = self.create_publisher(Odometry, '/odom', 10)
        self.cmd_vel_sub = self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10)
        self.timer = self.create_timer(0.1, self.publish_odom)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.pos = np.array([0.0, 0.0])  # Initial position
        self.vel = np.array([0.0, 0.0])  # Initial velocity
        self.last_time = self.get_clock().now()
        self.get_logger().info("Simulated Odom Node initialized with pos=[0.0, 0.0]")

    def cmd_vel_callback(self, msg):
        self.vel = np.array([msg.linear.x, msg.linear.y])
        self.get_logger().info(f"Received and updated vel: x={msg.linear.x}, y={msg.linear.y}, vel={self.vel}")

    def publish_odom(self):
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds / 1e9
        self.get_logger().info(f"Calculated dt: {dt}, current vel={self.vel}")
        if dt > 0 and dt < 1.0:
            # Calculate new position with fresh array
            new_pos = self.pos + self.vel * dt
            self.get_logger().info(f"New position before clip: x={new_pos[0]}, y={new_pos[1]}")
            self.pos = np.clip(new_pos, -20.0, 20.0)
            self.get_logger().info(f"New position after clip: x={self.pos[0]}, y={self.pos[1]}")
        else:
            self.get_logger().warn(f"Invalid dt: {dt}, skipping update")
        self.last_time = current_time

        # Publish Odometry
        odom_msg = Odometry()
        odom_msg.header.stamp = current_time.to_msg()
        odom_msg.header.frame_id = 'odom'
        odom_msg.child_frame_id = 'base_link'
        odom_msg.pose.pose.position.x = float(self.pos[0])
        odom_msg.pose.pose.position.y = float(self.pos[1])
        odom_msg.pose.pose.position.z = 0.0
        odom_msg.pose.pose.orientation.w = 1.0
        odom_msg.twist.twist.linear.x = float(self.vel[0])
        odom_msg.twist.twist.linear.y = float(self.vel[1])
        self.publisher_.publish(odom_msg)

        # Publish TF Transform
        transformStamped = TransformStamped()
        transformStamped.header.stamp = current_time.to_msg()
        transformStamped.header.frame_id = 'odom'
        transformStamped.child_frame_id = 'base_link'
        transformStamped.transform.translation.x = float(self.pos[0])
        transformStamped.transform.translation.y = float(self.pos[1])
        transformStamped.transform.translation.z = 0.0
        transformStamped.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(transformStamped)
        self.get_logger().info(f"TF published: x={self.pos[0]}, y={self.pos[1]}")
        #self.get_logger().info(f"Published odom: x={self.pos[0]}, y={self.pos[1]}, vel={self.vel}, dt={dt}")

def main(args=None):
    rclpy.init(args=args)
    node = SimulatedOdomNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
