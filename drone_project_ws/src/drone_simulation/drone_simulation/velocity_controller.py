import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class VelocityController(Node):
    def __init__(self):
        super().__init__('velocity_controller')
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.timer = self.create_timer(0.1, self.publish_velocity)
        self.linear_x = 0.0  # Initial linear velocity in x
        self.linear_y = 0.0  # Initial linear velocity in y
        self.angular_z = 0.0  # Initial angular velocity

    def publish_velocity(self):
        msg = Twist()
        msg.linear.x = self.linear_x
        msg.linear.y = self.linear_y
        msg.angular.z = self.angular_z
        self.publisher_.publish(msg)
        self.get_logger().info(f"Publishing velocity: linear_x={self.linear_x}, linear_y={self.linear_y}, angular_z={self.angular_z}")

def main(args=None):
    rclpy.init(args=args)
    node = VelocityController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
