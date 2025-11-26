import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SetEntityState
from gazebo_msgs.msg import EntityState
import time

WAYPOINTS = [
    {"x": 0.0, "y": 0.0, "z": 1.0},
    {"x": 2.0, "y": 0.0, "z": 1.0},
    {"x": 2.0, "y": 2.0, "z": 1.0},
    {"x": 0.0, "y": 2.0, "z": 1.0},
    {"x": 0.0, "y": 0.0, "z": 1.0},
]

class WaypointNavigator(Node):
    def __init__(self):  
        super().__init__('waypoint_navigator')  
        self.cli = self.create_client(SetEntityState, '/gazebo/set_entity_state')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /gazebo/set_entity_state service...')
        self.navigate()

    def navigate(self):
        for i, wp in enumerate(WAYPOINTS):
            state = EntityState()
            state.name = 'drone'
            state.pose.position.x = wp['x']
            state.pose.position.y = wp['y']
            state.pose.position.z = wp['z']
            state.pose.orientation.w = 1.0

            req = SetEntityState.Request()
            req.state = state
            self.cli.call_async(req)
            self.get_logger().info(f'⏩ Sent waypoint {i+1}: {wp}')
            time.sleep(2.0)

def main(args=None):
    rclpy.init(args=args)
    node = WaypointNavigator()
    rclpy.spin_once(node, timeout_sec=0)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':  
    main()
