#!/usr/bin/env python
import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray
import numpy as np

# Global variable to store the drone state
latest_drone_state = None

def drone_pose_callback(msg):
    global latest_drone_state
    # Get drone's xyz position from PoseStamped
    latest_drone_state = np.array([
        msg.pose.position.x,
        msg.pose.position.y,
        msg.pose.position.z
    ])
    rospy.loginfo("Updated Drone State: %s", latest_drone_state)

def pattern_callback(msg):
    # Get pattern xyz from the CNN output (assumed to be a Float32MultiArray)
    pattern_state = np.array(msg.data)
    rospy.loginfo("Received Pattern State: %s", pattern_state)
    
    # Verify that we have a valid drone state before combining
    global latest_drone_state
    if latest_drone_state is not None:
        combined_state = np.concatenate((latest_drone_state, pattern_state))
        rospy.loginfo("Combined State (Drone + Pattern): %s", combined_state)
    else:
        rospy.logwarn("Drone state not available yet, cannot form combined state.")

if __name__ == '__main__':
    rospy.init_node('debug_state_node', anonymous=True)
    
    # Subscribers to drone's pose and pattern detection output
    rospy.Subscriber("/drone/pose", PoseStamped, drone_pose_callback)
    rospy.Subscriber("/cnn/pattern", Float32MultiArray, pattern_callback)
    
    rospy.spin()
