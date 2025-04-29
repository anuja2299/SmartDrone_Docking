# SmartDrone_Docking

Drone Navigation with CNN and DDPG \\
Project Overview\\
This project implements an autonomous drone navigation system using a Convolutional Neural Network (CNN) for pattern detection and Deep Deterministic Policy Gradient (DDPG) reinforcement learning for control. The drone navigates to a target pattern (e.g., a QR code) in a simulated Gazebo environment, leveraging real-time odometry feedback. The system is designed as a waterflow architecture, processing data from perception through learning, execution, and monitoring, with no manual initialization or control inputs.

Key Features:
CNN detects the target pattern's position and outputs coordinates.
DDPG learns optimal navigation policies based on rewards derived from distance to the target.
Real-time control in Gazebo with ROS 2 integration.
Autonomous operation without user intervention.
Target Audience: Robotics enthusiasts, reinforcement learning researchers, and ROS 2 developers.

Repository Structure
drone_project_ws/
├── src/
│   ├── drone_simulation/
│   │   ├── drone_simulation/          # Main scripts directory
│   │   │   ├── cnn.py                # CNN for pattern detection
│   │   │   ├── ddpg.py               # DDPG training and model
│   │   │   ├── test_ddpg.py          # Real-time DDPG testing
│   │   │   ├── simulated_odom.py     # Odometry node
│   │   │   └── pattern_position.txt  # Target position file
│   │   ├── worlds/                   # Gazebo world files
│   │   │   └── drone_world.sdf       # Simulation world
│   │   ├── config/                   # Configuration files
│   │   │   └── config.yaml           # Configuration parameters
│   │   └── launch/                   # Launch files
│   │       └── drone_docking.launch.py # Gazebo launch script
├── README.md                         # This file
└── .gitignore                        # Git ignore file
