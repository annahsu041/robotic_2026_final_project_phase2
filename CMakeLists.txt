cmake_minimum_required(VERSION 3.0.2)
project(drone_demo)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  nav_msgs
  trajectory_msgs
  mavros_msgs
)

catkin_package()

# 安裝 Python 腳本，讓 rosrun 可以找到
catkin_install_python(PROGRAMS
  ros_nodes/obstacle_avoidance_planner.py
  ros_nodes/ego_mavros_bridge.py
  drone_fsm.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# 安裝 launch 檔
install(FILES system_launch.launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)
