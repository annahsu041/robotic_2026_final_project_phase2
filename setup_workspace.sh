#!/bin/bash
# 視覺感知多旋翼機自主避障與精準降落 ROS1 工作空間安裝腳本
# 適用環境：Ubuntu 20.04 + ROS1 Noetic

set -e

echo "=== 開始安裝 ROS Noetic 相關依賴套件 ==="
sudo apt-get update
sudo apt-get install -y \
    ros-noetic-mavros ros-noetic-mavros-extras ros-noetic-mavros-msgs \
    ros-noetic-robot-localization \
    ros-noetic-apriltag-ros \
    ros-noetic-octomap-server \
    python3-wstool python3-catkin-tools python3-empy python3-pip \
    libarmadillo-dev

echo "=== 安裝 MAVROS 需要的 GeographicLib 數據集 ==="
wget -qO- https://raw.githubusercontent.com/mavlink/mavros/master/mavros/scripts/install_geographiclib_datasets.sh | sudo bash

echo "=== 建立 Catkin 工作空間 ==="
WS_DIR="$HOME/drone_ws"
mkdir -p "$WS_DIR/src"
cd "$WS_DIR/src"

echo "=== 複製開源套件 (Ego-Planner, Mavros Controllers, OpenVINS) ==="
# 1. 軌跡規劃 (Ego-Planner)
if [ ! -d "ego-planner" ]; then
    git clone https://github.com/ZJU-FAST-Lab/ego-planner.git
fi

# 2. 幾何控制器 (Mavros Controllers)
if [ ! -d "mavros_controllers" ]; then
    git clone https://github.com/Jaeyoung-Lim/mavros_controllers.git
fi

# 3. 狀態估測 VIO (OpenVINS)
if [ ! -d "open_vins" ]; then
    git clone https://github.com/rpng/open_vins.git
fi

echo "=== 編譯工作空間 ==="
cd "$WS_DIR"
catkin init || true
catkin config --extend /opt/ros/noetic
catkin build

echo "=== 安裝完成！ ==="
echo "請記得安裝 PX4 SITL (v1.13.x) 與 Gazebo Classic 11 進行模擬。"
echo "執行 'source $WS_DIR/devel/setup.bash' 來載入環境。"
