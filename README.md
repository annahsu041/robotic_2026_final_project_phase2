# 視覺感知多旋翼機自主避障與精準降落專案

此資料夾包含了根據「ROS1 模組總覽」所設計的起步環境與腳本範例，協助您在 Ubuntu 20.04 (ROS1 Noetic) 環境下快速搭建無人機模擬系統。

## 資料夾內容

1. **`setup_workspace.sh`**
   - **用途**：一鍵安裝所需的核心 ROS1 套件（MAVROS, robot_localization, apriltag_ros 等），並自動下載推薦的 GitHub 開源模組（Ego-Planner, Mavros Controllers, OpenVINS），最後建立並編譯 Catkin 工作空間。
   - **使用方式**：在 Ubuntu 20.04 中執行 `bash setup_workspace.sh`。

2. **`system_launch.launch`**
   - **用途**：ROS launch 骨架檔，示範如何將系統各模組（MAVROS、VIO、EKF、規劃器、AprilTag 偵測、FSM 狀態機）串聯起來。
   - **注意**：部分節點（如 open_vins 和 ekf）的參數檔需要根據您實際的 Gazebo 相機或真機進行客製化設定，腳本內已預留註解。

3. **`drone_fsm.py`**
   - **用途**：以 Python (rospy) 實作的三階段狀態機範例。
   - **邏輯**：
     - **TAKEOFF_EXPLORE**：等待 Ego-Planner 執行探索，持續監控是否看到 AprilTag。
     - **APPROACH**：偵測到標籤後，切換目標為標籤上方。
     - **PRECISION_LAND**：穩定追蹤標籤後，觸發 PX4 的 `AUTO.PRECLAND` 模式進行視覺輔助精準降落。

## 系統架構簡述

本專案採用的技術棧（推薦路線）：
- **模擬器**：PX4 SITL (v1.13.x) + Gazebo Classic 11
- **視覺定位 (VIO)**：OpenVINS
- **感測器融合**：robot_localization (雙重 EKF)
- **軌跡重規劃**：Ego-Planner (ESDF-free, B-spline)
- **精準降落**：apriltag_ros + PX4 內建 precision_land (透過 MAVROS)

## 開發與執行步驟

1. 確保您的 Ubuntu 20.04 已安裝 ROS1 Noetic。
2. 執行 `setup_workspace.sh` 安裝與編譯相依套件。
3. 參考 PX4 官方教學，安裝並啟動 `PX4-Autopilot` (版本 1.13.x) 搭配 Gazebo。
   - `make px4_sitl gazebo_iris_downward_camera` (或其他含有下視相機的模型)
4. 將 `system_launch.launch` 和 `drone_fsm.py` 移至您的自訂 ROS package 中。
5. 修改 launch 檔內的 `pkg="your_pkg"` 為您實際的套件名稱。
6. 啟動您的 Launch 檔：`roslaunch your_pkg system_launch.launch`。
7. 在 RViz 中給定 2D Nav Goal 讓 Ego-Planner 啟動，FSM 將自動接管降落邏輯。
