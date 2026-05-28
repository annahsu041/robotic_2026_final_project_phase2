# 🚁 Phase 2 Drone Project — Workflow & Architecture

## Project Overview

A fully autonomous drone system running in **PX4 SITL + Gazebo** simulation with:
- **Ego-Planner** for real-time trajectory generation
- **MAVROS** for PX4 flight control
- **AprilTag ROS** for precision landing target detection
- **State Machine FSM** for autonomous mission management

---

## Project Structure

```
Docker Container: ros_gui_final
│
├── ~/PX4-Autopilot/                    # PX4 firmware + Gazebo simulator
│   └── Tools/sitl_gazebo/              # Drone models (iris_opt_flow used)
│
├── ~/drone_ws/                         # Planner & controller workspace
│   └── src/
│       ├── ego-planner/                # Trajectory planner (B-spline optimiser)
│       │   └── src/planner/
│       │       ├── plan_manage/        # ego_planner_node, traj_server
│       │       ├── bspline_opt/        # B-spline optimisation
│       │       └── path_searching/     # A* initial path search
│       ├── mavros_controllers/         # Geometric controller
│       ├── open_vins/                  # Visual-inertial odometry (optional VIO)
│       └── waypoint_generator/         # Relays RViz 2D Nav Goal to Ego-Planner
│
└── ~/catkin_ws/                        # Main ROS workspace
    └── src/
        └── drone_demo/                 # YOUR project package
            ├── scripts/
            │   ├── drone_fsm.py        # 4-phase state machine (FSM)
            │   └── ego_mavros_bridge.py # Bridges Ego-Planner → MAVROS
            └── launch/
                └── system_launch.launch # Unified system launcher
```

---

## ROS Node Architecture

```
[RViz 2D Nav Goal]
        │ /move_base_simple/goal
        ▼
[waypoint_generator]
        │ /goal (to Ego-Planner)
        ▼
[ego_planner_node] ←── /mavros/local_position/odom (drone position)
        │ /planning/bspline
        ▼
[traj_server]
        │ /planning/pos_cmd  (quadrotor_msgs/PositionCommand @ 100Hz)
        ▼
[ego_mavros_bridge.py]  ← arms drone, switches to OFFBOARD automatically
        │ /mavros/setpoint_position/local  (@ 20Hz)
        ▼
[MAVROS] ──UDP──► [PX4 SITL] ──► [Gazebo Drone]
                                        │
                              /iris/cam_down/image_raw
                                        │
                                        ▼
                              [apriltag_ros]
                                        │ /tag_detections
                                        ▼
                              [drone_fsm.py]  ── triggers AUTO.PRECLAND
```

---

## 🚀 Startup Sequence

### Prerequisites (one-time setup, already done)
```bash
# Disable RC loss failsafe so PX4 accepts OFFBOARD without RC
rosrun mavros mavparam set NAV_RCL_ACT 0
rosrun mavros mavparam set COM_RCL_EXCEPT 4
rosrun mavros mavparam save
```

---

### Terminal 1 — PX4 + Gazebo Simulation
```bash
docker exec -it ros_gui_final bash (optional if not in docker)
cd ~/PX4-Autopilot
make px4_sitl gazebo_iris_opt_flow
```
⏳ Wait until Gazebo opens and drone appears on runway (~30-60 seconds)

---

### Terminal 2 — Full ROS System
```bash
docker exec -it ros_gui_final bash (optional if not in docker)
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
source ~/drone_ws/devel/setup.bash
roslaunch drone_demo system_launch.launch
```
✅ Wait for: `[Bridge] ✅ Drone armed!`

---

### Terminal 3 — Fix TF Frame (world = map)
```bash
docker exec -it ros_gui_final bash (optional if not in docker)
source /opt/ros/noetic/setup.bash
rosrun tf2_ros static_transform_publisher 0 0 0 0 0 0 map world
```
⚠️ Keep this running — needed for RViz to show Ego-Planner markers

---

### Terminal 4 — Arm the Drone (if bridge does not auto-arm)
```bash
docker exec -it ros_gui_final bash (optional if not in docker)
source /opt/ros/noetic/setup.bash
rosrun mavros mavparam set NAV_RCL_ACT 0
rosrun mavros mavparam set COM_RCL_EXCEPT 4
rosservice call /mavros/set_mode "custom_mode: 'STABILIZED'"
rosservice call /mavros/set_mode "custom_mode: 'OFFBOARD'"
rosservice call /mavros/cmd/arming "value: true"
```

---

### Terminal 5 — RViz Visualization
```bash
docker exec -it ros_gui_final bash (optional if not in docker)  
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
source ~/drone_ws/devel/setup.bash
rviz
```

**RViz Setup:**

| Display | Type | Topic |
|---|---|---|
| Drone position | Odometry | `/mavros/local_position/odom` |
| Planned trajectory | Marker | `/ego_planner_node/optimal_list` |
| Goal point | Marker | `/ego_planner_node/goal_point` |

- **Fixed Frame** → `world`
- Click **2D Nav Goal** → click grid → drone flies!

---

## 🤖 FSM State Machine (drone_fsm.py)

```
TAKEOFF_EXPLORE
     │  obstacle detected on /obstacle_alert
     ├──► OBSTACLE_REPLAN ──► (done) ──► TAKEOFF_EXPLORE
     │  AprilTag stable (>10 detections) on /tag_detections
     └──► APPROACH
               │  stable lock (>30 detections)
               └──► PRECISION_LAND  (triggers AUTO.PRECLAND in PX4)
```

| State | What happens |
|---|---|
| `TAKEOFF_EXPLORE` | Ego-Planner flies drone to 2D Nav Goal |
| `OBSTACLE_REPLAN` | Phase 2 replanner generates new trajectory |
| `APPROACH` | Drone centres above AprilTag |
| `PRECISION_LAND` | PX4 AUTO.PRECLAND triggered, drone descends |

---

## 🔧 Debug Commands

```bash
# Full system health check
rostopic echo /mavros/state -n1              # armed? OFFBOARD?
rostopic hz /planning/pos_cmd               # ~100Hz = Ego-Planner working
rostopic hz /mavros/setpoint_position/local # ~20Hz  = bridge working
rostopic echo /mavros/local_position/pose -n1   # drone XYZ position
rostopic echo /tag_detections -n1           # AprilTag detected?
rostopic echo /drone_fsm/state -n1          # current FSM state

# If stuck in AUTO.RTL
rosservice call /mavros/set_mode "custom_mode: 'STABILIZED'"
rosservice call /mavros/set_mode "custom_mode: 'OFFBOARD'"
rosservice call /mavros/cmd/arming "value: true"

# Check all active nodes and topics
rosnode list
rostopic list | grep -E "planning|mavros|tag|fsm"
```

---

## ⚠️ Known Issues & Fixes

| Issue | Cause | Fix |
|---|---|---|
| `Resource not found: ego_planner` | drone_ws not sourced last | Source order: noetic → catkin_ws → **drone_ws** |
| `AUTO.RTL` keeps returning | RC loss failsafe triggered | `mavparam set NAV_RCL_ACT 0` |
| Arming fails (`success: False`) | Wrong mode or pre-arm check | Switch STABILIZED → OFFBOARD → arm |
| No green trajectory in RViz | `world` frame not found | Run static_transform_publisher (Terminal 3) |
| `cannot find -lpose_utils` | catkin build isolation | Use `catkin_make` not `catkin build` |
| PX4 MAVLink compile error | MAVLink version mismatch | Hide ROS MAVLink headers before building PX4 |
