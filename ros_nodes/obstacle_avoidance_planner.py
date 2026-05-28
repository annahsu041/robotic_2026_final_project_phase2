#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
 階段二主整合節點：Obstacle Avoidance Planner (ROS Node)
==============================================================================
 功能：
   1. 訂閱 AprilTag_Localization 的位置估計（/apriltag_localization/pose）
   2. 訂閱點雲/障礙物偵測話題，即時更新障礙物地圖
   3. 偵測到障礙物後，自動觸發：
      a. ConvexPathOptimizer → 生成安全路徑點
      b. MinSnapTrajectory   → 生成平滑軌跡
   4. 將軌跡以 MultiDOFJointTrajectory 發布，供 ego-planner/controller 追蹤
   5. 整合 FSM，在 TAKEOFF_EXPLORE 和 APPROACH 階段均可觸發

 ROS Topics (訂閱)：
   /apriltag_localization/pose    geometry_msgs/PoseStamped   UAV 位置（來自 AprilTag 定位）
   /mavros/local_position/pose    geometry_msgs/PoseStamped   MAVROS 本地位置
   /obstacle_detector/obstacles   obstacle_msgs/Obstacles     障礙物列表（需配合 obstacle_detector）
   /goal_pose                     geometry_msgs/PoseStamped   目標位置（由 FSM 或操作員給定）

 ROS Topics (發布)：
   /phase2/trajectory             trajectory_msgs/MultiDOFJointTrajectory  平滑軌跡
   /phase2/waypoints              nav_msgs/Path                            可視化用路徑點
   /phase2/status                 std_msgs/String                          規劃狀態

 非 ROS 模式（STANDALONE_TEST=True）：
   直接執行此腳本，模擬完整避障規劃流程

 使用方式：
   # ROS 模式
   rosrun your_pkg obstacle_avoidance_planner.py

   # 獨立測試
   python obstacle_avoidance_planner.py
==============================================================================
"""

import numpy as np
import sys
import os
import time
import warnings

# ─────────────────────────────────────────────────────
# 路徑設定：讓本地測試能找到模組
# ─────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_HERE)
for _p in [
    os.path.join(_PKG_ROOT, 'minimum_snap'),
    os.path.join(_PKG_ROOT, 'convex_opt'),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from min_snap_trajectory import MinSnapTrajectory, CVXPY_AVAILABLE as SNAP_CVXPY
from convex_path_optimizer import (ConvexPathOptimizer, SphereObstacle,
                                   BoxObstacle, FlightEnvelope)

# ─────────────────────────────────────────────────────
# ROS 可用性偵測
# ─────────────────────────────────────────────────────
try:
    import rospy
    from geometry_msgs.msg import PoseStamped, Point
    from nav_msgs.msg import Path
    from std_msgs.msg import String, Bool
    from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint
    from geometry_msgs.msg import Transform, Twist
    try:
        from obstacle_detector.msg import Obstacles
    except ImportError:
        # obstacle_detector 套件未安裝時提供 placeholder，避免啟動崩潰
        Obstacles = None
        rospy.logwarn("[Phase2 Planner] obstacle_detector 套件未找到，"
                      "/obstacles topic 將不可用。"
                      "請安裝: https://github.com/tysik/obstacle_detector")
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    warnings.warn("[Planner] ROS 未安裝，以 STANDALONE 模式執行", ImportWarning, stacklevel=2)

STANDALONE_TEST = not ROS_AVAILABLE  # 自動偵測：ROS 可用時以節點模式執行，否則進行本地測試


# ─────────────────────────────────────────────────────
# 核心規劃器
# ─────────────────────────────────────────────────────

class ObstacleAvoidancePlanner:
    """
    階段二完整避障規劃器

    整合 ConvexPathOptimizer + MinSnapTrajectory，
    可在 ROS 環境或本地測試環境下運作。
    """

    def __init__(self,
                 drone_radius: float = 0.25,
                 v_max: float = 2.0,
                 a_max: float = 2.0,
                 replan_threshold: float = 0.5,
                 env: FlightEnvelope = None):
        """
        Parameters
        ----------
        drone_radius : float
            無人機等效安全半徑（m）
        v_max : float
            最大速度限制（m/s）
        a_max : float
            最大加速度限制（m/s²）
        replan_threshold : float
            觸發重新規劃的障礙物最近距離（m）
        env : FlightEnvelope
            飛行環境邊界
        """
        self.drone_radius = drone_radius
        self.v_max = v_max
        self.a_max = a_max
        self.replan_threshold = replan_threshold
        self.env = env or FlightEnvelope()

        # 內部狀態
        self.current_pos = np.array([0.0, 0.0, 1.0])
        self.goal_pos = np.array([5.0, 0.0, 1.0])
        self.current_trajectory: MinSnapTrajectory = None
        self.traj_start_time: float = None
        self.last_replan_time: float = 0.0
        self.replan_cooldown: float = 2.0  # 重新規劃冷卻時間（秒）

        # 凸優化器
        self.optimizer = ConvexPathOptimizer(
            drone_radius=drone_radius,
            v_max=v_max,
            a_max=a_max,
            env=self.env,
            lambda_smooth=2.0,
            lambda_safe=15.0,
        )

        # 規劃統計
        self.stats = {
            'total_replans': 0,
            'successful_replans': 0,
            'failed_replans': 0,
            'last_plan_time_ms': 0.0,
        }

        if not STANDALONE_TEST:
            self._init_ros()

    def _init_ros(self):
        """初始化 ROS 節點、訂閱者、發布者"""
        rospy.init_node('obstacle_avoidance_planner', anonymous=False)

        # 訂閱者
        rospy.Subscriber('/apriltag_localization/pose', PoseStamped,
                         self._apriltag_pose_cb, queue_size=5)
        rospy.Subscriber('/mavros/local_position/pose', PoseStamped,
                         self._local_pose_cb, queue_size=5)
        rospy.Subscriber('/goal_pose', PoseStamped,
                         self._goal_cb, queue_size=1)
        # 障礙物話題：接收 obstacle_detector 套件發布的障礙物列表
        if Obstacles is not None:
            rospy.Subscriber('/obstacles', Obstacles,
                             self._obs_cb, queue_size=5)
        else:
            rospy.logwarn("[Phase2 Planner] 跳過 /obstacles 訂閱（obstacle_detector 未安裝）")

        # 發布者
        self.traj_pub = rospy.Publisher(
            '/phase2/trajectory', MultiDOFJointTrajectory, queue_size=1)
        self.path_pub = rospy.Publisher(
            '/phase2/waypoints', Path, queue_size=1)
        self.status_pub = rospy.Publisher(
            '/phase2/status', String, queue_size=1)
        # FSM 監聽的障礙物警報（Bool）
        self.alert_pub = rospy.Publisher(
            '/obstacle_alert', Bool, queue_size=1)

        rospy.loginfo("[Phase2 Planner] ROS 節點已啟動")

    # ── ROS Callbacks ─────────────────────────────────

    def _apriltag_pose_cb(self, msg: 'PoseStamped'):
        """接收 AprilTag 定位結果"""
        self.current_pos = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ])

    def _local_pose_cb(self, msg: 'PoseStamped'):
        """接收 MAVROS 本地位置（備援）"""
        # 若 AprilTag 定位失效時使用
        if np.allclose(self.current_pos, 0):
            self.current_pos = np.array([
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z,
            ])

    def _goal_cb(self, msg: 'PoseStamped'):
        """接收新目標點，觸發重新規劃"""
        self.goal_pos = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ])
        rospy.loginfo(f"[Phase2] 新目標: {self.goal_pos}")
        self.replan()

    def _obs_cb(self, msg: 'Obstacles'):
        """
        接收 obstacle_detector 發布的障礙物列表，轉換格式後更新地圖，
        並在需要時向 FSM 發布 /obstacle_alert。
        """
        detected = []
        for circle in msg.circles:
            detected.append({
                'type': 'sphere',
                'center': [circle.center.x, circle.center.y, self.current_pos[2]],
                'radius': circle.radius,
                'margin': 0.3,
            })
        for segment in msg.segments:
            xs = [segment.first_point.x, segment.last_point.x]
            ys = [segment.first_point.y, segment.last_point.y]
            z = self.current_pos[2]
            detected.append({
                'type': 'box',
                'min': [min(xs) - 0.2, min(ys) - 0.2, z - 0.5],
                'max': [max(xs) + 0.2, max(ys) + 0.2, z + 0.5],
                'margin': 0.3,
            })

        if detected:
            self.update_obstacles_from_detection(detected)
            # 若當前軌跡不安全，通知 FSM
            if self.needs_replan():
                self.alert_pub.publish(Bool(data=True))
                rospy.logwarn(f"[Phase2] 偵測到 {len(detected)} 個障礙物，已通知 FSM")

    # ── 核心規劃函數 ───────────────────────────────────

    def update_obstacles_from_detection(self, detected_obstacles: list):
        """
        從偵測結果更新障礙物地圖

        Parameters
        ----------
        detected_obstacles : list of dict
            每個元素 {'type': 'sphere'/'box',
                      'center': [x,y,z], 'radius': r}
            或       {'type': 'box', 'min': [...], 'max': [...]}
        """
        self.optimizer.clear_obstacles()
        for obs_info in detected_obstacles:
            if obs_info.get('type') == 'sphere':
                self.optimizer.add_obstacle(
                    SphereObstacle(
                        center=obs_info['center'],
                        radius=obs_info['radius'],
                        safety_margin=obs_info.get('margin', 0.3)
                    )
                )
            elif obs_info.get('type') == 'box':
                self.optimizer.add_obstacle(
                    BoxObstacle(
                        min_pt=obs_info['min'],
                        max_pt=obs_info['max'],
                        safety_margin=obs_info.get('margin', 0.3)
                    )
                )

    def needs_replan(self) -> bool:
        """
        判斷是否需要重新規劃

        條件：
        1. 目前沒有軌跡
        2. 當前軌跡通過障礙物
        3. 距離最近障礙物過近
        """
        now = time.time()
        if now - self.last_replan_time < self.replan_cooldown:
            return False

        if self.current_trajectory is None:
            return True

        # 檢查當前位置到最近障礙物距離
        for obs in self.optimizer.obstacles:
            if not obs.is_safe(self.current_pos, self.drone_radius):
                return True

        # 檢查剩餘軌跡是否安全
        if self.traj_start_time is not None:
            elapsed = now - self.traj_start_time
            remaining = self.current_trajectory.total_time - elapsed
            if remaining > 0:
                check_pts = []
                for dt in np.arange(0, remaining, 0.5):
                    pos, _, _ = self.current_trajectory.sample(elapsed + dt)
                    check_pts.append(pos)
                if check_pts and not self.optimizer.is_path_safe(check_pts):
                    return True

        return False

    def replan(self) -> bool:
        """
        觸發完整避障重規劃流程

        Returns
        -------
        bool
            是否成功生成軌跡
        """
        t0 = time.time()
        self.stats['total_replans'] += 1
        self.last_replan_time = t0

        log = (rospy.loginfo if ROS_AVAILABLE else print)
        log(f"[Phase2] 開始重新規劃... (重規劃次數: {self.stats['total_replans']})")

        # Step 1: 凸優化求解安全路徑點
        original_path = [self.current_pos.tolist(), self.goal_pos.tolist()]
        try:
            safe_wps, opt_info = self.optimizer.optimize(
                original_path, n_intermediate=4
            )
        except Exception as e:
            warnings.warn(f"[Phase2] 凸優化失敗: {e}")
            self.stats['failed_replans'] += 1
            return False

        if not opt_info['path_safe']:
            log(f"[Phase2] ⚠️ 優化後路徑仍不安全，嘗試增加中間點...")
            safe_wps, opt_info = self.optimizer.optimize(
                original_path, n_intermediate=8
            )

        # Step 2: Minimum Snap 軌跡生成
        try:
            trajectory = MinSnapTrajectory(
                waypoints=[wp.tolist() if hasattr(wp, 'tolist') else wp
                           for wp in safe_wps],
                v_max=self.v_max,
                a_max=self.a_max,
                use_cvxpy=SNAP_CVXPY,
            )
        except Exception as e:
            warnings.warn(f"[Phase2] MinSnap 生成失敗: {e}")
            self.stats['failed_replans'] += 1
            return False

        # Step 3: 可行性驗證
        feasibility = trajectory.check_feasibility()
        if not feasibility['feasible']:
            log(f"[Phase2] ⚠️ 軌跡超出動態限制，嘗試延長時間...")
            # 若不可行，重新以更寬鬆時間求解
            extended_times = [t * 1.5 for t in trajectory.times]
            trajectory = MinSnapTrajectory(
                waypoints=[wp.tolist() if hasattr(wp, 'tolist') else wp
                           for wp in safe_wps],
                times=extended_times,
                v_max=self.v_max,
                a_max=self.a_max,
                use_cvxpy=SNAP_CVXPY,
            )
            feasibility = trajectory.check_feasibility()

        # Step 4: 更新當前軌跡
        self.current_trajectory = trajectory
        self.traj_start_time = time.time()

        elapsed_ms = (time.time() - t0) * 1000
        self.stats['successful_replans'] += 1
        self.stats['last_plan_time_ms'] = elapsed_ms

        log(f"[Phase2] ✅ 規劃完成！")
        log(f"  優化器狀態: {opt_info['status']}")
        log(f"  路徑點數: {len(safe_wps)}")
        log(f"  軌跡時間: {trajectory.total_time:.2f}s")
        log(f"  最大速度: {feasibility['max_velocity']:.3f} m/s")
        log(f"  最大加速: {feasibility['max_acceleration']:.3f} m/s²")
        log(f"  可行性: {feasibility['feasible']}")
        log(f"  規劃耗時: {elapsed_ms:.1f} ms")

        # Step 5: 發布（ROS 模式）
        if ROS_AVAILABLE:
            self._publish_trajectory(trajectory, safe_wps)

        return True

    def _publish_trajectory(self, traj: MinSnapTrajectory, waypoints: list):
        """發布軌跡到 ROS"""
        # 發布 MultiDOFJointTrajectory
        msg = MultiDOFJointTrajectory()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = 'world'

        data = traj.get_trajectory(dt=0.1)
        for i, t in enumerate(data['time']):
            pt = MultiDOFJointTrajectoryPoint()
            tf = Transform()
            pos = data['position'][i]
            tf.translation.x = pos[0]
            tf.translation.y = pos[1]
            tf.translation.z = pos[2]
            tf.rotation.w = 1.0  # 保持朝向水平
            pt.transforms.append(tf)

            vel = Twist()
            v = data['velocity'][i]
            vel.linear.x = v[0]
            vel.linear.y = v[1]
            vel.linear.z = v[2]
            pt.velocities.append(vel)

            pt.time_from_start = rospy.Duration(float(t))
            msg.points.append(pt)

        self.traj_pub.publish(msg)

        # 發布路徑點（RViz 可視化）
        path_msg = Path()
        path_msg.header.stamp = rospy.Time.now()
        path_msg.header.frame_id = 'world'
        for wp in waypoints:
            ps = PoseStamped()
            ps.header = path_msg.header
            arr = np.array(wp)
            ps.pose.position.x = arr[0]
            ps.pose.position.y = arr[1]
            ps.pose.position.z = arr[2]
            ps.pose.orientation.w = 1.0
            path_msg.poses.append(ps)
        self.path_pub.publish(path_msg)

        # 發布狀態
        self.status_pub.publish(String(data='REPLANNING_COMPLETE'))

    def spin_ros(self):
        """ROS 主循環"""
        rate = rospy.Rate(10)  # 10 Hz 監控
        while not rospy.is_shutdown():
            if self.needs_replan():
                self.replan()
            rate.sleep()

    def get_current_setpoint(self) -> tuple:
        """
        取得當前時刻的軌跡設定點

        Returns
        -------
        pos : np.ndarray (3,)
        vel : np.ndarray (3,)
        acc : np.ndarray (3,)
        """
        if self.current_trajectory is None or self.traj_start_time is None:
            return self.current_pos, np.zeros(3), np.zeros(3)

        elapsed = time.time() - self.traj_start_time
        return self.current_trajectory.sample(elapsed)


# ─────────────────────────────────────────────────────
# 本地獨立整合測試
# ─────────────────────────────────────────────────────
def _run_standalone_test():
    print("=" * 60)
    print("  Phase 2: Obstacle Avoidance Planner - Integrated Test")
    print(f"  CVXPY: {SNAP_CVXPY} | ROS: {ROS_AVAILABLE}")
    print("=" * 60)

    # 初始化規劃器
    # lambda_smooth 提高到 8.0，讓路徑更傾向走直線；
    # lambda_safe 降低到 8.0，避免路徑點被推離障礙物太遠
    env = FlightEnvelope(x_range=(-1, 8), y_range=(-3, 3), z_range=(0.5, 4.0))
    planner = ObstacleAvoidancePlanner(
        drone_radius=0.25, v_max=2.0, a_max=2.0,
        replan_threshold=0.5, env=env
    )
    # 覆蓋預設 lambda 參數：平衡安全與路徑效率
    planner.optimizer.lambda_smooth = 8.0
    planner.optimizer.lambda_safe   = 8.0

    # 設定起點和目標
    planner.current_pos = np.array([0.0, 0.0, 1.0])
    planner.goal_pos    = np.array([6.0, 0.0, 1.0])

    # 障礙物設定（margin 縮小，避免凸優化過度保守）
    OBS = [
        {'type': 'sphere', 'center': [2.5, 0.0, 1.0], 'radius': 0.6, 'margin': 0.25},
        {'type': 'sphere', 'center': [4.0, 0.3, 1.1], 'radius': 0.4, 'margin': 0.25},
    ]

    # ── 場景一：靜態障礙物 ────────────────────────────
    print("\n-- Scenario 1: Static obstacle avoidance --")
    planner.update_obstacles_from_detection(OBS)
    success = planner.replan()
    print(f"\nPlanning result: {'SUCCESS' if success else 'FAILED'}")

    # ── 安全距離驗證 ──────────────────────────────────
    if success and planner.current_trajectory:
        traj = planner.current_trajectory
        data = traj.get_trajectory(dt=0.05)
        pos_arr = data['position']

        print("\n-- Clearance verification (min distance to each obstacle) --")
        for obs_info in OBS:
            c = np.array(obs_info['center'])
            r_total = obs_info['radius'] + planner.drone_radius + obs_info['margin']
            dists = np.linalg.norm(pos_arr - c, axis=1)
            min_d = dists.min()
            status = "OK" if min_d >= r_total else "VIOLATION"
            print(f"  Obs {c} r={obs_info['radius']:.1f}: "
                  f"min_clearance={min_d:.3f}m  required={r_total:.3f}m  [{status}]")

        print("\n-- Trajectory samples (every 0.5s) --")
        print(f"  {'t':>6} | {'x':>7} {'y':>7} {'z':>7} | {'|v|':>7} | {'|a|':>7}")
        print("  " + "-" * 52)
        for t in np.arange(0, min(traj.total_time, 10), 0.5):
            p, v, a = traj.sample(t)
            print(f"  {t:6.1f} | {p[0]:7.3f} {p[1]:7.3f} {p[2]:7.3f} | "
                  f"{np.linalg.norm(v):7.3f} | {np.linalg.norm(a):7.3f}")

    # ── 場景二：動態重規劃 ────────────────────────────
    print("\n-- Scenario 2: Dynamic replanning with new obstacle --")
    time.sleep(0.1)
    planner.current_pos = np.array([1.5, 0.2, 1.0])
    planner.update_obstacles_from_detection([
        {'type': 'sphere', 'center': [3.0, 0.2, 1.0], 'radius': 0.7, 'margin': 0.25},
    ])
    print(f"\nNeeds replan? {planner.needs_replan()}")
    planner.last_replan_time = 0
    success2 = planner.replan()
    print(f"Replanning result: {'SUCCESS' if success2 else 'FAILED'}")

    # ── 統計資訊 ──────────────────────────────────────
    print(f"\n-- Planning statistics --")
    for k, v in planner.stats.items():
        print(f"  {k}: {v}")

    # ── 視覺化 ────────────────────────────────────────
    if success and planner.current_trajectory:
        try:
            import matplotlib
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D

            matplotlib.rcParams['font.family'] = 'DejaVu Sans'

            traj = planner.current_trajectory
            data = traj.get_trajectory(dt=0.05)
            pos  = data['position']
            vn   = np.linalg.norm(data['velocity'],     axis=1)
            an   = np.linalg.norm(data['acceleration'], axis=1)

            fig = plt.figure(figsize=(16, 8))

            # ── 左上：3D 軌跡（視角俯瞰，清楚看見繞行）
            ax1 = fig.add_subplot(221, projection='3d')
            ax1.plot(pos[:, 0], pos[:, 1], pos[:, 2],
                     'b-', lw=2.5, label='Trajectory')
            ax1.scatter([0], [0], [1.0], c='green', s=60, zorder=5, label='Start')
            ax1.scatter([6], [0], [1.0], c='red',   s=60, zorder=5, label='Goal')

            u  = np.linspace(0, 2 * np.pi, 30)
            v2 = np.linspace(0, np.pi, 20)
            for obs_info in OBS:
                c = obs_info['center']
                r = obs_info['radius'] + planner.drone_radius + obs_info['margin']
                xs = c[0] + r * np.outer(np.cos(u), np.sin(v2))
                ys = c[1] + r * np.outer(np.sin(u), np.sin(v2))
                zs = c[2] + r * np.outer(np.ones_like(u), np.cos(v2))
                ax1.plot_surface(xs, ys, zs, alpha=0.20, color='red')

            ax1.view_init(elev=35, azim=-60)   # 較好的俯視視角
            ax1.set_title('3D Avoidance Trajectory')
            ax1.set_xlabel('X (m)'); ax1.set_ylabel('Y (m)'); ax1.set_zlabel('Z (m)')
            ax1.legend(fontsize=8)

            # ── 右上：俯視圖 XY（最直觀看清楚繞行路徑）
            ax2 = fig.add_subplot(222)
            ax2.plot(pos[:, 0], pos[:, 1], 'b-', lw=2.5, label='Trajectory')
            ax2.scatter([0], [0], c='green', s=80, zorder=5, label='Start')
            ax2.scatter([6], [0], c='red',   s=80, zorder=5, label='Goal')
            ax2.annotate('Start', (0, 0),   textcoords='offset points', xytext=(5, 5), fontsize=8)
            ax2.annotate('Goal',  (6, 0),   textcoords='offset points', xytext=(5, 5), fontsize=8)

            theta = np.linspace(0, 2 * np.pi, 100)
            for obs_info in OBS:
                c  = obs_info['center']
                r  = obs_info['radius'] + planner.drone_radius + obs_info['margin']
                r0 = obs_info['radius']
                # 安全邊界
                ax2.fill(c[0] + r  * np.cos(theta),
                         c[1] + r  * np.sin(theta),
                         alpha=0.20, color='red', label='Safety zone')
                # 實體障礙物
                ax2.fill(c[0] + r0 * np.cos(theta),
                         c[1] + r0 * np.sin(theta),
                         alpha=0.50, color='darkred')
            ax2.set_aspect('equal')
            ax2.set_title('Top-down View (XY plane)')
            ax2.set_xlabel('X (m)'); ax2.set_ylabel('Y (m)')
            ax2.grid(alpha=0.3)
            handles, labels = ax2.get_legend_handles_labels()
            # 去除重複 legend
            by_label = dict(zip(labels, handles))
            ax2.legend(by_label.values(), by_label.keys(), fontsize=8)

            # ── 左下：速度剖面
            ax3 = fig.add_subplot(223)
            ax3.plot(data['time'], vn, 'b-', lw=1.5, label='Velocity |v|')
            ax3.axhline(planner.v_max, color='b', ls='--', alpha=0.6, label=f'v_max={planner.v_max}')
            ax3.set_title('Velocity Profile')
            ax3.set_xlabel('t (s)'); ax3.set_ylabel('m/s')
            ax3.legend(); ax3.grid(alpha=0.3)

            # ── 右下：加速度剖面
            ax4 = fig.add_subplot(224)
            ax4.plot(data['time'], an, 'r-', lw=1.5, label='Acceleration |a|')
            ax4.axhline(planner.a_max, color='r', ls='--', alpha=0.6, label=f'a_max={planner.a_max}')
            ax4.set_title('Acceleration Profile')
            ax4.set_xlabel('t (s)'); ax4.set_ylabel('m/s^2')
            ax4.legend(); ax4.grid(alpha=0.3)

            plt.suptitle('Phase 2: Minimum Snap + Convex Opt Obstacle Avoidance',
                         fontsize=13, fontweight='bold')
            plt.tight_layout()
            plt.savefig('phase2_integrated_result.png', dpi=150, bbox_inches='tight')
            print("\nFigure saved: phase2_integrated_result.png")
            plt.show()

        except ImportError:
            print("\n[Hint] pip install matplotlib to display figures")

    print("\nIntegrated test complete!")
    print("  ROS integration:")
    print("  1. Add this node to system_launch.launch")
    print("  2. FSM publishes /goal_pose to trigger replanning")
    print("  3. FSM listens to /phase2/status for completion")


if __name__ == '__main__':
    if STANDALONE_TEST:
        _run_standalone_test()
    else:
        try:
            planner = ObstacleAvoidancePlanner()
            planner.spin_ros()
        except rospy.ROSInterruptException:
            pass
