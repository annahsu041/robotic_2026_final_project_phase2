#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
 Phase 2 橋接節點：ego_mavros_bridge
==============================================================================
 功能：
   將 Phase 2 規劃器發布的 /phase2/trajectory (MultiDOFJointTrajectory)
   轉換為 MAVROS 可接受的 setpoint，並在 FSM 處於 TAKEOFF_EXPLORE /
   OBSTACLE_REPLAN 階段時持續追蹤軌跡設定點。

   FSM 進入 APPROACH / PRECISION_LAND 後，AprilTag 視覺伺服直接控制，
   此橋接節點退出追蹤（停止發布 setpoint）。

 ROS Topics (訂閱)：
   /phase2/trajectory    trajectory_msgs/MultiDOFJointTrajectory   Phase 2 軌跡
   /drone_fsm/state      std_msgs/String                           FSM 當前狀態
   /mavros/state         mavros_msgs/State                         飛控狀態

 ROS Topics (發布)：
   /mavros/setpoint_position/local   geometry_msgs/PoseStamped   位置設定點 @ 20Hz

 ROS Services：
   /mavros/set_mode    → 切換 OFFBOARD
   /mavros/cmd/arming  → 解鎖

 使用方式：
   rosrun your_pkg ego_mavros_bridge.py
==============================================================================
"""

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from trajectory_msgs.msg import MultiDOFJointTrajectory
from mavros_msgs.msg import State
from mavros_msgs.srv import SetMode, CommandBool

# FSM 狀態下橋接節點主動追蹤軌跡的白名單
TRACKING_STATES = {"TAKEOFF_EXPLORE", "OBSTACLE_REPLAN"}


class EgoMavrosBridge:
    """
    Phase 2 軌跡 → MAVROS setpoint 橋接器

    從 /phase2/trajectory 中按時間插值取出設定點，
    以 20Hz 發布給 MAVROS OFFBOARD 模式追蹤。
    """

    SETPOINT_RATE = 20  # Hz，MAVROS OFFBOARD 最低需求 2Hz，建議 ≥ 10Hz

    def __init__(self):
        rospy.init_node('ego_mavros_bridge', anonymous=False)

        # ── 狀態變數 ───────────────────────────────────
        self.fsm_state: str = "TAKEOFF_EXPLORE"
        self.mavros_state = State()
        self.trajectory_points: list = []   # list of (time_from_start_sec, x, y, z)
        self.traj_recv_time: float = None   # rospy.Time 記錄軌跡接收時刻
        self.offboard_sent: bool = False

        # ── 訂閱者 ─────────────────────────────────────
        rospy.Subscriber('/phase2/trajectory', MultiDOFJointTrajectory,
                         self._traj_cb, queue_size=1)
        rospy.Subscriber('/drone_fsm/state', String,
                         self._fsm_state_cb, queue_size=1)
        rospy.Subscriber('/mavros/state', State,
                         self._mavros_state_cb, queue_size=10)

        # ── 發布者 ─────────────────────────────────────
        self.setpoint_pub = rospy.Publisher(
            '/mavros/setpoint_position/local', PoseStamped, queue_size=10)

        # ── 服務客戶端 ─────────────────────────────────
        rospy.loginfo("[Bridge] 等待 MAVROS 服務...")
        rospy.wait_for_service('/mavros/set_mode', timeout=30)
        rospy.wait_for_service('/mavros/cmd/arming', timeout=30)
        self.set_mode_srv = rospy.ServiceProxy('/mavros/set_mode', SetMode)
        self.arm_srv = rospy.ServiceProxy('/mavros/cmd/arming', CommandBool)

        rospy.loginfo("[Bridge] 初始化完成，等待軌跡輸入...")

    # ── Callbacks ──────────────────────────────────────

    def _traj_cb(self, msg: MultiDOFJointTrajectory):
        """接收 Phase 2 新軌跡，解析為時間-位置序列"""
        points = []
        for pt in msg.points:
            t = pt.time_from_start.to_sec()
            if pt.transforms:
                tf = pt.transforms[0]
                points.append((t,
                                tf.translation.x,
                                tf.translation.y,
                                tf.translation.z))
        if points:
            self.trajectory_points = points
            self.traj_recv_time = rospy.Time.now().to_sec()
            rospy.loginfo(f"[Bridge] 接收到新軌跡，共 {len(points)} 個點，"
                          f"總時長 {points[-1][0]:.2f}s")

    def _fsm_state_cb(self, msg: String):
        self.fsm_state = msg.data

    def _mavros_state_cb(self, msg: State):
        self.mavros_state = msg

    # ── 輔助函數 ────────────────────────────────────────

    def _interpolate_setpoint(self) -> np.ndarray:
        """
        根據當前時間插值出軌跡上的位置設定點。

        Returns
        -------
        np.ndarray shape (3,)  — [x, y, z]，若軌跡結束回傳最後一點。
        """
        if not self.trajectory_points or self.traj_recv_time is None:
            return None

        elapsed = rospy.Time.now().to_sec() - self.traj_recv_time
        times = [p[0] for p in self.trajectory_points]

        # 軌跡播放完畢：停在終點
        if elapsed >= times[-1]:
            p = self.trajectory_points[-1]
            return np.array([p[1], p[2], p[3]])

        # 線性插值
        for i in range(len(times) - 1):
            if times[i] <= elapsed < times[i + 1]:
                alpha = (elapsed - times[i]) / (times[i + 1] - times[i])
                p0 = self.trajectory_points[i]
                p1 = self.trajectory_points[i + 1]
                x = p0[1] + alpha * (p1[1] - p0[1])
                y = p0[2] + alpha * (p1[2] - p0[2])
                z = p0[3] + alpha * (p1[3] - p0[3])
                return np.array([x, y, z])

        return None

    def _switch_to_offboard_and_arm(self):
        """嘗試切換 OFFBOARD 模式並解鎖（僅執行一次）"""
        if self.offboard_sent:
            return
        try:
            resp = self.set_mode_srv(0, 'OFFBOARD')
            if resp.mode_sent:
                rospy.loginfo("[Bridge] ✅ 切換至 OFFBOARD 模式")
            arm_resp = self.arm_srv(True)
            if arm_resp.success:
                rospy.loginfo("[Bridge] ✅ 無人機已解鎖")
            self.offboard_sent = True
        except rospy.ServiceException as e:
            rospy.logerr(f"[Bridge] 模式切換失敗: {e}")

    def _make_setpoint(self, pos: np.ndarray) -> PoseStamped:
        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = 'world'
        msg.pose.position.x = float(pos[0])
        msg.pose.position.y = float(pos[1])
        msg.pose.position.z = float(pos[2])
        msg.pose.orientation.w = 1.0
        return msg

    # ── 主循環 ─────────────────────────────────────────

    def run(self):
        rate = rospy.Rate(self.SETPOINT_RATE)
        rospy.loginfo(f"[Bridge] 主循環啟動 @ {self.SETPOINT_RATE}Hz")

        # OFFBOARD 模式要求在切換前已持續發布 setpoint
        # → 先以當前位置暖機發布 2 秒
        warmup_end = rospy.Time.now() + rospy.Duration(2.0)
        while not rospy.is_shutdown() and rospy.Time.now() < warmup_end:
            warmup_sp = PoseStamped()
            warmup_sp.header.stamp = rospy.Time.now()
            warmup_sp.header.frame_id = 'world'
            warmup_sp.pose.position.z = 0.1  # 低姿態暖機
            warmup_sp.pose.orientation.w = 1.0
            self.setpoint_pub.publish(warmup_sp)
            rate.sleep()

        self._switch_to_offboard_and_arm()

        while not rospy.is_shutdown():
            # 只在 FSM 主動探索 / 重規劃時追蹤軌跡
            if self.fsm_state in TRACKING_STATES:
                pos = self._interpolate_setpoint()
                if pos is not None:
                    self.setpoint_pub.publish(self._make_setpoint(pos))
                    rospy.logdebug(f"[Bridge] setpoint → "
                                   f"({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
            # APPROACH / PRECISION_LAND：AprilTag 視覺伺服主導，橋接節點停止介入

            rate.sleep()


if __name__ == '__main__':
    try:
        bridge = EgoMavrosBridge()
        bridge.run()
    except rospy.ROSInterruptException:
        pass
