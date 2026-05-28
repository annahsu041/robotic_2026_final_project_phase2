#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
多旋翼機四階段狀態機 (FSM) - 含階段二動態避障
==============================================================================
 階段一：TAKEOFF_EXPLORE  起飛與自主探索（ego-planner 主導）
 階段二：OBSTACLE_REPLAN  偵測到障礙物 → 凸優化 + Minimum Snap 重規劃
 階段三：APPROACH          AprilTag 鎖定後接近降落點
 階段四：PRECISION_LAND    觸發 PX4 AUTO.PRECLAND 精準降落

 [階段二整合]
   障礙物偵測 → /phase2/trigger 話題 → 觸發 ObstacleAvoidancePlanner
   規劃完成 → /phase2/status 發布 'REPLANNING_COMPLETE'
   軌跡 → /phase2/trajectory (MultiDOFJointTrajectory) → 飛控追蹤
==============================================================================
"""

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String, Bool
from apriltag_ros.msg import AprilTagDetectionArray
from mavros_msgs.srv import SetMode, CommandTOL
from mavros_msgs.msg import State


class DroneFSM:
    """
    四階段無人機狀態機

    狀態轉移圖：
    TAKEOFF_EXPLORE
         │  (偵測到障礙物且需要重規劃)
         ├─── OBSTACLE_REPLAN ──→ TAKEOFF_EXPLORE (規劃完成後)
         │  (AprilTag 穩定偵測)
         └─── APPROACH ──→ PRECISION_LAND
    """

    def __init__(self):
        rospy.init_node('drone_fsm_node', anonymous=True)

        # ── 狀態變數 ───────────────────────────────────
        self.state = "TAKEOFF_EXPLORE"
        self.tag_detected = False
        self.detection_count = 0
        self.current_mavros_state = State()
        self.current_pos = np.zeros(3)

        # 階段二狀態
        self.obstacle_detected = False
        self.phase2_planning = False
        self.phase2_done = False
        self.replan_trigger_distance = 1.5  # 距離障礙物 1.5m 內觸發重規劃（m）

        # ── 訂閱者 ─────────────────────────────────────
        rospy.Subscriber('/mavros/state', State, self._mavros_state_cb)
        rospy.Subscriber('/mavros/local_position/pose', PoseStamped, self._local_pos_cb)
        rospy.Subscriber('/tag_detections', AprilTagDetectionArray, self._tag_cb)

        # 階段二：障礙物警報（由障礙物偵測節點發布）
        rospy.Subscriber('/obstacle_alert', Bool, self._obstacle_alert_cb)
        # 階段二：規劃器狀態回報
        rospy.Subscriber('/phase2/status', String, self._phase2_status_cb)

        # ── 發布者 ─────────────────────────────────────
        # 降落目標（階段四）
        self.landing_target_pub = rospy.Publisher(
            '/mavros/landing_target/pose', PoseStamped, queue_size=10)
        # 觸發階段二重規劃
        self.replan_trigger_pub = rospy.Publisher(
            '/goal_pose', PoseStamped, queue_size=1)
        # FSM 狀態廣播（供其他節點監聽）
        self.fsm_state_pub = rospy.Publisher(
            '/drone_fsm/state', String, queue_size=1)

        # ── 服務客戶端 ─────────────────────────────────
        rospy.wait_for_service('/mavros/set_mode')
        rospy.wait_for_service('/mavros/cmd/land')
        self.set_mode_srv = rospy.ServiceProxy('/mavros/set_mode', SetMode)
        self.land_cmd_srv = rospy.ServiceProxy('/mavros/cmd/land', CommandTOL)

        rospy.loginfo("[FSM] 初始化完成 - 四階段避障狀態機已就緒")

    # ── Callbacks ──────────────────────────────────────

    def _mavros_state_cb(self, msg: State):
        self.current_mavros_state = msg

    def _local_pos_cb(self, msg: PoseStamped):
        self.current_pos = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ])

    def _tag_cb(self, msg: AprilTagDetectionArray):
        """AprilTag 偵測回調"""
        if len(msg.detections) > 0:
            self.detection_count += 1
            if self.detection_count > 10:
                self.tag_detected = True

            # 發布降落目標姿態
            landing_pose = PoseStamped()
            landing_pose.header = msg.header
            landing_pose.pose = msg.detections[0].pose.pose.pose
            self.landing_target_pub.publish(landing_pose)
        else:
            self.detection_count = max(0, self.detection_count - 1)
            if self.detection_count == 0:
                self.tag_detected = False

    def _obstacle_alert_cb(self, msg: Bool):
        """
        接收障礙物警報
        若在 TAKEOFF_EXPLORE 階段偵測到障礙物，切換至 OBSTACLE_REPLAN
        """
        if msg.data and self.state == "TAKEOFF_EXPLORE":
            self.obstacle_detected = True
            rospy.logwarn("[FSM] ⚠️  偵測到障礙物！準備觸發 Phase 2 重規劃...")

    def _phase2_status_cb(self, msg: String):
        """接收 Phase 2 規劃器的狀態回報"""
        if msg.data == "REPLANNING_COMPLETE":
            self.phase2_done = True
            self.phase2_planning = False
            rospy.loginfo("[FSM] ✅ Phase 2 重規劃完成，繼續探索")

    # ── 輔助函數 ────────────────────────────────────────

    def _trigger_phase2_replan(self):
        """觸發 Phase 2 避障重規劃"""
        self.phase2_planning = True
        self.phase2_done = False

        # 發布當前位置作為規劃起點
        goal_msg = PoseStamped()
        goal_msg.header.stamp = rospy.Time.now()
        goal_msg.header.frame_id = 'world'
        # 目標點：繼續向前（可由上層任務規劃器提供）
        goal_msg.pose.position.x = self.current_pos[0] + 3.0
        goal_msg.pose.position.y = self.current_pos[1]
        goal_msg.pose.position.z = self.current_pos[2]
        goal_msg.pose.orientation.w = 1.0
        self.replan_trigger_pub.publish(goal_msg)

        rospy.loginfo(f"[FSM] 已觸發 Phase 2 重規劃，目標: "
                      f"({goal_msg.pose.position.x:.2f}, "
                      f"{goal_msg.pose.position.y:.2f}, "
                      f"{goal_msg.pose.position.z:.2f})")

    def _broadcast_state(self):
        """廣播 FSM 狀態"""
        self.fsm_state_pub.publish(String(data=self.state))

    # ── 主循環 ─────────────────────────────────────────

    def run(self):
        rate = rospy.Rate(10)  # 10 Hz

        rospy.loginfo("[FSM] 開始主循環")

        while not rospy.is_shutdown():

            self._broadcast_state()

            # ──────────────────────────────────────────
            # 階段一：起飛與探索
            # ──────────────────────────────────────────
            if self.state == "TAKEOFF_EXPLORE":
                # 優先級 1：偵測到障礙物 → 觸發重規劃
                if self.obstacle_detected and not self.phase2_planning:
                    rospy.loginfo("[FSM] 切換至 OBSTACLE_REPLAN 階段")
                    self.state = "OBSTACLE_REPLAN"
                    self._trigger_phase2_replan()

                # 優先級 2：穩定看到 AprilTag → 直接進入接近
                elif self.tag_detected:
                    rospy.loginfo("[FSM] AprilTag 連續偵測成功！切換至 APPROACH 階段")
                    self.state = "APPROACH"

            # ──────────────────────────────────────────
            # 階段二：動態避障重規劃 (Phase 2 核心)
            # ──────────────────────────────────────────
            elif self.state == "OBSTACLE_REPLAN":
                if self.phase2_planning:
                    # 等待規劃器完成（規劃器非同步執行）
                    rospy.loginfo_throttle(2.0, "[FSM] 等待 Phase 2 規劃完成...")

                elif self.phase2_done:
                    # 規劃完成，重置狀態並繼續探索
                    rospy.loginfo("[FSM] 重規劃完成，回到 TAKEOFF_EXPLORE 繼續")
                    self.obstacle_detected = False
                    self.phase2_done = False
                    self.state = "TAKEOFF_EXPLORE"

                else:
                    # 若 phase2 未在規劃中也未完成（異常情況），重新觸發
                    rospy.logwarn("[FSM] Phase 2 狀態異常，重新觸發規劃")
                    self._trigger_phase2_replan()

            # ──────────────────────────────────────────
            # 階段三：接近降落點
            # ──────────────────────────────────────────
            elif self.state == "APPROACH":
                # 過程中若仍有障礙物，插入一次重規劃
                if self.obstacle_detected and not self.phase2_planning:
                    rospy.logwarn("[FSM] APPROACH 中偵測到障礙物，插入重規劃")
                    self._trigger_phase2_replan()
                    self.phase2_planning = True

                elif self.detection_count > 30:  # 穩定追蹤後切換
                    rospy.loginfo("[FSM] 目標鎖定，切換至 PRECISION_LAND 階段")
                    self.state = "PRECISION_LAND"

            # ──────────────────────────────────────────
            # 階段四：精準降落
            # ──────────────────────────────────────────
            elif self.state == "PRECISION_LAND":
                if self.current_mavros_state.mode != "AUTO.PRECLAND":
                    rospy.loginfo("[FSM] 觸發 AUTO.PRECLAND 模式...")
                    try:
                        resp = self.set_mode_srv(0, 'AUTO.PRECLAND')
                        if resp.mode_sent:
                            rospy.loginfo("[FSM] ✅ 成功切換至 AUTO.PRECLAND！")
                    except rospy.ServiceException as e:
                        rospy.logerr(f"[FSM] SetMode 服務失敗: {e}")

                rospy.sleep(5)
                rospy.loginfo("[FSM] 降落完成，狀態機結束。")
                break

            rate.sleep()


if __name__ == '__main__':
    try:
        fsm = DroneFSM()
        fsm.run()
    except rospy.ROSInterruptException:
        pass
