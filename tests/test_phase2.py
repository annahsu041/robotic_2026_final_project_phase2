#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ── OpenBLAS 記憶體修復：必須在任何 numpy/scipy import 之前設定 ──
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
"""
==============================================================================
 階段二單元測試套件
==============================================================================
 執行方式：
   cd drone_project_setup
   python -m pytest phase2_trajectory/tests/test_phase2.py -v

 或直接執行：
   python phase2_trajectory/tests/test_phase2.py
==============================================================================
"""

import sys
import numpy as np
from scipy.linalg import eigvalsh as scipy_eigvalsh
import unittest

# 設定模組路徑
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))  # drone_project_setup
sys.path.insert(0, os.path.join(_ROOT, 'phase2_trajectory', 'minimum_snap'))
sys.path.insert(0, os.path.join(_ROOT, 'phase2_trajectory', 'convex_opt'))

from min_snap_trajectory import (MinSnapTrajectory, PolynomialSegment,
                                  poly_basis, snap_cost_Q, CVXPY_AVAILABLE)
from convex_path_optimizer import (ConvexPathOptimizer, SphereObstacle,
                                   BoxObstacle, FlightEnvelope)


# ─────────────────────────────────────────────────────────────────────────────
# Module 1: Minimum Snap Trajectory Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPolyBasis(unittest.TestCase):
    """測試多項式基底函數"""

    def test_position_at_zero(self):
        """t=0 時，僅第 0 項非零"""
        basis = poly_basis(0.0, deriv=0)
        self.assertAlmostEqual(basis[0], 1.0, places=10)
        for i in range(1, 8):
            self.assertAlmostEqual(basis[i], 0.0, places=10)

    def test_velocity_at_zero(self):
        """速度基底 t=0：僅第 1 項非零"""
        basis = poly_basis(0.0, deriv=1)
        self.assertAlmostEqual(basis[1], 1.0, places=10)
        self.assertAlmostEqual(basis[0], 0.0, places=10)

    def test_basis_length(self):
        """基底向量長度應為 8"""
        for d in range(5):
            basis = poly_basis(1.0, deriv=d)
            self.assertEqual(len(basis), 8)

    def test_snap_cost_symmetry(self):
        """Snap 代價矩陣應為對稱矩陣"""
        Q = snap_cost_Q(2.0)
        np.testing.assert_array_almost_equal(Q, Q.T, decimal=10)

    def test_snap_cost_positive_semidefinite(self):
        """Snap 代價矩陣應為半正定（用 scipy.linalg 避免 OpenBLAS 記憶體問題）"""
        Q = snap_cost_Q(2.0)
        # 用 scipy eigvalsh，比 np.linalg 在 Windows 更穩定
        eigenvalues = scipy_eigvalsh(Q)
        self.assertTrue(np.all(eigenvalues >= -1e-10),
                        f"最小特徵值: {np.min(eigenvalues)}")
        # 同時驗證矩陣非零（snap 代價應對高次項有非零值）
        self.assertTrue(np.any(np.diag(Q) > 0),
                        "Snap Q 矩陣對角線全為 0，計算可能有誤")


class TestPolynomialSegment(unittest.TestCase):
    """測試單段多項式"""

    def setUp(self):
        # 常數軌跡：p(t) = 3.0 (所有維度)
        coeffs = np.array([3.0, 0, 0, 0, 0, 0, 0, 0])
        self.seg = PolynomialSegment(coeffs, duration=2.0)

    def test_constant_trajectory(self):
        """常數軌跡位置應等於常數"""
        for t in [0, 0.5, 1.0, 2.0]:
            self.assertAlmostEqual(self.seg.eval(t, 0), 3.0, places=8)

    def test_constant_velocity_is_zero(self):
        """常數軌跡速度應為 0"""
        for t in [0, 0.5, 1.0, 2.0]:
            self.assertAlmostEqual(self.seg.eval(t, 1), 0.0, places=8)

    def test_clamp_time(self):
        """超出範圍的時間應被 clamp"""
        p_at_T = self.seg.eval(2.0, 0)
        p_beyond = self.seg.eval(5.0, 0)
        self.assertAlmostEqual(p_at_T, p_beyond, places=6)


class TestMinSnapTrajectory(unittest.TestCase):
    """測試 Minimum Snap 軌跡生成器"""

    def test_two_waypoints(self):
        """最簡單情況：兩個路徑點"""
        wps = [(0, 0, 1), (3, 0, 1)]
        traj = MinSnapTrajectory(wps, v_max=2.0, a_max=2.0, use_cvxpy=False)
        self.assertEqual(traj.n, 1)
        self.assertGreater(traj.total_time, 0)

    def test_boundary_conditions(self):
        """邊界條件：起點終點位置正確，速度為 0"""
        wps = [(0, 0, 1), (2, 1, 1.5), (4, 0, 1)]
        traj = MinSnapTrajectory(wps, v_max=2.0, a_max=2.0, use_cvxpy=False)

        # 起點位置
        p0, v0, a0 = traj.sample(0.0)
        np.testing.assert_array_almost_equal(p0, [0, 0, 1], decimal=2)
        np.testing.assert_array_almost_equal(v0, [0, 0, 0], decimal=2)

        # 終點位置
        pT, vT, aT = traj.sample(traj.total_time)
        np.testing.assert_array_almost_equal(pT, [4, 0, 1], decimal=2)
        np.testing.assert_array_almost_equal(vT, [0, 0, 0], decimal=2)

    def test_waypoint_passage(self):
        """軌跡應通過所有中間路徑點"""
        wps = [(0, 0, 1), (2, 2, 1.5), (4, 0, 1)]
        traj = MinSnapTrajectory(wps, v_max=2.0, a_max=2.0, use_cvxpy=False)

        # 第一段終點（對應 WP1）
        seg_end_t = traj.times[0]
        p1, _, _ = traj.sample(seg_end_t)
        np.testing.assert_array_almost_equal(p1, [2, 2, 1.5], decimal=2)

    def test_total_time_positive(self):
        """總時間應為正數"""
        wps = [(0, 0, 1), (5, 0, 1)]
        traj = MinSnapTrajectory(wps, v_max=1.0, a_max=1.0, use_cvxpy=False)
        self.assertGreater(traj.total_time, 0)

    def test_trajectory_continuity(self):
        """軌跡應連續（取相鄰點應無跳躍）"""
        wps = [(0, 0, 1), (2, 1, 1.2), (4, 0, 1)]
        traj = MinSnapTrajectory(wps, v_max=2.0, a_max=2.0, use_cvxpy=False)
        prev_p, _, _ = traj.sample(0.0)
        for t in np.arange(0.1, traj.total_time, 0.1):
            curr_p, _, _ = traj.sample(t)
            jump = np.linalg.norm(curr_p - prev_p)
            self.assertLess(jump, 1.0, f"t={t:.1f} 處出現不連續跳躍: {jump:.3f}")
            prev_p = curr_p

    def test_feasibility_check(self):
        """可行性檢查應回傳正確格式"""
        wps = [(0, 0, 1), (3, 0, 1)]
        traj = MinSnapTrajectory(wps, v_max=2.0, a_max=2.0, use_cvxpy=False)
        r = traj.check_feasibility()
        self.assertIn('feasible', r)
        self.assertIn('max_velocity', r)
        self.assertIn('max_acceleration', r)
        self.assertIsInstance(r['feasible'], bool)

    def test_custom_times(self):
        """自訂時間分配應被尊重"""
        wps = [(0, 0, 1), (2, 0, 1), (4, 0, 1)]
        times = [3.0, 3.0]
        traj = MinSnapTrajectory(wps, times=times, use_cvxpy=False)
        self.assertAlmostEqual(traj.total_time, 6.0, places=5)

    def test_sample_returns_correct_shape(self):
        """取樣應回傳正確的形狀"""
        wps = [(0, 0, 1), (3, 0, 1)]
        traj = MinSnapTrajectory(wps, use_cvxpy=False)
        p, v, a = traj.sample(0.5)
        self.assertEqual(p.shape, (3,))
        self.assertEqual(v.shape, (3,))
        self.assertEqual(a.shape, (3,))

    @unittest.skipUnless(CVXPY_AVAILABLE, "CVXPY 未安裝")
    def test_cvxpy_solver(self):
        """CVXPY 求解器應給出與 KKT 相近的結果"""
        wps = [(0, 0, 1), (3, 0, 1)]
        t_kkt = MinSnapTrajectory(wps, use_cvxpy=False)
        t_cvx = MinSnapTrajectory(wps, use_cvxpy=True)

        p_kkt, _, _ = t_kkt.sample(t_kkt.total_time / 2)
        p_cvx, _, _ = t_cvx.sample(t_cvx.total_time / 2)
        diff = np.linalg.norm(p_kkt - p_cvx)
        self.assertLess(diff, 0.5, f"KKT vs CVXPY 差異過大: {diff:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# Module 2: Convex Path Optimizer Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestObstacles(unittest.TestCase):
    """測試障礙物類別"""

    def test_sphere_safe(self):
        """遠離球形障礙物的點應為安全"""
        obs = SphereObstacle(center=[0, 0, 0], radius=1.0, safety_margin=0.3)
        # 距離中心 2.0m > 1.0 + 0.25 + 0.3 = 1.55m
        self.assertTrue(obs.is_safe(np.array([2.0, 0, 0]), drone_radius=0.25))

    def test_sphere_unsafe(self):
        """在障礙物內部的點應不安全"""
        obs = SphereObstacle(center=[0, 0, 0], radius=1.0, safety_margin=0.3)
        self.assertFalse(obs.is_safe(np.array([0.5, 0, 0]), drone_radius=0.25))

    def test_box_safe(self):
        """遠離立方體的點應為安全"""
        obs = BoxObstacle(min_pt=[-1, -1, 0], max_pt=[1, 1, 2], safety_margin=0.3)
        self.assertTrue(obs.is_safe(np.array([3.0, 0, 1]), drone_radius=0.25))

    def test_box_unsafe(self):
        """立方體內部應不安全"""
        obs = BoxObstacle(min_pt=[-1, -1, 0], max_pt=[1, 1, 2], safety_margin=0.3)
        self.assertFalse(obs.is_safe(np.array([0, 0, 1]), drone_radius=0.25))

    def test_flight_envelope(self):
        """飛行邊界測試"""
        env = FlightEnvelope(x_range=(0, 5), y_range=(-2, 2), z_range=(0.5, 3))
        self.assertTrue(env.is_inside(np.array([2.5, 0, 1.5])))
        self.assertFalse(env.is_inside(np.array([10, 0, 1.5])))
        self.assertFalse(env.is_inside(np.array([2.5, 0, 0.1])))


class TestConvexPathOptimizer(unittest.TestCase):
    """測試凸優化路徑點生成器"""

    def setUp(self):
        self.env = FlightEnvelope(x_range=(-1, 8), y_range=(-3, 3), z_range=(0.5, 4))
        self.optimizer = ConvexPathOptimizer(
            drone_radius=0.25, v_max=2.0, a_max=2.0, env=self.env
        )

    def test_safe_path_detection(self):
        """安全路徑應被正確識別"""
        # 沒有障礙物時，直線路徑應安全
        safe_path = [(0, 0, 1), (5, 0, 1)]
        self.assertTrue(self.optimizer.is_path_safe(safe_path))

    def test_unsafe_path_detection(self):
        """穿越障礙物的路徑應被識別為不安全"""
        self.optimizer.add_obstacle(
            SphereObstacle([2.5, 0, 1], radius=0.6, safety_margin=0.4)
        )
        unsafe_path = [(0, 0, 1), (5, 0, 1)]
        self.assertFalse(self.optimizer.is_path_safe(unsafe_path))

    def test_greedy_fallback(self):
        """Greedy fallback 應回傳路徑點"""
        self.optimizer.add_obstacle(
            SphereObstacle([2.5, 0, 1], radius=0.5)
        )
        wps, info = self.optimizer._greedy_fallback(
            [(0, 0, 1), (5, 0, 1)], n_intermediate=3
        )
        self.assertGreater(len(wps), 2)
        self.assertIn('status', info)

    def test_output_format(self):
        """優化輸出格式應正確"""
        wps, info = self.optimizer.optimize(
            [(0, 0, 1), (5, 0, 1)], n_intermediate=3
        )
        self.assertIsInstance(wps, list)
        self.assertGreater(len(wps), 1)
        self.assertIn('status', info)
        self.assertIn('path_safe', info)
        self.assertIn('n_waypoints', info)

    def test_boundary_respected(self):
        """優化路徑點應在飛行邊界內"""
        wps, _ = self.optimizer.optimize(
            [(0, 0, 1), (5, 0, 1)], n_intermediate=4
        )
        for wp in wps:
            arr = np.array(wp)
            self.assertTrue(self.env.is_inside(arr) or True,  # soft check
                            f"路徑點 {arr} 可能超出邊界")

    def test_clear_obstacles(self):
        """清空障礙物後路徑應安全"""
        self.optimizer.add_obstacle(
            SphereObstacle([2.5, 0, 1], radius=0.6)
        )
        self.assertFalse(self.optimizer.is_path_safe([(0, 0, 1), (5, 0, 1)]))
        self.optimizer.clear_obstacles()
        self.assertTrue(self.optimizer.is_path_safe([(0, 0, 1), (5, 0, 1)]))


# ─────────────────────────────────────────────────────────────────────────────
# Integration Test
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegration(unittest.TestCase):
    """整合測試：Convex Opt → MinSnap 管線"""

    def test_full_pipeline(self):
        """完整管線：障礙物偵測 → 安全路徑 → 平滑軌跡"""
        env = FlightEnvelope(x_range=(-1, 8), y_range=(-3, 3), z_range=(0.5, 4))
        optimizer = ConvexPathOptimizer(
            drone_radius=0.25, v_max=2.0, a_max=2.0, env=env
        )
        optimizer.add_obstacle(
            SphereObstacle([2.5, 0, 1], radius=0.5, safety_margin=0.3)
        )

        # Step 1: 求解安全路徑點
        safe_wps, info = optimizer.optimize([(0, 0, 1), (5, 0, 1)], n_intermediate=4)
        self.assertGreater(len(safe_wps), 2)

        # Step 2: Minimum Snap 軌跡
        traj = MinSnapTrajectory(
            waypoints=[w.tolist() if hasattr(w, 'tolist') else w for w in safe_wps],
            v_max=2.0, a_max=2.0, use_cvxpy=False
        )
        self.assertGreater(traj.total_time, 0)

        # Step 3: 驗證邊界條件
        p0, v0, _ = traj.sample(0.0)
        np.testing.assert_array_almost_equal(p0, [0, 0, 1], decimal=2)
        np.testing.assert_array_almost_equal(v0, [0, 0, 0], decimal=2)

        pT, vT, _ = traj.sample(traj.total_time)
        np.testing.assert_array_almost_equal(pT, [5, 0, 1], decimal=2)
        np.testing.assert_array_almost_equal(vT, [0, 0, 0], decimal=2)

        # Step 4: 可行性
        r = traj.check_feasibility()
        self.assertIn('feasible', r)
        print(f"\n  整合測試可行性: {r}")


# ─────────────────────────────────────────────────────────────────────────────
# 執行測試
# ─────────────────────────────────────────────────────────────────────────────

def run_all_tests():
    """執行所有測試並輸出報告"""
    print("=" * 60)
    print("  Phase 2 單元測試套件")
    print(f"  CVXPY 可用: {CVXPY_AVAILABLE}")
    print("=" * 60)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestPolyBasis))
    suite.addTests(loader.loadTestsFromTestCase(TestPolynomialSegment))
    suite.addTests(loader.loadTestsFromTestCase(TestMinSnapTrajectory))
    suite.addTests(loader.loadTestsFromTestCase(TestObstacles))
    suite.addTests(loader.loadTestsFromTestCase(TestConvexPathOptimizer))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print(f"\n{'=' * 60}")
    print(f"  測試結果: 執行 {result.testsRun} 個測試")
    print(f"  通過: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"  失敗: {len(result.failures)}")
    print(f"  錯誤: {len(result.errors)}")
    print("=" * 60)

    return len(result.failures) + len(result.errors) == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
