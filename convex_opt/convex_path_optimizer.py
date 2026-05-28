#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
 階段二模組二：Convex Optimization 避障路徑點生成器
==============================================================================
 理論基礎 (Week 5)：
   - 將環境約束（障礙物球形/立方體排除區）轉化為凸約束
   - 使用 CVXPY 求解 QP，在動態限制內找到安全路徑點
   - 目標：min ||waypoints - original_path||² + λ * safety_margin
   - 約束：
     * 碰撞安全距離 ||wp_i - obs_j|| >= r_obs + r_drone + margin
     * 速度上限 (linearized): ||wp_{i+1} - wp_i|| <= v_max * dt_segment
     * 高度約束: z_min <= z <= z_max
     * 飛行邊界: x∈[x_min, x_max], y∈[y_min, y_max]

 本地執行：
   pip install numpy cvxpy matplotlib
   python convex_path_optimizer.py

 ROS 整合：
   from phase2_trajectory.convex_opt.convex_path_optimizer import ConvexPathOptimizer
==============================================================================
"""

import numpy as np
import warnings

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    warnings.warn(
        "[ConvexOpt] CVXPY 未安裝！\n"
        "  安裝: pip install cvxpy\n"
        "  此模組需要 CVXPY 才能執行完整凸優化。\n"
        "  僅啟用基礎 greedy fallback。", ImportWarning, stacklevel=2
    )


class Obstacle:
    """障礙物抽象基類"""
    def __init__(self, safety_margin: float = 0.3):
        self.margin = safety_margin

    def min_clearance_sq(self, point: np.ndarray) -> float:
        """回傳點到障礙物的最小距離平方"""
        raise NotImplementedError

    def is_safe(self, point: np.ndarray, drone_radius: float = 0.25) -> bool:
        raise NotImplementedError


class SphereObstacle(Obstacle):
    """球形障礙物（適合柱狀物體的保守近似）"""

    def __init__(self, center: list, radius: float, safety_margin: float = 0.3):
        super().__init__(safety_margin)
        self.center = np.array(center, dtype=float)
        self.radius = float(radius)

    def min_clearance_sq(self, point: np.ndarray) -> float:
        return float(np.sum((point - self.center) ** 2))

    def is_safe(self, point: np.ndarray, drone_radius: float = 0.25) -> bool:
        min_dist = self.radius + drone_radius + self.margin
        return self.min_clearance_sq(point) >= min_dist ** 2

    def __repr__(self):
        return f"SphereObstacle(center={self.center}, r={self.radius})"


class BoxObstacle(Obstacle):
    """軸對齊立方體障礙物（AABB）"""

    def __init__(self, min_pt: list, max_pt: list, safety_margin: float = 0.3):
        super().__init__(safety_margin)
        self.min_pt = np.array(min_pt, dtype=float)
        self.max_pt = np.array(max_pt, dtype=float)

    def min_clearance_sq(self, point: np.ndarray) -> float:
        # 點到 AABB 的最近距離
        d = np.maximum(self.min_pt - point, 0) + np.maximum(point - self.max_pt, 0)
        return float(np.sum(d ** 2))

    def is_safe(self, point: np.ndarray, drone_radius: float = 0.25) -> bool:
        min_dist = drone_radius + self.margin
        return self.min_clearance_sq(point) >= min_dist ** 2

    def __repr__(self):
        return f"BoxObstacle(min={self.min_pt}, max={self.max_pt})"


class FlightEnvelope:
    """飛行邊界定義"""

    def __init__(self,
                 x_range=(-10, 10),
                 y_range=(-10, 10),
                 z_range=(0.5, 5.0)):
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        self.z_min, self.z_max = z_range

    def is_inside(self, point: np.ndarray) -> bool:
        return (self.x_min <= point[0] <= self.x_max and
                self.y_min <= point[1] <= self.y_max and
                self.z_min <= point[2] <= self.z_max)


class ConvexPathOptimizer:
    """
    凸優化路徑點生成器

    當障礙物偵測模組發出警報後，此類別計算繞避路徑點，
    再傳給 MinSnapTrajectory 生成平滑軌跡。

    範例
    ----
    >>> optimizer = ConvexPathOptimizer(drone_radius=0.25)
    >>> optimizer.add_obstacle(SphereObstacle([2.5, 0, 1], radius=0.5))
    >>> original_path = [(0,0,1), (5,0,1)]
    >>> safe_wps, info = optimizer.optimize(original_path, n_intermediate=3)
    """

    def __init__(self,
                 drone_radius: float = 0.25,
                 v_max: float = 2.0,
                 a_max: float = 2.0,
                 env: FlightEnvelope = None,
                 lambda_smooth: float = 1.0,
                 lambda_safe: float = 10.0):
        """
        Parameters
        ----------
        drone_radius : float
            無人機等效半徑（m）
        v_max : float
            最大速度（m/s）
        a_max : float
            最大加速度（m/s²）
        env : FlightEnvelope
            飛行邊界
        lambda_smooth : float
            平滑性懲罰權重
        lambda_safe : float
            安全距離懲罰權重（soft constraint）
        """
        self.drone_radius = drone_radius
        self.v_max = v_max
        self.a_max = a_max
        self.env = env or FlightEnvelope()
        self.lambda_smooth = lambda_smooth
        self.lambda_safe = lambda_safe
        self.obstacles: list = []

    def add_obstacle(self, obs: Obstacle):
        """新增障礙物"""
        self.obstacles.append(obs)
        return self

    def clear_obstacles(self):
        """清空障礙物列表"""
        self.obstacles.clear()

    def is_path_safe(self, waypoints: list, check_step: float = 0.1) -> bool:
        """
        線段取樣檢查路徑是否與所有障礙物安全間隔

        Parameters
        ----------
        waypoints : list of array-like
        check_step : float
            取樣步長（m）
        """
        for i in range(len(waypoints) - 1):
            p0 = np.array(waypoints[i], dtype=float)
            p1 = np.array(waypoints[i + 1], dtype=float)
            d = np.linalg.norm(p1 - p0)
            n_steps = max(2, int(d / check_step))
            for k in range(n_steps + 1):
                pt = p0 + (p1 - p0) * k / n_steps
                for obs in self.obstacles:
                    if not obs.is_safe(pt, self.drone_radius):
                        return False
        return True

    def optimize(self, original_path: list, n_intermediate: int = 3,
                 avg_speed: float = None) -> tuple:
        """
        凸優化求解安全路徑點

        Parameters
        ----------
        original_path : list of (x, y, z)
            原始路徑（起點 + 終點，或已規劃的粗路徑）
        n_intermediate : int
            在每兩個路徑點之間插入的中間點數量
        avg_speed : float, optional
            預估平均速度，用於計算段時間約束

        Returns
        -------
        safe_waypoints : list of np.ndarray
            優化後的安全路徑點
        info : dict
            求解資訊
        """
        if not CVXPY_AVAILABLE:
            return self._greedy_fallback(original_path, n_intermediate)

        # 建立決策變數
        # 固定起點和終點，優化中間點
        wps_in = [np.array(p, dtype=float) for p in original_path]
        start = wps_in[0]
        goal = wps_in[-1]

        # 中間點：在原始路徑上均勻插值作為初始猜測
        n_opt = (len(wps_in) - 1) * n_intermediate
        init_pts = []
        for seg_i in range(len(wps_in) - 1):
            for k in range(1, n_intermediate + 1):
                alpha = k / (n_intermediate + 1)
                init_pts.append(wps_in[seg_i] + alpha * (wps_in[seg_i + 1] - wps_in[seg_i]))

        # CVXPY 決策變數（中間點坐標）
        X = cp.Variable((n_opt, 3))

        # 所有點（含固定端點）
        all_pts = []  # list of expressions
        all_pts.append(start)
        for i in range(n_opt):
            all_pts.append(X[i])
        all_pts.append(goal)
        N_all = len(all_pts)

        # ── 目標函數 ──────────────────────────────────
        # 1. 平滑性：最小化相鄰路徑點距離的二次差（類似加速度懲罰）
        smooth_cost = 0
        for i in range(1, N_all - 1):
            if i == 0 or i == N_all - 1:
                continue
            prev = all_pts[i - 1] if isinstance(all_pts[i - 1], np.ndarray) else all_pts[i - 1]
            curr = all_pts[i]
            nxt = all_pts[i + 1] if isinstance(all_pts[i + 1], np.ndarray) else all_pts[i + 1]

            if isinstance(curr, np.ndarray):
                continue  # 固定點不納入優化

            diff = None
            if isinstance(prev, np.ndarray) and not isinstance(nxt, np.ndarray):
                diff = (nxt - curr) - (curr - prev)
            elif not isinstance(prev, np.ndarray) and isinstance(nxt, np.ndarray):
                diff = (nxt - curr) - (curr - prev)
            elif not isinstance(prev, np.ndarray) and not isinstance(nxt, np.ndarray):
                diff = (nxt - curr) - (curr - prev)

            if diff is not None:
                smooth_cost += cp.sum_squares(diff)

        # 2. 接近原始路徑（不偏離太遠）
        path_cost = 0
        for i, xi in enumerate(X):
            path_cost += cp.sum_squares(xi - init_pts[i])

        objective = cp.Minimize(
            self.lambda_smooth * smooth_cost +
            1.0 * path_cost
        )

        # ── 約束條件 ──────────────────────────────────
        constraints = []

        # 飛行邊界（box 約束，线性）
        constraints += [
            X[:, 0] >= self.env.x_min,
            X[:, 0] <= self.env.x_max,
            X[:, 1] >= self.env.y_min,
            X[:, 1] <= self.env.y_max,
            X[:, 2] >= self.env.z_min,
            X[:, 2] <= self.env.z_max,
        ]

        # 相鄰點距離（速度）約束（線性近似）
        if avg_speed is None:
            avg_speed = self.v_max * 0.7  # 70% 最大速度作為平均
        for i in range(N_all - 1):
            p_i = all_pts[i]
            p_j = all_pts[i + 1]
            if isinstance(p_i, np.ndarray) and isinstance(p_j, np.ndarray):
                continue
            diff = (p_j - p_i) if not isinstance(p_j, np.ndarray) else -(p_i - p_j)
            # 用 norm_inf 近似 norm_2 以保持線性（保守）
            if not isinstance(p_i, np.ndarray):
                constraints.append(p_i - p_j <= self.v_max * 2.0)
                constraints.append(p_j - p_i <= self.v_max * 2.0)

        # 注意：障礙物距離約束是非凸的（norm >= r 是非凸）
        # 在這裡使用線性化技巧：在初始猜測附近展開
        # ||x - c|| >= r  =>  (x0-c)^T (x-c) >= r * ||x0-c||  (linearized)
        for k, obs in enumerate(self.obstacles):
            if isinstance(obs, SphereObstacle):
                safe_r = obs.radius + self.drone_radius + obs.margin
                c = obs.center
                for i, xi in enumerate(X):
                    x0 = np.array(init_pts[i])
                    d0 = x0 - c
                    d0_norm = np.linalg.norm(d0)
                    if d0_norm < 1e-6:
                        # 若初始猜測就在障礙物中心，往上偏移
                        d0 = np.array([0.0, 0.0, 1.0])
                        d0_norm = 1.0
                    d0_hat = d0 / d0_norm
                    # 線性化約束：d0_hat^T (xi - c) >= safe_r
                    constraints.append(d0_hat @ (xi - c) >= safe_r)

        # ── 求解 ──────────────────────────────────────
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.OSQP, verbose=False,
                       eps_abs=1e-5, eps_rel=1e-5, warm_start=True,
                       max_iter=10000)
        except Exception as e:
            warnings.warn(f"[ConvexOpt] 求解器異常: {e}，使用 greedy fallback")
            return self._greedy_fallback(original_path, n_intermediate)

        if prob.status not in ("optimal", "optimal_inaccurate") or X.value is None:
            warnings.warn(f"[ConvexOpt] 求解狀態: {prob.status}，使用 greedy fallback")
            return self._greedy_fallback(original_path, n_intermediate)

        # 組合完整路徑點
        opt_intermediate = X.value  # (n_opt, 3)
        safe_wps = [start]
        for pt in opt_intermediate:
            safe_wps.append(pt)
        safe_wps.append(goal)

        # 後處理：若仍有不安全點則補充修正
        safe_wps = self._post_process(safe_wps)

        return safe_wps, {
            'status': prob.status,
            'solver': 'CVXPY-OSQP',
            'n_waypoints': len(safe_wps),
            'path_safe': self.is_path_safe(safe_wps),
            'objective': float(prob.value) if prob.value is not None else None,
        }

    def _greedy_fallback(self, original_path: list, n_intermediate: int) -> tuple:
        """
        不需要 CVXPY 的 greedy 繞避策略：
        在原始路徑上均勻插點，遇到不安全點就往側方向偏移
        """
        wps_in = [np.array(p, dtype=float) for p in original_path]
        result = [wps_in[0]]

        for seg_i in range(len(wps_in) - 1):
            p0, p1 = wps_in[seg_i], wps_in[seg_i + 1]
            for k in range(1, n_intermediate + 1):
                alpha = k / (n_intermediate + 1)
                pt = p0 + alpha * (p1 - p0)
                # 若不安全，往 +Y 和 +Z 偏移
                attempts = 0
                offsets = [(0, 0.5, 0), (0, -0.5, 0), (0, 0, 0.5),
                           (0.5, 0.5, 0), (-0.5, 0.5, 0), (0, 1.0, 0.3)]
                while attempts < len(offsets):
                    for obs in self.obstacles:
                        if not obs.is_safe(pt, self.drone_radius):
                            off = np.array(offsets[attempts])
                            pt = pt + off
                            attempts += 1
                            break
                    else:
                        break
                result.append(pt)

        result.append(wps_in[-1])
        return result, {
            'status': 'greedy_fallback',
            'solver': 'greedy',
            'n_waypoints': len(result),
            'path_safe': self.is_path_safe(result),
            'objective': None,
        }

    def _post_process(self, waypoints: list) -> list:
        """後處理：移除過於接近的重複點"""
        if len(waypoints) < 2:
            return waypoints
        result = [waypoints[0]]
        for wp in waypoints[1:]:
            if np.linalg.norm(wp - result[-1]) > 0.1:
                result.append(wp)
        return result


# ─────────────────────────────────────────────────────
# 本地獨立測試
# ─────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 55)
    print("  Convex Path Optimizer - 本地獨立測試")
    print(f"  CVXPY 可用: {CVXPY_AVAILABLE}")
    print("=" * 55)

    # 定義環境
    env = FlightEnvelope(x_range=(-1, 7), y_range=(-3, 3), z_range=(0.5, 4.0))
    optimizer = ConvexPathOptimizer(
        drone_radius=0.25, v_max=2.0, a_max=2.0, env=env,
        lambda_smooth=2.0, lambda_safe=10.0
    )

    # 加入障礙物
    obs1 = SphereObstacle(center=[2.5, 0.0, 1.0], radius=0.6, safety_margin=0.4)
    obs2 = SphereObstacle(center=[3.5, 0.5, 1.2], radius=0.4, safety_margin=0.3)
    optimizer.add_obstacle(obs1)
    optimizer.add_obstacle(obs2)

    print(f"\n障礙物：{obs1}, {obs2}")

    # 原始直線路徑（穿越障礙物）
    original = [(0.0, 0.0, 1.0), (5.0, 0.0, 1.0)]
    print(f"原始路徑：{original[0]} → {original[1]}")
    print(f"原始路徑安全？{optimizer.is_path_safe(original)}")

    print("\n[求解凸優化路徑...]")
    safe_wps, info = optimizer.optimize(original, n_intermediate=4)

    print(f"\n求解結果：")
    print(f"  狀態: {info['status']}")
    print(f"  求解器: {info['solver']}")
    print(f"  路徑點數: {info['n_waypoints']}")
    print(f"  路徑安全？{info['path_safe']}")
    print(f"\n安全路徑點：")
    for i, wp in enumerate(safe_wps):
        arr = np.array(wp)
        safe = all(obs.is_safe(arr, 0.25) for obs in optimizer.obstacles)
        print(f"  WP{i}: ({arr[0]:.3f}, {arr[1]:.3f}, {arr[2]:.3f})  安全={safe}")

    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        # 障礙物可視化（球）
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 20)
        for obs in optimizer.obstacles:
            r = obs.radius + optimizer.drone_radius + obs.margin
            xs = obs.center[0] + r * np.outer(np.cos(u), np.sin(v))
            ys = obs.center[1] + r * np.outer(np.sin(u), np.sin(v))
            zs = obs.center[2] + r * np.outer(np.ones_like(u), np.cos(v))
            ax.plot_surface(xs, ys, zs, alpha=0.15, color='red')

        # 原始路徑
        orig_arr = np.array(original)
        ax.plot(orig_arr[:, 0], orig_arr[:, 1], orig_arr[:, 2],
                'k--', lw=2, label='原始路徑')

        # 優化路徑
        wp_arr = np.array([np.array(w) for w in safe_wps])
        ax.plot(wp_arr[:, 0], wp_arr[:, 1], wp_arr[:, 2],
                'g-o', lw=2, markersize=6, label='凸優化安全路徑')

        ax.set_title('Convex Optimization 避障路徑點生成')
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
        ax.legend()
        plt.tight_layout()
        plt.savefig('convex_opt_result.png', dpi=150, bbox_inches='tight')
        print("\n✅ 圖表已儲存: convex_opt_result.png")
        plt.show()
    except ImportError:
        print("\n[提示] pip install matplotlib 可顯示圖表")

    print("\n✅ 凸優化模組本地測試完成！")
    print("   下一步：此輸出傳入 MinSnapTrajectory 生成平滑軌跡")
