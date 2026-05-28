#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
 階段二模組一：Minimum Snap Trajectory Generator
==============================================================================
 理論基礎 (Week 9)：
   - 最小化 snap (4th derivative of position)，確保加速度/jerk 平滑
   - 多項式分段軌跡：每段 7 次多項式，共 3 軸 (x, y, z)
   - 透過 QP 求解最小化目標函數：min ∫ ||d⁴p/dt⁴||² dt

 模組設計原則（認領模組後在本地跑起來）：
   1. 純 Python，不依賴 ROS，可在任何機器上獨立執行
   2. 輸入/輸出為標準 numpy 格式，方便之後貼入 ROS 框架
   3. 完整單元測試在 tests/ 目錄

 本地執行：
   pip install numpy scipy cvxpy matplotlib
   python min_snap_trajectory.py

 ROS 整合：
   from phase2_trajectory.minimum_snap.min_snap_trajectory import MinSnapTrajectory
==============================================================================
"""

import numpy as np
from scipy.linalg import block_diag
import warnings

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    warnings.warn(
        "[MinSnap] CVXPY 未安裝，改用 KKT pseudo-inverse 求解。\n"
        "  安裝指令: pip install cvxpy", ImportWarning, stacklevel=2
    )

# 7 次多項式：8 個係數（minimum snap 需至少 7 次）
POLY_DEG = 7
POLY_N = POLY_DEG + 1  # 8


# ─────────────────────────────────────────────────────
# 工具函數：多項式基底與 snap 代價矩陣
# ─────────────────────────────────────────────────────

def poly_basis(t: float, deriv: int = 0) -> np.ndarray:
    """
    計算 7 次多項式在時刻 t 的基底向量（含 deriv 階導數）
    回傳 shape (8,)
    """
    basis = np.zeros(POLY_N)
    for i in range(deriv, POLY_N):
        c = 1.0
        for j in range(deriv):
            c *= (i - j)
        basis[i] = c * (t ** (i - deriv))
    return basis


def snap_cost_Q(T: float) -> np.ndarray:
    """
    單段 snap 代價矩陣 Q，shape (8, 8)
    Q_ij = ∫₀ᵀ p⁽⁴⁾_i * p⁽⁴⁾_j dt
    """
    Q = np.zeros((POLY_N, POLY_N))
    for i in range(4, POLY_N):
        for j in range(4, POLY_N):
            ci, cj = 1.0, 1.0
            for k in range(4):
                ci *= (i - k)
                cj *= (j - k)
            pw = i + j - 7
            if pw >= 0:
                Q[i, j] = ci * cj * (T ** pw) / pw
    return Q


# ─────────────────────────────────────────────────────
# 核心類別
# ─────────────────────────────────────────────────────

class PolynomialSegment:
    """單段 7 次多項式，支援位置/速度/加速度取樣"""

    def __init__(self, coeffs: np.ndarray, duration: float):
        assert len(coeffs) == POLY_N
        self.c = np.array(coeffs, dtype=float)
        self.T = float(duration)

    def eval(self, t: float, deriv: int = 0) -> float:
        t = float(np.clip(t, 0, self.T))
        return float(np.dot(poly_basis(t, deriv), self.c))


class MinSnapTrajectory:
    """
    Minimum Snap 軌跡生成器

    範例
    ----
    >>> waypoints = [(0,0,1), (2,1,1.5), (4,0,1)]
    >>> traj = MinSnapTrajectory(waypoints, v_max=2.0, a_max=2.0)
    >>> pos, vel, acc = traj.sample(t=1.0)
    >>> report = traj.check_feasibility()
    """

    def __init__(self,
                 waypoints: list,
                 times: list = None,
                 v_max: float = 2.0,
                 a_max: float = 2.0,
                 use_cvxpy: bool = True):
        assert len(waypoints) >= 2, "至少需要 2 個路徑點"
        self.wps = [np.array(w, dtype=float) for w in waypoints]
        self.n = len(waypoints) - 1   # 段數
        self.v_max = v_max
        self.a_max = a_max
        self.use_cvxpy = use_cvxpy and CVXPY_AVAILABLE

        if times is None:
            self.times = self._auto_time()
        else:
            assert len(times) == self.n
            self.times = [float(t) for t in times]

        # 求解三軸係數，shape (n_seg, 8)
        self.cx = self._solve([w[0] for w in self.wps])
        self.cy = self._solve([w[1] for w in self.wps])
        self.cz = self._solve([w[2] for w in self.wps])

        # 建立段物件
        self.segs = [
            {ax: PolynomialSegment(getattr(self, f'c{ax}')[i], self.times[i])
             for ax in 'xyz'}
            for i in range(self.n)
        ]

    # ── 自動時間分配 ──────────────────────────────────
    def _auto_time(self) -> list:
        times = []
        for i in range(self.n):
            d = float(np.linalg.norm(self.wps[i + 1] - self.wps[i]))
            t_v = d / max(self.v_max, 0.1)
            t_a = np.sqrt(2.0 * d / max(self.a_max, 0.1))
            times.append(max(t_v, t_a) * 1.2 + 0.3)
        return times

    # ── 建立等式約束 A, b ────────────────────────────
    def _constraints(self, vals: list):
        n, N = self.n, POLY_N
        total = n * N
        A_rows, b_rows = [], []

        def add(seg_idx, t_local, deriv, rhs):
            row = np.zeros(total)
            row[seg_idx * N:(seg_idx + 1) * N] = poly_basis(t_local, deriv)
            A_rows.append(row)
            b_rows.append(rhs)

        # 起點：位置、速度=0、加速=0
        for d, v in [(0, vals[0]), (1, 0.0), (2, 0.0)]:
            add(0, 0.0, d, v)

        # 終點：位置、速度=0、加速=0
        for d, v in [(0, vals[-1]), (1, 0.0), (2, 0.0)]:
            add(n - 1, self.times[-1], d, v)

        # 中間路徑點
        for i in range(1, n):
            T_i = self.times[i - 1]
            # 前段終點位置
            add(i - 1, T_i, 0, vals[i])
            # 後段起點位置
            add(i, 0.0, 0, vals[i])
            # 速度、加速、jerk 連續
            for d in [1, 2, 3]:
                row = np.zeros(total)
                row[(i - 1) * N:i * N] = poly_basis(T_i, d)
                row[i * N:(i + 1) * N] = -poly_basis(0.0, d)
                A_rows.append(row)
                b_rows.append(0.0)

        return np.vstack(A_rows), np.array(b_rows)

    # ── 建立全局 Q 矩陣 ───────────────────────────────
    def _global_Q(self) -> np.ndarray:
        return block_diag(*[snap_cost_Q(T) for T in self.times])

    # ── 單維度求解 ────────────────────────────────────
    def _solve(self, vals: list) -> np.ndarray:
        n, N = self.n, POLY_N
        total = n * N
        Q = self._global_Q()
        A, b = self._constraints(vals)

        if self.use_cvxpy:
            x = cp.Variable(total)
            prob = cp.Problem(
                cp.Minimize(cp.quad_form(x, cp.psd_wrap(Q + 1e-7 * np.eye(total)))),
                [A @ x == b]
            )
            prob.solve(solver=cp.OSQP, verbose=False,
                       eps_abs=1e-6, eps_rel=1e-6, warm_start=True)
            if prob.status in ("optimal", "optimal_inaccurate"):
                raw = x.value
            else:
                raw = self._kkt_solve(Q, A, b, total)
        else:
            raw = self._kkt_solve(Q, A, b, total)

        return raw.reshape(n, N)

    # ── KKT fallback ─────────────────────────────────
    @staticmethod
    def _kkt_solve(Q, A, b, total) -> np.ndarray:
        m = len(b)
        K = np.zeros((total + m, total + m))
        K[:total, :total] = 2.0 * Q + 1e-6 * np.eye(total)
        K[:total, total:] = A.T
        K[total:, :total] = A
        rhs = np.zeros(total + m)
        rhs[total:] = b
        sol = np.linalg.lstsq(K, rhs, rcond=None)[0]
        return sol[:total]

    # ── 取樣介面 ──────────────────────────────────────
    def _seg_and_local_t(self, t: float):
        t = float(np.clip(t, 0, self.total_time))
        cum = 0.0
        for i, Ti in enumerate(self.times):
            if t <= cum + Ti or i == self.n - 1:
                return i, t - cum
            cum += Ti
        return self.n - 1, self.times[-1]

    def sample(self, t: float):
        """
        取樣軌跡

        Returns
        -------
        pos : np.ndarray (3,)
        vel : np.ndarray (3,)
        acc : np.ndarray (3,)
        """
        i, lt = self._seg_and_local_t(t)
        pos = np.array([self.segs[i][ax].eval(lt, 0) for ax in 'xyz'])
        vel = np.array([self.segs[i][ax].eval(lt, 1) for ax in 'xyz'])
        acc = np.array([self.segs[i][ax].eval(lt, 2) for ax in 'xyz'])
        return pos, vel, acc

    @property
    def total_time(self) -> float:
        return sum(self.times)

    def get_trajectory(self, dt: float = 0.05) -> dict:
        """取樣完整軌跡，回傳 dict"""
        ts = np.arange(0, self.total_time + dt, dt)
        P, V, A = [], [], []
        for t in ts:
            p, v, a = self.sample(t)
            P.append(p); V.append(v); A.append(a)
        return {'time': ts,
                'position': np.array(P),
                'velocity': np.array(V),
                'acceleration': np.array(A)}

    def check_feasibility(self) -> dict:
        """檢查動態可行性"""
        traj = self.get_trajectory(dt=0.02)
        vn = np.linalg.norm(traj['velocity'], axis=1)
        an = np.linalg.norm(traj['acceleration'], axis=1)
        return {
            'feasible': bool(np.max(vn) <= self.v_max * 1.05 and
                             np.max(an) <= self.a_max * 1.05),
            'max_velocity': float(np.max(vn)),
            'max_acceleration': float(np.max(an)),
            'v_max': self.v_max,
            'a_max': self.a_max,
            'total_time': self.total_time,
            'n_segments': self.n,
        }


# ─────────────────────────────────────────────────────
# 本地獨立測試
# ─────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 55)
    print("  Minimum Snap Trajectory - 本地獨立測試")
    print(f"  CVXPY 可用: {CVXPY_AVAILABLE}")
    print("=" * 55)

    # 障礙物情境：原路線被阻擋，需繞行
    waypoints = [
        (0.0, 0.0, 1.0),
        (1.5, 0.8, 1.2),
        (2.5, 1.5, 1.3),   # 繞避障礙物
        (3.5, 0.8, 1.2),
        (5.0, 0.0, 1.0),
    ]

    print("\n[1] 建立軌跡...")
    traj = MinSnapTrajectory(waypoints, v_max=2.0, a_max=2.0)
    print(f"    段數: {traj.n}")
    print(f"    時間: {[f'{t:.2f}s' for t in traj.times]}")
    print(f"    總時: {traj.total_time:.2f}s")

    print("\n[2] 動態可行性...")
    r = traj.check_feasibility()
    print(f"    可行: {r['feasible']}")
    print(f"    最大速度: {r['max_velocity']:.3f} / {r['v_max']} m/s")
    print(f"    最大加速: {r['max_acceleration']:.3f} / {r['a_max']} m/s²")

    print("\n[3] 軌跡取樣 (每 1s):")
    print(f"    {'t':>5} | {'x':>7} {'y':>7} {'z':>7} | {'|v|':>7}")
    print("    " + "-" * 42)
    for t in np.arange(0, traj.total_time + 1, 1.0):
        p, v, a = traj.sample(t)
        print(f"    {t:5.1f} | {p[0]:7.3f} {p[1]:7.3f} {p[2]:7.3f} | "
              f"{np.linalg.norm(v):7.3f}")

    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        data = traj.get_trajectory(dt=0.05)
        pos = data['position']
        wp = np.array(waypoints)

        fig = plt.figure(figsize=(13, 5))
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.plot(pos[:, 0], pos[:, 1], pos[:, 2], 'b-', lw=2, label='Minimum Snap 軌跡')
        ax1.scatter(wp[:, 0], wp[:, 1], wp[:, 2], c='red', s=80, label='Waypoints')
        ax1.scatter([2.5], [0.0], [1.0], c='orange', s=200, marker='X', label='障礙物')
        ax1.set_title('Minimum Snap 避障軌跡')
        ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
        ax1.legend(fontsize=8)

        ax2 = fig.add_subplot(122)
        vn = np.linalg.norm(data['velocity'], axis=1)
        an = np.linalg.norm(data['acceleration'], axis=1)
        ax2.plot(data['time'], vn, 'b-', label='速度 |v|')
        ax2.plot(data['time'], an, 'r-', label='加速度 |a|')
        ax2.axhline(traj.v_max, color='b', ls='--', alpha=0.5)
        ax2.axhline(traj.a_max, color='r', ls='--', alpha=0.5)
        ax2.set_title('動態特性'); ax2.set_xlabel('t (s)')
        ax2.legend(); ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig('min_snap_result.png', dpi=150, bbox_inches='tight')
        print("\n✅ 圖表已儲存: min_snap_result.png")
        plt.show()
    except ImportError:
        print("\n[提示] pip install matplotlib 可顯示圖表")

    print("\n✅ 本地測試完成！下一步：整合到 obstacle_avoidance_planner.py")
