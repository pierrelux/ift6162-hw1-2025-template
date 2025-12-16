"""
Trajectory Optimization for Supermarket Refrigeration Control

Multiple shooting approach with SLSQP:
- States and controls are decision variables
- Hard path constraints on temperatures and pressure
- JAX automatic differentiation for fast gradients
- Receding horizon: optimize 3-minute windows, apply controls, repeat

Performance: ~5 minutes for full 4-hour optimization
"""

import os
os.environ['JAX_ENABLE_X64'] = '1'

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, grad
from scipy.optimize import minimize, Bounds, NonlinearConstraint
import time
from typing import Tuple
import matplotlib.pyplot as plt

import supermarket as sm
from supermarket import (
    make_dynamics_step, make_forward_simulate, power_factor_jax,
    density_suction, d_density_dP, calculate_performance
)


class TrajectoryOptimizer:
    """Multiple shooting trajectory optimizer with SLSQP and JAX gradients"""
    
    def __init__(self, n_cases: int, comp_capacities: list, V_sl: float, 
                 horizon: int, dt: float = 10.0):
        self.n_cases = n_cases
        self.horizon = horizon
        self.dt = dt
        self.n_u = n_cases + 1
        self.n_x = 4 * n_cases + 1
        
        self.params = {
            'n_cases': n_cases, 'dt': dt,
            'M_goods': 200.0, 'Cp_goods': 1000.0, 'UA_goods_air': 300.0,
            'M_wall': 260.0, 'Cp_wall': 385.0, 'UA_air_wall': 500.0,
            'M_air': 50.0, 'Cp_air': 1000.0,
            'UA_wall_ref_max': 4000.0, 'M_ref_max': 1.0, 'tau_fill': 40.0,
            'V_suc': 5.0, 'eta_vol': 0.81, 'V_sl': V_sl,
            'w_con': 10000.0, 'w_pow': 0.001, 'w_switch': 10.0,
        }
        
        self.dynamics_step = make_dynamics_step(self.params)
        self.forward_simulate = make_forward_simulate(self.params)
        self._setup_jax_functions()
    
    def _setup_jax_functions(self):
        """JIT-compiled objective and constraints"""
        dynamics_step = self.dynamics_step
        n_x, n_u, horizon = self.n_x, self.n_u, self.horizon
        n_cases = self.n_cases
        V_sl = self.params['V_sl']
        eta_vol = self.params['eta_vol']
        
        self.x0_jax = None
        self.d_traj_jax = None
        
        @jit
        def objective(z):
            n_u_total = horizon * n_u
            u_traj = z[:n_u_total].reshape(horizon, n_u)
            x_inner = z[n_u_total:].reshape(horizon, n_x)
            x_full = jnp.concatenate([self.x0_jax[None, :], x_inner], axis=0)
            
            
            # TODO: Implement the objective function for multiple shooting optimization
            # 
            # Your objective should minimize:
            # 1. Power consumption (primary objective)
            # 2. Control switching (secondary objective)
            #
            # Available variables:
            # - u_traj: control trajectory [horizon, n_u]
            #   - u_traj[:, :n_cases] are valve commands [0,1]
            #   - u_traj[:, n_cases] is compressor percentage [0,100]
            # - x_full: state trajectory [horizon+1, n_x] (includes initial state)
            #   - x_full[:, -1] is P_suc (suction pressure)
            #
            # Hints:
            # - Compute power using Eq. 15 from paper.md:
            #   power = V_comp * power_factor_jax(P_suc)
            #   where V_comp = eta_vol * V_sl * (comp_percentage / 100.0)
            # - Switching cost = compressor switches + valve switches / 100
            #   Use jnp.diff() and jnp.abs() to count switches
            # - Return: power_cost / 1000.0 + 0.01 * switch_cost
            #   (power in kW, switching penalty weighted at 0.01)
            #
            # See question.md for detailed mathematical formulation.
            raise NotImplementedError("Implement objective function")
        
        @jit
        def dynamics_defects(z):
            n_u_total = horizon * n_u
            u_traj = z[:n_u_total].reshape(horizon, n_u)
            x_inner = z[n_u_total:].reshape(horizon, n_x)
            x_full = jnp.concatenate([self.x0_jax[None, :], x_inner], axis=0)
            
            
            # TODO: Implement dynamics defects for multiple shooting
            #
            # In multiple shooting, both states AND controls are decision variables.
            # The dynamics must be enforced as equality constraints.
            #
            # For each time step k = 0, ..., horizon-1:
            #   1. Propagate dynamics: x_predicted = dynamics_step(x[k], u[k], d[k])
            #   2. Compute defect: defect[k] = x_predicted - x[k+1]
            #
            # The optimizer will drive these defects to zero, ensuring x[k+1] = f(x[k], u[k], d[k])
            #
            # Available:
            # - x_full: state trajectory [horizon+1, n_x]
            # - u_traj: control trajectory [horizon, n_u]
            # - dynamics_step(x, u, d): one-step dynamics function
            # - self.d_traj_jax: disturbance trajectory [horizon, n+1]
            #
            # Return: flat vector of all defects, shape [horizon * n_x]
            #
            # This is the KEY difference between single and multiple shooting!
            raise NotImplementedError("Implement dynamics defects")
        
        self.objective_jax = objective
        self.objective_grad_jax = jit(grad(objective))
        self.dynamics_defects_jax = dynamics_defects
        self.dynamics_jacobian_jax = jit(jax.jacfwd(dynamics_defects))
    
    def optimize(self, x0: np.ndarray, d_traj: np.ndarray, P_ref: float, 
                max_iter: int = 20, verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve multiple shooting optimization problem using SLSQP
        
        Args:
            x0: Initial state [4n+1]
            d_traj: Disturbance trajectory [horizon, n+1]
            max_iter: Maximum SLSQP iterations
            verbose: Print optimization details
            
        Returns:
            u_optimal: Optimal control sequence [horizon, n+1]
            x_optimal: Optimal state trajectory [horizon+1, 4n+1]
        """
        
        self.x0_jax = jnp.array(x0)
        self.d_traj_jax = jnp.array(d_traj)
        
        # Initialize from PID baseline
        u_init = self._get_pid_warmstart(x0, d_traj, P_ref)
        x_init = self.forward_simulate(self.x0_jax, u_init, self.d_traj_jax)[1:]
        z0 = np.concatenate([np.array(u_init).flatten(), np.array(x_init).flatten()]).astype(np.float64)
        
        # Define bounds
        n_u_total = self.horizon * self.n_u
        
        # Control bounds: valves ∈ [0,1], compressor ∈ [0,100]
        lb_u = np.zeros(n_u_total)
        ub_u = np.ones(n_u_total)
        for k in range(self.horizon):
            ub_u[k * self.n_u + self.n_cases] = 100.0
        
        
        # TODO: Set up state bounds for the optimization
        #
        # State vector structure (for each k): [T_goods, T_wall, T_air, M_ref, P_suc]
        # - T_goods, T_wall: no explicit bounds (use -inf, inf)
        # - T_air: MUST be in [2, 5]°C (food safety constraint)
        # - M_ref: Physical bound [0, 1] kg (refrigerant mass)
        # - P_suc: MUST be in [0.8, 1.7] bar (system constraint)
        #
        # You need to set bounds for ALL time steps k = 0, ..., horizon-1
        #
        # Indexing: For state j at time k: idx = k * self.n_x + j
        # State order: [T_goods_0, T_goods_1, T_wall_0, T_wall_1, T_air_0, T_air_1, M_ref_0, M_ref_1, P_suc]
        #
        # Start with all bounds as -inf/inf:
        lb_x = np.full(self.horizon * self.n_x, -np.inf)
        ub_x = np.full(self.horizon * self.n_x, np.inf)
        
        # Then set specific bounds for T_air, P_suc, and M_ref
        # (See question.md for constraint specifications)
        raise NotImplementedError("Set up state bounds")
        
        bounds = Bounds(
            lb=np.concatenate([lb_u, lb_x]).astype(np.float64),
            ub=np.concatenate([ub_u, ub_x]).astype(np.float64)
        )
        
        # Objective with JAX gradients
        def objective_np(z):
            return np.float64(self.objective_jax(jnp.array(z, dtype=jnp.float64)))
        
        def objective_grad_np(z):
            return np.array(self.objective_grad_jax(jnp.array(z, dtype=jnp.float64)), dtype=np.float64)
        
        
        # TODO: Set up and solve the constrained optimization problem with SLSQP
        #
        # You need to create a NonlinearConstraint for the dynamics:
        # - fun: function that computes dynamics defects
        # - lb, ub: bounds on defects (should be near zero)
        # - jac: Jacobian of defects (for faster convergence)
        #
        # Then call scipy.optimize.minimize with:
        # - fun: objective function (objective_np)
        # - x0: initial guess (z0)
        # - method: 'SLSQP' (Sequential Least Squares Programming)
        # - jac: objective gradient (objective_grad_np)
        # - bounds: box constraints on decision variables
        # - constraints: list of NonlinearConstraint objects
        # - options: {'maxiter': max_iter, 'ftol': 1e-6}
        #
        # Note: SLSQP handles three types of constraints:
        # 1. Box constraints (bounds= argument): simple lb ≤ z ≤ ub
        # 2. Equality constraints (NonlinearConstraint with lb=ub): g(z) = 0
        # 3. Inequality constraints (NonlinearConstraint): lb ≤ g(z) ≤ ub
        #
        # For dynamics, we want g(z) ≈ 0 (slight tolerance for numerical stability)
        #
        # Hint: Wrap your JAX functions to convert between numpy and jax arrays
        raise NotImplementedError("Set up SLSQP optimization")
        
        # Extract solution
        u_optimal = result.x[:n_u_total].reshape(self.horizon, self.n_u)
        x_optimal_inner = result.x[n_u_total:].reshape(self.horizon, self.n_x)
        x_optimal = np.vstack([x0, x_optimal_inner])
        
        # Round valves
        u_optimal[:, :self.n_cases] = np.round(u_optimal[:, :self.n_cases])
        
        return u_optimal, x_optimal
    
    def _get_pid_warmstart(self, x0: np.ndarray, d_traj: np.ndarray, P_ref: float) -> jnp.ndarray:
        """PID baseline for warm start"""
        system = sm.RefrigerationSystem(self.n_cases, [50.0, 50.0], 0.08, False)
        
        # Extract state components from flat vector [4n+1]
        # State order: [T_goods, T_wall, T_air, M_ref, P_suc]
        n = self.n_cases
        T_goods = x0[0:n]
        T_wall = x0[n:2*n]
        T_air = x0[2*n:3*n]
        M_ref = x0[3*n:4*n]
        P_suc = x0[4*n]
        
        # Set initial state
        for i in range(self.n_cases):
            system.cases[i].state = np.array([T_goods[i], T_wall[i], T_air[i], M_ref[i]])
        system.P_suc = P_suc
        
        u_init = np.zeros((self.horizon, self.n_u))
        for t in range(self.horizon):
            system.Q_airload = d_traj[t, 0]
            system.m_ref_const = d_traj[t, -1]
            system.P_ref = P_ref

            valves, comp_on, _, _ = system.simulate_step(self.dt, t * self.dt)
            u_init[t, :self.n_cases] = valves
            u_init[t, self.n_cases] = sum(comp_on)
        
        return jnp.array(u_init)


def optimize_full_trajectory(scenario='2d-2c', duration=14400, window_size=180,
                             dt=10.0, max_iter=50):
    """
    Receding horizon optimization over full duration
    
    Args:
        scenario: '2d-2c' (2 display cases, 2 compressors)
        duration: Total simulation time [s]
        window_size: Planning horizon per window [s] (180s = 3 min)
        dt: Timestep for Euler integration [s]
        max_iter: Maximum SLSQP iterations per window
    """
    
    # Setup
    n_cases = 2
    system = sm.RefrigerationSystem(n_cases, [50.0, 50.0], 0.08, False)
    system.cases[0].state = np.array([2.0, 0.0, 5.1, 0.0])
    system.cases[1].state = np.array([2.0, 0.0, 0.0, 1.0])
    system.P_suc = 1.40
    system.set_day_mode()
    
    def get_state_vector(system, n_cases):
        """Extract state as flat vector [4n+1]"""
        state_vec = []
        for i in range(n_cases):
            state_vec.append(system.cases[i].state)  # [T_goods, T_wall, T_air, M_ref]
        state_vec = list(np.array(state_vec).transpose().flatten())
        state_vec.append(system.P_suc)
        return np.array(state_vec)
    
    def get_disturbance(system, n_cases):
        """Get current disturbances [Q_airload per case + m_ref_const]"""
        return np.array([system.Q_airload] * n_cases + [system.m_ref_const])
    
    horizon_steps = int(window_size / dt)
    n_windows = int(duration / window_size)
    
    print("="*70)
    print("TRAJECTORY OPTIMIZATION")
    print("="*70)
    print(f"Duration: {duration/3600:.1f} hours")
    print(f"Window: {window_size/60:.1f} min ({horizon_steps} steps)")
    print(f"Windows: {n_windows}")
    print(f"Max iterations: {max_iter}")
    print(f"Estimated time: {n_windows * 5:.0f}-{n_windows * 10:.0f}s = {n_windows*5/60:.1f}-{n_windows*10/60:.1f} min")
    print("="*70)
    
    optimizer = TrajectoryOptimizer(n_cases, [50.0, 50.0], 0.08, 
                                          horizon=horizon_steps, dt=dt)
    
    # Storage
    time_opt, T_air_opt, P_suc_opt, power_opt, u_opt_hist = [], [], [], [], []
    
    # Optimize
    t_start = time.time()
    
    for window_idx in range(n_windows):
        t_window = window_idx * window_size
        
        if window_idx % 10 == 0:
            elapsed = time.time() - t_start
            eta = (elapsed / (window_idx + 1)) * (n_windows - window_idx - 1) if window_idx > 0 else 0
            print(f"Window {window_idx+1}/{n_windows} | Elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s", flush=True)
        
        if t_window >= 7200:
            system.set_night_mode()
        
        x0 = get_state_vector(system, n_cases)
        d_traj = np.tile(get_disturbance(system, n_cases), (horizon_steps, 1))
        
        u_window, x_window = optimizer.optimize(x0, d_traj, P_ref=system.P_ref, max_iter=max_iter, verbose=False)
        
        # Apply the ENTIRE optimized window (not just first step!)
        for step_in_window in range(horizon_steps):
            t_current = t_window + step_in_window * dt
            
            # Apply optimized control DIRECTLY (bypass PID controller)
            u_apply = u_window[step_in_window]
            
            # Set valve states
            for i in range(n_cases):
                system.cases[i].valve = int(u_apply[i] > 0.5)
            
            # Set compressor state (map continuous to discrete for 2d-2c)
            comp_total = u_apply[n_cases]  # [0-100]
            if comp_total < 25:
                system.current_comp_on = [0.0, 0.0]
            elif comp_total < 75:
                system.current_comp_on = [50.0, 0.0]
            else:
                system.current_comp_on = [50.0, 50.0]
            
            # Now simulate with our controls (not PID!)
            # We need to manually step the system, bypassing the controller
            valves = [case.valve for case in system.cases]
            
            # Update display cases
            m_in_suc = 0.0
            for case in system.cases:
                Q_load = system.Q_airload
                new_state, Q_e = case.dynamics(case.state, system.P_suc, case.valve, Q_load, dt)
                case.state = new_state
                m_in_suc += case.mass_flow_out(Q_e, system.P_suc)
            
            # Update suction pressure
            comp_on = system.current_comp_on
            V_comp = system.volume_flow(comp_on)
            rho = density_suction(system.P_suc)
            drho_dP = max(1e-3, d_density_dP(system.P_suc))
            dP = (m_in_suc + system.m_ref_const - V_comp * rho) / (system.V_suc * drho_dP)
            system.P_suc += dP * dt
            system.P_suc = np.clip(system.P_suc, 0.8, 3.0)
            
            # Calculate power
            power = system.power_consumption(system.P_suc, comp_on)
            comp_switches = 0  # We're not tracking switches within the window
            
            time_opt.append(t_current)
            T_air_opt.append([case.state[2] for case in system.cases])
            P_suc_opt.append(system.P_suc)
            power_opt.append(power)
            u_opt_hist.append(u_apply)
    
    t_total = time.time() - t_start
    
    # Convert
    time_opt = np.array(time_opt)
    T_air_opt = np.array(T_air_opt)
    P_suc_opt = np.array(P_suc_opt)
    power_opt = np.array(power_opt)
    u_opt_hist = np.array(u_opt_hist)
    
    # Metrics (split u_opt_hist into valves and compressor)
    valve_states = u_opt_hist[:, :n_cases]
    comp_switches = np.abs(np.diff(jnp.round(u_opt_hist[:, n_cases] / system.comp_capacities[0])))  # All compressor switches
    gamma_con_opt, gamma_switch_opt, gamma_pow_opt = sm.calculate_performance(
        time_opt, T_air_opt, P_suc_opt, comp_switches, power_opt, valve_states, P_ref=1.7
    )
    
    print(f"\n{'='*70}")
    print(f"COMPLETE in {t_total/60:.2f} minutes ({t_total/n_windows:.1f}s/window)")
    print(f"{'='*70}")
    print(f"OPTIMIZED: γ_con={gamma_con_opt:.3f}, γ_pow={gamma_pow_opt/1000:.3f} kW, γ_switch={gamma_switch_opt:.6f}")
    
    # Baseline
    print(f"\nComparing with baseline...")
    time_b, T_air_b, P_suc_b, _, _, power_b, valve_states_b, comp_switches_b, _, _ = sm.run_scenario(
        scenario, duration=duration, dt=dt, seed=42
    )
    gamma_con_b, gamma_switch_b, gamma_pow_b = sm.calculate_performance(
        time_b, T_air_b, P_suc_b, comp_switches_b, power_b, valve_states_b, P_ref=1.7
    )
    
    print(f"BASELINE:  γ_con={gamma_con_b:.3f}, γ_pow={gamma_pow_b/1000:.3f} kW, γ_switch={gamma_switch_b:.6f}")
    print(f"IMPROVEMENT: {(gamma_con_b-gamma_con_opt)/gamma_con_b*100:+.1f}% con, "
          f"{(gamma_pow_b-gamma_pow_opt)/gamma_pow_b*100:+.1f}% pow, "
          f"{(gamma_switch_b-gamma_switch_opt)/gamma_switch_b*100:+.1f}% switch")
    
    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    
    for i in range(n_cases):
        axes[0].plot(time_b/3600, T_air_b[:, i], '--', alpha=0.7, label=f'Baseline {i+1}')
        axes[0].plot(time_opt/3600, T_air_opt[:, i], '-', linewidth=2, label=f'Optimized {i+1}')
    axes[0].axhline(5.0, color='r', linestyle=':', label='T_max')
    axes[0].axhline(2.0, color='b', linestyle=':', label='T_min')
    axes[0].set_ylabel('T_air [°C]')
    axes[0].legend(loc='upper right', ncol=2, fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(f'Trajectory Optimization: γ_con={gamma_con_opt:.2f} vs baseline={gamma_con_b:.2f}')
    
    axes[1].plot(time_b/3600, P_suc_b, '--', alpha=0.7, label='Baseline')
    axes[1].plot(time_opt/3600, P_suc_opt, '-', linewidth=2, label='Optimized')
    axes[1].axhline(1.7, color='r', linestyle=':', label='P_max')
    axes[1].set_ylabel('P_suc [bar]')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(time_b/3600, power_b, '--', alpha=0.7, label='Baseline')
    axes[2].plot(time_opt/3600, power_opt, '-', linewidth=2, label='Optimized')
    axes[2].set_xlabel('Time [hours]')
    axes[2].set_ylabel('Power [kW]')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('optimization_results.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: optimization_results.png")
    
    return time_opt, T_air_opt, P_suc_opt, power_opt


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Trajectory Optimization for Supermarket Refrigeration')
    parser.add_argument('--full', action='store_true', 
                        help='Run full 4-hour optimization (default: 30-min quick test)')
    parser.add_argument('--duration', type=int, default=None,
                        help='Custom duration in seconds')
    parser.add_argument('--max-iter', type=int, default=50,
                        help='Maximum SLSQP iterations per window (default: 50)')
    
    args = parser.parse_args()
    
    if args.duration is not None:
        duration = args.duration
    elif args.full:
        duration = 14400  # 4 hours
    else:
        duration = 1800   # 30 minutes (quick test)
    
    optimize_full_trajectory(duration=duration, window_size=180, max_iter=args.max_iter)
    
    # For faster testing during development, use:
    # optimize_full_trajectory(duration=1800, window_size=180, max_iter=50)  # 30 min test

