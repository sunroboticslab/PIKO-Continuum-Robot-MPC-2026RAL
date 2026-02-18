import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')

# JAX imports - Force CPU usage
import os
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
import jax
import jax.numpy as jnp
from jax import jit, grad, vmap
from jax.lax import scan
from functools import partial

# Import core JAX functions from ParamIdeV1_JAX
from ParamIdeV1_JAX import (
    hat, compute_muscle_contributions, cosserat_muscle_ode, 
    RK4_step, integrate_shape_scan, compute_single_muscle_length,
    shooting_residual_fused
)

def generate_ramp_hold(taus, dt, pressure_max, hold_time):
    """
    Generate ramp-hold pressure sequence
    
    Args:
        taus: Number of transitions
        dt: Time step
        pressure_max: Maximum pressure value
        hold_time: Hold time at each target
        
    Returns:
        np.array: Pressure sequence with shape (n_steps, 3)
    """
    ramp_times = 0.5 + np.random.uniform(0.0, 6.0, size=(taus + 1,))
    seq = []
    prev = np.zeros(3)
    targets = pressure_max * np.random.rand(taus + 2, 3)
    targets[0] = prev
    # Always set the final target to zero pressure
    targets[-1] = np.zeros(3)
    
    for i in range(taus + 1):
        steps_r = int(ramp_times[i]/dt)
        for k in range(steps_r):
            frac = (k+1)/steps_r
            seq.append(prev + (targets[i+1] - prev)*frac)
        steps_h = int(hold_time/dt)
        for _ in range(steps_h):
            seq.append(targets[i+1])
        prev = targets[i+1]
    
    return np.vstack(seq)

# --- Manual Parameters Section ---
manual_params = {
    'E': 1.187504e+10,         # Young's modulus [Pa]
    'rho': 8.236995e+05,      # Density [kg/m^3]
    'L': 1.993596e-01,         # Length [m]
    'ux': 1.640484e+00,         # Reference strain x
    'uy': -2.121691e+00,        # Reference strain y
    'k1': 8.026306e+01, 'k2': 7.011259e+01, 'k3': 7.035667e+01,  # Muscle spring constants
    'c1': 1.183499e-02, 'c2': 7.855970e-03, 'c3': 7.642034e-03,     # Pressure coefficients
    'b1': 3.003308e-04, 'b2': 3.102560e-04, 'b3': 2.989365e-04,  # Quadratic pressure coefficients
    'l01': 9.843773e-02, 'l02': 8.846369e-02, 'l03': 8.772880e-02,  # Muscle original lengths
    'mz0': -7.236826e-04,         # Tip moment offset
    'A': 1.732520e-04           # Tip moment pressure coefficient
}

# --- JAX-Optimized Simulation Functions ---
def solve_single_point_jax(rod_params, muscle_params, target_pressures, W_dist, f_t=None, l_t=None, target_xi_star=None, n_steps=20):
    """
    JAX-optimized single point simulation with proper pressure and xi_star ramping
    """
    E, r, G, n, L, rho = rod_params
    
    # Initialize rod properties
    A = jnp.pi * r ** 2
    Ix = jnp.pi * r ** 4 / 4
    J = 2 * Ix
    Kbt = jnp.diag(jnp.array([E*Ix, E*Ix, G*J]))
    Kse = jnp.diag(jnp.array([G*A, G*A, E*A]))
    ds = L/(n-1)
    
    # Extract muscle parameters
    muscles_r_i = jnp.array([jnp.array([m['r_i'] * jnp.cos(m['angle_i']), m['r_i'] * jnp.sin(m['angle_i']), 0]) for m in muscle_params])
    muscles_params_array = jnp.array([[m['k'], m['l0'], m['c'], m['b']] for m in muscle_params])
    
    # Initialize shape
    p_init = jnp.linspace(jnp.array([0., 0., 0.]), jnp.array([0., 0., L]), n)
    R_init = jnp.tile(jnp.eye(3).reshape(1, 9), (n, 1))
    initial_xi_star = jnp.tile(jnp.array([0., 0., 0., 0., 0., 1.]), (n, 1))  # Straight configuration
    u_init = jnp.tile(initial_xi_star[0, :3], (n, 1))
    v_init = jnp.tile(initial_xi_star[0, 3:], (n, 1))
    X_init = jnp.hstack([p_init, R_init, u_init, v_init])
    
    # Set target xi_star (if not provided, use initial)
    if target_xi_star is None:
        target_xi_star = initial_xi_star.copy()
    
    # Initial guesses
    tau_guess = jnp.array([5.0 for _ in range(len(muscles_r_i))])
    xi0_guess = jnp.array([0., 0., 0., 0., 0., 1.])
    
    # Create pressure and xi_star steps (use provided n_steps parameter)
    pressure_steps = jnp.linspace(jnp.zeros(len(muscles_r_i)), jnp.array(target_pressures), n_steps)
    xi_star_steps = jnp.linspace(initial_xi_star, target_xi_star, n_steps)
    
    # Pre-compile the shooting function
    @jit
    def shooting_wrapper(vars, pressures, xi_star_current):
        return shooting_residual_fused(vars, muscles_r_i, pressures, 
                                     muscles_params_array, W_dist, xi_star_current, 
                                     Kse, Kbt, X_init, ds, f_t, l_t)
    
    # Python loop for pressure and xi_star ramping
    for step in range(n_steps):
        current_pressures = pressure_steps[step]
        current_xi_star = xi_star_steps[step]
        
        # Build initial guess vector
        vars0 = jnp.hstack([xi0_guess, tau_guess])
        
        # Use scipy fsolve with pre-compiled JAX function
        from scipy.optimize import fsolve
        
        def objective_np(vars_np):
            vars_jax = jnp.array(vars_np)
            result_jax = shooting_wrapper(vars_jax, current_pressures, current_xi_star)
            return np.array(result_jax)
        
        sol = fsolve(objective_np, np.array(vars0), maxfev=1000)
        
        xi0_guess = jnp.array(sol[:6])
        tau_guess = jnp.array(sol[6:])
    
    # Compute final shape with target xi_star
    x0 = X_init[0].copy()
    x0 = x0.at[12:18].set(xi0_guess)
    X_final = integrate_shape_scan(x0, muscles_r_i, tau_guess, W_dist, ds, target_xi_star, Kse, Kbt, f_t, l_t)
    
    return X_final, tau_guess

def solve_single_point_jax_with_initial_guess(rod_params, muscle_params, target_pressures, W_dist, initial_xi0, initial_taus, f_t=None, l_t=None, target_xi_star=None, n_steps=20):
    """
    JAX-optimized single point simulation with provided initial guesses and target xi_star for faster convergence
    """
    E, r, G, n, L, rho = rod_params
    
    # Initialize rod properties
    A = jnp.pi * r ** 2
    Ix = jnp.pi * r ** 4 / 4
    J = 2 * Ix
    Kbt = jnp.diag(jnp.array([E*Ix, E*Ix, G*J]))
    Kse = jnp.diag(jnp.array([G*A, G*A, E*A]))
    ds = L/(n-1)
    
    # Extract muscle parameters
    muscles_r_i = jnp.array([jnp.array([m['r_i'] * jnp.cos(m['angle_i']), m['r_i'] * jnp.sin(m['angle_i']), 0]) for m in muscle_params])
    muscles_params_array = jnp.array([[m['k'], m['l0'], m['c'], m['b']] for m in muscle_params])
    
    # Initialize shape
    p_init = jnp.linspace(jnp.array([0., 0., 0.]), jnp.array([0., 0., L]), n)
    R_init = jnp.tile(jnp.eye(3).reshape(1, 9), (n, 1))
    initial_xi_star = jnp.tile(jnp.array([0., 0., 0., 0., 0., 1.]), (n, 1))  # Straight configuration
    u_init = jnp.tile(initial_xi_star[0, :3], (n, 1))
    v_init = jnp.tile(initial_xi_star[0, 3:], (n, 1))
    X_init = jnp.hstack([p_init, R_init, u_init, v_init])
    
    # Set target xi_star (if not provided, use initial)
    if target_xi_star is None:
        target_xi_star = initial_xi_star.copy()
    
    # Use provided initial guesses
    tau_guess = initial_taus.copy()
    xi0_guess = initial_xi0.copy()
    
    # Create pressure and xi_star steps (use provided n_steps parameter)
    n_steps = 100  # Set to 100 as requested
    pressure_steps = jnp.linspace(jnp.zeros(len(muscles_r_i)), jnp.array(target_pressures), n_steps)
    xi_star_steps = jnp.linspace(initial_xi_star, target_xi_star, n_steps)
    
    # Pre-compile the shooting function
    @jit
    def shooting_wrapper(vars, pressures, xi_star_current):
        return shooting_residual_fused(vars, muscles_r_i, pressures, 
                                     muscles_params_array, W_dist, xi_star_current, 
                                     Kse, Kbt, X_init, ds, f_t, l_t)
    
    # Python loop for pressure and xi_star ramping
    for step in range(n_steps):
        current_pressures = pressure_steps[step]
        current_xi_star = xi_star_steps[step]
        
        # Build initial guess vector
        vars0 = jnp.hstack([xi0_guess, tau_guess])
        
        # Use scipy fsolve with pre-compiled JAX function (fewer iterations since we have good guesses)
        from scipy.optimize import fsolve
        
        def objective_np(vars_np):
            vars_jax = jnp.array(vars_np)
            result_jax = shooting_wrapper(vars_jax, current_pressures, current_xi_star)
            return np.array(result_jax)
        
        # Use fewer iterations since we have good initial guesses
        sol = fsolve(objective_np, np.array(vars0), maxfev=500)  # Reduced from 1000
        
        xi0_guess = jnp.array(sol[:6])
        tau_guess = jnp.array(sol[6:])
    
    # Compute final shape with target xi_star
    x0 = X_init[0].copy()
    x0 = x0.at[12:18].set(xi0_guess)
    X_final = integrate_shape_scan(x0, muscles_r_i, tau_guess, W_dist, ds, target_xi_star, Kse, Kbt, f_t, l_t)
    
    return X_final, tau_guess

def simulate_rod_tip_jax_optimized_with_initial_guess(E, rho, L, xi_star, muscle_params, tip_world_wrench, target_pressures, n=51, r=0.001, G=None, g=9.81, f_t=None, l_t=None, initial_xi0=None, initial_taus=None, target_xi_star=None, n_steps=20):
    """
    JAX-optimized rod tip simulation with initial guesses and target xi_star for faster convergence
    """
    if G is None:
        G = E/(2*1.3)
    
    rod_params = (E, r, G, n, L, rho)
    
    # Set up distributed gravity
    W_dist = jnp.zeros((n, 6))
    mass_per_length = jnp.pi * r**2 * rho
    f_gravity = mass_per_length * L * g / n
    W_dist = W_dist.at[:, 5].set(f_gravity)
    W_dist = W_dist.at[-1].set(tip_world_wrench)
    
    # Run simulation with initial guesses and target xi_star
    X_final, tau_final = solve_single_point_jax_with_initial_guess(
        rod_params, muscle_params, target_pressures, W_dist, 
        initial_xi0, initial_taus, f_t, l_t, target_xi_star, n_steps
    )
    
    tip_x, tip_y = X_final[-1, 0], X_final[-1, 1]
    
    return tip_x, tip_y, X_final, tau_final

def simulate_rod_tip_jax_optimized_with_shape_continuity(E, rho, L, xi_star, muscle_params, tip_world_wrench, target_pressures, n=51, r=0.001, G=None, g=9.81, f_t=None, l_t=None, initial_xi0=None, initial_taus=None, previous_X_final=None, target_xi_star=None):
    """
    JAX-optimized rod tip simulation with shape continuity for smoother trajectories
    
    Args:
        previous_X_final: Previous simulation's final shape configuration for continuity
        target_xi_star: Target xi_star for ramping
    """
    if G is None:
        G = E/(2*1.3)
    
    rod_params = (E, r, G, n, L, rho)
    
    # Set up distributed gravity
    W_dist = jnp.zeros((n, 6))
    mass_per_length = jnp.pi * r**2 * rho
    f_gravity = mass_per_length * L * g / n
    W_dist = W_dist.at[:, 5].set(f_gravity)
    W_dist = W_dist.at[-1].set(tip_world_wrench)
    
    # If we have a previous shape, use it to improve initial guesses
    if previous_X_final is not None:
        # Extract strains from previous shape for better initial guess
        previous_strains = previous_X_final[0, 12:18]  # Extract strains from first node
        if initial_xi0 is None:
            initial_xi0 = previous_strains
        else:
            # Blend previous strains with default strains for stability
            blend_factor = 1.0# Use 70% of previous strains, 30% default
            initial_xi0 = blend_factor * previous_strains + (1 - blend_factor) * initial_xi0
    
    # Run simulation with improved initial guesses and target xi_star
    X_final, tau_final = solve_single_point_jax_with_initial_guess(
        rod_params, muscle_params, target_pressures, W_dist, 
        initial_xi0, initial_taus, f_t, l_t, target_xi_star
    )
    
    tip_x, tip_y = X_final[-1, 0], X_final[-1, 1]
    
    return tip_x, tip_y, X_final, tau_final

def simulate_rod_tip_jax_optimized_with_gradual_transition(E, rho, L, xi_star, muscle_params, tip_world_wrench, target_pressures, n=51, r=0.001, G=None, g=9.81, f_t=None, l_t=None, initial_xi0=None, initial_taus=None, previous_X_final=None, n_convergence_steps=10, target_xi_star=None):
    """
    JAX-optimized rod tip simulation with gradual pressure transitions and shape continuity
    
    Args:
        previous_X_final: Previous simulation's final shape configuration for continuity
        n_convergence_steps: Number of convergence steps for pressure transition
        target_xi_star: Target xi_star for ramping
    """
    if G is None:
        G = E/(2*1.3)
    
    rod_params = (E, r, G, n, L, rho)
    
    # Set up distributed gravity
    W_dist = jnp.zeros((n, 6))
    mass_per_length = jnp.pi * r**2 * rho
    f_gravity = mass_per_length * L * g / n
    W_dist = W_dist.at[:, 5].set(f_gravity)
    W_dist = W_dist.at[-1].set(tip_world_wrench)
    
    # If we have a previous shape, use it to improve initial guesses
    if previous_X_final is not None:
        # Extract strains from previous shape for better initial guess
        previous_strains = previous_X_final[0, 12:18]  # Extract strains from first node
        if initial_xi0 is None:
            initial_xi0 = previous_strains
        else:
            # Blend previous strains with default strains for stability
            blend_factor = 0.7  # Use 70% of previous strains, 30% default
            initial_xi0 = blend_factor * previous_strains + (1 - blend_factor) * initial_xi0
    
    # Run simulation with gradual convergence and target xi_star
    X_final, tau_final = solve_single_point_jax_with_gradual_convergence(
        rod_params, muscle_params, target_pressures, W_dist, 
        initial_xi0, initial_taus, f_t, l_t, n_convergence_steps, target_xi_star
    )
    
    tip_x, tip_y = X_final[-1, 0], X_final[-1, 1]
    
    return tip_x, tip_y, X_final, tau_final

def solve_single_point_jax_with_gradual_convergence(rod_params, muscle_params, target_pressures, W_dist, initial_xi0, initial_taus, f_t=None, l_t=None, n_convergence_steps=10, target_xi_star=None):
    """
    JAX-optimized single point simulation with gradual convergence for pressure transitions
    Uses multiple convergence steps to ensure stable solution with target xi_star
    """
    E, r, G, n, L, rho = rod_params
    
    # Initialize rod properties
    A = jnp.pi * r ** 2
    Ix = jnp.pi * r ** 4 / 4
    J = 2 * Ix
    Kbt = jnp.diag(jnp.array([E*Ix, E*Ix, G*J]))
    Kse = jnp.diag(jnp.array([G*A, G*A, E*A]))
    ds = L/(n-1)
    
    # Extract muscle parameters
    muscles_r_i = jnp.array([jnp.array([m['r_i'] * jnp.cos(m['angle_i']), m['r_i'] * jnp.sin(m['angle_i']), 0]) for m in muscle_params])
    muscles_params_array = jnp.array([[m['k'], m['l0'], m['c'], m['b']] for m in muscle_params])
    
    # Initialize shape
    p_init = jnp.linspace(jnp.array([0., 0., 0.]), jnp.array([0., 0., L]), n)
    R_init = jnp.tile(jnp.eye(3).reshape(1, 9), (n, 1))
    initial_xi_star = jnp.tile(jnp.array([0., 0., 0., 0., 0., 1.]), (n, 1))  # Straight configuration
    u_init = jnp.tile(initial_xi_star[0, :3], (n, 1))
    v_init = jnp.tile(initial_xi_star[0, 3:], (n, 1))
    X_init = jnp.hstack([p_init, R_init, u_init, v_init])
    
    # Set target xi_star (if not provided, use initial)
    if target_xi_star is None:
        target_xi_star = initial_xi_star.copy()
    
    # Use provided initial guesses
    tau_guess = initial_taus.copy()
    xi0_guess = initial_xi0.copy()
    
    # Create pressure and xi_star steps for gradual convergence
    n_steps = n_convergence_steps
    pressure_steps = jnp.linspace(jnp.zeros(len(muscles_r_i)), jnp.array(target_pressures), n_steps)
    xi_star_steps = jnp.linspace(initial_xi_star, target_xi_star, n_steps)
    
    # Pre-compile the shooting function
    @jit
    def shooting_wrapper(vars, pressures, xi_star_current):
        return shooting_residual_fused(vars, muscles_r_i, pressures, 
                                     muscles_params_array, W_dist, xi_star_current, 
                                     Kse, Kbt, X_init, ds, f_t, l_t)
    
    # Gradual convergence with multiple steps
    from scipy.optimize import fsolve
    
    def objective_np(vars_np):
        vars_jax = jnp.array(vars_np)
        result_jax = shooting_wrapper(vars_jax, target_pressures, target_xi_star)
        return np.array(result_jax)
    
    # Multiple convergence steps for stability with pressure and xi_star ramping
    for step in range(n_steps):
        current_pressures = pressure_steps[step]
        current_xi_star = xi_star_steps[step]
        
        # Build initial guess vector
        vars0 = jnp.hstack([xi0_guess, tau_guess])
        
        # Use scipy fsolve with pre-compiled JAX function
        def objective_np_step(vars_np):
            vars_jax = jnp.array(vars_np)
            result_jax = shooting_wrapper(vars_jax, current_pressures, current_xi_star)
            return np.array(result_jax)
        
        # Solve with current guesses
        sol = fsolve(objective_np_step, np.array(vars0), maxfev=300)  # Fewer iterations per step
        
        # Update guesses for next iteration
        xi0_guess = jnp.array(sol[:6])
        tau_guess = jnp.array(sol[6:])
    
    # Compute final shape with target xi_star
    x0 = X_init[0].copy()
    x0 = x0.at[12:18].set(xi0_guess)
    X_final = integrate_shape_scan(x0, muscles_r_i, tau_guess, W_dist, ds, target_xi_star, Kse, Kbt, f_t, l_t)
    
    return X_final, tau_guess

def simulate_rod_tip_jax_optimized_with_continuous_ramping(E, rho, L, muscle_params, tip_world_wrench, initial_pressures, target_pressures, initial_xi_star, target_xi_star, n=51, r=0.001, G=None, g=9.81, f_t=None, l_t=None, initial_xi0=None, initial_taus=None, n_steps=20):
    """
    JAX-optimized rod tip simulation with continuous pressure and xi_star ramping for trajectory continuity
    
    Args:
        initial_pressures: Starting pressure [p1, p2, p3]
        target_pressures: Target pressure [p1, p2, p3]
        initial_xi_star: Starting xi_star configuration
        target_xi_star: Target xi_star configuration
        n_steps: Number of steps for pressure and xi_star ramping
    """
    if G is None:
        G = E/(2*1.3)
    
    rod_params = (E, r, G, n, L, rho)
    
    # Set up distributed gravity
    W_dist = jnp.zeros((n, 6))
    mass_per_length = jnp.pi * r**2 * rho
    f_gravity = mass_per_length * L * g / n
    W_dist = W_dist.at[:, 5].set(f_gravity)
    W_dist = W_dist.at[-1].set(tip_world_wrench)
    
    # Run simulation with continuous ramping
    X_final, tau_final, xi0_guess_out = solve_single_point_jax_with_continuous_ramping(
        rod_params, muscle_params, initial_pressures, target_pressures, 
        initial_xi_star, target_xi_star, W_dist, 
        initial_xi0, initial_taus, f_t, l_t, n_steps
    )
    
    tip_x, tip_y = X_final[-1, 0], X_final[-1, 1]
    
    return tip_x, tip_y, X_final, tau_final, xi0_guess_out

def solve_single_point_jax_with_continuous_ramping(rod_params, muscle_params, initial_pressures, target_pressures, initial_xi_star, target_xi_star, W_dist, initial_xi0, initial_taus, f_t=None, l_t=None, n_steps=20):
    """
    JAX-optimized single point simulation with continuous pressure and xi_star ramping
    Ramps from initial conditions to target conditions using previous guesses for continuity
    """
    E, r, G, n, L, rho = rod_params
    
    # Initialize rod properties
    A = jnp.pi * r ** 2
    Ix = jnp.pi * r ** 4 / 4
    J = 2 * Ix
    Kbt = jnp.diag(jnp.array([E*Ix, E*Ix, G*J]))
    Kse = jnp.diag(jnp.array([G*A, G*A, E*A]))
    ds = L/(n-1)
    
    # Extract muscle parameters
    muscles_r_i = jnp.array([jnp.array([m['r_i'] * jnp.cos(m['angle_i']), m['r_i'] * jnp.sin(m['angle_i']), 0]) for m in muscle_params])
    muscles_params_array = jnp.array([[m['k'], m['l0'], m['c'], m['b']] for m in muscle_params])
    
    # Initialize shape
    p_init = jnp.linspace(jnp.array([0., 0., 0.]), jnp.array([0., 0., L]), n)
    R_init = jnp.tile(jnp.eye(3).reshape(1, 9), (n, 1))
    u_init = jnp.tile(initial_xi_star[0, :3], (n, 1))
    v_init = jnp.tile(initial_xi_star[0, 3:], (n, 1))
    X_init = jnp.hstack([p_init, R_init, u_init, v_init])
    
    # Use provided initial guesses (from previous simulation for continuity)
    tau_guess = initial_taus.copy()
    xi0_guess = initial_xi0.copy()
    
    # Create pressure and xi_star steps for continuous ramping
    pressure_steps = jnp.linspace(jnp.array(initial_pressures), jnp.array(target_pressures), n_steps)
    xi_star_steps = jnp.linspace(initial_xi_star, target_xi_star, n_steps)
    
    # Pre-compile the shooting function
    @jit
    def shooting_wrapper(vars, pressures, xi_star_current):
        return shooting_residual_fused(vars, muscles_r_i, pressures, 
                                     muscles_params_array, W_dist, xi_star_current, 
                                     Kse, Kbt, X_init, ds, f_t, l_t)
    
    # Python loop for continuous pressure and xi_star ramping
    for step in range(n_steps):
        current_pressures = pressure_steps[step]
        current_xi_star = xi_star_steps[step]
        
        # Build initial guess vector using previous guesses (maintains continuity)
        vars0 = jnp.hstack([xi0_guess, tau_guess])
        

        
        # Use scipy fsolve with pre-compiled JAX function
        from scipy.optimize import fsolve
        
        def objective_np(vars_np):
            vars_jax = jnp.array(vars_np)
            result_jax = shooting_wrapper(vars_jax, current_pressures, current_xi_star)
            return np.array(result_jax)
        
        # Solve with current step using previous guesses as initial values
        sol = fsolve(objective_np, np.array(vars0), maxfev=500)
        
        # Update guesses for next iteration (this maintains continuity)
        xi0_guess = jnp.array(sol[:6])
        tau_guess = jnp.array(sol[6:])
        

    
    # Compute final shape with target xi_star
    x0 = X_init[0].copy()
    x0 = x0.at[12:18].set(xi0_guess)
    X_final = integrate_shape_scan(x0, muscles_r_i, tau_guess, W_dist, ds, target_xi_star, Kse, Kbt, f_t, l_t)
    
    return X_final, tau_guess, xi0_guess

# --- Main Simulation Function for Ramp-Hold Sequences ---
def simulate_ramp_hold_sequence(params, P_seq, dt, save_to_csv=True, csv_filename='simulation_results.csv', transition_steps=30):
    """
    Simulate rod tip position for a ramp-hold pressure sequence
    Uses adaptive step counts for efficient convergence with trajectory continuity
    
    Args:
        params: Dictionary of rod and muscle parameters
        P_seq: Pressure sequence array with shape (n_steps, 3)
        dt: Time step
        save_to_csv: Whether to save results to CSV
        csv_filename: Output CSV filename
        transition_steps: Number of steps for pressure and xi_star ramping (used for all steps)
        
    Returns:
        tuple: (time_sequence, tip_positions, pressure_sequence)
    """
    print(f"Starting ramp-hold simulation with {len(P_seq)} time steps...")
    print(f"Time step: {dt}s, Total simulation time: {len(P_seq)*dt:.2f}s")
    print(f"Using transition_steps = {transition_steps} for all pressure and xi_star ramping")
    
    # Simulation parameters
    n = 51
    r = 0.001
    G = params['E']/(2*1.3)
    
    # Create target xi_star using identified parameters (this is what we're ramping to)
    target_xi_star = jnp.tile(jnp.array([params['ux'], params['uy'], 0, 0, 0, 1]), (n, 1))
    
    muscle_params = [
        {'r_i': 0.02, 'angle_i': angle, 'k': params[f'k{i+1}'], 'c': params[f'c{i+1}'], 'b': params[f'b{i+1}'], 'l0': params[f'l0{i+1}']}
        for i, angle in enumerate([0, 2*jnp.pi/3, 4*jnp.pi/3])
    ]
    tip_world_wrench = jnp.zeros(6)
    
    # Initialize arrays to store results
    time_sequence = np.arange(len(P_seq)) * dt
    tip_positions = np.zeros((len(P_seq), 2))  # [x, y] positions
    pressure_sequence = P_seq.copy()
    
    # Pre-compute zero-pressure configuration for initial guess with more steps for convergence
    print("Computing zero-pressure configuration for initial guess...")
    print(f"Using target xi_star: ux={params['ux']:.6f}, uy={params['uy']:.6f}")
    zero_pressures = jnp.array([0., 0., 0.])
    zero_mz = params['A'] * jnp.sum(zero_pressures) + params['mz0']
    
    # Get zero-pressure configuration using JAX simulation with more steps for convergence
    zero_tip_x, zero_tip_y, zero_X_final, zero_tau_final = simulate_rod_tip_jax_optimized_with_initial_guess(
        E=params['E'],
        rho=params['rho'],
        L=params['L'],
        xi_star=target_xi_star,  # This is now the target, not the initial
        muscle_params=muscle_params,
        tip_world_wrench=tip_world_wrench,
        target_pressures=zero_pressures,
        n=n,
        r=r,
        G=G,
        f_t=jnp.zeros(3),
        l_t=jnp.array([0., 0., zero_mz]),
        initial_xi0=jnp.array([0., 0., 0., 0., 0., 1.]),
        initial_taus=jnp.array([5.0, 5.0, 5.0]),
        target_xi_star=target_xi_star,  # Pass the target xi_star
        n_steps=transition_steps  # Use transition_steps for internal ramping
    )
    
    # Extract zero-pressure initial guesses
    zero_xi0_guess = jnp.array([0., 0., 0., 0., 0., 1.])  # Default initial strains
    zero_tau_guess = zero_tau_final  # Use the computed tensions from zero-pressure simulation
    
    print(f"Zero-pressure configuration computed:")
    print(f"  Zero-pressure tip position: ({zero_tip_x:.6f}, {zero_tip_y:.6f})")
    print(f"  Initial tensions: {zero_tau_guess}")
    
    # Store zero-pressure position for plotting
    zero_pressure_position = np.array([float(zero_tip_x), float(zero_tip_y)])
    
    # Initialize current guesses and shape state for continuity
    current_xi0_guess = zero_xi0_guess.copy()
    current_tau_guess = zero_tau_guess.copy()
    current_X_final = zero_X_final.copy()  # Start with zero-pressure shape
    
    # Simulate each time step with continuous pressure and xi_star ramping for trajectory continuity
    simulation_start_time = time.time()
    successful_simulations = 0
    
    # Initialize starting conditions for the first step
    current_pressure = jnp.array([0., 0., 0.])  # Start from zero pressure
    current_xi_star = jnp.tile(jnp.array([0., 0., 0., 0., 0., 1.]), (n, 1))  # Start from straight configuration
    is_first_step = True  # Flag to handle first step differently
    
    for i, target_pressures in enumerate(P_seq):
        try:
            if i % 50 == 0:  # Progress update every 50 steps
                elapsed_time = time.time() - simulation_start_time
                print(f"Progress: {i}/{len(P_seq)} steps ({i/len(P_seq)*100:.1f}%) - Elapsed: {elapsed_time:.1f}s")
            
            # Handle xi_star ramping differently for first step vs subsequent steps
            if is_first_step:
                # First step: ramp from straight to target xi_star
                initial_xi_star = jnp.tile(jnp.array([0., 0., 0., 0., 0., 1.]), (n, 1))  # Straight configuration
                print(f"\nSimulating step {i+1}/{len(P_seq)} (FIRST STEP):")
                print(f"  From pressure: {current_pressure}")
                print(f"  To target pressure: {target_pressures}")
                print(f"  From xi_star: [0,0,0,0,0,1] to target: [{params['ux']:.6f},{params['uy']:.6f},0,0,0,1]")
                print(f"  Using n_steps = {transition_steps} for initial convergence")
                is_first_step = False  # Mark that first step is done
            else:
                # Subsequent steps: stay at target xi_star (no xi_star ramping)
                initial_xi_star = target_xi_star.copy()  # Stay at target xi_star
                print(f"\nSimulating step {i+1}/{len(P_seq)}:")
                print(f"  From pressure: {current_pressure}")
                print(f"  To target pressure: {target_pressures}")
                print(f"  Xi_star: [{params['ux']:.6f},{params['uy']:.6f},0,0,0,1] (constant)")
                print(f"  Using n_steps = {transition_steps} (previous solution as initial guess)")
            
            # For each target pressure, ramp from current pressure to target pressure
            # This maintains continuity by using previous solution as starting point
            
            # Compute tip moment mz for this target pressure
            mz = params['A'] * jnp.sum(target_pressures) + params['mz0']
            
            # Simulate tip position with continuous pressure and xi_star ramping
            tip_x, tip_y, X_final, tau_final, xi0_guess_out = simulate_rod_tip_jax_optimized_with_continuous_ramping(
                E=params['E'],
                rho=params['rho'],
                L=params['L'],
                muscle_params=muscle_params,
                tip_world_wrench=tip_world_wrench,
                initial_pressures=current_pressure,  # Start from current pressure
                target_pressures=jnp.array(target_pressures),  # Ramp to target pressure
                initial_xi_star=initial_xi_star,  # Use appropriate initial xi_star
                target_xi_star=target_xi_star,  # Ramp to target xi_star
                n=n,
                r=r,
                G=G,
                f_t=jnp.zeros(3),
                l_t=jnp.array([0., 0., mz]),
                initial_xi0=current_xi0_guess,
                initial_taus=current_tau_guess,
                n_steps=transition_steps  # Use the transition_steps parameter
            )
            
            # Store results
            tip_positions[i, 0] = float(tip_x)
            tip_positions[i, 1] = float(tip_y)
            
            # Update current conditions for next iteration (maintain continuity)
            current_pressure = jnp.array(target_pressures)  # Current becomes the target we just reached
            # Note: current_xi_star stays at target_xi_star after first step
            
            # Update guesses for next iteration using the final state from current simulation
            # This ensures trajectory continuity by using previous solution as initial guess
            current_xi0_guess = xi0_guess_out  # Use the shooting solution's final guess for next step
            current_tau_guess = tau_final  # Use computed tensions as next guess
            

            
            # Store the final shape state for continuity
            current_X_final = X_final.copy()  # Store the final shape configuration for next iteration
            
            successful_simulations += 1
            
        except Exception as e:
            print(f"Simulation failed at step {i} with target pressures {target_pressures}: {e}")
            tip_positions[i, :] = np.nan
            # Keep previous guesses for next iteration
    
    simulation_time = time.time() - simulation_start_time
    print(f"\nSimulation completed in {simulation_time:.2f} seconds")
    print(f"Successful simulations: {successful_simulations}/{len(P_seq)} ({successful_simulations/len(P_seq)*100:.1f}%)")
    
    # Check trajectory continuity
    continuity_analysis = analyze_trajectory_continuity(tip_positions, time_sequence)
    print(f"Trajectory continuity analysis: {continuity_analysis}")
    

    
    # Save results to CSV if requested
    if save_to_csv:
        save_simulation_results_to_csv(time_sequence, pressure_sequence, tip_positions, csv_filename)
    
    return time_sequence, tip_positions, pressure_sequence, zero_pressure_position

def analyze_trajectory_continuity(tip_positions, time_sequence):
    """
    Analyze the continuity of the tip trajectory
    
    Args:
        tip_positions: Array of tip positions (n_steps, 2)
        time_sequence: Array of time points
        
    Returns:
        dict: Analysis results
    """
    # Find valid positions
    valid_mask = ~(np.isnan(tip_positions[:, 0]) | np.isnan(tip_positions[:, 1]))
    
    if not np.any(valid_mask):
        return {"status": "No valid positions", "discontinuities": 0, "max_jump": 0}
    
    valid_positions = tip_positions[valid_mask]
    valid_times = time_sequence[valid_mask]
    
    if len(valid_positions) < 2:
        return {"status": "Insufficient valid positions", "discontinuities": 0, "max_jump": 0}
    
    # Calculate position differences between consecutive steps
    position_diffs = np.diff(valid_positions, axis=0)
    distances = np.sqrt(np.sum(position_diffs**2, axis=1))
    
    # Define threshold for discontinuity (in meters)
    discontinuity_threshold = 0.01  # 1 cm
    
    # Find discontinuities
    discontinuities = distances > discontinuity_threshold
    num_discontinuities = np.sum(discontinuities)
    max_jump = np.max(distances) if len(distances) > 0 else 0
    
    # Calculate average step size
    avg_step_size = np.mean(distances) if len(distances) > 0 else 0
    
    # Determine status
    if num_discontinuities == 0:
        status = "Continuous"
    elif num_discontinuities <= len(distances) * 0.1:  # Less than 10% discontinuities
        status = "Mostly continuous"
    else:
        status = "Discontinuous"
    
    return {
        "status": status,
        "discontinuities": int(num_discontinuities),
        "max_jump": float(max_jump),
        "avg_step_size": float(avg_step_size),
        "total_steps": len(valid_positions),
        "discontinuity_threshold": discontinuity_threshold
    }

def save_simulation_results_to_csv(time_sequence, pressure_sequence, tip_positions, filename):
    """
    Save simulation results to CSV file
    
    Args:
        time_sequence: Array of time points
        pressure_sequence: Array of pressure values (n_steps, 3)
        tip_positions: Array of tip positions (n_steps, 2)
        filename: Output CSV filename
    """
    # Create DataFrame with the specified column structure
    df = pd.DataFrame({
        'time': time_sequence,
        'p1': pressure_sequence[:, 0],
        'p2': pressure_sequence[:, 1],
        'p3': pressure_sequence[:, 2],
        'x': tip_positions[:, 0],
        'y': tip_positions[:, 1]
    })
    
    # First convert all numeric columns to float, keeping input_type as is
    numeric_cols = ['time', 'p1', 'p2', 'p3', 'x', 'y']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"Simulation results saved to {filename}")
    print(f"CSV contains {len(df)} data points")
    
    # Print summary statistics
    valid_positions = df[~df['x'].isna()]
    if len(valid_positions) > 0:
        print(f"Valid simulations: {len(valid_positions)}/{len(df)}")
        print(f"Tip position range: x=[{valid_positions['x'].min():.6f}, {valid_positions['x'].max():.6f}], "
              f"y=[{valid_positions['y'].min():.6f}, {valid_positions['y'].max():.6f}]")
    else:
        print("No valid simulations found")

def plot_simulation_results(time_sequence, pressure_sequence, tip_positions, save_plot=True, plot_filename='simulation_plot.png'):
    """
    Plot simulation results
    
    Args:
        time_sequence: Array of time points
        pressure_sequence: Array of pressure values (n_steps, 3)
        tip_positions: Array of tip positions (n_steps, 2)
        save_plot: Whether to save the plot
        plot_filename: Output plot filename
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Pressures over time
    axes[0, 0].plot(time_sequence, pressure_sequence[:, 0], 'r-', label='P1', linewidth=2)
    axes[0, 0].plot(time_sequence, pressure_sequence[:, 1], 'g-', label='P2', linewidth=2)
    axes[0, 0].plot(time_sequence, pressure_sequence[:, 2], 'b-', label='P3', linewidth=2)
    axes[0, 0].set_xlabel('Time [s]')
    axes[0, 0].set_ylabel('Pressure [kPa]')
    axes[0, 0].set_title('Pressure Input Sequence')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot 2: Tip x position over time
    valid_mask = ~np.isnan(tip_positions[:, 0])
    if np.any(valid_mask):
        axes[0, 1].plot(time_sequence[valid_mask], tip_positions[valid_mask, 0], 'b-', linewidth=2)
    axes[0, 1].set_xlabel('Time [s]')
    axes[0, 1].set_ylabel('Tip X Position [m]')
    axes[0, 1].set_title('Tip X Position vs Time')
    axes[0, 1].grid(True)
    
    # Plot 3: Tip y position over time
    valid_mask = ~np.isnan(tip_positions[:, 1])
    if np.any(valid_mask):
        axes[1, 0].plot(time_sequence[valid_mask], tip_positions[valid_mask, 1], 'g-', linewidth=2)
    axes[1, 0].set_xlabel('Time [s]')
    axes[1, 0].set_ylabel('Tip Y Position [m]')
    axes[1, 0].set_title('Tip Y Position vs Time')
    axes[1, 0].grid(True)
    
    # Plot 4: Tip trajectory (x vs y)
    valid_mask = ~(np.isnan(tip_positions[:, 0]) | np.isnan(tip_positions[:, 1]))
    if np.any(valid_mask):
        axes[1, 1].plot(tip_positions[valid_mask, 0], tip_positions[valid_mask, 1], 'k-', linewidth=2)
        axes[1, 1].plot(tip_positions[valid_mask, 0], tip_positions[valid_mask, 1], 'ko', markersize=3, alpha=0.5)
    axes[1, 1].set_xlabel('Tip X Position [m]')
    axes[1, 1].set_ylabel('Tip Y Position [m]')
    axes[1, 1].set_title('Tip Trajectory')
    axes[1, 1].grid(True)
    axes[1, 1].axis('equal')
    
    plt.tight_layout()
    
    if save_plot:
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {plot_filename}")
    else:
        plt.show()
    
    plt.close()





def print_device_info():
    """Print JAX device information"""
    print("=" * 60)
    print("JAX DEVICE CONFIGURATION")
    print("=" * 60)
    print(f"Available devices: {jax.devices()}")
    print(f"Default backend: {jax.default_backend()}")
    
    # Check for GPU
    gpu_devices = [d for d in jax.devices() if 'gpu' in str(d).lower()]
    if gpu_devices:
        print(f"GPU acceleration available: {len(gpu_devices)} GPU device(s)")
        print(f"GPU devices: {gpu_devices}")
    else:
        print("GPU not available, using CPU")
    
    print("=" * 60)

# --- Main Execution ---
if __name__ == "__main__":
    start_time = time.time()
    
    # Print JAX device information
    print_device_info()
    
    # Set simulation parameters
    dt = 0.005  # Time step1

    taus =  40 # Number of transitions  40
    pressure_max = 65.0  # Maximum pressure
    hold_time = 2.0     # Hold time
    
    print(f"\nSimulation Parameters:")
    print(f"  Time step (dt): {dt}s")
    print(f"  Transitions (taus): {taus}")
    print(f"  Maximum pressure: {pressure_max} kPa")
    print(f"  Hold time: {hold_time}s")
    
    # Generate ramp-hold pressure sequence
    print(f"\nGenerating ramp-hold pressure sequence...")
    P_seq = generate_ramp_hold(taus=taus, dt=dt, pressure_max=pressure_max, hold_time=hold_time)
    print(f"Generated pressure sequence with {len(P_seq)} time steps")
    print(f"Total simulation time: {len(P_seq)*dt:.2f}s")
    
    # Print parameters being used
    print("\nSimulating with manual parameters:")
    for key, value in manual_params.items():
        print(f"  {key}: {value}")
    
    # Run simulation
    print("\nStarting ramp-hold simulation...")
    transition_steps = 100
    print(f"Using transition_steps = {transition_steps} for all pressure and xi_star ramping")
    time_sequence, tip_positions, pressure_sequence, zero_pressure_position = simulate_ramp_hold_sequence(
        params=manual_params,
        P_seq=P_seq,
        dt=dt,
        save_to_csv=True,
        csv_filename='simulation_results.csv',
        transition_steps=transition_steps  # Number of steps for pressure and xi_star ramping
    )
    
    # Plot simulation results
    print("\nGenerating basic simulation plot...")
    
    # Plot basic simulation results and show in window
    plot_simulation_results(time_sequence, pressure_sequence, tip_positions, 
                           save_plot=False, plot_filename='simulation_plot.png')
    
    print("Basic simulation plot displayed in window!")
    
    end_time = time.time()
    print(f"\nTotal calculation time: {end_time - start_time:.2f} seconds")
    
    # Print final summary
    print("\n" + "="*60)
    print("SIMULATION GENERATOR SUMMARY")
    print("="*60)
    print(f"Simulation approach: Continuous pressure ramping with configurable step counts")
    print(f"Ramping steps: transition_steps = {transition_steps} for all steps")
    print(f"Continuity scheme: Each target uses previous solution as starting point")
    print(f"Xi_star scheme: First step ramps from straight to target, subsequent steps stay at target")
    print(f"Target xi_star: Settled configuration [{manual_params['ux']:.6f},{manual_params['uy']:.6f},0,0,0,1]")
    print(f"Simulation time steps: {len(P_seq)}")
    print(f"Total simulation time: {len(P_seq)*dt:.2f}s")
    print(f"Time step: {dt}s")
    print(f"Pressure range: [{pressure_sequence.min():.1f}, {pressure_sequence.max():.1f}] kPa")
    print(f"Xi_star ramping: First step: [0,0,0,0,0,1] → [{manual_params['ux']:.6f},{manual_params['uy']:.6f},0,0,0,1], then constant")
    
    valid_positions = tip_positions[~np.isnan(tip_positions[:, 0])]
    if len(valid_positions) > 0:
        print(f"Valid simulations: {len(valid_positions)}/{len(tip_positions)}")
        print(f"Tip position range: x=[{valid_positions[:, 0].min():.6f}, {valid_positions[:, 0].max():.6f}], "
              f"y=[{valid_positions[:, 1].min():.6f}, {valid_positions[:, 1].max():.6f}]")
        
        # Print zero-pressure position information
        if zero_pressure_position is not None:
            print(f"Zero-pressure start position: ({zero_pressure_position[0]:.6f}, {zero_pressure_position[1]:.6f})")
            print(f"Initial deflection from origin: {np.sqrt(zero_pressure_position[0]**2 + zero_pressure_position[1]**2)*1000:.2f} mm")
        
        # Analyze trajectory continuity
        continuity_analysis = analyze_trajectory_continuity(tip_positions, time_sequence)
        print(f"Trajectory continuity: {continuity_analysis['status']}")
        print(f"Discontinuities: {continuity_analysis['discontinuities']}/{continuity_analysis['total_steps']-1}")
        print(f"Maximum jump: {continuity_analysis['max_jump']*1000:.2f} mm")
        print(f"Average step size: {continuity_analysis['avg_step_size']*1000:.2f} mm")
    else:
        print("No valid simulations found")
    
    print(f"Results saved to: simulation_results.csv")
    print("="*60) 