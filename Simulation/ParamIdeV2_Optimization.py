import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')

# JAX imports
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

# --- Data Loading from average_static_points.csv ---
def load_experiment_data(static_points_file='average_static_points.csv'):
    """
    Load static points data from average_static_points.csv file.
    
    Args:
        static_points_file: Path to the CSV file containing static points data
        
    Returns: static_positions (N,2), static_pressures (N,3)
    """
    try:
        # Load static points data
        static_data = pd.read_csv(static_points_file)
        print(f"Successfully loaded static points file: {static_points_file}")
        print(f"Number of static points: {len(static_data)}")
        
        # Extract positions (x, y columns correspond to dz, dx in our coordinate system)
        static_positions = np.column_stack([
            static_data['x'],  # dz (z2 - z1)
            static_data['y']   # dx (x2 - x1)
        ])
        
        # Extract pressures
        static_pressures = np.column_stack([
            static_data['p1'],
            static_data['p2'],
            static_data['p3']
        ])
        
        print(f"Processed {len(static_positions)} static positions and pressures")
        print("\nStatic positions (dz, dx):")
        for i, pos in enumerate(static_positions):
            print(f"Point {i+1}: [{pos[0]:.6f}, {pos[1]:.6f}]")
        
        print("\nStatic pressures (P1, P2, P3):")
        for i, press in enumerate(static_pressures):
            print(f"Point {i+1}: [{press[0]:.6f}, {press[1]:.6f}, {press[2]:.6f}]")
        
        return static_positions, static_pressures
        
    except FileNotFoundError:
        print(f"Error: Static points file '{static_points_file}' not found.")
        print("Please run Init_points_validation.py first to generate the static points file.")
        raise
    except Exception as e:
        print(f"Error loading static points file: {e}")
        raise
# --- Manual Parameters Section --- initial values
manual_params = {
    'E': 1.187504e+10,         # Young's modulus [Pa]
    'rho': 8.236995e+05,      # Density [kg/m^3]
    'L': 2.000000e-01,         # Length [m]
    'ux': 1.634339e+00,         # Reference strain x
    'uy': -2.103838e+00,        # Reference strain y
    'k1': 8.028175e+01, 'k2': 7.011625e+01, 'k3': 7.034013e+01,  # Muscle spring constants
    'c1': 1.219215e-02, 'c2': 7.504683e-03, 'c3': 7.864152e-03,     # Pressure coefficients
    'b1': 4.289537e-04, 'b2': 4.007239e-04, 'b3': 4.115483e-04,  # Quadratic pressure coefficients
    'l01': 9.935136e-02, 'l02': 8.907066e-02, 'l03': 8.831651e-02,  # Muscle original lengths
    'mz0': 2.531363e-03,         # Tip moment offset
    'A': 2.738800e-04           # Tip moment pressure coefficient
}


# # --- Manual Parameters Section --- initial values
# manual_params = {
#     'E': 11875040734.254396,         # Young's modulus [Pa]
#     'rho': 823699.4570762948,      # Density [kg/m^3]
#     'L': 0.2,         # Length [m]
#     'ux': 1.634339267162691,         # Reference strain x
#     'uy': -2.1038382647760769,        # Reference strain y
#     'k1': 80.28174703431087, 'k2': 70.11624708945146, 'k3': 70.34012776765677,  # Muscle spring constants
#     'c1': 0.012192152029299403, 'c2': 0.007504683154198564, 'c3': 0.007864151776160305,     # Pressure coefficients
#     'b1': 0.00042895373175965865, 'b2': 0.00040072386896266675, 'b3': 0.000411548284865392,  # Quadratic pressure coefficients
#     'l01': 0.09935135940772408, 'l02': 0.08907066016110076, 'l03': 0.08831651322063939,  # Muscle original lengths
#     'mz0': 0.00253136259627531,         # Tip moment offset
#     'A': 0.0002738799758540097           # Tip moment pressure coefficient
# }

# --- Static Point Sets Definition ---
static_point_sets = {
    'set1': [1, 2, 3, 4, 5, 6,7, 14, 15, 16, 17, 18, 19],  # 13 points
    'set2': [1, 2, 4, 6, 15, 17, 19],                     # 7 points  
    'set3': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 20, 21, 22, 23, 24, 25],  # 19 points
    'set4': [1, 2, 4, 6]                                  # 4 points
}

# User-defined set to use for simulation (change this to select different sets)
selected_set = 'set1'  # Options: 'set1', 'set2', 'set3', 'set4'

# --- Parameter Bounds Definition (using scaled values for easier setup) ---
param_bounds = {
    'E': (1.1e10, 1.5e10),                                                 # Initial: 11875040734.254396 Pa (≈1.19e10)
    'rho': (7.0e5, 12.0e5),                                                 # Initial: 823699.4570762948 kg/m³ (≈8.24e5)
    'L': (0.18, 0.22),                                                     # Initial: 0.2 m
    'ux': (-3.0, 3.0),                                                      # Initial: 1.634339267162691
    'uy': (-3.0, 3.0),                                                    # Initial: -2.1038382647760769
    'k1': (70, 120),                                                        # Initial: 80.28174703431087
    'k2': (70, 120),                                                        # Initial: 70.11624708945146
    'k3': (70, 120),                                                        # Initial: 70.34012776765677
    'c1': (0.001, 0.05),                                                  # Initial: 0.012192152029299403
    'c2': (0.001, 0.05),                                                  # Initial: 0.007504683154198564
    'c3': (0.001, 0.05),                                                  # Initial: 0.007864151776160305
    'b1': (0.00001, 0.001),                                                # Initial: 0.00042895373175965865
    'b2': (0.00001, 0.001),                                                # Initial: 0.00040072386896266675
    'b3': (0.00001, 0.001),                                                # Initial: 0.000411548284865392
    'l01': (0.07, 0.12),                                                   # Initial: 0.09935135940772408 m
    'l02': (0.07, 0.12),                                                   # Initial: 0.08907066016110076 m
    'l03': (0.07, 0.12),                                                   # Initial: 0.08831651322063939 m
    'mz0': (-0.10, 0.10),                                               # Initial: 0.00253136259627531
    'A': (-0.001, 0.001)                                                  # Initial: 0.0002738799758540097
}

# --- JAX-Optimized Simulation Functions (EXACT COPY FROM ParamIdeV2_JAX.py) ---
def solve_single_point_jax(rod_params, muscle_params, target_pressures, W_dist, f_t=None, l_t=None, target_xi_star=None):
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
    
    # Create pressure and xi_star steps
    n_steps = 100
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

def simulate_rod_tip_jax_optimized(E, rho, L, xi_star, muscle_params, tip_world_wrench, target_pressures, n=51, r=0.001, G=None, g=9.81, f_t=None, l_t=None, target_xi_star=None):
    """
    JAX-optimized rod tip simulation with target xi_star support
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
    
    # Run simulation with target xi_star
    X_final, tau_final = solve_single_point_jax(rod_params, muscle_params, target_pressures, W_dist, f_t, l_t, target_xi_star)
    
    tip_x, tip_y = X_final[-1, 0], X_final[-1, 1]
    
    return tip_x, tip_y, X_final, tau_final

def simulate_rod_tip_jax_optimized_with_initial_guess(E, rho, L, xi_star, muscle_params, tip_world_wrench, target_pressures, n=51, r=0.001, G=None, g=9.81, f_t=None, l_t=None, initial_xi0=None, initial_taus=None, target_xi_star=None):
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
        initial_xi0, initial_taus, f_t, l_t, target_xi_star
    )
    
    tip_x, tip_y = X_final[-1, 0], X_final[-1, 1]
    
    return tip_x, tip_y, X_final, tau_final

def solve_single_point_jax_with_initial_guess(rod_params, muscle_params, target_pressures, W_dist, initial_xi0, initial_taus, f_t=None, l_t=None, target_xi_star=None):
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
    
    # Create pressure and xi_star steps (fewer steps since we have good initial guesses)
    n_steps = 30  # Reduced from 30 since we have good initial guesses
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

# --- JAX Simulation and Comparison (EXACT COPY FROM ParamIdeV2_JAX.py) ---
def simulate_all_jax(static_pressures, params, selected_indices=None):
    """
    Simulate selected static points with given parameters using JAX acceleration with target xi_star
    
    Args:
        static_pressures: All static pressures (N, 3)
        params: Model parameters
        selected_indices: List of indices to simulate (0-based). If None, simulate all points.
    """
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
    
    # Determine which points to simulate
    if selected_indices is None:
        selected_indices = list(range(len(static_pressures)))
    
    print(f"Starting simulation of {len(selected_indices)} selected points out of {len(static_pressures)} total points...")
    print(f"Selected point indices: {[i+1 for i in selected_indices]} (1-based)")
    print(f"Using target xi_star: ux={params['ux']:.6f}, uy={params['uy']:.6f}")
    
    sim_positions = []
    
    # Pre-compute zero-pressure configuration once (same for all points)
    print("Computing zero-pressure configuration for initial guess...")
    zero_pressures = jnp.array([0., 0., 0.])
    zero_mz = params['A'] * jnp.sum(zero_pressures) + params['mz0']
    
    # Get zero-pressure configuration using JAX simulation with target xi_star
    zero_tip_x, zero_tip_y, zero_X_final, zero_tau_final = simulate_rod_tip_jax_optimized(
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
        target_xi_star=target_xi_star  # Pass the target xi_star
    )
    
    # Extract zero-pressure initial guesses
    zero_xi0_guess = jnp.array([0., 0., 0., 0., 0., 1.])  # Default initial strains (straight)
    zero_tau_guess = zero_tau_final  # Use the computed tensions from zero-pressure simulation
    
    print(f"Zero-pressure configuration computed. Initial tensions: {zero_tau_guess}")
    
    for i, pressures in enumerate(static_pressures):
        # Skip points not in the selected set
        if i not in selected_indices:
            sim_positions.append([np.nan, np.nan])
            continue
        try:
            print(f"\nSimulating point {i+1}/{len(static_pressures)} with pressures: {pressures}")
            point_start_time = time.time()
            
            # Compute tip moment mz for this point
            mz = params['A'] * jnp.sum(pressures) + params['mz0']
            
            # Use the pre-computed zero-pressure configuration as initial guess
            # This is much more efficient than starting from zero pressure each time
            tip_x, tip_y, _, _ = simulate_rod_tip_jax_optimized_with_initial_guess(
                E=params['E'],
                rho=params['rho'],
                L=params['L'],
                xi_star=target_xi_star,  # This is now the target, not the initial
                muscle_params=muscle_params,
                tip_world_wrench=tip_world_wrench,
                target_pressures=pressures,
                n=n,
                r=r,
                G=G,
                f_t=jnp.zeros(3),
                l_t=jnp.array([0., 0., mz]),
                initial_xi0=zero_xi0_guess,
                initial_taus=zero_tau_guess,
                target_xi_star=target_xi_star  # Pass the target xi_star
            )
            
            point_time = time.time() - point_start_time
            sim_positions.append([float(tip_x), float(tip_y)])
            print(f"Point {i+1}: Simulated position = ({float(tip_x)*1000:.1f}, {float(tip_y)*1000:.1f}) mm in {point_time:.2f}s")
            
        except Exception as e:
            print(f"Simulation failed for point {i+1} with pressures {pressures}: {e}")
            sim_positions.append([np.nan, np.nan])
    
    return np.array(sim_positions)

# --- Visualization (EXACT COPY FROM ParamIdeV2_JAX.py) ---
def plot_comparison_jax(sim_positions, exp_positions, selected_indices=None, title_suffix=""):
    """
    Create comparison plot with error analysis for JAX simulation
    
    Args:
        sim_positions: Simulated positions
        exp_positions: Experimental positions
        selected_indices: List of selected point indices (0-based)
        title_suffix: Additional text for the title
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Determine which points to highlight
    if selected_indices is None:
        selected_indices = list(range(len(exp_positions)))
    
    # Plot all experimental points (smaller, lighter)
    for i, pos in enumerate(exp_positions):
        if i in selected_indices:
            # Selected points - larger, darker
            ax.plot(pos[0], pos[1], 'ro', markersize=12, alpha=1.0, label='Selected Experimental' if i==selected_indices[0] else "")
            ax.annotate(f'{i+1}\n({pos[0]*1000:.1f}, {pos[1]*1000:.1f})mm',
                        (pos[0], pos[1]), xytext=(10, 10), textcoords='offset points',
                        color='red', fontweight='bold', fontsize=10)
        else:
            # Non-selected points - smaller, lighter
            ax.plot(pos[0], pos[1], 'ro', markersize=6, alpha=0.3, label='Other Experimental' if i==0 else "")
            ax.annotate(f'{i+1}', (pos[0], pos[1]), xytext=(5, 5), textcoords='offset points',
                        color='red', alpha=0.5, fontsize=8)
    
    # Simulated points (only for selected points)
    total_error = 0
    valid_simulations = 0
    for i, pos in enumerate(sim_positions):
        # Only process selected points
        if i not in selected_indices:
            continue
            
        # Check for NaN values
        if np.isnan(pos[0]) or np.isnan(pos[1]):
            ax.plot(exp_positions[i][0], exp_positions[i][1], 'kx', markersize=15, label='Failed Sim' if i==selected_indices[0] else "")
            ax.annotate(f'Sim {i+1}\nFAILED',
                        (exp_positions[i][0], exp_positions[i][1]), xytext=(-10, -10), textcoords='offset points',
                        color='black', fontweight='bold')
            continue
        
        ax.plot(pos[0], pos[1], 'b*', markersize=12, label='JAX Simulation' if i==selected_indices[0] else "")
        ax.annotate(f'Sim {i+1}\n({pos[0]*1000:.1f}, {pos[1]*1000:.1f})mm',
                    (pos[0], pos[1]), xytext=(-10, -10), textcoords='offset points',
                    color='blue', fontweight='bold')
        
        # Calculate error for valid simulations
        exp_pos = exp_positions[i]
        error = np.sqrt((exp_pos[0] - pos[0])**2 + (exp_pos[1] - pos[1])**2)
        total_error += error
        valid_simulations += 1
        
        # Draw error line
        ax.plot([exp_pos[0], pos[0]], [exp_pos[1], pos[1]], 'g--', alpha=0.7, linewidth=2, label='Error' if i==selected_indices[0] else "")
        mid_x = (exp_pos[0] + pos[0]) / 2
        mid_y = (exp_pos[1] + pos[1]) / 2
        ax.annotate(f'{error*1000:.1f}mm', (mid_x, mid_y), xytext=(0, 5), textcoords='offset points',
                    ha='center', va='bottom', color='green', alpha=0.8, fontweight='bold')
    
    ax.plot(0, 0, 'k+', label='Origin', markersize=15)
    ax.set_xlabel('dz [m]')
    ax.set_ylabel('dx [m]')
    
    # Update title with error information
    title = f'JAX Simulation vs Experimental Static Tip Positions\nSet: {selected_set} ({len(selected_indices)} points)'
    if title_suffix:
        title += f" - {title_suffix}"
    
    if valid_simulations > 0:
        avg_error = total_error / valid_simulations
        title += f'\nTotal Error: {total_error*1000:.1f}mm (Avg: {avg_error*1000:.1f}mm, {valid_simulations}/{len(selected_indices)} valid)'
    else:
        title += '\nNo valid simulations'
    
    ax.set_title(title)
    ax.grid(True)
    ax.axis('equal')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    # Save plot based on context
    if 'Optimized' in title_suffix:
        plt.savefig('optimized_plot.png', dpi=300, bbox_inches='tight')
    elif 'Initial' in title_suffix:
        plt.savefig('initial_guess_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    return total_error

# --- Optimization Functions ---
def params_to_array(params_dict):
    """Convert parameters dictionary to array for optimization"""
    param_names = ['E', 'rho', 'L', 'ux', 'uy', 'k1', 'k2', 'k3', 
                  'c1', 'c2', 'c3', 'b1', 'b2', 'b3', 'l01', 'l02', 'l03', 'mz0', 'A']
    return np.array([params_dict[name] for name in param_names])

def array_to_params(params_array):
    """Convert array back to parameters dictionary"""
    param_names = ['E', 'rho', 'L', 'ux', 'uy', 'k1', 'k2', 'k3', 
                  'c1', 'c2', 'c3', 'b1', 'b2', 'b3', 'l01', 'l02', 'l03', 'mz0', 'A']
    return {name: value for name, value in zip(param_names, params_array)}

def cost_function(params_array, static_positions, static_pressures, selected_indices):
    """
    Cost function: sum of squared distances between simulated and experimental positions
    Only calculates for selected points
    """
    try:
        # Convert array back to parameters dictionary
        params = array_to_params(params_array)
        
        # Run simulation with current parameters
        sim_positions = simulate_all_jax(static_pressures, params, selected_indices)
        
        # Calculate cost (sum of squared distances) only for selected points
        total_cost = 0.0
        valid_points = 0
        
        for i in selected_indices:
            if not np.isnan(sim_positions[i, 0]):
                # Calculate distance between experimental and simulated positions
                exp_pos = static_positions[i]
                sim_pos = sim_positions[i]
                distance = np.sqrt((exp_pos[0] - sim_pos[0])**2 + (exp_pos[1] - sim_pos[1])**2)
                total_cost += distance**2  # Square the distance
                valid_points += 1
        
        if valid_points == 0:
            return 1e10  # Return high cost if no valid simulations
        
        return total_cost
        
    except Exception as e:
        print(f"Error in cost function: {e}")
        return 1e10

def optimize_parameters(static_positions, static_pressures, selected_indices, initial_params=None, max_iterations=100):
    """
    Optimize parameters using scipy.optimize.minimize with boundary conditions
    """
    from scipy.optimize import minimize
    
    if initial_params is None:
        initial_params = manual_params.copy()
    
    # Convert initial parameters to array
    initial_params_array = params_to_array(initial_params)
    
    # Create bounds for optimization (using param_bounds)
    bounds = []
    param_names = ['E', 'rho', 'L', 'ux', 'uy', 'k1', 'k2', 'k3', 
                  'c1', 'c2', 'c3', 'b1', 'b2', 'b3', 'l01', 'l02', 'l03', 'mz0', 'A']
    
    for param_name in param_names:
        if param_name in param_bounds:
            bounds.append(param_bounds[param_name])
        else:
            # Default bounds if not specified
            bounds.append((initial_params[param_name] * 0.5, initial_params[param_name] * 1.5))
    
    print(f"Optimization bounds:")
    for i, (param_name, (lower, upper)) in enumerate(zip(param_names, bounds)):
        print(f"  {param_name}: [{lower:.6e}, {upper:.6e}]")
    
    # Define the objective function
    def objective(params_array):
        return cost_function(params_array, static_positions, static_pressures, selected_indices)
    
    # Run optimization
    print(f"\nStarting optimization with {max_iterations} maximum iterations...")
    result = minimize(
        objective,
        initial_params_array,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': max_iterations, 'disp': True}
    )
    
    # Convert result back to parameters dictionary
    optimized_params = array_to_params(result.x)
    
    return optimized_params, result

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
    
    # Load static points data from average_static_points.csv
    static_points_file = 'average_static_points.csv'
    print(f"\nLoading static points from: {static_points_file}")
    
    static_positions, static_pressures = load_experiment_data(static_points_file)
    
    # Get selected static point indices (convert from 1-based to 0-based)
    selected_indices = [i-1 for i in static_point_sets[selected_set]]
    print(f"\nUsing static point set: {selected_set}")
    print(f"Selected points: {static_point_sets[selected_set]} (1-based)")
    print(f"Number of points to simulate: {len(selected_indices)}")
    
    # Print initial parameters
    print("\nInitial parameters:")
    for key, value in manual_params.items():
        print(f"  {key}: {value}")
    
    # STEP 1: Run initial simulation with manual parameters
    print("\n" + "="*60)
    print("STEP 1: INITIAL SIMULATION WITH MANUAL PARAMETERS")
    print("="*60)
    
    initial_sim_start_time = time.time()
    initial_sim_positions = simulate_all_jax(static_pressures, manual_params, selected_indices)
    initial_sim_end_time = time.time()
    
    print(f"\nInitial simulation completed in {initial_sim_end_time - initial_sim_start_time:.2f} seconds")
    
    # Calculate initial cost
    initial_cost = cost_function(params_to_array(manual_params), static_positions, static_pressures, selected_indices)
    print(f"Initial cost (sum of squared distances): {initial_cost:.6f}")
    
    # Show initial guess figure
    print("\n" + "="*60)
    print("SHOWING INITIAL GUESS FIGURE")
    print("="*60)
    initial_error = plot_comparison_jax(initial_sim_positions, static_positions, selected_indices, "Initial Parameters")
    print(f"Initial guess total error: {initial_error*1000:.1f} mm")
    print("="*60)
    
    # Continue automatically with optimization
    
    # STEP 2: Run parameter optimization
    print("\n" + "="*60)
    print("STEP 2: PARAMETER OPTIMIZATION")
    print("="*60)
    
    optimization_start_time = time.time()
    optimized_params, optimization_result = optimize_parameters(
        static_positions, static_pressures, selected_indices, 
        initial_params=manual_params, max_iterations=50
    )
    optimization_end_time = time.time()
    
    print(f"\nOptimization completed in {optimization_end_time - optimization_start_time:.2f} seconds")
    print(f"Optimization success: {optimization_result.success}")
    print(f"Number of iterations: {optimization_result.nit}")
    print(f"Number of function evaluations: {optimization_result.nfev}")
    
    # STEP 3: Run optimized simulation
    print("\n" + "="*60)
    print("STEP 3: OPTIMIZED SIMULATION")
    print("="*60)
    
    optimized_sim_start_time = time.time()
    optimized_sim_positions = simulate_all_jax(static_pressures, optimized_params, selected_indices)
    optimized_sim_end_time = time.time()
    
    print(f"\nOptimized simulation completed in {optimized_sim_end_time - optimized_sim_start_time:.2f} seconds")
    
    # Calculate optimized cost
    optimized_cost = cost_function(params_to_array(optimized_params), static_positions, static_pressures, selected_indices)
    print(f"Optimized cost (sum of squared distances): {optimized_cost:.6f}")
    
    # Show optimized results figure
    print("\n" + "="*60)
    print("SHOWING OPTIMIZED RESULTS FIGURE")
    print("="*60)
    optimized_error = plot_comparison_jax(optimized_sim_positions, static_positions, selected_indices, "Optimized Parameters")
    print(f"Optimized total error: {optimized_error*1000:.1f} mm")
    print("="*60)
    
    # Final summary
    end_time = time.time()
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"Total calculation time: {end_time - start_time:.2f} seconds")
    print(f"Initial cost: {initial_cost:.6f}")
    print(f"Optimized cost: {optimized_cost:.6f}")
    print(f"Cost improvement: {initial_cost - optimized_cost:.6f}")
    print(f"Improvement percentage: {((initial_cost - optimized_cost) / initial_cost * 100):.2f}%")
    
    # Print parameter comparison
    print("\n" + "="*60)
    print("PARAMETER COMPARISON")
    print("="*60)
    print("Parameter | Initial Value | Optimized Value | Change")
    print("----------|---------------|-----------------|--------")
    
    for key in manual_params.keys():
        initial_val = manual_params[key]
        optimized_val = optimized_params[key]
        change = optimized_val - initial_val
        print(f"{key:9} | {initial_val:13.6e} | {optimized_val:15.6e} | {change:+8.2e}")
    
    print("="*60)
    print("OPTIMIZATION RESULT SUMMARY")
    print("="*60)
    print(f"success: {optimization_result.success}")
    print(f"nfev: {optimization_result.nfev}")
    print(f"nit: {optimization_result.nit}")
    print(f"message: {optimization_result.message}")
    print("="*60) 