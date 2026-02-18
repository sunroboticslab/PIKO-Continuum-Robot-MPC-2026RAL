import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import warnings
from scipy.interpolate import interp1d
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

def generate_sinusoidal_pressures(total_time, dt, pressure_max=65.0, frequency=0.1):
    """
    Generate sinusoidal pressure signals for the three muscles
    Creates a sinusoidal trajectory by using same frequency with 2π/3 phase difference
    between all three muscles for one complete cycle
    Pressures are offset to ensure 0-60 kPa range
    
    Args:
        total_time: Total simulation time [s] (should be 1/frequency for one cycle)
        dt: Time step [s]
        pressure_max: Maximum pressure amplitude [kPa] (will be offset to 0-pressure_max range)
        frequency: Frequency for sinusoidal motion [Hz]
        
    Returns:
        np.array: Pressure sequence with shape (n_steps, 3)
    """
    n_steps = int(total_time / dt)
    time_sequence = np.linspace(0, total_time, n_steps)
    
    # Generate sinusoidal pressures with offset to ensure positive values
    pressures = np.zeros((n_steps, 3))
    
    # First muscle: sin(2π*f*t) + offset to make it 0 to pressure_max
    sin_signal_1 = pressure_max/2 * np.sin(2 * np.pi * frequency * time_sequence)
    pressures[:, 0] = pressure_max/2 + sin_signal_1
    
    # Second muscle: sin(2π*f*t + 2π/3) + offset to make it 0 to pressure_max
    sin_signal_2 = pressure_max/2 * np.sin(2 * np.pi * frequency * time_sequence + 2*np.pi/3)
    pressures[:, 1] = pressure_max/2 + sin_signal_2
    
    # Third muscle: sin(2π*f*t + 4π/3) + offset to make it 0 to pressure_max
    sin_signal_3 = pressure_max/2 * np.sin(2 * np.pi * frequency * time_sequence + 4*np.pi/3)
    pressures[:, 2] = pressure_max/2 + sin_signal_3
    
    return pressures, time_sequence

# --- Manual Parameters Section ---
# manual_params = {
#     'E': 1.187504e+10,         # Young's modulus [Pa]
#     'rho': 8.236995e+05,      # Density [kg/m^3]
#     'L': 2.005849e-01,         # Length [m]
#     'ux': 1.634464e+00,         # Reference strain x
#     'uy': -2.102465e+00,        # Reference strain y
#     'k1': 8.028893e+01, 'k2': 7.011634e+01, 'k3': 7.034240e+01,  # Muscle spring constants
#     'c1': 1.165667e-02, 'c2': 8.456121e-03, 'c3': 7.705992e-03,     # Pressure coefficients
#     'b1': 3.533195e-04, 'b2': 4.151490e-04, 'b3': 4.194060e-04,  # Quadratic pressure coefficients
#     'l01': 9.983082e-02, 'l02': 8.826556e-02, 'l03': 8.983696e-02,  # Muscle original lengths
#     'mz0': 2.787942e-03,         # Tip moment offset
#     'A': 2.301960e-04           # Tip moment pressure coefficient
# }

manual_params = {
    'E': 1.187504e+10,         # Young's modulus [Pa]
    'rho': 8.236995e+05,      # Density [kg/m^3]
    'L': 1.946427e-01,         # Length [m]
    'ux': 1.596104e+00,         # Reference strain x
    'uy': -2.125766e+00,        # Reference strain y
    'k1': 8.030672e+01, 'k2': 7.008507e+01, 'k3': 7.034977e+01,  # Muscle spring constants
    'c1': 1.368191e-02, 'c2': 5.761903e-03, 'c3': 6.032072e-03,     # Pressure coefficients
    'b1': 3.326281e-04, 'b2': 4.922282e-04, 'b3': 4.648332e-04,  # Quadratic pressure coefficients
    'l01': 9.148767e-02, 'l02': 8.396637e-02, 'l03': 8.339843e-02,  # Muscle original lengths
    'mz0': 2.867236e-02,         # Tip moment offset
    'A': -5.864442e-05           # Tip moment pressure coefficient
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

def simulate_sinusoidal_sequence(params, pressure_sequence, time_sequence, save_to_csv=True, csv_filename='simulation_results.csv', transition_steps=20):
    """
    Simulate rod tip position for a sinusoidal pressure sequence
    Uses the same structure as V1 but with sinusoidal pressure input
    
    Args:
        params: Dictionary of rod and muscle parameters
        pressure_sequence: Pressure sequence array with shape (n_steps, 3)
        time_sequence: Time sequence array with shape (n_steps,)
        save_to_csv: Whether to save results to CSV
        csv_filename: Output CSV filename
        transition_steps: Number of steps for pressure and xi_star ramping (used for all steps)
        
    Returns:
        tuple: (time_sequence, tip_positions, pressure_sequence)
    """
    print(f"Starting sinusoidal simulation with {len(pressure_sequence)} time steps...")
    print(f"Time step: {time_sequence[1]-time_sequence[0]:.3f}s, Total simulation time: {time_sequence[-1]:.2f}s")
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
    tip_positions = np.zeros((len(pressure_sequence), 2))  # [x, y] positions
    
    # Pre-compute zero-pressure configuration for initial guess with more steps for convergence
    print("Computing zero-pressure configuration for initial guess...")
    print(f"Using target xi_star: ux={params['ux']:.6f}, uy={params['uy']:.6f}")
    zero_pressures = jnp.array([0., 0., 0.])
    zero_mz = params['A'] * jnp.sum(zero_pressures) + params['mz0']
    
    # Get zero-pressure configuration using JAX simulation with gradual transition
    zero_tip_x, zero_tip_y, zero_X_final, zero_tau_final = simulate_rod_tip_jax_optimized_with_initial_guess(
        E=params['E'],
        rho=params['rho'],
        L=params['L'],
        xi_star=target_xi_star,
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
        target_xi_star=target_xi_star,
        n_steps=transition_steps
    )
    
    # Extract zero-pressure initial guesses
    zero_xi0_guess = jnp.array([0., 0., 0., 0., 0., 1.])  # Default initial strains
    zero_tau_guess = zero_tau_final  # Use the computed tensions from zero-pressure simulation
    
    print(f"Zero-pressure configuration computed. Initial tensions: {zero_tau_guess}")
    
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
    
    for i, target_pressures in enumerate(pressure_sequence):
        try:
            if i % 50 == 0:  # Progress update every 50 steps
                elapsed_time = time.time() - simulation_start_time
                print(f"Progress: {i}/{len(pressure_sequence)} steps ({i/len(pressure_sequence)*100:.1f}%) - Elapsed: {elapsed_time:.1f}s")
            
            # Handle xi_star ramping differently for first step vs subsequent steps
            if is_first_step:
                # First step: ramp from straight to target xi_star
                initial_xi_star = jnp.tile(jnp.array([0., 0., 0., 0., 0., 1.]), (n, 1))  # Straight configuration
                print(f"\nSimulating step {i+1}/{len(pressure_sequence)} (FIRST STEP):")
                print(f"  From pressure: {current_pressure}")
                print(f"  To target pressure: {target_pressures}")
                print(f"  From xi_star: [0,0,0,0,0,1] to target: [{params['ux']:.6f},{params['uy']:.6f},0,0,0,1]")
                print(f"  Using n_steps = {transition_steps} for initial convergence")
                is_first_step = False  # Mark that first step is done
            else:
                # Subsequent steps: stay at target xi_star (no xi_star ramping)
                initial_xi_star = target_xi_star.copy()  # Stay at target xi_star
                print(f"\nSimulating step {i+1}/{len(pressure_sequence)}:")
                print(f"  From pressure: {current_pressure}")
                print(f"  To target pressure: {target_pressures}")
                print(f"  Xi_star: [{params['ux']:.6f},{params['uy']:.6f},0,0,0,1] (constant)")
                print(f"  Using n_steps = {transition_steps} (previous solution as initial guess)")
            
            # For each target pressure, ramp from current pressure to target pressure
            # This maintains continuity by using previous solution as starting point
            
            # Compute tip moment mz for this target pressure
            mz = params['A'] * jnp.sum(target_pressures) + params['mz0']
            
            # Simulate tip position with gradual transition for better continuity
            tip_x, tip_y, X_final, tau_final = simulate_rod_tip_jax_optimized_with_initial_guess(
                E=params['E'],
                rho=params['rho'],
                L=params['L'],
                xi_star=target_xi_star,
                muscle_params=muscle_params,
                tip_world_wrench=tip_world_wrench,
                target_pressures=jnp.array(target_pressures),
                n=n,
                r=r,
                G=G,
                f_t=jnp.zeros(3),
                l_t=jnp.array([0., 0., mz]),
                initial_xi0=current_xi0_guess,
                initial_taus=current_tau_guess,
                target_xi_star=target_xi_star,
                n_steps=transition_steps
            )
            
            # Store results
            tip_positions[i, 0] = float(tip_x)
            tip_positions[i, 1] = float(tip_y)
            
            # Update current conditions for next iteration (maintain continuity)
            current_pressure = jnp.array(target_pressures)  # Current becomes the target we just reached
            # Note: current_xi_star stays at target_xi_star after first step
            
            # Update guesses for next iteration using the final state from current simulation
            # This ensures trajectory continuity by using previous solution as initial guess
            # Extract strains from the final shape for next iteration
            current_xi0_guess = X_final[0, 12:18]  # Extract strains from first node of final shape
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
    print(f"Successful simulations: {successful_simulations}/{len(pressure_sequence)} ({successful_simulations/len(pressure_sequence)*100:.1f}%)")
    
    # Check trajectory continuity
    continuity_analysis = analyze_trajectory_continuity(tip_positions, time_sequence)
    print(f"Trajectory continuity analysis: {continuity_analysis}")
    
    # Save results to CSV if requested
    if save_to_csv:
        save_simulation_results_to_csv(time_sequence, pressure_sequence, tip_positions, csv_filename)
    
    return time_sequence, tip_positions, pressure_sequence

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

def plot_sinusoidal_simulation_results(time_sequence, pressure_sequence, tip_positions, save_plot=True, plot_filename='sinusoidal_simulation_plot.png'):
    """
    Plot sinusoidal simulation results
    
    Args:
        time_sequence: Array of time points
        pressure_sequence: Array of pressure values (n_steps, 3)
        tip_positions: Array of tip positions (n_steps, 2)
        save_plot: Whether to save the plot
        plot_filename: Output plot filename
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Sinusoidal pressures over time
    axes[0, 0].plot(time_sequence, pressure_sequence[:, 0], 'r-', label='P1', linewidth=2)
    axes[0, 0].plot(time_sequence, pressure_sequence[:, 1], 'g-', label='P2', linewidth=2)
    axes[0, 0].plot(time_sequence, pressure_sequence[:, 2], 'b-', label='P3', linewidth=2)
    axes[0, 0].set_xlabel('Time [s]')
    axes[0, 0].set_ylabel('Pressure [kPa]')
    axes[0, 0].set_title('Sinusoidal Pressure Input Sequence')
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
    
    # Don't show the plot
    plt.close()

def load_average_trajectory_from_circle_data():
    """
    Load and calculate the average trajectory from circle validation data
    Returns the average trajectory coordinates (x, y)
    """
    
    # File names (only first 4 files)
    file_names = [f'collected_data_circle{i}.csv' for i in range(1, 5)]
    
    # Store all trajectories for averaging
    all_trajectories = []
    
    for file_name in file_names:
        try:
            print(f"Loading {file_name} for average trajectory calculation...")
            
            # Read CSV file
            df = pd.read_csv(file_name)
            
            # Calculate trajectory coordinates: (x,y) = (z2-z1, x2-x1)
            x_coords = df['z2'] - df['z1']
            y_coords = df['x2'] - df['x1']
            
            # Each circle has 2000 data points, so total data points = 8000 (4 circles)
            # We want to use only the last 3 circles (data points 2000-7999)
            start_idx = 2000  # Skip first circle
            end_idx = len(df) - 1  # Include all remaining data
            
            # Extract the last 3 circles (6000 data points)
            x_trajectory = x_coords[start_idx:end_idx]
            y_trajectory = y_coords[start_idx:end_idx]
            
            # Store trajectory for averaging
            all_trajectories.append((x_trajectory, y_trajectory))
            
        except FileNotFoundError:
            print(f"Warning: {file_name} not found, skipping...")
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
    
    # Calculate average trajectory
    if all_trajectories:
        print(f"Calculating average trajectory from {len(all_trajectories)} files...")
        average_trajectory = calculate_average_trajectory(all_trajectories)
        return average_trajectory
    else:
        print("No valid trajectory data found for averaging")
        return None

def calculate_average_trajectory(trajectories):
    """
    Calculate the average trajectory from multiple trajectories
    Uses interpolation to handle different trajectory lengths
    """
    if not trajectories:
        return None
    
    # Find the minimum length among all trajectories
    min_length = min(len(traj[0]) for traj in trajectories)
    print(f"Minimum trajectory length: {min_length}")
    
    # Normalize all trajectories to the same length
    normalized_trajectories = []
    
    for x_traj, y_traj in trajectories:
        # Create normalized parameter t from 0 to 1
        t_original = np.linspace(0, 1, len(x_traj))
        t_normalized = np.linspace(0, 1, min_length)
        
        # Interpolate x and y coordinates
        x_interp = interp1d(t_original, x_traj, kind='linear', bounds_error=False, fill_value='extrapolate')
        y_interp = interp1d(t_original, y_traj, kind='linear', bounds_error=False, fill_value='extrapolate')
        
        x_normalized = x_interp(t_normalized)
        y_normalized = y_interp(t_normalized)
        
        normalized_trajectories.append((x_normalized, y_normalized))
    
    # Calculate average
    x_avg = np.mean([traj[0] for traj in normalized_trajectories], axis=0)
    y_avg = np.mean([traj[1] for traj in normalized_trajectories], axis=0)
    
    return x_avg, y_avg

def calculate_trajectory_rmse(sim_trajectory, exp_trajectory):
    """
    Calculate RMSE between simulated and experimental trajectories
    Interpolates experimental trajectory to match simulation trajectory length
    
    Args:
        sim_trajectory: Tuple of (x_sim, y_sim) from simulation
        exp_trajectory: Tuple of (x_exp, y_exp) from experimental data
        
    Returns:
        dict: RMSE analysis results and interpolated experimental trajectory
    """
    x_sim, y_sim = sim_trajectory
    x_exp, y_exp = exp_trajectory
    
    # Find valid points in simulation trajectory
    sim_valid_mask = ~(np.isnan(x_sim) | np.isnan(y_sim))
    x_sim_valid = x_sim[sim_valid_mask]
    y_sim_valid = y_sim[sim_valid_mask]
    
    if len(x_sim_valid) == 0:
        return {
            "rmse": np.nan, "rmse_x": np.nan, "rmse_y": np.nan, "n_points": 0, 
            "status": "No valid simulation points",
            "x_exp_interp": None, "y_exp_interp": None
        }
    
    # Interpolate experimental trajectory to match simulation trajectory length
    sim_length = len(x_sim_valid)
    exp_length = len(x_exp)
    
    print(f"Interpolating experimental trajectory from {exp_length} points to {sim_length} points")
    
    # Create normalized parameter t from 0 to 1 for both trajectories
    t_sim = np.linspace(0, 1, sim_length)
    t_exp = np.linspace(0, 1, exp_length)
    
    # Interpolate experimental trajectory to simulation length
    x_exp_interp = interp1d(t_exp, x_exp, kind='linear', bounds_error=False, fill_value='extrapolate')
    y_exp_interp = interp1d(t_exp, y_exp, kind='linear', bounds_error=False, fill_value='extrapolate')
    
    # Get interpolated experimental coordinates at simulation points
    x_exp_interpolated = x_exp_interp(t_sim)
    y_exp_interpolated = y_exp_interp(t_sim)
    
    # Calculate RMSE for x and y components
    rmse_x = np.sqrt(np.mean((x_sim_valid - x_exp_interpolated)**2))
    rmse_y = np.sqrt(np.mean((y_sim_valid - y_exp_interpolated)**2))
    
    # Calculate overall RMSE (Euclidean distance)
    distances = np.sqrt((x_sim_valid - x_exp_interpolated)**2 + (y_sim_valid - y_exp_interpolated)**2)
    rmse = np.sqrt(np.mean(distances**2))
    
    return {
        "rmse": float(rmse),
        "rmse_x": float(rmse_x),
        "rmse_y": float(rmse_y),
        "n_points": sim_length,
        "status": "Valid comparison",
        "max_distance": float(np.max(distances)),
        "mean_distance": float(np.mean(distances)),
        "std_distance": float(np.std(distances)),
        "x_exp_interp": x_exp_interpolated,
        "y_exp_interp": y_exp_interpolated,
        "x_sim_valid": x_sim_valid,
        "y_sim_valid": y_sim_valid
    }

def plot_trajectory_comparison(tip_positions, average_trajectory=None, rmse_results=None, save_plot=True, plot_filename='trajectory_comparison.png'):
    """
    Plot sinusoidal trajectory compared with average trajectory from circle data
    Includes start point markers and interpolated experimental trajectory
    
    Args:
        tip_positions: Array of tip positions from sinusoidal simulation (n_steps, 2)
        average_trajectory: Tuple of (x_avg, y_avg) from circle data
        rmse_results: Results from calculate_trajectory_rmse function
        save_plot: Whether to save the plot
        plot_filename: Output plot filename
    """
    plt.figure(figsize=(12, 10))
    
    # Plot sinusoidal trajectory
    valid_mask = ~(np.isnan(tip_positions[:, 0]) | np.isnan(tip_positions[:, 1]))
    if np.any(valid_mask):
        plt.plot(tip_positions[valid_mask, 0], tip_positions[valid_mask, 1], 
                'b-', linewidth=3, label='Sinusoidal Simulation (Set 4)', alpha=0.8)
        plt.plot(tip_positions[valid_mask, 0], tip_positions[valid_mask, 1], 
                'bo', markersize=4, alpha=0.6)
        
        # Mark start point of simulation trajectory with star
        plt.plot(tip_positions[valid_mask, 0][0], tip_positions[valid_mask, 1][0], 
                'b*', markersize=15, markeredgewidth=2, markeredgecolor='black', 
                label='Simulation Start', alpha=1.0)
    
    # Plot experimental trajectory (original and interpolated)
    if average_trajectory is not None:
        x_avg, y_avg = average_trajectory
        
        # Plot original experimental trajectory (thin line)
        plt.plot(x_avg, y_avg, 'r-', linewidth=1, alpha=0.5, label='Original Experimental')
        
        # Plot interpolated experimental trajectory if available
        if rmse_results is not None and rmse_results['x_exp_interp'] is not None:
            plt.plot(rmse_results['x_exp_interp'], rmse_results['y_exp_interp'], 
                    'r-', linewidth=3, label='Interpolated Experimental', alpha=0.8)
            plt.plot(rmse_results['x_exp_interp'], rmse_results['y_exp_interp'], 
                    'ro', markersize=4, alpha=0.6)
            
            # Mark start point of interpolated experimental trajectory with star
            plt.plot(rmse_results['x_exp_interp'][0], rmse_results['y_exp_interp'][0], 
                    'r*', markersize=15, markeredgewidth=2, markeredgecolor='black', 
                    label='Experimental Start', alpha=1.0)
        else:
            # Mark start point of original experimental trajectory with star
            plt.plot(x_avg[0], y_avg[0], 'r*', markersize=15, markeredgewidth=2, markeredgecolor='black', 
                    label='Experimental Start', alpha=1.0)
    
    # Customize the plot
    plt.xlabel('X Position [m]', fontsize=14)
    plt.ylabel('Y Position [m]', fontsize=14)
    plt.title('Trajectory Comparison: Sinusoidal vs Circle Data', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')  # Equal aspect ratio
    
    # Add RMSE information to plot if available
    if rmse_results is not None and rmse_results['status'] == 'Valid comparison':
        rmse_text = f'RMSE: {rmse_results["rmse"]*1000:.2f} mm\n'
        rmse_text += f'RMSE X: {rmse_results["rmse_x"]*1000:.2f} mm\n'
        rmse_text += f'RMSE Y: {rmse_results["rmse_y"]*1000:.2f} mm'
        plt.text(0.02, 0.98, rmse_text, transform=plt.gca().transAxes, 
                fontsize=12, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add some padding around the plot
    plt.tight_layout()
    
    if save_plot:
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Trajectory comparison plot saved to {plot_filename}")
    
    # Don't show the plot
    plt.close()

def save_sinusoidal_trajectory_to_csv(tip_positions, filename='circle_set.csv'):
    """
    Save sinusoidal trajectory to CSV file in the format expected by circle validation
    
    Args:
        tip_positions: Array of tip positions (n_steps, 2)
        filename: Output CSV filename
    """
    # Filter out NaN values
    valid_mask = ~(np.isnan(tip_positions[:, 0]) | np.isnan(tip_positions[:, 1]))
    valid_positions = tip_positions[valid_mask]
    
    if len(valid_positions) == 0:
        print("No valid positions to save")
        return
    
    # Create DataFrame with x, y columns
    df = pd.DataFrame({
        'x': valid_positions[:, 0],
        'y': valid_positions[:, 1]
    })
    
    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"Sinusoidal trajectory saved to {filename}")
    print(f"CSV contains {len(df)} data points")
    
    # Print summary statistics
    print(f"Trajectory range: x=[{valid_positions[:, 0].min():.6f}, {valid_positions[:, 0].max():.6f}], "
          f"y=[{valid_positions[:, 1].min():.6f}, {valid_positions[:, 1].max():.6f}]")

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
    periods = 1  # Number of complete cycles
    frequency = 0.05  # Frequency for sinusoidal motion [Hz]
    dt = 0.1  # Time step [s]
    total_time = periods / frequency  # Total simulation time [s]
    pressure_max = 65.0  # Maximum pressure [kPa]
    
    print(f"\nSinusoidal Motion Simulation Parameters:")
    print(f"  Number of periods: {periods}")
    print(f"  Frequency: {frequency} Hz")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Time step (dt): {dt}s")
    print(f"  Maximum pressure: {pressure_max} kPa")
    print(f"  Phase difference: 2π/3 (120°) between all three muscles")
    print(f"  All three muscles active with sinusoidal signals")
    
    # Generate sinusoidal pressure sequence
    print(f"\nGenerating sinusoidal pressure sequence...")
    pressure_sequence, time_sequence = generate_sinusoidal_pressures(
        total_time=total_time, 
        dt=dt, 
        pressure_max=pressure_max,
        frequency=frequency
    )
    print(f"Generated pressure sequence with {len(pressure_sequence)} time steps")
    print(f"Total simulation time: {time_sequence[-1]:.2f}s")
    
    # Print parameters being used
    print("\nSimulating with manual parameters:")
    for key, value in manual_params.items():
        print(f"  {key}: {value}")
    
    # Run simulation
    print("\nStarting sinusoidal motion simulation...")
    transition_steps = 100
    print(f"Using transition_steps = {transition_steps} for all pressure and xi_star ramping")
    
    time_sequence, tip_positions, pressure_sequence = simulate_sinusoidal_sequence(
        params=manual_params,
        pressure_sequence=pressure_sequence,
        time_sequence=time_sequence,
        save_to_csv=True,
        csv_filename='simulation_results.csv',
        transition_steps=transition_steps
    )
    
    # Load average trajectory from circle data and create comparison
    print("\nLoading average trajectory from circle validation data...")
    average_trajectory = load_average_trajectory_from_circle_data()
    
    # Calculate RMSE between simulated and experimental trajectories
    print("\nCalculating RMSE between simulated and experimental trajectories...")
    rmse_results = None
    if average_trajectory is not None:
        # Extract valid simulation trajectory
        valid_mask = ~(np.isnan(tip_positions[:, 0]) | np.isnan(tip_positions[:, 1]))
        sim_x = tip_positions[valid_mask, 0]
        sim_y = tip_positions[valid_mask, 1]
        
        # Calculate RMSE
        rmse_results = calculate_trajectory_rmse((sim_x, sim_y), average_trajectory)
        
        # Print RMSE results
        print("\n" + "="*60)
        print("TRAJECTORY RMSE ANALYSIS")
        print("="*60)
        print(f"Status: {rmse_results['status']}")
        print(f"Number of comparison points: {rmse_results['n_points']}")
        print(f"Overall RMSE: {rmse_results['rmse']*1000:.3f} mm")
        print(f"RMSE X-component: {rmse_results['rmse_x']*1000:.3f} mm")
        print(f"RMSE Y-component: {rmse_results['rmse_y']*1000:.3f} mm")
        print(f"Maximum distance: {rmse_results['max_distance']*1000:.3f} mm")
        print(f"Mean distance: {rmse_results['mean_distance']*1000:.3f} mm")
        print(f"Standard deviation of distances: {rmse_results['std_distance']*1000:.3f} mm")
        print("="*60)
    else:
        print("No experimental trajectory available for RMSE calculation")
    
    # Plot trajectory comparison
    print("\nCreating trajectory comparison plot...")
    plot_trajectory_comparison(tip_positions, average_trajectory, rmse_results,
                              save_plot=True, plot_filename='trajectory_comparison.png')
    
    end_time = time.time()
    print(f"\nTotal calculation time: {end_time - start_time:.2f} seconds")
    
    # Print final summary
    print("\n" + "="*60)
    print("SINUSOIDAL MOTION SIMULATION SUMMARY")
    print("="*60)
    print(f"Simulation approach: Sinusoidal pressure input with 2π/3 phase difference")
    print(f"Ramping steps: transition_steps = {transition_steps} for all steps")
    print(f"Continuity scheme: Each target uses previous solution as starting point")
    print(f"Xi_star scheme: First step ramps from straight to target, subsequent steps stay at target")
    print(f"Target xi_star: Settled configuration [{manual_params['ux']:.6f},{manual_params['uy']:.6f},0,0,0,1]")
    print(f"Simulation time steps: {len(pressure_sequence)}")
    print(f"Total simulation time: {time_sequence[-1]:.2f}s (one complete cycle)")
    print(f"Time step: {dt}s")
    print(f"Pressure range: [{pressure_sequence.min():.1f}, {pressure_sequence.max():.1f}] kPa")
    print(f"Frequency: {frequency} Hz")
    print(f"Phase difference: 2π/3 (120°) between all three muscles")
    print(f"All three muscles active with sinusoidal signals")
    
    valid_positions = tip_positions[~np.isnan(tip_positions[:, 0])]
    if len(valid_positions) > 0:
        print(f"Valid simulations: {len(valid_positions)}/{len(tip_positions)}")
        print(f"Tip position range: x=[{valid_positions[:, 0].min():.6f}, {valid_positions[:, 0].max():.6f}], "
              f"y=[{valid_positions[:, 1].min():.6f}, {valid_positions[:, 1].max():.6f}]")
        
        # Analyze trajectory continuity
        continuity_analysis = analyze_trajectory_continuity(tip_positions, time_sequence)
        print(f"Trajectory continuity: {continuity_analysis['status']}")
        print(f"Discontinuities: {continuity_analysis['discontinuities']}/{continuity_analysis['total_steps']-1}")
        print(f"Maximum jump: {continuity_analysis['max_jump']*1000:.2f} mm")
        print(f"Average step size: {continuity_analysis['avg_step_size']*1000:.2f} mm")
    else:
        print("No valid simulations found")
    
    # Print RMSE summary if available
    if average_trajectory is not None and len(valid_positions) > 0:
        rmse_results = calculate_trajectory_rmse((valid_positions[:, 0], valid_positions[:, 1]), average_trajectory)
        print(f"\nTrajectory RMSE Summary:")
        print(f"  Overall RMSE: {rmse_results['rmse']*1000:.3f} mm")
        print(f"  RMSE X-component: {rmse_results['rmse_x']*1000:.3f} mm")
        print(f"  RMSE Y-component: {rmse_results['rmse_y']*1000:.3f} mm")
        print(f"  Maximum distance: {rmse_results['max_distance']*1000:.3f} mm")
        print(f"  Mean distance: {rmse_results['mean_distance']*1000:.3f} mm")
    
    print(f"\nResults saved to: simulation_results.csv")
    print(f"Trajectory comparison saved to: trajectory_comparison.png")
    print("="*60) 