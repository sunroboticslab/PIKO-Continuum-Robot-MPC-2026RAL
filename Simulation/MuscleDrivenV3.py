import numpy as np
from MuscleDrivenV2 import Backbone, Muscle, ramp_pressures_and_solve, visualize

def simulate_rod_tip(E, rho, L, xi_star, muscle_params, tip_world_wrench, target_pressures, n=51, r=0.001, G=None, g=9.81, n_steps=20, f_t=None, l_t=None):
    """
    Simulate Cosserat rod and return tip position (x, y).
    E, rho, L: rod parameters
    xi_star: (n,6) reference strain array
    muscle_params: list of dicts [{k, c, b, l0, r_i, angle_i} for each muscle]
    tip_world_wrench: (6,) world frame [lwx, lwy, lwz, fwx, fwy, fwz]
    target_pressures: list of muscle pressures
    n: number of points
    r: rod radius
    G: shear modulus (if None, computed as E/(2*1.3))
    g: gravity
    n_steps: ramping steps
    f_t: tip force in body frame (3,)
    l_t: tip moment in body frame (3,)
    """
    if G is None:
        G = E/(2*1.3)
    rod = Backbone(E, r, G, n, L, rho, xi0_star=xi_star)
    # Setup muscles
    muscles = [Muscle(m['r_i'], m['angle_i'], 0, m['l0'], m['k'], m['c'], m['b']) for m in muscle_params]
    # Set up distributed gravity
    W_dist = np.zeros((n, 6))
    mass_per_length = np.pi * r**2 * rho
    f_gravity = mass_per_length * L * g / n
    W_dist[:, 5] = f_gravity
    W_dist[-1] = tip_world_wrench
    # Run simulation
    X_final, tau_final = ramp_pressures_and_solve(rod, muscles, target_pressures, n_steps, W_dist, f_t=f_t, l_t=l_t)
    tip_x, tip_y = X_final[-1, 0], X_final[-1, 1]
    return tip_x, tip_y, X_final, tau_final, muscles

if __name__ == "__main__":
    # Example parameters
    E = 11.469e9
    rho = 8.3e5
    L = 0.160359
    n = 51
    xi_star = np.tile([0, 0, 0, 0, 0, 1], (n, 1))
    # Define muscle parameters as separate lists
    L_orig_values = [0.2/2, 0.2/2, 0.2/2]  # Original lengths [m]
    K_values = [108.0, 108.0, 108.0]        # Spring constants [N/m]
    c_values = [0.12, 0.12, 0.12]           # Pressure coefficients [N/Pa]
    b_values = [0.001, 0.001, 0.001]        # Quadratic pressure coefficients [N/Pa^2]
    r_i_values = [0.02, 0.02, 0.02]
    angle_i_values = [0, 2*np.pi/3, 4*np.pi/3]
    muscle_params = [
        {'r_i': r_i, 'angle_i': angle_i, 'k': K, 'c': c, 'b': b, 'l0': l0}
        for r_i, angle_i, K, c, b, l0 in zip(r_i_values, angle_i_values, K_values, c_values, b_values, L_orig_values)
    ]
    # Tip moment as A * sum(pressures)
    A = 0.001  # Example area or scaling
    mz = 0.0
    target_pressures = [0, 40, 0]
    tip_world_wrench = np.array([0, 0, 0, 0, 0, 0])  # [lwx, lwy, lwz, fwx, fwy, fwz]
    # Body-frame tip wrench
    f_t = np.array([0, 0, 0])      # body frame force at tip
    l_t = np.array([0, 0, A * sum(target_pressures) + mz])    # body frame moment at tip

    tip_x, tip_y, X_final, tau_final, muscles = simulate_rod_tip(
        E, rho, L, xi_star, muscle_params, tip_world_wrench, target_pressures, n=n, f_t=f_t, l_t=l_t
    )
    print(f"Tip position: x = {tip_x:.4f}, y = {tip_y:.4f}")
    visualize(X_final, muscles) 