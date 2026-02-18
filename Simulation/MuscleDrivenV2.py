import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def hat(v):
    return np.array([[0, -v[2], v[1]], 
                     [v[2], 0, -v[0]], 
                     [-v[1], v[0], 0]])

class Muscle:
    def __init__(self, r_i, angle_i, pressure, L_orig, K, c, b):
        self.r_i = np.array([r_i * np.cos(angle_i), r_i * np.sin(angle_i), 0])
        self.pressure = pressure
        self.L_orig = L_orig  # Original length
        self.K = K  # Spring constant
        self.c = c  # Pressure coefficient
        self.b = b  # Quadratic pressure coefficient
        self.tau = 0.0  # Will be solved
        self.length = 0.0  # Will be calculated

    def compute_length(self, X):
        """Calculate muscle length along the backbone shape X (n×18)"""
        try:
            n = X.shape[0]
            pts = np.zeros((n,3))
            for i in range(n):
                p_i = X[i,:3]
                R_i = X[i,3:12].reshape(3,3)
                pts[i] = p_i + R_i @ self.r_i
            segs = np.diff(pts,axis=0)
            lengths = np.linalg.norm(segs,axis=1)
            if np.any(np.isnan(lengths)):
                print("Warning: NaN detected in length calculation")
                return self.length  # Return previous valid length
            self.length = np.sum(lengths)
            return self.length
        except Exception as e:
            print(f"Error in length calculation: {e}")
            return self.length  # Return previous valid length

class Backbone:
    def __init__(self, E, r, G, n, L, rho, xi0_star=None):
        self.A = np.pi * r ** 2
        self.Ix = np.pi * r ** 4 / 4
        self.J = 2 * self.Ix
        self.Kbt = np.diag([E*self.Ix, E*self.Ix, G*self.J])
        self.Kse = np.diag([G*self.A, G*self.A, E*self.A])
        self.n, self.L, self.ds = n, L, L/(n-1)
        self.rho = rho  # Density for gravity calculation

        p_init = np.linspace([0,0,0], [0,0,self.L], n)
        R_init = np.tile(np.eye(3).reshape(1,9), (n,1))
        if xi0_star is None:
            xi0_star = np.tile([0, 0, 0, 0, 0, 1], (n, 1))
        self.xi_star = xi0_star
        u_init = np.tile(self.xi_star[0,:3], (n,1))
        v_init = np.tile(self.xi_star[0,3:], (n,1))

        self.X_init = np.hstack([p_init, R_init, u_init, v_init])

    def cosserat_muscle_ode(self, x, muscles, W_ext, xi_ref, f_t=None, l_t=None):
        R = x[3:12].reshape(3,3)
        u = x[12:15]
        v = x[15:18]

        A = np.zeros((3,3))
        B = np.zeros((3,3))
        G = np.zeros((3,3))
        H = np.zeros((3,3))
        a = np.zeros(3)
        b = np.zeros(3)

        for muscle in muscles:
            pb_si = np.cross(u, muscle.r_i) + v
            norm_pb_si = np.linalg.norm(pb_si)
            Ai = -muscle.tau * hat(pb_si) @ hat(pb_si) / norm_pb_si**3
            Bi = hat(muscle.r_i) @ Ai
            ai = Ai @ (np.cross(u, pb_si))
            bi = np.cross(muscle.r_i, ai)

            A += Ai
            B += Bi
            G -= Ai @ hat(muscle.r_i)
            H -= Bi @ hat(muscle.r_i)
            a += ai
            b += bi

        # baseline strains from reference xi
        u_star = xi_ref[:3]
        v_star = xi_ref[3:]

        # internal stresses
        nb = self.Kse @ (v - v_star)
        mb = self.Kbt @ (u - u_star)

        # external wrench parts (world frame)
        f_e = W_ext[3:]
        l_e = W_ext[:3]
        # tip wrench in body frame (if provided)
        if f_t is None:
            f_t = np.zeros(3)
        if l_t is None:
            l_t = np.zeros(3)

        # corrected RHS
        d = -(hat(u) @ nb) - R.T @ f_e - a - f_t
        c = -(hat(u) @ mb) - (hat(v) @ nb) - R.T @ l_e - b - l_t

        K_total = np.block([[self.Kse + A, G],
                            [B, self.Kbt + H]])
        vu_s = np.linalg.solve(K_total, np.hstack([d, c]))
        v_s, u_s = vu_s[:3], vu_s[3:]

        p_s = R @ v
        R_s = R @ hat(u)
        return np.hstack([p_s, R_s.flatten(), u_s, v_s])

    def RK4_step(self, x, muscles, W_ext, ds, xi_ref, f_t=None, l_t=None):
        k1 = self.cosserat_muscle_ode(x, muscles, W_ext, xi_ref, f_t, l_t)
        k2 = self.cosserat_muscle_ode(x + ds*k1/2, muscles, W_ext, xi_ref, f_t, l_t)
        k3 = self.cosserat_muscle_ode(x + ds*k2/2, muscles, W_ext, xi_ref, f_t, l_t)
        k4 = self.cosserat_muscle_ode(x + ds*k3, muscles, W_ext, xi_ref, f_t, l_t)

        return x + ds/6*(k1 + 2*k2 + 2*k3 + k4)

    def shooting(self, vars, muscles, W_tip, f_t=None, l_t=None):
        """Shooting method for both xi0 and muscle tensions"""
        try:
            # Split variables into xi0 and taus
            xi0 = vars[:6]
            taus = vars[6:]
            
            # Set tensions for muscles
            for muscle, tau in zip(muscles, taus):
                muscle.tau = tau
            
            # Initialize shape
            X = np.zeros((self.n,18))
            X[0] = self.X_init[0].copy()
            X[0,12:18] = xi0
            
            # Integrate shape
            for i in range(1,self.n):
                W_i = np.zeros(6) if i < self.n-1 else W_tip
                xi_ref = self.xi_star[i]
                # Only apply f_t, l_t at the tip
                if i == self.n-1:
                    X[i] = self.RK4_step(X[i-1], muscles, W_i, self.ds, xi_ref, f_t, l_t)
                else:
                    X[i] = self.RK4_step(X[i-1], muscles, W_i, self.ds, xi_ref)
                if np.any(np.isnan(X[i])):
                    print(f"Warning: NaN detected in integration at step {i}")
                    return np.ones_like(vars) * 1e6  # Return large residuals to help solver
            
            # Get final strains
            xi_end = X[-1,12:18]
            
            # Build tip wrench
            W_tip_calc = W_tip.copy()
            for muscle in muscles:
                pb_s = np.cross(X[-1,12:15], muscle.r_i) + X[-1,15:18]
                pb_s_norm = np.linalg.norm(pb_s)
                if pb_s_norm < 1e-10:  # Check for singularity
                    print("Warning: Singularity detected in tip wrench calculation")
                    return np.ones_like(vars) * 1e6
                f_tip = -muscle.tau * pb_s / pb_s_norm
                moment_tip = np.cross(muscle.r_i, f_tip)
                W_tip_calc += np.concatenate([moment_tip, f_tip])
            
            # Tip BC residuals (use xi0_star)
            moment_residual = self.Kbt @ (xi_end[:3] - self.xi_star[-1][:3]) - W_tip_calc[:3]
            force_residual = self.Kse @ (xi_end[3:] - self.xi_star[-1][3:]) - W_tip_calc[3:]
            # If f_t or l_t are nonzero, add them to the residuals (transform to body frame)
            if f_t is not None:
                R_tip = X[-1,3:12].reshape(3,3)
                force_residual -= R_tip @ f_t
            if l_t is not None:
                R_tip = X[-1,3:12].reshape(3,3)
                moment_residual -= R_tip @ l_t
            # Muscle model residuals
            muscle_residuals = []
            for muscle, tau in zip(muscles, taus):
                L = muscle.compute_length(X)
                if np.isnan(L):
                    print(f"Warning: NaN detected in muscle length calculation")
                    return np.ones_like(vars) * 1e6
                muscle_residuals.append(tau - (muscle.K*(L - muscle.L_orig) + muscle.c*muscle.pressure + muscle.b*muscle.pressure**2))
            
            return np.hstack([moment_residual, force_residual, muscle_residuals])
        except Exception as e:
            print(f"Error in shooting method: {e}")
            return np.ones_like(vars) * 1e6

    def solve(self, muscles, W_dist, f_t=None, l_t=None):
        """Solve for both xi0 and muscle tensions"""
        try:
            # Initial guess for xi0 and taus
            tau0 = np.array([muscle.K*(muscle.compute_length(self.X_init) - muscle.L_orig) + muscle.c*muscle.pressure + muscle.b*muscle.pressure**2 for muscle in muscles])
            vars0 = np.hstack([self.xi_star[0], tau0])
            
            # Get tip wrench from W_dist
            W_tip = W_dist[-1].copy()
            
            # Solve shooting method with maximum iterations
            sol = fsolve(lambda v: self.shooting(v, muscles, W_tip, f_t, l_t), vars0, maxfev=1000)
            
            # Extract solution
            xi0 = sol[:6]
            sol_tau = sol[6:]
            
            # Set final tensions
            for muscle, tau in zip(muscles, sol_tau):
                muscle.tau = tau
            
            # Compute final shape
            X = np.zeros((self.n,18))
            X[0] = self.X_init[0].copy()
            X[0,12:18] = xi0
            
            for i in range(1,self.n):
                W_i = W_dist[i].copy()
                xi_ref = self.xi_star[i]
                if i == self.n-1:
                    X[i] = self.RK4_step(X[i-1], muscles, W_i, self.ds, xi_ref, f_t, l_t)
                else:
                    X[i] = self.RK4_step(X[i-1], muscles, W_i, self.ds, xi_ref)
            
            return X
        except Exception as e:
            print(f"Error in solve method: {e}")
            return self.X_init  # Return initial configuration if solve fails

def visualize(X_final, muscles, num_plates=3, plate_radius=0.05):
    """
    X_final : (n×18) array of [p, R.flatten(), u, v]
    muscles : list of Muscle objects with .r_i offsets
    """
    # 1) build the reflection operator about the XY plane
    F = np.diag([1, 1, -1])

    # 2) reflect the centreline Z
    X = X_final.copy()
    X[:, 2] *= -1

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # plot the rod centreline
    ax.plot(X[:,0], X[:,1], X[:,2], 'b-', lw=2, label='Rod')
    ax.scatter(X[-1,0], X[-1,1], X[-1,2], c='r', s=50, label='Tip')

    # annotate the tip (x,y) in the 3D view
    xt, yt, zt = X[-1, :3]
    ax.text(xt, yt, zt,
            f"({xt:.3f}, {yt:.3f})",
            color='black', fontsize=10,
            horizontalalignment='left',
            verticalalignment='bottom')

    # plot each plate as a filled 3D circle
    thetas = np.linspace(0, 2*np.pi, 64)
    circle2D = np.stack([plate_radius*np.cos(thetas),
                        plate_radius*np.sin(thetas),
                        np.zeros_like(thetas)], axis=1)  # (64×3)

    for idx in np.linspace(0, len(X)-1, num_plates, dtype=int):
        p = X[idx, :3]
        R_orig = X_final[idx, 3:12].reshape(3,3)
        # mirror the local frame on both sides
        R_hang = F @ R_orig @ F

        # transform the unit-circle points and translate
        plate_pts = (R_hang @ circle2D.T).T + p  # (64×3)

        # make a Poly3DCollection so it's filled
        poly = Poly3DCollection([plate_pts], facecolor='gray', alpha=0.5)
        ax.add_collection3d(poly)

    # plot each muscle
    for muscle in muscles:
        pts = []
        for i in range(X.shape[0]):
            p_i = X[i, :3]
            R_orig = X_final[i, 3:12].reshape(3,3)
            R_hang = F @ R_orig @ F
            pts.append(p_i + R_hang @ (muscle.r_i * 1.5))
        pts = np.vstack(pts)
        ax.plot(pts[:,0], pts[:,1], pts[:,2], 'g--', lw=1)

    ax.set(xlim=[-0.2,0.2],
           ylim=[-0.2,0.2],
           zlim=[-0.20, 0.0],
           xlabel='X [m]',
           ylabel='Y [m]',
           zlabel='Z [m]')
    ax.legend()
    plt.title('Cosserat Rod Hanging from XY-Plane')
    plt.tight_layout()
    plt.show()

def ramp_pressures_and_solve(rod, muscles, target_pressures, n_steps, W_dist, f_t=None, l_t=None):
    """
    Ramps up the muscle pressures in n_steps, using previous solution as initial guess.
    Returns the final shape and muscle tensions.
    """
    pressures = np.array([muscle.pressure for muscle in muscles])
    target_pressures = np.array(target_pressures)
    pressure_steps = np.linspace(pressures, target_pressures, n_steps)

    # Initial guesses
    tau_guess = np.array([muscle.K*(muscle.compute_length(rod.X_init) - muscle.L_orig) + muscle.c*muscle.pressure + muscle.b*muscle.pressure**2 for muscle in muscles])
    xi0_guess = rod.xi_star[0].copy()

    for step, p_vec in enumerate(pressure_steps):
        for i, muscle in enumerate(muscles):
            muscle.pressure = p_vec[i]
        # Build initial guess vector
        vars0 = np.hstack([xi0_guess, tau_guess])
        # Get tip wrench from W_dist
        W_tip = W_dist[-1].copy()
        # Solve
        sol = fsolve(rod.shooting, vars0, args=(muscles, W_tip, f_t, l_t), maxfev=1000)
        xi0_guess = sol[:6]
        tau_guess = sol[6:]
        # Set final tensions for next step
        for muscle, tau in zip(muscles, tau_guess):
            muscle.tau = tau
        # Optionally print progress
        # print(f"Step {step+1}/{n_steps}: pressures = {p_vec}, tau = {tau_guess}")
    # After ramping, compute final shape
    X = np.zeros((rod.n,18))
    X[0] = rod.X_init[0].copy()
    X[0,12:18] = xi0_guess
    for i in range(1, rod.n):
        W_i = W_dist[i].copy()
        xi_ref = rod.xi_star[i]
        X[i] = rod.RK4_step(X[i-1], muscles, W_i, rod.ds, xi_ref, f_t, l_t)
    # Update muscle lengths
    for muscle in muscles:
        muscle.compute_length(X)
    return X, tau_guess

if __name__ == "__main__":
    # Parameters
    E, G, rho = 11.469e9, 11.469e9/(2*1.3), 8.3e5
    r, n = 0.001, 51
    L = 0.160359  # Optimized length
    # User can set xi0_star here:
    xi0_star = np.tile([0, 0, 0, 0, 0, 1], (n, 1))
    rod = Backbone(E, r, G, n, L, rho, xi0_star=xi0_star)

    # Muscle setup with new distribution (3 muscles, 2pi/3 apart)
    pressures = [0, 0, 0]  # Start from zero
    L_orig_values = [0.2/2, 0.2/2, 0.2/2]  # Original lengths [m]
    K_values = [108.0, 108.0, 108.0]  # Spring constants [N/m]
    c_values = [0.12, 0.12, 0.12]  # Pressure coefficients [N/Pa]
    b_values = [0.001, 0.001, 0.001]  # Quadratic pressure coefficients [N/Pa^2]
    muscles = [Muscle(0.02, 2*np.pi*i/3, p, L_orig, K, c, b) 
              for i, (p, L_orig, K, c, b) in enumerate(zip(pressures, L_orig_values, K_values, c_values, b_values))]

    # Set tip force and gravity distribution
    W_dist = np.zeros((n, 6))
    # Add gravity force (in -z direction)
    g = 9.81  # gravity [m/s^2]
    mass_per_length = np.pi * r**2 * rho  # [kg/m]
    f_gravity = mass_per_length * L * g / n  # distribute total force across n points
    W_dist[:, 5] += f_gravity  # apply gravity force in z direction to all points

    # Add tip wrench (world frame)
    W_dist[-1] += np.array([0, 0, 0, 0.1, 0.1, 0])

    # Tip wrench in body frame (at tip)
    f_t = np.array([0, 0, 0])  # body frame force at tip
    l_t = np.array([0, 0, 0.1])    # body frame moment at tip

    # Ramping pressures to [0.1, 0, 0] in 10 steps
    target_pressures = [40, 0, 0]
    n_steps = 40
    X_final, tau_final = ramp_pressures_and_solve(rod, muscles, target_pressures, n_steps, W_dist, f_t=f_t, l_t=l_t)

    # Print results
    print("\nMuscle tensions and lengths:")
    for i, muscle in enumerate(muscles):
        print(f"Muscle {i+1}: τ = {muscle.tau:.2f} N, L = {muscle.length:.3f} m")
    # Print tip position
    print(f"Tip position: x = {X_final[-1,0]:.4f}, y = {X_final[-1,1]:.4f}")

    # Visualization
    visualize(X_final, muscles) 