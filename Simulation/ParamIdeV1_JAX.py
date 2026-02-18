import jax
import jax.numpy as jnp
from jax import jit, vmap, grad, lax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import warnings
import time
from functools import partial
warnings.filterwarnings('ignore')

# Force CPU usage if GPU is not available to prevent RuntimeError
try:
    devices = jax.devices()
    gpu_devices = [d for d in devices if 'gpu' in str(d).lower() or 'cuda' in str(d).lower()]
    if not gpu_devices:
        print("No GPU devices found - forcing CPU usage")
        import os
        os.environ['JAX_PLATFORM_NAME'] = 'cpu'
        # Re-import jax after setting environment variable
        import jax
        import jax.numpy as jnp
        from jax import jit, vmap, grad, lax
    else:
        # Test if GPU is actually functional
        try:
            test_array = jnp.array([1.0, 2.0, 3.0])
            test_result = jnp.sum(test_array)
            print(f"GPU devices detected and functional: {gpu_devices}")
        except Exception as gpu_error:
            print(f"GPU devices detected but not functional: {gpu_error}")
            print("Forcing CPU usage")
            import os
            os.environ['JAX_PLATFORM_NAME'] = 'cpu'
            # Re-import jax after setting environment variable
            import jax
            import jax.numpy as jnp
            from jax import jit, vmap, grad, lax
except Exception as e:
    print(f"Warning: Could not check devices, using default configuration: {e}")

# Timing utilities for JAX compilation and execution
class JAXTimer:
    """Utility class to track JIT compilation and execution times"""
    
    def __init__(self):
        self.compilation_times = {}
        self.execution_times = {}
        self.total_compilation_time = 0.0
        self.total_execution_time = 0.0
    
    def time_compilation(self, func_name):
        """Decorator to time JIT compilation"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                # First call triggers compilation
                result = func(*args, **kwargs)
                compilation_time = time.time() - start_time
                
                # Store compilation time
                if func_name not in self.compilation_times:
                    self.compilation_times[func_name] = compilation_time
                    self.total_compilation_time += compilation_time
                    print(f"JIT compilation time for {func_name}: {compilation_time:.4f}s")
                
                return result
            return wrapper
        return decorator
    
    def time_execution(self, func_name):
        """Decorator to time execution (excluding compilation)"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # Store execution time
                if func_name not in self.execution_times:
                    self.execution_times[func_name] = 0.0
                self.execution_times[func_name] += execution_time
                self.total_execution_time += execution_time
                
                return result
            return wrapper
        return decorator
    
    def print_summary(self):
        """Print timing summary"""
        print("\n" + "="*60)
        print("JAX TIMING SUMMARY")
        print("="*60)
        print(f"Total compilation time: {self.total_compilation_time:.4f}s")
        print(f"Total execution time: {self.total_execution_time:.4f}s")
        print(f"Total time: {self.total_compilation_time + self.total_execution_time:.4f}s")
        
        if self.compilation_times:
            print("\nCompilation times:")
            for func_name, comp_time in self.compilation_times.items():
                print(f"  {func_name}: {comp_time:.4f}s")
        
        if self.execution_times:
            print("\nExecution times:")
            for func_name, exec_time in self.execution_times.items():
                print(f"  {func_name}: {exec_time:.4f}s")
        print("="*60)
    
    def save_timing_log(self, filename='jax_timing_log.txt'):
        """Save timing information to file"""
        with open(filename, 'w') as f:
            f.write("JAX TIMING LOG\n")
            f.write("="*40 + "\n")
            f.write(f"Total compilation time: {self.total_compilation_time:.4f}s\n")
            f.write(f"Total execution time: {self.total_execution_time:.4f}s\n")
            f.write(f"Total time: {self.total_compilation_time + self.total_execution_time:.4f}s\n\n")
            
            if self.compilation_times:
                f.write("Compilation times:\n")
                for func_name, comp_time in self.compilation_times.items():
                    f.write(f"  {func_name}: {comp_time:.4f}s\n")
                f.write("\n")
            
            if self.execution_times:
                f.write("Execution times:\n")
                for func_name, exec_time in self.execution_times.items():
                    f.write(f"  {func_name}: {exec_time:.4f}s\n")
        
        print(f"Timing log saved to {filename}")

# Global timer instance
jax_timer = JAXTimer()

# Smart device configuration - detect and configure available devices
def configure_jax_devices():
    """Configure JAX devices intelligently based on availability"""
    try:
        devices = jax.devices()
        if not devices:
            print("No JAX devices found, using default configuration")
            return 'cpu', None
        
        gpu_devices = [d for d in devices if 'gpu' in str(d).lower() or 'cuda' in str(d).lower()]
        cpu_devices = [d for d in devices if 'cpu' in str(d).lower()]
        
        print(f"Available devices: {devices}")
        
        if gpu_devices:
            # GPU is available - enable GPU acceleration
            try:
                # Test if GPU is actually functional
                test_array = jnp.array([1.0, 2.0, 3.0])
                gpu_test = jnp.sum(test_array)
                print(f"GPU acceleration enabled. Found {len(gpu_devices)} GPU device(s): {gpu_devices}")
                print(f"CPU devices also available: {cpu_devices}")
                return 'gpu', gpu_devices[0]  # Return GPU as primary device
            except Exception as gpu_error:
                print(f"GPU configuration failed: {gpu_error}")
                print("Falling back to CPU configuration")
                return 'cpu', cpu_devices[0] if cpu_devices else devices[0]
        else:
            # Only CPU available
            print(f"GPU not available, using CPU. Found {len(cpu_devices)} CPU device(s): {cpu_devices}")
            return 'cpu', cpu_devices[0] if cpu_devices else devices[0]
            
    except Exception as e:
        print(f"Error detecting devices: {e}")
        print("Falling back to default CPU configuration")
        try:
            return 'cpu', jax.devices()[0]
        except:
            return 'cpu', None

# Configure devices
device_type, primary_device = configure_jax_devices()

def check_gpu_availability():
    """
    Comprehensive function to check GPU availability and capabilities.
    
    Returns:
        dict: Dictionary containing detailed GPU information
    """
    gpu_info = {
        'gpu_available': False,
        'gpu_count': 0,
        'gpu_devices': [],
        'gpu_names': [],
        'jax_platform': None,
        'backend_info': {},
        'recommendations': []
    }
    
    try:
        # Check JAX platform
        gpu_info['jax_platform'] = jax.default_backend()
        
        # Get all devices
        all_devices = jax.devices()
        gpu_info['total_devices'] = len(all_devices)
        
        # Filter GPU devices
        gpu_devices = [d for d in all_devices if 'gpu' in str(d).lower()]
        gpu_info['gpu_count'] = len(gpu_devices)
        gpu_info['gpu_devices'] = gpu_devices
        gpu_info['gpu_available'] = len(gpu_devices) > 0
        
        # Extract GPU names
        for device in gpu_devices:
            device_str = str(device)
            if 'gpu' in device_str.lower():
                # Try to extract GPU name from device string
                if ':' in device_str:
                    gpu_name = device_str.split(':')[0]
                else:
                    gpu_name = device_str
                gpu_info['gpu_names'].append(gpu_name)
        
        # Check backend-specific information
        if gpu_info['gpu_available']:
            try:
                # Try to get CUDA information if available
                import jaxlib
                if hasattr(jaxlib, 'xla_extension'):
                    gpu_info['backend_info']['jaxlib_version'] = jaxlib.__version__
                
                # Test GPU computation
                test_array = jnp.array([1.0, 2.0, 3.0])
                gpu_test = jnp.sum(test_array)
                gpu_info['backend_info']['gpu_computation_test'] = 'PASSED'
                
            except Exception as e:
                gpu_info['backend_info']['gpu_computation_test'] = f'FAILED: {str(e)}'
        
        # Generate recommendations
        if gpu_info['gpu_available']:
            gpu_info['recommendations'].append("GPU is available - consider using GPU acceleration for better performance")
            gpu_info['recommendations'].append(f"Found {gpu_info['gpu_count']} GPU device(s): {gpu_info['gpu_names']}")
        else:
            gpu_info['recommendations'].append("No GPU detected - using CPU for computations")
            gpu_info['recommendations'].append("Consider installing CUDA and cuDNN for GPU acceleration")
        
        # Check for common issues
        if gpu_info['jax_platform'] == 'cpu' and gpu_info['gpu_available']:
            gpu_info['recommendations'].append("WARNING: GPU devices found but JAX is using CPU backend")
            gpu_info['recommendations'].append("Check CUDA installation and JAX-GPU compatibility")
        
    except Exception as e:
        gpu_info['error'] = str(e)
        gpu_info['recommendations'].append(f"Error checking GPU: {str(e)}")
    
    return gpu_info

def print_detailed_gpu_info():
    """
    Print detailed information about GPU availability and configuration.
    """
    print("=" * 60)
    print("DETAILED GPU AVAILABILITY CHECK")
    print("=" * 60)
    
    gpu_info = check_gpu_availability()
    
    print(f"GPU Available: {gpu_info['gpu_available']}")
    print(f"GPU Count: {gpu_info['gpu_count']}")
    print(f"JAX Platform: {gpu_info['jax_platform']}")
    print(f"Total Devices: {gpu_info.get('total_devices', 'Unknown')}")
    
    if gpu_info['gpu_available']:
        print(f"\nGPU Devices:")
        for i, (device, name) in enumerate(zip(gpu_info['gpu_devices'], gpu_info['gpu_names'])):
            print(f"  {i+1}. {name} -> {device}")
    
    if gpu_info['backend_info']:
        print(f"\nBackend Information:")
        for key, value in gpu_info['backend_info'].items():
            print(f"  {key}: {value}")
    
    if gpu_info['recommendations']:
        print(f"\nRecommendations:")
        for rec in gpu_info['recommendations']:
            print(f"  • {rec}")
    
    if 'error' in gpu_info:
        print(f"\nError: {gpu_info['error']}")
    
    print("=" * 60)

def test_gpu_performance():
    """
    Test GPU performance with a simple computation.
    """
    print("\n" + "=" * 60)
    print("GPU PERFORMANCE TEST")
    print("=" * 60)
    
    gpu_info = check_gpu_availability()
    
    if not gpu_info['gpu_available']:
        print("No GPU available for performance testing")
        return
    
    try:
        import time
        
        # Test data
        size = 1000
        a = jnp.random.random((size, size))
        b = jnp.random.random((size, size))
        
        # Warm up
        _ = jnp.dot(a, b)
        
        # CPU test
        print("Testing CPU performance...")
        start_time = time.time()
        for _ in range(10):
            result_cpu = jnp.dot(a, b)
        cpu_time = time.time() - start_time
        print(f"CPU time (10 iterations): {cpu_time:.4f} seconds")
        
        # GPU test (if available)
        if gpu_info['gpu_available']:
            print("Testing GPU performance...")
            # Move computation to GPU
            a_gpu = jax.device_put(a, gpu_info['gpu_devices'][0])
            b_gpu = jax.device_put(b, gpu_info['gpu_devices'][0])
            
            # Warm up GPU
            _ = jnp.dot(a_gpu, b_gpu)
            
            start_time = time.time()
            for _ in range(10):
                result_gpu = jnp.dot(a_gpu, b_gpu)
            gpu_time = time.time() - start_time
            print(f"GPU time (10 iterations): {gpu_time:.4f} seconds")
            
            if gpu_time < cpu_time:
                speedup = cpu_time / gpu_time
                print(f"GPU speedup: {speedup:.2f}x faster than CPU")
            else:
                slowdown = gpu_time / cpu_time
                print(f"GPU slowdown: {slowdown:.2f}x slower than CPU (may be due to overhead)")
        
    except Exception as e:
        print(f"Performance test failed: {e}")

def get_optimal_device():
    """Get the optimal device for computation based on availability"""
    return primary_device

def jit_with_device(fn, device=None):
    """JIT compile function with optional device specification"""
    if device is None:
        device = get_optimal_device()
    
    try:
        if device_type == 'gpu' and device is not None and ('gpu' in str(device).lower() or 'cuda' in str(device).lower()):
            # Use GPU for computationally intensive operations
            return jit(fn, device=device)
        else:
            # Use CPU
            return jit(fn)
    except Exception as e:
        print(f"Warning: Device-specific JIT failed ({e}), falling back to default JIT")
        return jit(fn)

def hat(v):
    """Convert 3D vector to skew-symmetric matrix"""
    return jnp.array([[0, -v[2], v[1]], 
                     [v[2], 0, -v[0]], 
                     [-v[1], v[0], 0]])

# Use device-optimized JIT for computationally intensive functions with timing
@jit_with_device
@jax_timer.time_compilation("compute_muscle_contributions")
def compute_muscle_contributions(muscles_r_i, muscles_tau, u, v):
    """Vectorized computation of muscle contributions with device optimization"""
    def single_muscle_contribution(r_i, tau):
        pb_si = jnp.cross(u, r_i) + v
        norm_pb_si = jnp.linalg.norm(pb_si)
        # Use a more robust approach for numerical stability
        norm_pb_si = jnp.maximum(norm_pb_si, 1e-12)
        
        Ai = -tau * hat(pb_si) @ hat(pb_si) / norm_pb_si**3
        Bi = hat(r_i) @ Ai
        ai = Ai @ jnp.cross(u, pb_si)
        bi = jnp.cross(r_i, ai)
        
        return Ai, Bi, ai, bi
    
    # Vectorize over muscles
    Ais, Bis, ais, bis = vmap(single_muscle_contribution)(muscles_r_i, muscles_tau)
    
    A = jnp.sum(Ais, axis=0)
    B = jnp.sum(Bis, axis=0)
    
    # Vectorize hat operation for muscles_r_i
    hat_r_i = vmap(hat)(muscles_r_i)  # Shape: (num_muscles, 3, 3)
    
    G = -jnp.sum(Ais @ hat_r_i, axis=0)
    H = -jnp.sum(Bis @ hat_r_i, axis=0)
    a = jnp.sum(ais, axis=0)
    b = jnp.sum(bis, axis=0)
    
    return A, B, G, H, a, b

@jit_with_device
@jax_timer.time_compilation("cosserat_muscle_ode")
def cosserat_muscle_ode(x, muscles_r_i, muscles_tau, W_ext, xi_ref, Kse, Kbt, f_t=None, l_t=None):
    """JIT compiled version of the Cosserat muscle ODE with device optimization"""
    R = x[3:12].reshape(3, 3)
    u = x[12:15]
    v = x[15:18]

    # Vectorized muscle contributions
    A, B, G, H, a, b = compute_muscle_contributions(muscles_r_i, muscles_tau, u, v)

    # baseline strains from reference xi
    u_star = xi_ref[:3]
    v_star = xi_ref[3:]

    # internal stresses
    nb = Kse @ (v - v_star)
    mb = Kbt @ (u - u_star)

    # external wrench parts (world frame)
    f_e = W_ext[3:]
    l_e = W_ext[:3]
    
    # tip wrench in body frame (use lax.cond for JAX compatibility)
    def use_provided_f_t(f_t):
        return f_t
    
    def use_zero_f_t(f_t):
        return jnp.zeros(3)
    
    def use_provided_l_t(l_t):
        return l_t
    
    def use_zero_l_t(l_t):
        return jnp.zeros(3)
    
    f_t_final = lax.cond(f_t is not None, use_provided_f_t, use_zero_f_t, f_t if f_t is not None else jnp.zeros(3))
    l_t_final = lax.cond(l_t is not None, use_provided_l_t, use_zero_l_t, l_t if l_t is not None else jnp.zeros(3))

    # corrected RHS
    d = -(hat(u) @ nb) - R.T @ f_e - a - f_t_final
    c = -(hat(u) @ mb) - (hat(v) @ nb) - R.T @ l_e - b - l_t_final

    K_total = jnp.block([[Kse + A, G],
                        [B, Kbt + H]])
    vu_s = jnp.linalg.solve(K_total, jnp.hstack([d, c]))
    v_s, u_s = vu_s[:3], vu_s[3:]

    p_s = R @ v
    R_s = R @ hat(u)
    return jnp.hstack([p_s, R_s.flatten(), u_s, v_s])

@jit_with_device
@jax_timer.time_compilation("RK4_step")
def RK4_step(x, muscles_r_i, muscles_tau, W_ext, ds, xi_ref, Kse, Kbt, f_t=None, l_t=None):
    """JIT compiled RK4 step with device optimization"""
    k1 = cosserat_muscle_ode(x, muscles_r_i, muscles_tau, W_ext, xi_ref, Kse, Kbt, f_t, l_t)
    k2 = cosserat_muscle_ode(x + ds*k1/2, muscles_r_i, muscles_tau, W_ext, xi_ref, Kse, Kbt, f_t, l_t)
    k3 = cosserat_muscle_ode(x + ds*k2/2, muscles_r_i, muscles_tau, W_ext, xi_ref, Kse, Kbt, f_t, l_t)
    k4 = cosserat_muscle_ode(x + ds*k3, muscles_r_i, muscles_tau, W_ext, xi_ref, Kse, Kbt, f_t, l_t)

    return x + ds/6*(k1 + 2*k2 + 2*k3 + k4)

@jit_with_device
@jax_timer.time_compilation("integrate_shape_scan")
def integrate_shape_scan(x0, muscles_r_i, muscles_tau, W_dist, ds, xi_star, Kse, Kbt, f_t=None, l_t=None):
    """Fused integrator using lax.scan with device optimization"""
    n = W_dist.shape[0]
    
    def integration_step(carry, i):
        x_prev, X = carry
        
        # Get current wrench and reference strains
        W_i = W_dist[i]
        xi_ref = xi_star[i]
        
        # Use lax.cond instead of if statement for JAX compatibility
        def apply_tip_forces(x_prev, W_i, xi_ref):
            return RK4_step(x_prev, muscles_r_i, muscles_tau, W_i, ds, xi_ref, Kse, Kbt, f_t, l_t)
        
        def no_tip_forces(x_prev, W_i, xi_ref):
            return RK4_step(x_prev, muscles_r_i, muscles_tau, W_i, ds, xi_ref, Kse, Kbt)
        
        x_next = lax.cond(
            i == n-1,
            apply_tip_forces,
            no_tip_forces,
            x_prev, W_i, xi_ref
        )
        
        X = X.at[i].set(x_next)
        return (x_next, X), None
    
    X = jnp.zeros((n, 18))
    X = X.at[0].set(x0)
    
    (_, X), _ = lax.scan(integration_step, (x0, X), jnp.arange(1, n))
    return X

@jit_with_device
@jax_timer.time_compilation("compute_single_muscle_length")
def compute_single_muscle_length(X, r_i):
    """Compute length for a single muscle with device optimization"""
    n = X.shape[0]
    
    def muscle_path(carry, i):
        p_i = X[i, :3]
        R_i = X[i, 3:12].reshape(3, 3)
        pt = p_i + R_i @ r_i
        return carry, pt
    
    # Use a dummy carry value and get the muscle points
    _, muscle_pts = lax.scan(muscle_path, jnp.zeros(3), jnp.arange(n))
    
    # Calculate segment lengths
    segs = jnp.diff(muscle_pts, axis=0)
    lengths = jnp.linalg.norm(segs, axis=1)
    total_length = jnp.sum(lengths)
    
    # Use a more robust approach for numerical stability
    return jnp.maximum(total_length, 1e-12)

@jit_with_device
@jax_timer.time_compilation("shooting_residual_fused")
def shooting_residual_fused(vars, muscles_r_i, muscles_pressures, muscles_params, W_dist, xi_star, Kse, Kbt, X_init, ds, f_t=None, l_t=None):
    """Fused shooting method with device optimization"""
    # Split variables into xi0 and taus
    xi0 = vars[:6]
    taus = vars[6:]
    
    # Initialize shape
    x0 = X_init[0].copy()
    x0 = x0.at[12:18].set(xi0)
    
    # Integrate shape
    X = integrate_shape_scan(x0, muscles_r_i, taus, W_dist, ds, xi_star, Kse, Kbt, f_t, l_t)
    
    # Get final strains
    xi_end = X[-1, 12:18]
    
    # Build tip wrench
    W_tip_calc = W_dist[-1].copy()
    for i, (r_i, tau) in enumerate(zip(muscles_r_i, taus)):
        pb_s = jnp.cross(X[-1, 12:15], r_i) + X[-1, 15:18]
        pb_s_norm = jnp.linalg.norm(pb_s)
        # Use a more robust approach for numerical stability
        pb_s_norm = jnp.maximum(pb_s_norm, 1e-12)
        f_tip = -tau * pb_s / pb_s_norm
        moment_tip = jnp.cross(r_i, f_tip)
        W_tip_calc += jnp.concatenate([moment_tip, f_tip])
    
    # Tip BC residuals
    moment_residual = Kbt @ (xi_end[:3] - xi_star[-1][:3]) - W_tip_calc[:3]
    force_residual = Kse @ (xi_end[3:] - xi_star[-1][3:]) - W_tip_calc[3:]
    
    # If f_t or l_t are nonzero, add them to the residuals
    if f_t is not None:
        R_tip = X[-1, 3:12].reshape(3, 3)
        force_residual -= R_tip @ f_t
    if l_t is not None:
        R_tip = X[-1, 3:12].reshape(3, 3)
        moment_residual -= R_tip @ l_t
    
    # Muscle model residuals - vectorized using actual pressures
    def muscle_residual(r_i, pressure, params, tau):
        K, L_orig, c, b = params
        
        # Compute muscle length
        length = compute_single_muscle_length(X, r_i)
        
        # Compute expected tension using the actual pressure
        tau_expected = K*(length - L_orig) + c*pressure + b*pressure**2
        residual = tau - tau_expected
        
        return residual
    
    # Vectorize over muscles using the provided pressures
    muscle_residuals = vmap(muscle_residual)(muscles_r_i, muscles_pressures, muscles_params, taus)
    
    return jnp.concatenate([moment_residual, force_residual, muscle_residuals])

class Muscle:
    def __init__(self, r_i, angle_i, pressure, L_orig, K, c, b):
        self.r_i = jnp.array([r_i * jnp.cos(angle_i), r_i * jnp.sin(angle_i), 0])
        self.pressure = pressure
        self.L_orig = L_orig
        self.K = K
        self.c = c
        self.b = b
        self.tau = 0.0
        self.length = 0.0

class Backbone:
    def __init__(self, E, r, G, n, L, rho, xi0_star=None):
        self.E = E
        self.r = r
        self.G = G
        self.A = jnp.pi * r ** 2
        self.Ix = jnp.pi * r ** 4 / 4
        self.J = 2 * self.Ix
        self.Kbt = jnp.diag(jnp.array([E*self.Ix, E*self.Ix, G*self.J]))
        self.Kse = jnp.diag(jnp.array([G*self.A, G*self.A, E*self.A]))
        self.n, self.L, self.ds = n, L, L/(n-1)
        self.rho = rho

        p_init = jnp.linspace(jnp.array([0., 0., 0.]), jnp.array([0., 0., self.L]), n)
        R_init = jnp.tile(jnp.eye(3).reshape(1, 9), (n, 1))
        if xi0_star is None:
            xi0_star = jnp.tile(jnp.array([0., 0., 0., 0., 0., 1.]), (n, 1))
        self.xi_star = xi0_star
        u_init = jnp.tile(self.xi_star[0, :3], (n, 1))
        v_init = jnp.tile(self.xi_star[0, 3:], (n, 1))

        self.X_init = jnp.hstack([p_init, R_init, u_init, v_init])

    def solve_fused(self, muscles, W_dist, f_t=None, l_t=None):
        """Fused solve using SciPy fsolve with GPU-accelerated function evaluations"""
        # Prepare muscle data for JIT functions
        muscles_r_i = jnp.array([muscle.r_i for muscle in muscles])
        muscles_pressures = jnp.array([muscle.pressure for muscle in muscles])
        muscles_params = jnp.array([[muscle.K, muscle.L_orig, muscle.c, muscle.b] for muscle in muscles])
        
        # Better initial guess based on gravity and geometry
        xi0_guess = jnp.array([0., 0., 0., 0., 0., 1.])
        tau0 = jnp.array([5.0, 5.0, 5.0])
        vars0 = jnp.hstack([xi0_guess, tau0])
        
        # Define objective function with GPU acceleration
        @jit_with_device
        def objective_jax(vars):
            residuals = shooting_residual_fused(vars, muscles_r_i, muscles_pressures, 
                                              muscles_params, W_dist, self.xi_star, 
                                              self.Kse, self.Kbt, self.X_init, self.ds, f_t, l_t)
            return residuals
        
        # Wrapper function for SciPy fsolve (converts JAX arrays to numpy)
        def objective_scipy(vars):
            vars_jax = jnp.array(vars)
            residuals_jax = objective_jax(vars_jax)
            return np.array(residuals_jax)
        
        # Use SciPy's fsolve with GPU-accelerated function evaluations
        from scipy.optimize import fsolve
        sol = fsolve(objective_scipy, np.array(vars0), maxfev=1000, xtol=1e-6)
        
        # Convert solution back to JAX array
        sol_jax = jnp.array(sol)
        
        # Extract solution
        xi0 = sol_jax[:6]
        sol_tau = sol_jax[6:]
        
        # Set final tensions
        for muscle, tau in zip(muscles, sol_tau):
            muscle.tau = tau
        
        # Compute final shape
        x0 = self.X_init[0].copy()
        x0 = x0.at[12:18].set(xi0)
        
        X = integrate_shape_scan(x0, muscles_r_i, sol_tau, W_dist, self.ds, 
                               self.xi_star, self.Kse, self.Kbt, f_t, l_t)
        
        # Compute and store muscle lengths
        for i, muscle in enumerate(muscles):
            muscle.length = float(compute_single_muscle_length(X, muscle.r_i))
        
        return X

def simulate_rod_tip_jax(E, rho, L, xi_star, muscle_params, tip_world_wrench, target_pressures, n=51, r=0.001, G=None, g=9.81, n_steps=100, f_t=None, l_t=None):
    """
    JAX version of simulate_rod_tip for parameter identification
    """
    if G is None:
        G = E/(2*1.3)
    
    rod = Backbone(E, r, G, n, L, rho, xi0_star=xi_star)
    
    # Setup muscles with initial pressure 0 (same as Python version)
    muscles = []
    for i, m in enumerate(muscle_params):
        muscle = Muscle(m['r_i'], m['angle_i'], 0, m['l0'], m['k'], m['c'], m['b'])
        muscles.append(muscle)
    
    # Set up distributed gravity
    W_dist = jnp.zeros((n, 6))
    mass_per_length = jnp.pi * r**2 * rho
    f_gravity = mass_per_length * L * g / n
    W_dist = W_dist.at[:, 5].add(f_gravity)
    W_dist = W_dist.at[-1].add(tip_world_wrench)
    
    # Run simulation with proper pressure ramping (same as Python version)
    start_time = time.time()
    
    # Pressure ramping approach - match Python version exactly
    current_pressures = jnp.zeros(len(muscles))
    target_pressures_jax = jnp.array(target_pressures)
    
    # Use linspace for pressure steps (same as Python version)
    pressure_steps = jnp.linspace(current_pressures, target_pressures_jax, n_steps)
    
    # Ramp pressures in steps
    for step in range(n_steps):
        # Get current pressure step
        new_pressures = pressure_steps[step]
        
        # Update muscle pressures
        for i, muscle in enumerate(muscles):
            muscle.pressure = float(new_pressures[i])
        
        # Solve with current pressures
    X_final = rod.solve_fused(muscles, W_dist, f_t=f_t, l_t=l_t)
    
    simulation_time = time.time() - start_time
    
    # Track execution time
    jax_timer.execution_times['simulate_rod_tip_jax'] = jax_timer.execution_times.get('simulate_rod_tip_jax', 0.0) + simulation_time
    jax_timer.total_execution_time += simulation_time
    
    tip_x, tip_y = X_final[-1, 0], X_final[-1, 1]
    
    return tip_x, tip_y, X_final, muscles

def load_experiment_data(data_file):
    """
    Load experimental data from CSV file and extract static positions and pressures.
    Returns: static_positions (N,2), static_pressures (N,3)
    """
    try:
        data = pd.read_csv(data_file)
        print(f"Successfully loaded data file: {data_file}")
        print(f"Number of data points: {len(data)}")
    except Exception as e:
        print(f"Error loading data file: {e}")
        raise

    # Parameters for static point extraction
    sampling_freq = 100  # 100 Hz
    ramp_time = 3  # 3 seconds for ramp
    hold_time = 10  # 10 seconds for hold
    points_per_segment = (ramp_time + hold_time) * sampling_freq  # 1300 points per segment
    
    # Get initial point (first point of the experiment)
    initial_point = data.iloc[0]
    
    # Get static points (last point of each 13-second segment)
    static_indices = []
    for i in range(0, len(data), points_per_segment):
        if i + points_per_segment <= len(data):
            static_idx = i + points_per_segment - 1
            static_indices.append(static_idx)
    
    print(f"Found {len(static_indices)} static points")
    
    # Combine initial point with static points
    static_data = pd.concat([pd.DataFrame([initial_point]), data.iloc[static_indices]])
    
    # Calculate relative positions (dx, dz) for static points
    static_data['dx'] = static_data['x2'] - static_data['x1']
    static_data['dz'] = static_data['z2'] - static_data['z1']
    
    # Create static positions array [dz, dx] to match the coordinate system
    static_positions = np.column_stack([
        static_data['dz'],
        static_data['dx']
    ])
    
    # Extract pressures
    static_pressures = np.column_stack([
        static_data['P1'],
        static_data['P2'],
        static_data['P3']
    ])
    
    print(f"Processed {len(static_positions)} static positions and pressures")
    print("\nStatic positions (dz, dx):")
    for i, pos in enumerate(static_positions):
        print(f"Point {i+1}: [{pos[0]:.6f}, {pos[1]:.6f}]")
    
    return static_positions, static_pressures

def visualize_experiment(static_positions, static_pressures, data_file):
    """
    Visualize the experimental data, including the full trajectory and static points.
    """
    try:
        # Load the full experimental data
        data = pd.read_csv(data_file)
        
        # Calculate relative positions for all points
        data['dx'] = data['x2'] - data['x1']
        data['dz'] = data['z2'] - data['z1']
        
        # Create full positions array [dz, dx]
        full_positions = np.column_stack([
            data['dz'],
            data['dx']
        ])
        
        # Create figure with non-interactive backend for server compatibility
        plt.switch_backend('Agg')  # Use Agg backend for non-interactive plotting
        plt.figure(figsize=(8,6))
        
        # Plot the full trajectory and static points
        plt.plot(full_positions[:,0], full_positions[:,1], 'b-', alpha=1, label='Full Trajectory')
        plt.plot(static_positions[:,0], static_positions[:,1], 'ro', label='Static Points')
        
        # Add point labels
        for i, (x, y) in enumerate(static_positions):
            plt.text(x, y, f'{i+1}', fontsize=9, color='green')
        
        plt.xlabel('dz [m]')
        plt.ylabel('dx [m]')
        plt.title('Experimental Tip Trajectory and Static Points')
        plt.legend()
        plt.axis('equal')
        plt.grid(True)
        plt.tight_layout()
        
        # Save plot instead of showing
        plt.savefig('experimental_trajectory.png', dpi=300, bbox_inches='tight')
        print("Experimental trajectory plot saved as experimental_trajectory.png")
        plt.close()
        
    except Exception as e:
        print(f"Error in visualization: {e}")
        raise

def objective_function_jax(params, static_positions, static_pressures):
    """
    JAX-accelerated objective function for parameter fitting.
    """
    start_time = time.time()
    
    try:
        # Unpack parameters
        E, rho, L, ux, uy, k1, k2, k3, c1, c2, c3, b1, b2, b3, l01, l02, l03, mz0, A = params
        
        # Calculate G from E
        G = E/(2*1.3)
        
        # Fixed parameters
        r = 0.001  # rod radius
        n = 51     # number of discretization points
        
        # Create reference strain array
        xi_star = jnp.tile(jnp.array([ux, uy, 0, 0, 0, 1]), (n, 1))
        
        # Create muscle parameters list
        muscle_params = [
            {'r_i': 0.02, 'angle_i': angle, 'k': k, 'c': c, 'b': b, 'l0': l0}
            for angle, k, c, b, l0 in zip(
                [0, 2*jnp.pi/3, 4*jnp.pi/3],
                [k1, k2, k3],
                [c1, c2, c3],
                [b1, b2, b3],
                [l01, l02, l03]
            )
        ]
        
        # Initialize error
        total_error = 0.0
        
        # For each static point
        for pos, pressures in zip(static_positions, static_pressures):
            try:
                # Create tip wrench (no pressure-dependent moment here)
                tip_world_wrench = jnp.zeros(6)  # [lwx, lwy, lwz, fwx, fwy, fwz]
                
                # Compute total tip moment: pressure-dependent + offset
                mz = A * jnp.sum(pressures) + mz0
                
                # Simulate tip position
                tip_x, tip_y, _, _ = simulate_rod_tip_jax(
                    E=E,
                    rho=rho,
                    L=L,
                    xi_star=xi_star,
                    muscle_params=muscle_params,
                    tip_world_wrench=tip_world_wrench,
                    target_pressures=pressures,
                    n=n,
                    r=r,
                    G=G,
                    f_t=jnp.zeros(3),  # No tip force
                    l_t=jnp.array([0, 0, mz])  # Total moment in body frame
                )
                
                # Calculate error
                error = jnp.linalg.norm(jnp.array([tip_x - pos[0], tip_y - pos[1]]))
                total_error += error
                
            except Exception as e:
                print(f"Error in simulation: {e}")
                return 1e10  # Return large error for failed simulations
        
        objective_time = time.time() - start_time
        jax_timer.execution_times['objective_function_jax'] = jax_timer.execution_times.get('objective_function_jax', 0.0) + objective_time
        jax_timer.total_execution_time += objective_time
        
        return float(total_error)
        
    except Exception as e:
        print(f"Error in objective function: {e}")
        return 1e10

def simulate_all(static_pressures, params):
    """
    Simulate all static points with given parameters using JAX arrays and minimal Python objects.
    Returns: sim_positions (N,2) array of simulated tip positions
    """
    start_time = time.time()
    
    n = 51
    r = 0.001
    G = params['E']/(2*1.3)
    xi_star = jnp.tile(jnp.array([params['ux'], params['uy'], 0, 0, 0, 1]), (n, 1))
    # Prepare muscle parameters as JAX arrays
    muscle_r_i = jnp.array([
        [0.02 * jnp.cos(angle), 0.02 * jnp.sin(angle), 0.0]
        for angle in [0, 2*jnp.pi/3, 4*jnp.pi/3]
    ])
    muscle_k = jnp.array([params['k1'], params['k2'], params['k3']])
    muscle_c = jnp.array([params['c1'], params['c2'], params['c3']])
    muscle_b = jnp.array([params['b1'], params['b2'], params['b3']])
    muscle_l0 = jnp.array([params['l01'], params['l02'], params['l03']])
    tip_world_wrench = jnp.zeros(6)
    sim_positions = []
    
    print(f"Starting simulation of {len(static_pressures)} points...")
    
    # Pre-compute zero-pressure configuration once (same for all points)
    print("Computing zero-pressure configuration for initial guess...")
    zero_pressures = jnp.zeros(3)
    zero_mz = params['A'] * jnp.sum(zero_pressures) + params['mz0']
    
    # Use JAX arrays for muscle parameters - pass all necessary parameters for correct xi_star
    zero_params = {
        'E': params['E'],
        'rho': params['rho'],
        'L': params['L'],
        'ux': params['ux'],  # Include ux for correct xi_star
        'uy': params['uy'],  # Include uy for correct xi_star
        'xi_star': None,  # Will be computed in get_zero_pressure_initial_guess_jax
        'muscle_r_i': muscle_r_i,
        'muscle_k': muscle_k,
        'muscle_c': muscle_c,
        'muscle_b': muscle_b,
        'muscle_l0': muscle_l0,
        'tip_world_wrench': tip_world_wrench,
        'target_pressures': zero_pressures,
        'n': n,
        'r': r,
        'G': G,
        'f_t': jnp.zeros(3),
        'l_t': jnp.array([0., 0., zero_mz])
    }
    zero_xi0_guess, zero_tau_guess = get_zero_pressure_initial_guess_jax(zero_params)
    print(f"Zero-pressure configuration computed. Initial tensions: {zero_tau_guess}")
    
    for i, pressures in enumerate(static_pressures):
        try:
            print(f"\nSimulating point {i+1}/{len(static_pressures)} with pressures: {pressures}")
            point_start_time = time.time()
            
            # Compute tip moment mz for this point
            mz = params['A'] * jnp.sum(pressures) + params['mz0']
            
            # Use the pre-computed zero-pressure configuration as initial guess
            # But use the correct xi_star (with ux, uy) for the actual simulation
            tip_x, tip_y = simulate_single_point_jax(
                params, xi_star, muscle_r_i, muscle_k, muscle_c, muscle_b, muscle_l0,
                tip_world_wrench, pressures, n, r, G, zero_xi0_guess, zero_tau_guess, mz
            )
            
            point_time = time.time() - point_start_time
            sim_positions.append([float(tip_x), float(tip_y)])
            print(f"Point {i+1}: Simulated position = ({float(tip_x)*1000:.1f}, {float(tip_y)*1000:.1f}) mm in {point_time:.2f}s")
            
        except Exception as e:
            print(f"Simulation failed for point {i+1} with pressures {pressures}: {e}")
            sim_positions.append([np.nan, np.nan])
    
    simulation_time = time.time() - start_time
    jax_timer.execution_times['simulate_all'] = jax_timer.execution_times.get('simulate_all', 0.0) + simulation_time
    jax_timer.total_execution_time += simulation_time
    
    return np.array(sim_positions)

def get_zero_pressure_initial_guess_jax(params):
    """
    Compute the zero-pressure initial guess using a pressure ramp (from [0,0,0] to [0,0,0]) for n_steps.
    This should match V2_JAX exactly.
    """
    E, rho, L = params['E'], params['rho'], params['L']
    ux, uy = params['ux'], params['uy']  # Get ux, uy from params
    muscle_r_i, muscle_k, muscle_c, muscle_b, muscle_l0 = params['muscle_r_i'], params['muscle_k'], params['muscle_c'], params['muscle_b'], params['muscle_l0']
    tip_world_wrench, target_pressures = params['tip_world_wrench'], params['target_pressures']
    n, r, G = params['n'], params['r'], params['G']
    f_t, l_t = params['f_t'], params['l_t']

    # Build muscle_params as list of dicts for compatibility
    muscle_params = [
        {'r_i': float(muscle_r_i[i,0]), 'angle_i': float(jnp.arctan2(muscle_r_i[i,1], muscle_r_i[i,0])),
         'k': float(muscle_k[i]), 'c': float(muscle_c[i]), 'b': float(muscle_b[i]), 'l0': float(muscle_l0[i])}
        for i in range(3)
    ]

    rod_params = (E, r, G, n, L, rho)

    # Set up distributed gravity
    W_dist = jnp.zeros((n, 6))
    mass_per_length = jnp.pi * r**2 * rho
    f_gravity = mass_per_length * L * 9.81 / n
    W_dist = W_dist.at[:, 5].set(f_gravity)
    W_dist = W_dist.at[-1].set(tip_world_wrench)

    # Use the exact same approach as V2_JAX: solve_single_point_jax with correct xi_star
    # For zero-pressure, we should use the correct xi_star (with ux, uy) like V2_JAX does
    correct_xi_star = jnp.tile(jnp.array([ux, uy, 0, 0, 0, 1]), (n, 1))
    X_final, tau_final = solve_single_point_jax_direct(
        rod_params, muscle_params, target_pressures, W_dist, 
        correct_xi_star,  # Use correct xi_star with ux, uy like V2_JAX
        f_t=f_t, l_t=l_t
    )

    # Return the computed xi0 from the final shape, not the initial guess
    xi0_final = X_final[0, 12:18]  # Extract the computed initial strains from the final shape
    return xi0_final, tau_final

def solve_single_point_jax_direct(rod_params, muscle_params, target_pressures, W_dist, xi_star, f_t=None, l_t=None):
    """
    Direct JAX-optimized single point simulation (exact copy of V2_JAX approach)
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
    
    # Initialize shape - use the provided xi_star (like V2_JAX)
    p_init = jnp.linspace(jnp.array([0., 0., 0.]), jnp.array([0., 0., L]), n)
    R_init = jnp.tile(jnp.eye(3).reshape(1, 9), (n, 1))
    u_init = jnp.tile(xi_star[0, :3], (n, 1))
    v_init = jnp.tile(xi_star[0, 3:], (n, 1))
    X_init = jnp.hstack([p_init, R_init, u_init, v_init])
    
    # Initial guesses - exact same as V2_JAX
    tau_guess = jnp.array([5.0 for _ in range(len(muscles_r_i))])
    xi0_guess = jnp.array([0., 0., 0., 0., 0., 1.])
    
    # Create pressure steps - exact same as V2_JAX
    n_steps = 100
    pressure_steps = jnp.linspace(jnp.zeros(len(muscles_r_i)), jnp.array(target_pressures), n_steps)
    
    # Pre-compile the shooting function - exact same as V2_JAX
    @jit
    def shooting_wrapper(vars, pressures):
        return shooting_residual_fused(vars, muscles_r_i, pressures, 
                                     muscles_params_array, W_dist, xi_star, 
                                     Kse, Kbt, X_init, ds, f_t, l_t)
    
    # Python loop for pressure ramping - exact same as V2_JAX
    for step in range(n_steps):
        current_pressures = pressure_steps[step]
        
        # Build initial guess vector
        vars0 = jnp.hstack([xi0_guess, tau_guess])
        
        # Use scipy fsolve with pre-compiled JAX function - exact same as V2_JAX
        from scipy.optimize import fsolve
        
        def objective_np(vars_np):
            vars_jax = jnp.array(vars_np)
            result_jax = shooting_wrapper(vars_jax, current_pressures)
            return np.array(result_jax)
        
        sol = fsolve(objective_np, np.array(vars0), maxfev=1000)
        
        xi0_guess = jnp.array(sol[:6])
        tau_guess = jnp.array(sol[6:])
    
    # Compute final shape - exact same as V2_JAX
    x0 = X_init[0].copy()
    x0 = x0.at[12:18].set(xi0_guess)
    X_final = integrate_shape_scan(x0, muscles_r_i, tau_guess, W_dist, ds, xi_star, Kse, Kbt, f_t, l_t)
    
    return X_final, tau_guess

def solve_single_point_jax_direct_with_initial_guess(rod_params, muscle_params, target_pressures, W_dist, xi_star, initial_xi0, initial_taus, f_t=None, l_t=None):
    """
    Direct JAX-optimized single point simulation with provided initial guesses (uses correct xi_star)
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
    
    # Initialize shape - use the provided xi_star (with ux, uy) for actual simulation
    p_init = jnp.linspace(jnp.array([0., 0., 0.]), jnp.array([0., 0., L]), n)
    R_init = jnp.tile(jnp.eye(3).reshape(1, 9), (n, 1))
    u_init = jnp.tile(xi_star[0, :3], (n, 1))
    v_init = jnp.tile(xi_star[0, 3:], (n, 1))
    X_init = jnp.hstack([p_init, R_init, u_init, v_init])
    
    # Use provided initial guesses
    tau_guess = initial_taus.copy()
    xi0_guess = initial_xi0.copy()
    
    # Create pressure steps (fewer steps since we have good initial guesses)
    n_steps = 100  # Reduced from 30 since we have good initial guesses
    pressure_steps = jnp.linspace(jnp.zeros(len(muscles_r_i)), jnp.array(target_pressures), n_steps)
    
    # Pre-compile the shooting function
    @jit
    def shooting_wrapper(vars, pressures):
        return shooting_residual_fused(vars, muscles_r_i, pressures, 
                                     muscles_params_array, W_dist, xi_star, 
                                     Kse, Kbt, X_init, ds, f_t, l_t)
    
    # Python loop for pressure ramping (minimal overhead)
    for step in range(n_steps):
        current_pressures = pressure_steps[step]
        
        # Build initial guess vector
        vars0 = jnp.hstack([xi0_guess, tau_guess])
        
        # Use scipy fsolve with pre-compiled JAX function (fewer iterations since we have good guesses)
        from scipy.optimize import fsolve
        
        def objective_np(vars_np):
            vars_jax = jnp.array(vars_np)
            result_jax = shooting_wrapper(vars_jax, current_pressures)
            return np.array(result_jax)
        
        # Use fewer iterations since we have good initial guesses
        sol = fsolve(objective_np, np.array(vars0), maxfev=500)  # Reduced from 1000
        
        xi0_guess = jnp.array(sol[:6])
        tau_guess = jnp.array(sol[6:])
    
    # Compute final shape
    x0 = X_init[0].copy()
    x0 = x0.at[12:18].set(xi0_guess)
    X_final = integrate_shape_scan(x0, muscles_r_i, tau_guess, W_dist, ds, xi_star, Kse, Kbt, f_t, l_t)
    
    return X_final, tau_guess

def simulate_single_point_jax(params, xi_star, muscle_r_i, muscle_k, muscle_c, muscle_b, muscle_l0,
                             tip_world_wrench, pressures, n, r, G, xi0_guess, tau_guess, mz):
    """
    Simulate a single point using JAX arrays and initial guesses.
    """
    # Build muscle_params as list of dicts for compatibility
    muscle_params = [
        {'r_i': float(muscle_r_i[i,0]), 'angle_i': float(jnp.arctan2(muscle_r_i[i,1], muscle_r_i[i,0])),
         'k': float(muscle_k[i]), 'c': float(muscle_c[i]), 'b': float(muscle_b[i]), 'l0': float(muscle_l0[i])}
        for i in range(3)
    ]
    
    # Use the correct xi_star (with ux, uy) for the actual simulation
    # This ensures we get the correct results while using the zero-pressure initial guesses
    tip_x, tip_y, _, _, _ = simulate_rod_tip_jax_with_initial_guess(
                E=params['E'],
                rho=params['rho'],
                L=params['L'],
                xi_star=xi_star,  # Use the correct xi_star with ux, uy
                muscle_params=muscle_params,
                tip_world_wrench=tip_world_wrench,
                target_pressures=pressures,
                n=n,
                r=r,
                G=G,
                f_t=jnp.zeros(3),
                l_t=jnp.array([0., 0., mz]),
                initial_xi0=xi0_guess,
                initial_taus=tau_guess
    )
    return tip_x, tip_y

def simulate_rod_tip_jax_with_initial_guess(E, rho, L, xi_star, muscle_params, tip_world_wrench, target_pressures, n=51, r=0.001, G=None, g=9.81, n_steps=100, f_t=None, l_t=None, initial_xi0=None, initial_taus=None):
    """
    JAX version of simulate_rod_tip with initial guesses for faster convergence
    """
    if G is None:
        G = E/(2*1.3)
    
    # Use the direct JAX approach like V2_JAX instead of Backbone class
    rod_params = (E, r, G, n, L, rho)
    
    # Set up distributed gravity
    W_dist = jnp.zeros((n, 6))
    mass_per_length = jnp.pi * r**2 * rho
    f_gravity = mass_per_length * L * g / n
    W_dist = W_dist.at[:, 5].set(f_gravity)
    W_dist = W_dist.at[-1].set(tip_world_wrench)
    
    # Run simulation with proper pressure ramping and initial guesses
    start_time = time.time()
    
    # Use provided initial guesses if available
    if initial_xi0 is not None and initial_taus is not None:
        # Use provided initial guesses for faster convergence
        X_final, tau_final = solve_single_point_jax_direct_with_initial_guess(
            rod_params, muscle_params, target_pressures, W_dist, xi_star, 
            initial_xi0, initial_taus, f_t=f_t, l_t=l_t
        )
    else:
        # Use default ramping (slower but more robust)
        X_final, tau_final = solve_single_point_jax_direct(rod_params, muscle_params, target_pressures, W_dist, xi_star, f_t=f_t, l_t=l_t)
    
    simulation_time = time.time() - start_time
    
    # Track execution time
    jax_timer.execution_times['simulate_rod_tip_jax_with_initial_guess'] = jax_timer.execution_times.get('simulate_rod_tip_jax_with_initial_guess', 0.0) + simulation_time
    jax_timer.total_execution_time += simulation_time
    
    tip_x, tip_y = X_final[-1, 0], X_final[-1, 1]
    
    return tip_x, tip_y, X_final, tau_final, None  # Return None for muscles since we don't use Backbone class

def ramp_pressures_and_solve_with_initial_guess(rod, muscles, target_pressures, n_steps, W_dist, initial_xi0, initial_taus, f_t=None, l_t=None):
    """
    Ramps up the muscle pressures in n_steps, using provided initial guesses.
    This is optimized for cases where we have good initial guesses from previous simulations.
    Returns the final shape and muscle tensions.
    """
    from scipy.optimize import fsolve
    
    pressures = jnp.array([muscle.pressure for muscle in muscles])
    target_pressures = jnp.array(target_pressures)
    pressure_steps = jnp.linspace(pressures, target_pressures, n_steps)

    # Use provided initial guesses
    xi0_guess = initial_xi0.copy()
    tau_guess = initial_taus.copy()

    # Extract muscle parameters for JAX functions
    muscles_r_i = jnp.array([muscle.r_i for muscle in muscles])
    muscles_params_array = jnp.array([[muscle.K, muscle.L_orig, muscle.c, muscle.b] for muscle in muscles])
    
    # Initialize rod properties for JAX functions
    E, r, G, n, L, rho = rod.E, rod.r, rod.G, rod.n, rod.L, rod.rho
    A = jnp.pi * r ** 2
    Ix = jnp.pi * r ** 4 / 4
    J = 2 * Ix
    Kbt = jnp.diag(jnp.array([E*Ix, E*Ix, G*J]))
    Kse = jnp.diag(jnp.array([G*A, G*A, E*A]))
    ds = L/(n-1)

    # Pre-compile the shooting function
    @jit
    def shooting_wrapper(vars, pressures):
        return shooting_residual_fused(vars, muscles_r_i, pressures, 
                                     muscles_params_array, W_dist, rod.xi_star, 
                                     Kse, Kbt, rod.X_init, ds, f_t, l_t)

    for step, p_vec in enumerate(pressure_steps):
        for i, muscle in enumerate(muscles):
            muscle.pressure = p_vec[i]
        
        # Build initial guess vector
        vars0 = jnp.hstack([xi0_guess, tau_guess])
        
        # Use scipy fsolve with pre-compiled JAX function (fewer iterations since we have good guesses)
        def objective_np(vars_np):
            vars_jax = jnp.array(vars_np)
            result_jax = shooting_wrapper(vars_jax, p_vec)
            return np.array(result_jax)
        
        # Use fewer iterations since we have good initial guesses
        sol = fsolve(objective_np, np.array(vars0), maxfev=500)
        
        xi0_guess = jnp.array(sol[:6])
        tau_guess = jnp.array(sol[6:])
        
        # Set final tensions for next step
        for muscle, tau in zip(muscles, tau_guess):
            muscle.tau = tau
    
    # After ramping, compute final shape using JAX integration
    x0 = rod.X_init[0].copy()
    x0 = x0.at[12:18].set(xi0_guess)
    X_final = integrate_shape_scan(x0, muscles_r_i, tau_guess, W_dist, ds, rod.xi_star, Kse, Kbt, f_t, l_t)
    
    # Update muscle lengths
    for muscle in muscles:
        muscle.length = float(compute_single_muscle_length(X_final, muscle.r_i))
    
    return X_final, tau_guess

def ramp_pressures_and_solve(rod, muscles, target_pressures, n_steps, W_dist, f_t=None, l_t=None):
    """
    Default ramping function without initial guesses (slower but more robust).
    This is the fallback when no initial guesses are provided.
    Returns the final shape and muscle tensions.
    """
    from scipy.optimize import fsolve
    
    pressures = jnp.array([muscle.pressure for muscle in muscles])
    target_pressures = jnp.array(target_pressures)
    pressure_steps = jnp.linspace(pressures, target_pressures, n_steps)

    # Initial guesses (default approach)
    tau_guess = jnp.array([5.0, 5.0, 5.0])  # Default tension guesses
    xi0_guess = rod.xi_star[0].copy()

    # Extract muscle parameters for JAX functions
    muscles_r_i = jnp.array([muscle.r_i for muscle in muscles])
    muscles_params_array = jnp.array([[muscle.K, muscle.L_orig, muscle.c, muscle.b] for muscle in muscles])
    
    # Initialize rod properties for JAX functions
    E, r, G, n, L, rho = rod.E, rod.r, rod.G, rod.n, rod.L, rod.rho
    A = jnp.pi * r ** 2
    Ix = jnp.pi * r ** 4 / 4
    J = 2 * Ix
    Kbt = jnp.diag(jnp.array([E*Ix, E*Ix, G*J]))
    Kse = jnp.diag(jnp.array([G*A, G*A, E*A]))
    ds = L/(n-1)

    # Pre-compile the shooting function
    @jit
    def shooting_wrapper(vars, pressures):
        return shooting_residual_fused(vars, muscles_r_i, pressures, 
                                     muscles_params_array, W_dist, rod.xi_star, 
                                     Kse, Kbt, rod.X_init, ds, f_t, l_t)

    for step, p_vec in enumerate(pressure_steps):
        for i, muscle in enumerate(muscles):
            muscle.pressure = p_vec[i]
        
        # Build initial guess vector
        vars0 = jnp.hstack([xi0_guess, tau_guess])
        
        # Use scipy fsolve with pre-compiled JAX function
        def objective_np(vars_np):
            vars_jax = jnp.array(vars_np)
            result_jax = shooting_wrapper(vars_jax, p_vec)
            return np.array(result_jax)
        
        # Use more iterations for default approach (no initial guesses)
        sol = fsolve(objective_np, np.array(vars0), maxfev=1000)
        
        xi0_guess = jnp.array(sol[:6])
        tau_guess = jnp.array(sol[6:])
        
        # Set final tensions for next step
        for muscle, tau in zip(muscles, tau_guess):
            muscle.tau = tau
    
    # After ramping, compute final shape using JAX integration
    x0 = rod.X_init[0].copy()
    x0 = x0.at[12:18].set(xi0_guess)
    X_final = integrate_shape_scan(x0, muscles_r_i, tau_guess, W_dist, ds, rod.xi_star, Kse, Kbt, f_t, l_t)
    
    # Update muscle lengths
    for muscle in muscles:
        muscle.length = float(compute_single_muscle_length(X_final, muscle.r_i))
    
    return X_final, tau_guess

def plot_comparison(sim_positions, exp_positions, title, filename=None):
    """
    Plot comparison between simulated and experimental positions.
    Args:
        sim_positions: (N,2) array of simulated positions
        exp_positions: (N,2) array of experimental positions
        title: Title for the plot
        filename: Optional filename to save the plot
    """
    # Use non-interactive backend for server compatibility
    plt.switch_backend('Agg')
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Experimental points
    for i, pos in enumerate(exp_positions):
        ax.plot(pos[0], pos[1], 'ro', markersize=10, label='Exp' if i==0 else "")
        ax.annotate(f'Exp {i+1}\n({pos[0]*1000:.1f}, {pos[1]*1000:.1f})mm',
                    (pos[0], pos[1]), xytext=(10, 10), textcoords='offset points',
                    color='red', fontweight='bold')
    
    # Simulated points
    total_error = 0
    valid_simulations = 0
    for i, pos in enumerate(sim_positions):
        # Check for NaN values
        if np.isnan(pos[0]) or np.isnan(pos[1]):
            ax.plot(exp_positions[i][0], exp_positions[i][1], 'kx', markersize=15, label='Failed Sim' if i==0 else "")
            ax.annotate(f'Sim {i+1}\nFAILED',
                        (exp_positions[i][0], exp_positions[i][1]), xytext=(-10, -10), textcoords='offset points',
                        color='black', fontweight='bold')
            continue
        
        ax.plot(pos[0], pos[1], 'b*', markersize=10, label='Sim' if i==0 else "")
        ax.annotate(f'Sim {i+1}\n({pos[0]*1000:.1f}, {pos[1]*1000:.1f})mm',
                    (pos[0], pos[1]), xytext=(-10, -10), textcoords='offset points',
                    color='blue', fontweight='bold')
        
        # Calculate error for valid simulations
        exp_pos = exp_positions[i]
        error = np.sqrt((exp_pos[0] - pos[0])**2 + (exp_pos[1] - pos[1])**2)
        total_error += error
        valid_simulations += 1
        
        # Draw error line
        ax.plot([exp_pos[0], pos[0]], [exp_pos[1], pos[1]], 'g--', alpha=0.5, label='Error' if i==0 else "")
        mid_x = (exp_pos[0] + pos[0]) / 2
        mid_y = (exp_pos[1] + pos[1]) / 2
        ax.annotate(f'{error*1000:.1f}mm', (mid_x, mid_y), xytext=(0, 5), textcoords='offset points',
                    ha='center', va='bottom', color='green', alpha=0.7)
    
    ax.plot(0, 0, 'k+', label='Origin', markersize=15)
    ax.set_xlabel('dz [m]')
    ax.set_ylabel('dx [m]')
    
    # Update title with error information
    if valid_simulations > 0:
        avg_error = total_error / valid_simulations
        ax.set_title(f'{title}\nTotal Error: {total_error*1000:.1f}mm (Avg: {avg_error*1000:.1f}mm, {valid_simulations}/{len(exp_positions)} valid)')
    else:
        ax.set_title(f'{title}\nNo valid simulations')
    
    ax.grid(True)
    ax.axis('equal')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as {filename}")
    
    plt.close()
    
    return total_error

def params_to_dict(params_array):
    """
    Convert parameter array to dictionary format.
    """
    param_names = ['E', 'rho', 'L', 'ux', 'uy', 'k1', 'k2', 'k3', 
                  'c1', 'c2', 'c3', 'b1', 'b2', 'b3', 'l01', 'l02', 'l03', 'mz0', 'A']
    return dict(zip(param_names, params_array))

def save_optimization_log(result, initial_params, static_positions, static_pressures, filename='optimization_log.txt'):
    """
    Save optimization results and log to a text file.
    """
    with open(filename, 'w') as f:
        f.write("=== PARAMETER IDENTIFICATION OPTIMIZATION LOG ===\n\n")
        
        # Optimization results
        f.write("OPTIMIZATION RESULTS:\n")
        f.write(f"Success: {result.success}\n")
        f.write(f"Message: {result.message}\n")
        f.write(f"Final error: {result.fun:.6f}\n")
        f.write(f"Number of iterations: {result.nit}\n")
        f.write(f"Number of function evaluations: {result.nfev}\n\n")
        
        # Parameter comparison
        param_names = ['E', 'rho', 'L', 'ux', 'uy', 'k1', 'k2', 'k3', 
                      'c1', 'c2', 'c3', 'b1', 'b2', 'b3', 'l01', 'l02', 'l03', 'mz0', 'A']
        
        f.write("PARAMETER COMPARISON:\n")
        f.write(f"{'Parameter':<10} {'Initial':<15} {'Optimal':<15} {'Change':<15}\n")
        f.write("-" * 60 + "\n")
        
        for name, init_val, opt_val in zip(param_names, initial_params, result.x):
            change = opt_val - init_val
            change_pct = (change / init_val) * 100 if init_val != 0 else 0
            f.write(f"{name:<10} {init_val:<15.6e} {opt_val:<15.6e} {change_pct:<15.2f}%\n")
        
        f.write("\n" + "="*60 + "\n\n")
        
        # Detailed parameter values
        f.write("DETAILED PARAMETER VALUES:\n")
        f.write("Initial Parameters:\n")
        for name, value in zip(param_names, initial_params):
            f.write(f"  {name}: {value}\n")
        
        f.write("\nOptimal Parameters:\n")
        for name, value in zip(param_names, result.x):
            f.write(f"  {name}: {value}\n")
        
        f.write("\n" + "="*60 + "\n\n")
        
        # Experimental data summary
        f.write("EXPERIMENTAL DATA SUMMARY:\n")
        f.write(f"Number of static points: {len(static_positions)}\n")
        f.write("Static positions (dz, dx):\n")
        for i, pos in enumerate(static_positions):
            f.write(f"  Point {i+1}: [{pos[0]:.6f}, {pos[1]:.6f}]\n")
        
        f.write("\nStatic pressures (P1, P2, P3):\n")
        for i, pressures in enumerate(static_pressures):
            f.write(f"  Point {i+1}: [{pressures[0]:.6f}, {pressures[1]:.6f}, {pressures[2]:.6f}]\n")
    
    print(f"Optimization log saved to {filename}")

def fit_parameters_jax(static_positions, static_pressures):
    """
    Fit model parameters to experimental data using JAX-accelerated simulation.
    """
    print("\nStarting JAX-accelerated parameter fitting...")
    
    # Initial parameter guess using manually tuned values from V2
    initial_params = np.array([
        12e9,           # E
        8.3e5,          # rho
        0.15019191323880385,  # L
        0.3134339267162691,   # ux
        -0.8038382647760769,  # uy
        109.87942486687194, 99.8929386339798, 109.90263143856308,  # k1, k2, k3
        0.034189862080583834, 0.018846469408956925, 0.022616173522392316,   # c1, c2, c3
        0.0005, 0.0005, 0.0005,   # b1, b2, b3
        0.09904043380598078, 0.08076765295521537, 0.08076765295521537,     # l01, l02, l03
        0.0019191323880384421,    # mz0
        0.00010767652955215377    # A (tip moment pressure coefficient)
    ])
    
    # Bounds
    bounds = [
        (10e9, 15e9),      # E
        (5e5, 8.3e5),      # rho
        (0.12, 0.17),      # L
        (-3, 3),           # ux
        (-3, 3),           # uy
        (80, 130), (80, 130), (80, 130),   # k1, k2, k3
        (0.01, 0.06), (0.01, 0.06), (0.01, 0.06), # c1, c2, c3
        (0.0001, 0.002), (0.0001, 0.002), (0.0001, 0.002), # b1, b2, b3
        (0.05, 0.12), (0.05, 0.12), (0.05, 0.12), # l01, l02, l03
        (-0.01, 0.01),     # mz0
        (0.0, 0.0005)      # A
    ]
    
    # Optimization options
    options = {
        'maxiter': 100,    # Maximum iterations
        'disp': True,      # Display progress
        'ftol': 1e-6,      # Function tolerance
        'gtol': 1e-6       # Gradient tolerance
    }
    
    try:
        # Run optimization
        result = minimize(
            objective_function_jax,
            initial_params,
            args=(static_positions, static_pressures),
            method='TNC',  # Changed from L-BFGS-B to TNC for faster optimization
            bounds=bounds,
            options=options
        )
        
        print("\nOptimization completed:")
        print(f"Success: {result.success}")
        print(f"Message: {result.message}")
        print(f"Final error: {result.fun}")
        
        return result
        
    except Exception as e:
        print(f"Error in optimization: {e}")
        raise

def visualize_results(static_positions, static_pressures, optimal_params, data_files):
    """
    Visualize the results of parameter fitting.
    """
    try:
        # Load and average all data files
        all_data = []
        for data_file in data_files:
            data = pd.read_csv(data_file)
            all_data.append(data)
        
        # Ensure all datasets have the same length
        min_length = min(len(data) for data in all_data)
        all_data = [data.iloc[:min_length] for data in all_data]
        
        # Calculate average trajectory
        avg_data = pd.DataFrame()
        for col in all_data[0].columns:
            avg_data[col] = np.mean([data[col].values for data in all_data], axis=0)
        
        # Calculate relative positions for all points
        avg_data['dx'] = avg_data['x2'] - avg_data['x1']
        avg_data['dz'] = avg_data['z2'] - avg_data['z1']
        
        # Create full positions array [dz, dx]
        full_positions = np.column_stack([
            avg_data['dz'],
            avg_data['dx']
        ])
        
        # Simulate with optimal parameters
        E, rho, L, ux, uy, k1, k2, k3, c1, c2, c3, b1, b2, b3, l01, l02, l03, mz0, A = optimal_params
        G = E/(2*1.3)
        r = 0.001
        n = 51
        xi_star = jnp.tile(jnp.array([ux, uy, 0, 0, 0, 1]), (n, 1))
        
        muscle_params = [
            {'r_i': 0.02, 'angle_i': angle, 'k': k, 'c': c, 'b': b, 'l0': l0}
            for angle, k, c, b, l0 in zip(
                [0, 2*jnp.pi/3, 4*jnp.pi/3],
                [k1, k2, k3],
                [c1, c2, c3],
                [b1, b2, b3],
                [l01, l02, l03]
            )
        ]
        
        sim_positions = []
        for pressures in static_pressures:
            tip_world_wrench = jnp.zeros(6)
            tip_world_wrench = tip_world_wrench.at[2].add(A * jnp.sum(pressures))
            
            tip_x, tip_y, _, _ = simulate_rod_tip_jax(
                E=E, rho=rho, L=L, xi_star=xi_star, muscle_params=muscle_params,
                tip_world_wrench=tip_world_wrench, target_pressures=pressures,
                n=n, r=r, G=G, f_t=jnp.zeros(3), l_t=jnp.array([0, 0, mz0])
            )
            sim_positions.append([float(tip_x), float(tip_y)])
        
        sim_positions = np.array(sim_positions)
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Plot full trajectory
        plt.plot(full_positions[:,0], full_positions[:,1], 'b-', alpha=0.3, label='Averaged Trajectory')
        
        # Plot experimental static points
        plt.plot(static_positions[:,0], static_positions[:,1], 'ro', markersize=8, label='Experimental Static Points')
        
        # Plot simulated static points
        plt.plot(sim_positions[:,0], sim_positions[:,1], 'b*', markersize=10, label='Simulated Static Points')
        
        # Add error lines
        for i in range(len(static_positions)):
            exp_pos = static_positions[i]
            sim_pos = sim_positions[i]
            plt.plot([exp_pos[0], sim_pos[0]], [exp_pos[1], sim_pos[1]], 'g--', alpha=0.5)
            mid_x = (exp_pos[0] + sim_pos[0]) / 2
            mid_y = (exp_pos[1] + sim_pos[1]) / 2
            error = np.sqrt((exp_pos[0] - sim_pos[0])**2 + (exp_pos[1] - sim_pos[1])**2)
            plt.annotate(f'{error*1000:.1f}mm', (mid_x, mid_y), xytext=(0, 5), 
                        textcoords='offset points', ha='center', va='bottom', color='green', alpha=0.7)
        
        plt.xlabel('dz [m]')
        plt.ylabel('dx [m]')
        plt.title('Parameter Identification Results: Experimental vs Simulated')
        plt.legend()
        plt.axis('equal')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error in visualization: {e}")
        raise

def save_optimization_results(result, static_positions, static_pressures, data_files, output_dir='results'):
    """
    Save optimization results including parameters and comparison data.
    """
    import os
    import json
    from datetime import datetime
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Parameter names
    param_names = ['E', 'rho', 'L', 'ux', 'uy', 'k1', 'k2', 'k3', 
                  'c1', 'c2', 'c3', 'b1', 'b2', 'b3', 'l01', 'l02', 'l03', 'mz0', 'A']
    
    # Save final parameters
    params_dict = {name: float(value) for name, value in zip(param_names, result.x)}
    params_dict.update({
        'optimization_success': bool(result.success),
        'final_error': float(result.fun),
        'iterations': int(result.nit),
        'function_evaluations': int(result.nfev),
        'gradient_evaluations': int(result.njev),
        'optimization_message': str(result.message),
        'timestamp': timestamp,
        'data_files': data_files
    })
    
    params_file = os.path.join(output_dir, f'optimal_parameters_{timestamp}.json')
    with open(params_file, 'w') as f:
        json.dump(params_dict, f, indent=2)
    
    print(f"\nSaved optimal parameters to: {params_file}")
    
    # Save parameters in a more readable format
    readable_file = os.path.join(output_dir, f'optimal_parameters_{timestamp}.txt')
    with open(readable_file, 'w') as f:
        f.write("OPTIMAL PARAMETERS FROM JAX PARAMETER IDENTIFICATION\n")
        f.write("=" * 60 + "\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Optimization success: {result.success}\n")
        f.write(f"Final error: {result.fun:.6f}\n")
        f.write(f"Iterations: {result.nit}\n")
        f.write(f"Function evaluations: {result.nfev}\n")
        f.write(f"Gradient evaluations: {result.njev}\n")
        
        f.write("PARAMETER VALUES:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Material properties:\n")
        f.write(f"  E: {result.x[0]:.2e} Pa\n")
        f.write(f"  rho: {result.x[1]:.2e} kg/m³\n")
        f.write(f"  L: {result.x[2]:.6f} m\n\n")
        
        f.write(f"Reference strains:\n")
        f.write(f"  ux: {result.x[3]:.6f}\n")
        f.write(f"  uy: {result.x[4]:.6f}\n\n")
        
        f.write(f"Muscle spring constants:\n")
        f.write(f"  k1: {result.x[5]:.2f}\n")
        f.write(f"  k2: {result.x[6]:.2f}\n")
        f.write(f"  k3: {result.x[7]:.2f}\n\n")
        
        f.write(f"Pressure coefficients:\n")
        f.write(f"  c1: {result.x[8]:.6f}\n")
        f.write(f"  c2: {result.x[9]:.6f}\n")
        f.write(f"  c3: {result.x[10]:.6f}\n\n")
        
        f.write(f"Quadratic pressure coefficients:\n")
        f.write(f"  b1: {result.x[11]:.6f}\n")
        f.write(f"  b2: {result.x[12]:.6f}\n")
        f.write(f"  b3: {result.x[13]:.6f}\n\n")
        
        f.write(f"Muscle original lengths:\n")
        f.write(f"  l01: {result.x[14]:.6f}\n")
        f.write(f"  l02: {result.x[15]:.6f}\n")
        f.write(f"  l03: {result.x[16]:.6f}\n\n")
        
        f.write(f"Tip moment parameters:\n")
        f.write(f"  mz0: {result.x[17]:.6f}\n")
        f.write(f"  A: {result.x[18]:.6f}\n\n")
        
        f.write("EXPERIMENTAL DATA FILES:\n")
        f.write("-" * 30 + "\n")
        for i, file in enumerate(data_files):
            f.write(f"  {i+1}. {file}\n")
    
    print(f"Saved readable parameters to: {readable_file}")
    
    return params_file, readable_file

def plot_detailed_comparison(static_positions, static_pressures, optimal_params, data_files, save_plot=True, output_dir='results'):
    """
    Create detailed comparison plot between fitted model and experimental data.
    """
    import os
    from datetime import datetime
    
    # Create output directory if it doesn't exist
    if save_plot:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Simulate with optimal parameters
    E, rho, L, ux, uy, k1, k2, k3, c1, c2, c3, b1, b2, b3, l01, l02, l03, mz0, A = optimal_params
    G = E/(2*1.3)
    r = 0.001
    n = 51
    xi_star = jnp.tile(jnp.array([ux, uy, 0, 0, 0, 1]), (n, 1))
    
    muscle_params = [
        {'r_i': 0.02, 'angle_i': angle, 'k': k, 'c': c, 'b': b, 'l0': l0}
        for angle, k, c, b, l0 in zip(
            [0, 2*jnp.pi/3, 4*jnp.pi/3],
            [k1, k2, k3],
            [c1, c2, c3],
            [b1, b2, b3],
            [l01, l02, l03]
        )
    ]
    
    # Simulate positions for each pressure set
    sim_positions = []
    for pressures in static_pressures:
        tip_world_wrench = jnp.zeros(6)
        tip_world_wrench = tip_world_wrench.at[2].add(A * jnp.sum(pressures))
        
        tip_x, tip_y, _, _ = simulate_rod_tip_jax(
            E=E, rho=rho, L=L, xi_star=xi_star, muscle_params=muscle_params,
            tip_world_wrench=tip_world_wrench, target_pressures=pressures,
            n=n, r=r, G=G, f_t=jnp.zeros(3), l_t=jnp.array([0, 0, mz0])
        )
        sim_positions.append([float(tip_x), float(tip_y)])
    
    sim_positions = np.array(sim_positions)
    
    # Calculate errors
    errors = []
    for i in range(len(static_positions)):
        error = np.sqrt((static_positions[i,0] - sim_positions[i,0])**2 + 
                       (static_positions[i,1] - sim_positions[i,1])**2)
        errors.append(error)
    
    total_error = np.sum(errors)
    mean_error = np.mean(errors)
    max_error = np.max(errors)
    
    # Create detailed comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Main comparison plot
    for i, (exp_pos, sim_pos) in enumerate(zip(static_positions, sim_positions)):
        # Experimental points
        ax1.plot(exp_pos[0], exp_pos[1], 'ro', markersize=12, label='Experimental' if i==0 else "")
        ax1.annotate(f'Exp {i+1}\n({exp_pos[0]*1000:.1f}, {exp_pos[1]*1000:.1f})mm',
                    (exp_pos[0], exp_pos[1]), xytext=(10, 10), textcoords='offset points',
                    color='red', fontweight='bold', fontsize=9)
        
        # Simulated points
        ax1.plot(sim_pos[0], sim_pos[1], 'b*', markersize=12, label='Simulated' if i==0 else "")
        ax1.annotate(f'Sim {i+1}\n({sim_pos[0]*1000:.1f}, {sim_pos[1]*1000:.1f})mm',
                    (sim_pos[0], sim_pos[1]), xytext=(-10, -10), textcoords='offset points',
                    color='blue', fontweight='bold', fontsize=9)
        
        # Error lines and values
        ax1.plot([exp_pos[0], sim_pos[0]], [exp_pos[1], sim_pos[1]], 'g--', alpha=0.7, linewidth=2, label='Error' if i==0 else "")
        mid_x = (exp_pos[0] + sim_pos[0]) / 2
        mid_y = (exp_pos[1] + sim_pos[1]) / 2
        error = errors[i]
        ax1.annotate(f'{error*1000:.1f}mm', (mid_x, mid_y), xytext=(0, 5), textcoords='offset points',
                    ha='center', va='bottom', color='green', alpha=0.8)
    
    # Add origin marker
    ax1.plot(0, 0, 'k+', label='Origin', markersize=15, linewidth=3)
    
    ax1.set_xlabel('dz [m]', fontsize=12)
    ax1.set_ylabel('dx [m]', fontsize=12)
    
    # Update title with error information
    if len(errors) > 0:
        avg_error = total_error / len(errors)
        ax1.set_title(f'Fitted Model vs Experimental Static Positions\nTotal Error: {total_error*1000:.1f}mm (Avg: {avg_error*1000:.1f}mm)')
    else:
        ax1.set_title('No valid simulations')
    
    ax1.grid(True)
    ax1.axis('equal')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Error analysis plot
    point_numbers = list(range(1, len(errors) + 1))
    ax2.bar(point_numbers, [e*1000 for e in errors], color='orange', alpha=0.7, edgecolor='darkorange', linewidth=1)
    ax2.axhline(y=mean_error*1000, color='red', linestyle='--', linewidth=2, label=f'Mean Error: {mean_error*1000:.1f}mm')
    ax2.axhline(y=max_error*1000, color='darkred', linestyle=':', linewidth=2, label=f'Max Error: {max_error*1000:.1f}mm')
    
    ax2.set_xlabel('Point Number', fontsize=12)
    ax2.set_ylabel('Error [mm]', fontsize=12)
    ax2.set_title('Position Error Analysis', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xticks(point_numbers)
    
    # Add error statistics as text
    error_text = f'Total Error: {total_error*1000:.1f}mm\nMean Error: {mean_error*1000:.1f}mm\nMax Error: {max_error*1000:.1f}mm'
    ax2.text(0.02, 0.98, error_text, transform=ax2.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=11)
    
    plt.tight_layout()
    
    # Save plot if requested
    if save_plot:
        plot_file = os.path.join(output_dir, f'comparison_plot_{timestamp}.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Saved comparison plot to: {plot_file}")
    
    plt.show()
    
    # Print detailed error analysis
    print(f"\n{'='*60}")
    print("DETAILED ERROR ANALYSIS")
    print(f"{'='*60}")
    print(f"Total error: {total_error*1000:.2f} mm")
    print(f"Mean error: {mean_error*1000:.2f} mm")
    print(f"Max error: {max_error*1000:.2f} mm")
    print(f"Standard deviation: {np.std(errors)*1000:.2f} mm")
    print(f"\nPoint-by-point errors:")
    for i, (exp_pos, sim_pos, error) in enumerate(zip(static_positions, sim_positions, errors)):
        print(f"  Point {i+1}: {error*1000:.2f} mm")
        print(f"    Experimental: ({exp_pos[0]*1000:.1f}, {exp_pos[1]*1000:.1f}) mm")
        print(f"    Simulated:    ({sim_pos[0]*1000:.1f}, {sim_pos[1]*1000:.1f}) mm")
    
    return sim_positions, errors

def print_device_info():
    """Print detailed device information and performance capabilities"""
    print("=" * 60)
    print("JAX DEVICE CONFIGURATION")
    print("=" * 60)
    print(f"Device type: {device_type}")
    print(f"Primary device: {primary_device}")
    print(f"All available devices: {jax.devices()}")
    
    if device_type == 'gpu':
        print("\nGPU ACCELERATION ENABLED")
        print("- Computationally intensive operations will use GPU")
        print("- Matrix operations, ODE integration, and optimization will be accelerated")
        print("- CPU fallback available for compatibility")
    else:
        print("\nCPU EXECUTION MODE")
        print("- All operations will use CPU")
        print("- JIT compilation still provides performance benefits")
        print("- Compatible with all JAX operations")
    
    print("=" * 60)

# Main execution function for Jupyter notebook
def run_parameter_identification(data_files=None, save_results=True, output_dir='results'):
    """
    Main function to run parameter identification in Jupyter notebook.
    
    Args:
        data_files: List of CSV file paths to load and average. 
                   If None, uses default files.
        save_results: Whether to save results to files
        output_dir: Directory to save results
    """
    if data_files is None:
        data_files = [
           'collected_data_0615_initPoints.csv'
        ]
    
    print("=" * 60)
    print("JAX-ACCELERATED PARAMETER IDENTIFICATION")
    print("=" * 60)
    
    # Print device information
    print_device_info()
    
    try:
        start_time = time.time()
        
        # Load and process data
        print(f"\nLoading data from {len(data_files)} file(s):")
        for file in data_files:
            print(f"  - {file}")
        static_positions, static_pressures = load_experiment_data(data_files[0])
        
        # Visualize data
        print("\nVisualizing experimental data...")
        visualize_experiment(static_positions, static_pressures, data_files[0])
        
        # Initial parameters (same as in fit_parameters function)
        initial_params = np.array([
            12e9,           # E
            8.3e5,          # rho
            0.15019191323880385,  # L
            0.3134339267162691,   # ux
            -0.8038382647760769,  # uy
            109.87942486687194, 99.8929386339798, 109.90263143856308,  # k1, k2, k3
            0.034189862080583834, 0.018846469408956925, 0.022616173522392316,   # c1, c2, c3
            0.0005, 0.0005, 0.0005,   # b1, b2, b3
            0.09904043380598078, 0.08076765295521537, 0.08076765295521537,     # l01, l02, l03
            0.0019191323880384421,    # mz0
            0.00010767652955215377    # A (tip moment pressure coefficient)
        ])
        
        # Convert to dictionary for simulation
        initial_params_dict = params_to_dict(initial_params)
        
        # 1. Comparison plotting for initial guess
        print("\nSimulating with initial parameters...")
        initial_sim_positions = simulate_all(static_pressures, initial_params_dict)
        initial_error = plot_comparison(initial_sim_positions, static_positions, 
                                      "Initial Guess Comparison", "initial_guess_comparison.png")
        print(f"Initial guess total error: {initial_error*1000:.1f}mm")
        
        # 2. Fit parameters
        print("\nStarting parameter fitting...")
        result = fit_parameters_jax(static_positions, static_pressures)
        
        # 3. Save optimization log and final parameters
        save_optimization_log(result, initial_params, static_positions, static_pressures)
        
        # 4. Comparison plotting for optimal solution
        print("\nSimulating with optimal parameters...")
        optimal_params_dict = params_to_dict(result.x)
        optimal_sim_positions = simulate_all(static_pressures, optimal_params_dict)
        optimal_error = plot_comparison(optimal_sim_positions, static_positions, 
                                      "Optimal Solution Comparison", "optimal_solution_comparison.png")
        print(f"Optimal solution total error: {optimal_error*1000:.1f}mm")
        
        # Print timing summary
        jax_timer.print_summary()
        jax_timer.save_timing_log()
        
        # Print summary
        total_time = time.time() - start_time
        print("\n" + "="*60)
        print("OPTIMIZATION SUMMARY:")
        print(f"Initial error: {initial_error*1000:.1f}mm")
        print(f"Final error: {optimal_error*1000:.1f}mm")
        print(f"Improvement: {(initial_error - optimal_error)*1000:.1f}mm ({(1 - optimal_error/initial_error)*100:.1f}%)")
        print(f"Total calculation time: {total_time:.2f} seconds")
        print("="*60)
        
        # Print results
        print("\nOptimal parameters:")
        param_names = ['E', 'rho', 'L', 'ux', 'uy', 'k1', 'k2', 'k3', 
                      'c1', 'c2', 'c3', 'b1', 'b2', 'b3', 'l01', 'l02', 'l03', 'mz0', 'A']
        for name, value in zip(param_names, result.x):
            print(f"{name}: {value}")
            
    except Exception as e:
        print(f"\nError in main execution: {e}")
        # Print timing summary even if there's an error
        jax_timer.print_summary()
        jax_timer.save_timing_log()
        raise

def run_v1_parameter_optimization(data_files=None, max_iterations=20, save_results=True):
    """
    Main function to run V1_JAX parameter optimization.
    """
    if data_files is None:
        data_files = ['collected_data_0615_initPoints.csv']
    
    print("=" * 60)
    print("V1_JAX PARAMETER OPTIMIZATION")
    print("=" * 60)
    
    try:
        # Load data
        print(f"Loading data from {data_files[0]}...")
        static_positions, static_pressures = load_experiment_data(data_files[0])
        
        # Run optimization
        print("Starting parameter optimization...")
        optimized_params, optimization_history = optimize_parameters_v1_jax(
            static_positions, static_pressures, 
            max_iterations=max_iterations
        )
        
        # Save results
        if save_results:
            save_optimization_results_v1(optimization_history, static_positions, static_pressures)
        
        # Plot optimization history
        plot_optimization_history_v1(optimization_history, save_plot=save_results)
        
        # Final comparison
        print("\nFinal parameter comparison:")
        print("Optimized parameters:")
        for key, value in optimized_params.items():
            print(f"  {key}: {value}")
        
        print(f"\nFinal error: {optimization_history[-1]['total_error']*1000:.2f}mm")
        print(f"Final average error: {optimization_history[-1]['avg_error']*1000:.2f}mm")
        
        return optimized_params, optimization_history
        
    except Exception as e:
        print(f"Error in V1_JAX optimization: {e}")
        raise

if __name__ == "__main__":
    try:
        start_time = time.time()
        
        # Load and process data
        data_files = ['collected_data_0615_initPoints.csv']
        print(f"\nLoading data from {len(data_files)} file(s):")
        for file in data_files:
            print(f"  - {file}")
        static_positions, static_pressures = load_experiment_data(data_files[0])
        
        # Visualize data
        print("\nVisualizing experimental data...")
        visualize_experiment(static_positions, static_pressures, data_files[0])
        
        # Initial parameters
        initial_params = {
            'E': 12e9,
            'rho': 8.3e5,
            'L': 0.15019191323880385,
            'ux': 0.3134339267162691,
            'uy': -0.8038382647760769,
            'k1': 109.87942486687194, 'k2': 99.8929386339798, 'k3': 109.90263143856308,
            'c1': 0.034189862080583834, 'c2': 0.018846469408956925, 'c3': 0.022616173522392316,
            'b1': 0.0005, 'b2': 0.0005, 'b3': 0.0005,
            'l01': 0.09904043380598078, 'l02': 0.08076765295521537, 'l03': 0.08076765295521537,
            'mz0': 0.0019191323880384421,
            'A': 0.00010767652955215377
        }
        
        # 1. Comparison plotting for initial guess
        print("\nSimulating with initial parameters...")
        initial_sim_positions = simulate_all(static_pressures, initial_params)
        initial_error = plot_comparison(initial_sim_positions, static_positions, 
                                      "Initial Guess Comparison", "initial_guess_comparison.png")
        print(f"Initial guess total error: {initial_error*1000:.1f}mm")
        
        # 2. Run V1_JAX parameter optimization
        print("\nStarting V1_JAX parameter optimization...")
        optimized_params, optimization_history = run_v1_parameter_optimization(
            data_files=data_files, 
            max_iterations=10,  # Reduced for faster testing
            save_results=True
        )
        
        # 3. Comparison plotting for optimal solution
        print("\nSimulating with optimized parameters...")
        optimized_sim_positions = simulate_all(static_pressures, optimized_params)
        optimized_error = plot_comparison(optimized_sim_positions, static_positions, 
                                        "Optimized Solution Comparison", "optimized_solution_comparison.png")
        print(f"Optimized solution total error: {optimized_error*1000:.1f}mm")
        
        # Print timing summary
        jax_timer.print_summary()
        jax_timer.save_timing_log()
        
        # Print summary
        total_time = time.time() - start_time
        print("\n" + "="*60)
        print("V1_JAX OPTIMIZATION SUMMARY:")
        print(f"Initial error: {initial_error*1000:.1f}mm")
        print(f"Final error: {optimized_error*1000:.1f}mm")
        print(f"Improvement: {(initial_error - optimized_error)*1000:.1f}mm ({(1 - optimized_error/initial_error)*100:.1f}%)")
        print(f"Total calculation time: {total_time:.2f} seconds")
        print("="*60)
        
        # Print results
        print("\nOptimized parameters:")
        for key, value in optimized_params.items():
            print(f"{key}: {value}")
            
    except Exception as e:
        print(f"\nError in main execution: {e}")
        # Print timing summary even if there's an error
        jax_timer.print_summary()
        jax_timer.save_timing_log()
        raise 

# --- Parameter Optimization System for V1_JAX ---

def optimize_parameters_v1_jax(static_positions, static_pressures, initial_params=None, max_iterations=50, tolerance=1e-6):
    """
    Optimize parameters using V1_JAX approach with proper initial guess management.
    
    Args:
        static_positions: Experimental static positions
        static_pressures: Corresponding pressures
        initial_params: Initial parameter guess (if None, uses default)
        max_iterations: Maximum optimization iterations
        tolerance: Convergence tolerance
        
    Returns:
        optimized_params: Optimized parameters
        optimization_history: History of optimization
    """
    print("=" * 60)
    print("V1_JAX PARAMETER OPTIMIZATION")
    print("=" * 60)
    
    # Default parameters if none provided
    if initial_params is None:
        initial_params = {
            'E': 12e9,
            'rho': 8.3e5,
            'L': 0.15019191323880385,
            'ux': 0.3134339267162691,
            'uy': -0.8038382647760769,
            'k1': 109.87942486687194, 'k2': 99.8929386339798, 'k3': 109.90263143856308,
            'c1': 0.034189862080583834, 'c2': 0.018846469408956925, 'c3': 0.022616173522392316,
            'b1': 0.0005, 'b2': 0.0005, 'b3': 0.0005,
            'l01': 0.09904043380598078, 'l02': 0.08076765295521537, 'l03': 0.08076765295521537,
            'mz0': 0.0019191323880384421,
            'A': 0.00010767652955215377
        }
    
    # Parameter bounds
    bounds = {
        'E': (10e9, 15e9),
        'rho': (5e5, 8.3e5),
        'L': (0.12, 0.17),
        'ux': (-3, 3),
        'uy': (-3, 3),
        'k1': (80, 130), 'k2': (80, 130), 'k3': (80, 130),
        'c1': (0.01, 0.06), 'c2': (0.01, 0.06), 'c3': (0.01, 0.06),
        'b1': (0.0001, 0.002), 'b2': (0.0001, 0.002), 'b3': (0.0001, 0.002),
        'l01': (0.05, 0.12), 'l02': (0.05, 0.12), 'l03': (0.05, 0.12),
        'mz0': (-0.01, 0.01),
        'A': (0.0, 0.0005)
    }
    
    # Current parameters
    current_params = initial_params.copy()
    optimization_history = []
    
    print(f"Starting optimization with {max_iterations} maximum iterations")
    print(f"Initial parameters: {current_params}")
    
    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration + 1}/{max_iterations} ---")
        
        # 1. Recalculate zero-pressure configuration for current parameters
        print("Recalculating zero-pressure configuration...")
        zero_pressures = jnp.array([0., 0., 0.])
        zero_mz = current_params['A'] * jnp.sum(zero_pressures) + current_params['mz0']
        
        # Build parameters for zero-pressure simulation
        n = 51
        r = 0.001
        G = current_params['E']/(2*1.3)
        xi_star = jnp.tile(jnp.array([current_params['ux'], current_params['uy'], 0, 0, 0, 1]), (n, 1))
        
        # Prepare muscle parameters
        muscle_r_i = jnp.array([
            [0.02 * jnp.cos(angle), 0.02 * jnp.sin(angle), 0.0]
            for angle in [0, 2*jnp.pi/3, 4*jnp.pi/3]
        ])
        muscle_k = jnp.array([current_params['k1'], current_params['k2'], current_params['k3']])
        muscle_c = jnp.array([current_params['c1'], current_params['c2'], current_params['c3']])
        muscle_b = jnp.array([current_params['b1'], current_params['b2'], current_params['b3']])
        muscle_l0 = jnp.array([current_params['l01'], current_params['l02'], current_params['l03']])
        
        # Build muscle_params for simulation
        muscle_params = [
            {'r_i': float(muscle_r_i[i,0]), 'angle_i': float(jnp.arctan2(muscle_r_i[i,1], muscle_r_i[i,0])),
             'k': float(muscle_k[i]), 'c': float(muscle_c[i]), 'b': float(muscle_b[i]), 'l0': float(muscle_l0[i])}
            for i in range(3)
        ]
        
        # Get zero-pressure configuration - use hardcoded xi_star for initial guess
        zero_params = {
            'E': current_params['E'],
            'rho': current_params['rho'],
            'L': current_params['L'],
            'xi_star': None,  # Will use hardcoded xi_star in solve_single_point_jax_direct
            'muscle_r_i': muscle_r_i,
            'muscle_k': muscle_k,
            'muscle_c': muscle_c,
            'muscle_b': muscle_b,
            'muscle_l0': muscle_l0,
            'tip_world_wrench': jnp.zeros(6),
            'target_pressures': zero_pressures,
            'n': n,
            'r': r,
            'G': G,
            'f_t': jnp.zeros(3),
            'l_t': jnp.array([0., 0., zero_mz])
        }
        
        zero_xi0_guess, zero_tau_guess = get_zero_pressure_initial_guess_jax(zero_params)
        print(f"Zero-pressure initial tensions: {zero_tau_guess}")
        
        # 2. Simulate all points with current parameters and zero-pressure initial guess
        print("Simulating all points with current parameters...")
        sim_positions = simulate_all_with_initial_guess(static_pressures, current_params, zero_xi0_guess, zero_tau_guess)
        
        # 3. Calculate error
        total_error = 0.0
        valid_points = 0
        for i, (exp_pos, sim_pos) in enumerate(zip(static_positions, sim_positions)):
            if not np.isnan(sim_pos[0]):
                error = np.sqrt((exp_pos[0] - sim_pos[0])**2 + (exp_pos[1] - sim_pos[1])**2)
                total_error += error
                valid_points += 1
                print(f"Point {i+1}: Error = {error*1000:.2f}mm")
        
        avg_error = total_error / valid_points if valid_points > 0 else float('inf')
        print(f"Total error: {total_error*1000:.2f}mm, Average error: {avg_error*1000:.2f}mm")
        
        # 4. Store optimization history
        optimization_history.append({
            'iteration': iteration + 1,
            'params': current_params.copy(),
            'total_error': total_error,
            'avg_error': avg_error,
            'valid_points': valid_points,
            'zero_tau_guess': zero_tau_guess
        })
        
        # 5. Check convergence
        if iteration > 0:
            prev_error = optimization_history[-2]['total_error']
            error_improvement = prev_error - total_error
            print(f"Error improvement: {error_improvement*1000:.2f}mm")
            
            if abs(error_improvement) < tolerance:
                print(f"Converged! Error improvement below tolerance {tolerance}")
                break
        
        # 6. Update parameters (simplified gradient descent for demonstration)
        # In a real implementation, you would use a proper optimization algorithm
        if iteration < max_iterations - 1:
            print("Updating parameters...")
            # Simple parameter adjustment based on error
            adjustment_factor = 0.1 * avg_error
            for param_name in current_params:
                if param_name in bounds:
                    min_val, max_val = bounds[param_name]
                    current_val = current_params[param_name]
                    # Simple random adjustment within bounds
                    adjustment = np.random.uniform(-adjustment_factor, adjustment_factor)
                    new_val = current_val + adjustment
                    current_params[param_name] = np.clip(new_val, min_val, max_val)
    
    print(f"\nOptimization completed after {len(optimization_history)} iterations")
    print(f"Final average error: {optimization_history[-1]['avg_error']*1000:.2f}mm")
    
    return current_params, optimization_history

def simulate_all_with_initial_guess(static_pressures, params, zero_xi0_guess, zero_tau_guess):
    """
    Simulate all points using the provided zero-pressure initial guesses.
    This is optimized to reuse the initial guesses for all pressure points.
    """
    n = 51
    r = 0.001
    G = params['E']/(2*1.3)
    xi_star = jnp.tile(jnp.array([params['ux'], params['uy'], 0, 0, 0, 1]), (n, 1))
    
    # Prepare muscle parameters
    muscle_r_i = jnp.array([
        [0.02 * jnp.cos(angle), 0.02 * jnp.sin(angle), 0.0]
        for angle in [0, 2*jnp.pi/3, 4*jnp.pi/3]
    ])
    muscle_k = jnp.array([params['k1'], params['k2'], params['k3']])
    muscle_c = jnp.array([params['c1'], params['c2'], params['c3']])
    muscle_b = jnp.array([params['b1'], params['b2'], params['b3']])
    muscle_l0 = jnp.array([params['l01'], params['l02'], params['l03']])
    
    tip_world_wrench = jnp.zeros(6)
    sim_positions = []
    
    print(f"Simulating {len(static_pressures)} points with initial guesses...")
    
    for i, pressures in enumerate(static_pressures):
        try:
            print(f"Point {i+1}/{len(static_pressures)} with pressures: {pressures}")
            
            # Compute tip moment mz for this point
            mz = params['A'] * jnp.sum(pressures) + params['mz0']
            
            # Build muscle_params for this point
            muscle_params = [
                {'r_i': float(muscle_r_i[j,0]), 'angle_i': float(jnp.arctan2(muscle_r_i[j,1], muscle_r_i[j,0])),
                 'k': float(muscle_k[j]), 'c': float(muscle_c[j]), 'b': float(muscle_b[j]), 'l0': float(muscle_l0[j])}
                for j in range(3)
            ]
            
            # Use the provided zero-pressure initial guesses with correct xi_star
            tip_x, tip_y, _, _, _ = simulate_rod_tip_jax_with_initial_guess(
                E=params['E'],
                rho=params['rho'],
                L=params['L'],
                xi_star=xi_star,  # Use correct xi_star with ux, uy for actual simulation
                muscle_params=muscle_params,
                tip_world_wrench=tip_world_wrench,
                target_pressures=pressures,
                n=n,
                r=r,
                G=G,
                f_t=jnp.zeros(3),
                l_t=jnp.array([0., 0., mz]),
                initial_xi0=zero_xi0_guess,
                initial_taus=zero_tau_guess
            )
            
            sim_positions.append([float(tip_x), float(tip_y)])
            print(f"Simulated position: ({float(tip_x)*1000:.1f}, {float(tip_y)*1000:.1f}) mm")
            
        except Exception as e:
            print(f"Simulation failed for point {i+1}: {e}")
            sim_positions.append([np.nan, np.nan])
    
    return np.array(sim_positions)

def save_optimization_results_v1(optimization_history, static_positions, static_pressures, filename='v1_optimization_results.json'):
    """
    Save V1_JAX optimization results to a JSON file.
    """
    import json
    from datetime import datetime
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'optimization_history': optimization_history,
        'final_params': optimization_history[-1]['params'] if optimization_history else {},
        'final_error': optimization_history[-1]['total_error'] if optimization_history else float('inf'),
        'final_avg_error': optimization_history[-1]['avg_error'] if optimization_history else float('inf'),
        'static_positions': static_positions.tolist(),
        'static_pressures': static_pressures.tolist(),
        'num_iterations': len(optimization_history)
    }
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Optimization results saved to {filename}")
    return filename

def plot_optimization_history_v1(optimization_history, save_plot=True):
    """
    Plot the optimization history for V1_JAX.
    """
    if not optimization_history:
        print("No optimization history to plot")
        return
    
    iterations = [h['iteration'] for h in optimization_history]
    total_errors = [h['total_error']*1000 for h in optimization_history]  # Convert to mm
    avg_errors = [h['avg_error']*1000 for h in optimization_history]  # Convert to mm
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot error convergence
    ax1.plot(iterations, total_errors, 'b-o', linewidth=2, markersize=6, label='Total Error')
    ax1.plot(iterations, avg_errors, 'r-s', linewidth=2, markersize=6, label='Average Error')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Error [mm]')
    ax1.set_title('V1_JAX Optimization Convergence')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot parameter evolution (show key parameters)
    key_params = ['E', 'ux', 'uy', 'k1', 'c1', 'A']
    for param in key_params:
        if param in optimization_history[0]['params']:
            values = [h['params'][param] for h in optimization_history]
            ax2.plot(iterations, values, 'o-', linewidth=2, markersize=4, label=param)
    
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Parameter Value')
    ax2.set_title('Key Parameter Evolution')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    if save_plot:
        plt.savefig('v1_optimization_history.png', dpi=300, bbox_inches='tight')
        print("Optimization history plot saved as v1_optimization_history.png")
    
    plt.show()

# --- End of Parameter Optimization System ---

def run_v1_parameter_optimization(data_files=None, max_iterations=20, save_results=True):
    """
    Main function to run V1_JAX parameter optimization.
    """
    if data_files is None:
        data_files = ['collected_data_0615_initPoints.csv']
    
    print("=" * 60)
    print("V1_JAX PARAMETER OPTIMIZATION")
    print("=" * 60)
    
    try:
        # Load data
        print(f"Loading data from {data_files[0]}...")
        static_positions, static_pressures = load_experiment_data(data_files[0])
        
        # Run optimization
        print("Starting parameter optimization...")
        optimized_params, optimization_history = optimize_parameters_v1_jax(
            static_positions, static_pressures, 
            max_iterations=max_iterations
        )
        
        # Save results
        if save_results:
            save_optimization_results_v1(optimization_history, static_positions, static_pressures)
        
        # Plot optimization history
        plot_optimization_history_v1(optimization_history, save_plot=save_results)
        
        # Final comparison
        print("\nFinal parameter comparison:")
        print("Optimized parameters:")
        for key, value in optimized_params.items():
            print(f"  {key}: {value}")
        
        print(f"\nFinal error: {optimization_history[-1]['total_error']*1000:.2f}mm")
        print(f"Final average error: {optimization_history[-1]['avg_error']*1000:.2f}mm")
        
        return optimized_params, optimization_history
        
    except Exception as e:
        print(f"Error in V1_JAX optimization: {e}")
        raise