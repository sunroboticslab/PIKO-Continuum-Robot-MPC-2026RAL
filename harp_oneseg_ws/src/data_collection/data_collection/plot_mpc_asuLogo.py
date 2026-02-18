import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 1) Load the MPC log (update this path if needed)
df = pd.read_csv('mpc_log.csv')

# Extract only the true trajectory (exclude first 1000 and last 1000 points)
start_idx = 1000
end_idx = len(df) - 1000

# Get the true trajectory data
true_ref_x = df['ref_x'].values[start_idx:end_idx]
true_ref_y = df['ref_y'].values[start_idx:end_idx]
true_act_dx = df['dx'].values[start_idx:end_idx]
true_act_dz = df['dz'].values[start_idx:end_idx]

# 2) Plot the reference vs actual trajectory (true trajectory only)
fig, ax = plt.subplots(figsize=(8, 8))

ax.plot(true_ref_x, true_ref_y, 'b-', linewidth=2, alpha=0.4, label='Reference Path')
ax.plot(true_act_dx, true_act_dz, '-', color='darkorange', alpha=0.5, label='Actual Trajectory')

# Markers for start/end of true trajectory
ax.plot(true_ref_x[0], true_ref_y[0], 'go', markersize=8, label='Start (Ref)')
ax.plot(true_ref_x[-1], true_ref_y[-1], 'ro', markersize=8, label='End (Ref)')
ax.plot(true_act_dx[0], true_act_dz[0], 'g*', markersize=10, label='Start (Actual)')
ax.plot(true_act_dx[-1], true_act_dz[-1], 'r*', markersize=10, label='End (Actual)')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('ASU Logo: True Trajectory - Reference vs Actual')
ax.set_aspect('equal')
ax.legend()
plt.tight_layout()
plt.show()

# 3) Plot tracking error along the true trajectory with time axis
err_euclid = np.sqrt((true_act_dx - true_ref_x)**2 + (true_act_dz - true_ref_y)**2)

# Create time axis (0.01s per point)
time_axis = np.arange(len(err_euclid)) * 0.01  # 0.01s per point

fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(time_axis, err_euclid, 'crimson', alpha=0.8)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Euclidean Error')
ax.set_title('Tracking Error (Actual vs Reference) - True Trajectory')
ax.grid(True)
plt.tight_layout()
plt.show()

# Calculate MSE for the true trajectory
mse = np.mean(err_euclid**2)
print(f'True Trajectory Tracking:')
print(f'  Mean Error = {np.mean(err_euclid):.3f}')
print(f'  Max Error = {np.max(err_euclid):.3f}')
print(f'  MSE = {mse:.6f}')

# 4) Animate actual vs reference trajectory (true trajectory only)
fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(true_ref_x, true_ref_y, 'b-', linewidth=2, alpha=0.4, label='Reference Path')
trace_line, = ax.plot([], [], 'r-', linewidth=2, alpha=0.7, label='Actual Trace')
ref_pt, = ax.plot([], [], 'bo', markersize=8, label='Current Reference')
act_pt, = ax.plot([], [], 'ro', markersize=8, label='Current Actual')

padding = 0.01
xmin = min(true_ref_x.min(), true_act_dx.min()) - padding
xmax = max(true_ref_x.max(), true_act_dx.max()) + padding
ymin = min(true_ref_y.min(), true_act_dz.min()) - padding
ymax = max(true_ref_y.max(), true_act_dz.max()) + padding
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_aspect('equal')
ax.set_title('ASU Logo: True Trajectory Tracking Animation')
ax.legend()

trace_x, trace_y = [], []
def update(i):
    ref_pt.set_data(true_ref_x[i], true_ref_y[i])
    act_pt.set_data(true_act_dx[i], true_act_dz[i])
    trace_x.append(true_act_dx[i])
    trace_y.append(true_act_dz[i])
    trace_line.set_data(trace_x, trace_y)
    return ref_pt, act_pt, trace_line

ani = animation.FuncAnimation(fig, update, frames=len(true_ref_x), interval=1, blit=True)

# Save to video
writer = animation.FFMpegWriter(fps=100)
ani.save("mpc_asu_true_trajectory_tracking.mp4", writer=writer)
print("🎥 Saved true trajectory tracking animation to mpc_asu_true_trajectory_tracking.mp4")

plt.close()
