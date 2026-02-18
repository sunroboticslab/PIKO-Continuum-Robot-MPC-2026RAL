import pandas as pd
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# 1) Load the MPC log
df = pd.read_csv('mpc_log.csv')

# 2) Plot reference vs actual trajectory
fig, ax = plt.subplots(figsize=(6, 6))

# Explicitly convert to 1-D numpy arrays:
ref_x = df['ref_x'].values
ref_y = df['ref_y'].values
act_dx = df['dx'].values
act_dz = df['dz'].values

ax.plot(ref_x, ref_y, 'b--', label='Reference Trajectory')
ax.plot(act_dx, act_dz, '-', alpha=0.1, label='Actual dx,dz')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Reference vs Actual Trajectory')
ax.legend()
plt.tight_layout()
plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# 3) Slice into full‐circle segments and plot them on a 0…1999 x‐axis, then
#    shade the envelope (min→max) and draw the mean‐error line.
# ─────────────────────────────────────────────────────────────────────────────

# (A) Define constants for “circle_10x”
n_per_circle   = 2000   # each circle has 2000 points
guide1_len     = 1000   # points in guide1
guide2_len     = 0   # points in guide2
start_circle   = guide1_len + guide2_len

# (B) Extract only the “circle_10x” block (whatever length it is)
circle_block = df.iloc[start_circle : len(df)].reset_index(drop=True)

# Compute Euclidean error over all those points:
err_euclid = np.sqrt(
    (circle_block['dx'].values   - circle_block['ref_x'].values)**2 +
    (circle_block['dz'].values   - circle_block['ref_y'].values)**2
)
# err_euclid.shape == (total_points,)

# --- New line: compute and print the overall MSE of these circle‐tracking errors ---
mse_all = np.mean(err_euclid**2) **(1/2)
print(f"Overall MSE (all circle points): {mse_all:.6e}")

# (C) Figure out how many *full* 2000‐point circles we actually have:
n_full_reps = err_euclid.size // n_per_circle
if n_full_reps == 0:
    raise RuntimeError(
        f"Less than {n_per_circle} points found ({err_euclid.size} total). "
        "Cannot extract a single full circle."
    )

# Warn if there are leftover points beyond the last full circle:
remainder = err_euclid.size % n_per_circle
if remainder != 0:
    print(
        f"⚠️  Found {err_euclid.size} points, which is {n_full_reps} full circles "
        f"plus {remainder} leftover points. Dropping the last {remainder} points."
    )

# (D) Only keep the first (n_full_reps × 2000) points
n_to_keep     = n_full_reps * n_per_circle
err_full      = err_euclid[:n_to_keep]           # length = n_full_reps * 2000
err_matrix_raw = err_full.reshape(n_full_reps, n_per_circle)  # shape = (n_full_reps, 2000)
err_matrix     = err_matrix_raw.T                             # shape = (2000, n_full_reps)

# (E) Compute pointwise min, max, mean across the n_full_reps runs
mean_err = err_matrix.mean(axis=1)  # (2000,)
min_err  = err_matrix.min(axis=1)   # (2000,)
max_err  = err_matrix.max(axis=1)   # (2000,)

# (F) Plot everything on a 0…1999 x‐axis
x_axis = np.arange(n_per_circle)  # [0,1,…,1999]

fig, ax = plt.subplots(figsize=(8,4))

# 1) Plot each of the n_full_reps raw runs in light red (alpha=0.3)
for run_idx in range(n_full_reps):
    ax.plot(
        x_axis,
        err_matrix[:, run_idx],
        color='tab:red',
        alpha=0.3,
        linewidth=1,
        label='_nolegend_'
    )

# 2) Shade the “min→max” envelope in a translucent band
ax.fill_between(
    x_axis,
    min_err,
    max_err,
    color='lightcoral',
    alpha=0.2,
    label='Error Envelope (min→max)'
)

# 3) Plot the mean‐error curve on top in a darker red
ax.plot(
    x_axis,
    mean_err,
    color='darkred',
    linewidth=2,
    label=f'Mean Error (over {n_full_reps} circles)'
)

ax.set_xlabel(f'Circle Step (0 … {n_per_circle-1})')
ax.set_ylabel('Euclidean Error')
ax.set_title('Error Envelope & Mean Error per Circle Step')
ax.grid(True)
ax.legend(loc='upper right')
plt.tight_layout()
plt.show()

# 5) Animate dx/dz vs reference
# fig, ax = plt.subplots(figsize=(6,6))
# ax.plot(df['ref_x'], df['ref_y'], 'b--', alpha=0.5, label='Reference')
# trace_line, = ax.plot([], [], 'r-', linewidth=1.5, alpha=0.5, label='Actual Trace')
# ref_pt,     = ax.plot([], [], 'bo', markersize=5, label='Ref Point')
# act_pt,     = ax.plot([], [], 'ro', markersize=5, label='Actual Point')
# ax.set_xlabel("dx")
# ax.set_ylabel("dz")
# ax.set_title("Tracking Animation")
# ax.set_aspect('equal')
# ax.legend()

# # fixed axis limits
# padding = 0.005
# xmin = min(df['dx'].min(), df['ref_x'].min()) - padding
# xmax = max(df['dx'].max(), df['ref_x'].max()) + padding
# ymin = min(df['dz'].min(), df['ref_y'].min()) - padding
# ymax = max(df['dz'].max(), df['ref_y'].max()) + padding
# ax.set_xlim(xmin, xmax)
# ax.set_ylim(ymin, ymax)

# # buffer to hold trace
# trace_x, trace_y = [], []

# def update(i):
#     ref_pt.set_data([df['ref_x'][i]], [df['ref_y'][i]])
#     act_pt.set_data([df['dx'][i]],    [df['dz'][i]])
#     trace_x.append(df['dx'][i])
#     trace_y.append(df['dz'][i])
#     trace_line.set_data(trace_x, trace_y)
#     return ref_pt, act_pt, trace_line

# ani = animation.FuncAnimation(fig, update, frames=len(df), interval=1, blit=True)

# # Save to video
# writer = animation.FFMpegWriter(fps=60)
# ani.save("mpc_tracking.mp4", writer=writer)
# print("🎥 Saved to mpc_tracking.mp4")

# plt.close()  # to avoid showing again
