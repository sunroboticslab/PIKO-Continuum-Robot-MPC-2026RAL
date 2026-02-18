import numpy as np
import matplotlib.pyplot as plt
from svgpathtools import svg2paths
from matplotlib.animation import FuncAnimation
# Load SVG and extract paths
paths, attributes = svg2paths(r'/root/DataCollection/harp_oneseg_ws/src/data_collection/data_collection/RAL3.svg')
path = paths[0]
start_pt = np.array([-0.04, -0.02])
Scale = 0.1 / 1700.0
N = 9000

points = [path.point(t) for t in np.linspace(0, 1, N)]
x = np.array([Scale * p.real - .05 for p in points])
y = np.array([Scale * p.imag - .05 for p in points])
y = -y  # Optional: flip Y

# Offset by current start point
x = x + start_pt[0]
y = y + start_pt[1]

traj_svg = np.column_stack([x, y])

# Flip the trajectory direction: reverse the SVG trajectory
# traj_svg = traj_svg[::-1]  # Reverse the trajectory

# Hold the starting point (now the original end point) for 1000 steps
hold_start = np.repeat(traj_svg[0:1, :], 1000, axis=0)
# Hold the ending point (now the original start point) for 1000 steps
hold_end   = np.repeat(traj_svg[-1:, :], 1000, axis=0)

# Final trajectory: [hold_start, flipped SVG trajectory, hold_end]
# Now the robot will follow from the original end point back to the original start point
trajectory = np.vstack([hold_start, traj_svg, hold_end])

# -------------- animate (works across backends) --------------
fig, ax = plt.subplots(figsize=(6,6))
ax.set_title("Generated RAL Trajectory (animated)")
ax.set_xlabel("dx"); ax.set_ylabel("dz"); ax.set_aspect("equal", "box")
pad = 0.02
ax.set_xlim(trajectory[:,0].min()-pad, trajectory[:,0].max()+pad)
ax.set_ylim(trajectory[:,1].min()-pad, trajectory[:,1].max()+pad)
line, = ax.plot([], [], 'r-', lw=1)
dot,  = ax.plot([], [], 'bo', ms=4)

def init():
    line.set_data([], []); dot.set_data([], [])
    return line, dot

def update(i):
    line.set_data(trajectory[:i+1,0], trajectory[:i+1,1])
    dot.set_data(trajectory[i,0], trajectory[i,1])
    return line, dot

ani = FuncAnimation(fig, update, init_func=init,
                    frames=len(trajectory), interval=1, blit=True)
plt.show()
