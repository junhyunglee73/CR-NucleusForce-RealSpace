import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# ========================================
# System Configuration & Geometry
# ========================================
HEIGHT, WIDTH = 400, 400
dt = 1
T_steps = 400
Total_steps = int(T_steps / dt)
cy, cx = HEIGHT // 2, WIDTH // 2
y, x = np.ogrid[:HEIGHT, :WIDTH]
dist_map = np.sqrt((x - cx)**2 + (y - cy)**2) * dt
Z = 1 # Atom number (can be modified for different atoms)
ZS = 1 + ((Z - 1) * 0.002485)**2  # Slightly adjusted Z for scaling

Shell_spread_factor = (ZS * Z) ** 0.000335 # volume scaling factor for the shell spread
effective_dist = dist_map / Shell_spread_factor

# ========================================
# Spatial Structural Field (beta) & Input Force magnitude (A)
# ========================================
beta_field = np.ones((HEIGHT, WIDTH))
input_force_magnitude = (13.598434599702 * 4) * ((ZS * Z)**2)  
dissipation = 1 /(1 + (effective_dist * 0.02))**2

A = input_force_magnitude * beta_field * dissipation

# ========================================
# Physical Parameters (The Causes)
# ========================================
stiffness_k = 0.2
mass_m = 50
damping_c = 0.0
V_record = 1.0
    
gamma = damping_c / (2 * mass_m)
w_squared = (stiffness_k / mass_m) - gamma**2
natural_freq = np.sqrt(w_squared) if w_squared > 0 else 0.0

# ========================================
# Causal Operators & Hookean State Operator
# ========================================
def H_time(tau):
    return (tau >= 0).astype(float)

def hookean_state_operator(tau):
    return np.cos(natural_freq * tau)

# ========================================
# Governing Equation (Causal Recording Kernel)
# ========================================
def causal_record_kernel(t_current, r_dist):
    tau = t_current - (r_dist / V_record)
    return np.exp(-gamma * tau) * hookean_state_operator(tau) * H_time(tau)

# ========================================
# Front finder 
# ========================================
x_idx = np.arange(WIDTH)
dist_line = dist_map[cy, :]

def find_front_x(current_time):
    ct_line = current_time[cy, :] if np.ndim(current_time) == 2 else current_time
    tau_line = ct_line - dist_line / V_record
    reached = (tau_line >= 0) & (x_idx >= cx)    
    return int(x_idx[reached].max()) if reached.any() else cx

# ========================================
# Time Evolution & Snapshot Recording
# ========================================
snap_surface_2d     = []
snap_cross_1d      = []
snap_lagrangian_1d = []
capture_times      = []
capture_front_x    = []      # front at each snapshot
capture_front_val  = []      # force at front for each snapshot
capture_steps = [50 / dt, 150 / dt]  # steps at which to capture snapshots

front_x_hist   = []          # front position history over time
front_val_hist = []          # force at front history over time
time_hist      = []          # time history

# Source-position time histories: x = cx, y = cy
source_causal_hist = []
source_lagrangian_hist = []
source_conservative_hist = []

for step in range(Total_steps + 1):
    t_eff = step * dt 
    current_time = t_eff * beta_field
    theta = current_time - (dist_map / V_record)

    surface_record = A * causal_record_kernel(current_time, dist_map)
    lagrangian = A * np.exp(-gamma * theta) * hookean_state_operator(theta)
    
    # Record the response at the source position
    source_causal_hist.append(surface_record[cy, cx])
    source_lagrangian_hist.append(lagrangian[cy, cx])

    # Undamped conservative reference (c = 0)
    omega_0 = np.sqrt(stiffness_k / mass_m)
    source_conservative_hist.append(
        A[cy, cx] * np.cos(omega_0 * t_eff)
    )

    # Find front position and force
    fx = find_front_x(current_time)
    fv = surface_record[cy, fx]
    
    time_hist.append(t_eff)
    front_x_hist.append(fx)
    front_val_hist.append(fv)

    if step in capture_steps:
        snap_surface_2d.append(surface_record.copy())
        snap_cross_1d.append(surface_record[cy, :].copy())
        snap_lagrangian_1d.append(lagrangian[cy, :].copy())
        capture_times.append(step)
        capture_front_x.append(fx)
        capture_front_val.append(fv)
        print(f">> Snapshot step {step}: t_eff={t_eff:.1f}, front_x={fx}, force={fv:.12f}")

# ========================================
# CSV Export Logic for Origin / Excel
# ========================================
# 1. save time history data (Time, Front_X, Front_Force, Source_Causal, Source_Lagrangian)
history_data = np.column_stack((
    time_hist, 
    front_x_hist, 
    front_val_hist, 
    source_causal_hist, 
    source_lagrangian_hist
))
np.savetxt(
    "simulation_time_history.csv", 
    history_data, 
    delimiter=",", 
    header="Time,Front_X,Front_Force,Source_Causal,Source_Lagrangian", 
    comments=""
)
print(">> Saved: simulation_time_history.csv")

# 2. save snapshot cross-section data (X_Coord, Snap1_Causal, Snap2_Causal, Snap3_Causal, Snap4_Causal)
# : [X_Coord, Snap1_Causal, Snap2_Causal, Snap3_Causal, Snap4_Causal]
snapshot_matrix = [x_idx]
snap_header = "X_Coord"
for i, s_step in enumerate(capture_steps):
    snapshot_matrix.append(snap_cross_1d[i])
    snap_header += f",Snapshot_t{s_step}_Causal"

cross_section_data = np.column_stack(snapshot_matrix)
np.savetxt(
    "snapshot_cross_sections.csv", 
    cross_section_data, 
    delimiter=",", 
    header=snap_header, 
    comments=""
)
print(">> Saved: snapshot_cross_sections.csv")


# ========================================
# Visualization
# ========================================
num_snaps = len(snap_surface_2d)
fig, axes = plt.subplots(2, num_snaps, figsize=(16, 8), constrained_layout=True)
VISUAL_LIMIT = input_force_magnitude * 1.2
norm = TwoSlopeNorm(vmin=-VISUAL_LIMIT, vcenter=0, vmax=VISUAL_LIMIT)

for i in range(num_snaps):
    im1 = axes[0, i].imshow(snap_surface_2d[i], cmap="seismic", norm=norm, origin="lower")
    axes[0, i].set_title(f"Surface Record (t={capture_times[i]})")
    axes[0, i].axis("off")
    axes[0, i].axhline(cy, color="black", linestyle="--", alpha=0.3)

    axes[1, i].plot(x_idx, snap_cross_1d[i], color="black", linewidth=1.5, label="Causal Record")
    axes[1, i].plot(x_idx, snap_lagrangian_1d[i], color="red", linestyle="--", linewidth=1.5, label="Lagrangian Path")

    # front position and force annotation
    front_x = capture_front_x[i]
    force_value = capture_front_val[i]

    axes[1, i].scatter(front_x, force_value, color="blue", s=40, zorder=5, label="Front force")
    axes[1, i].axvline(front_x, color="blue", linestyle=":", alpha=0.4)
    axes[1, i].text(min(front_x + 10, WIDTH - 150), force_value + 1.5,
                    f"Force={force_value:.12f}", color="blue", fontsize=10)

    axes[1, i].set_xlim(0, WIDTH)
    axes[1, i].set_ylim(-VISUAL_LIMIT * 1.2, VISUAL_LIMIT * 1.2)
    axes[1, i].grid(True, linestyle=":", alpha=0.5)
    axes[1, i].legend()

cbar = fig.colorbar(im1, ax=axes[0, :], location="right", fraction=0.015, pad=0.02)

plt.show()