import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─────────────────────────────────────────
#  OPTICAL FIBER ATTENUATION SIMULATION
#  Models power decay in Step-Index and
#  Graded-Index fibers for varying alpha
# ─────────────────────────────────────────

# Fiber length (km)
z = np.linspace(0, 50, 1000)

# Input power (mW)
P0 = 1.0

# Attenuation coefficients to compare (dB/km)
alphas = [0.2, 0.5, 1.0, 2.0]

# Colors for each alpha curve
colors = ['#1d9e75', '#378add', '#ba7517', '#e24b4a']


def power_step_index(P0, alpha_dB, z):
    """
    Step-Index Fiber: uniform refractive index in core.
    Power decays exponentially: P(z) = P0 * 10^(-alpha*z/10)
    alpha in dB/km, z in km
    """
    return P0 * 10 ** (-alpha_dB * z / 10)


def power_graded_index(P0, alpha_dB, z, profile_exp=2.0):
    """
    Graded-Index Fiber: parabolic refractive index profile.
    Lower modal dispersion => effective attenuation is slightly reduced
    due to better modal confinement. We model this with a correction factor
    that reduces effective loss by ~15% for alpha=2 profile (typical GI benefit).
    Correction scales with alpha to reflect the GI advantage diminishing at low alpha.
    """
    correction = 1 - 0.08 * (alpha_dB / 2.0)  # GI reduces effective loss slightly
    alpha_eff = alpha_dB * correction
    return P0 * 10 ** (-alpha_eff * z / 10)


def alpha_dB_to_Npm(alpha_dB):
    """Convert attenuation from dB/km to Np/km (Neper per km)"""
    return alpha_dB / (20 * np.log10(np.e))


# ─────────────────────────────────────────
#  PLOTTING
# ─────────────────────────────────────────

fig = plt.figure(figsize=(16, 12))
fig.patch.set_facecolor('#0f1117')
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)

ax1 = fig.add_subplot(gs[0, 0])  # Step-Index linear
ax2 = fig.add_subplot(gs[0, 1])  # Step-Index dB scale
ax3 = fig.add_subplot(gs[1, 0])  # Graded-Index linear
ax4 = fig.add_subplot(gs[1, 1])  # Step vs Graded comparison

axes = [ax1, ax2, ax3, ax4]
for ax in axes:
    ax.set_facecolor('#1a1d27')
    ax.tick_params(colors='#aaaaaa', labelsize=9)
    ax.xaxis.label.set_color('#cccccc')
    ax.yaxis.label.set_color('#cccccc')
    ax.title.set_color('#eeeeee')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333344')

# ── Plot 1: Step-Index Linear Scale ──────
for alpha, color in zip(alphas, colors):
    P = power_step_index(P0, alpha, z)
    ax1.plot(z, P * 1000, color=color, linewidth=2,
             label=f'α = {alpha} dB/km')

ax1.set_title('Step-Index Fiber  —  Power vs Distance', fontsize=11, pad=10)
ax1.set_xlabel('Distance z (km)')
ax1.set_ylabel('Optical Power (µW)')
ax1.legend(fontsize=8, facecolor='#1a1d27', edgecolor='#333344', labelcolor='#cccccc')
ax1.grid(True, color='#2a2d3a', linewidth=0.5)
ax1.set_xlim(0, 50)

# ── Plot 2: Step-Index dB Scale ───────────
for alpha, color in zip(alphas, colors):
    P = power_step_index(P0, alpha, z)
    P_dBm = 10 * np.log10(P / 1e-3)  # convert to dBm
    ax2.plot(z, P_dBm, color=color, linewidth=2,
             label=f'α = {alpha} dB/km')

ax2.set_title('Step-Index Fiber  —  Power in dBm', fontsize=11, pad=10)
ax2.set_xlabel('Distance z (km)')
ax2.set_ylabel('Power (dBm)')
ax2.legend(fontsize=8, facecolor='#1a1d27', edgecolor='#333344', labelcolor='#cccccc')
ax2.grid(True, color='#2a2d3a', linewidth=0.5)
ax2.set_xlim(0, 50)

# ── Plot 3: Graded-Index Linear Scale ────
for alpha, color in zip(alphas, colors):
    P = power_graded_index(P0, alpha, z)
    ax3.plot(z, P * 1000, color=color, linewidth=2,
             label=f'α = {alpha} dB/km')

ax3.set_title('Graded-Index Fiber  —  Power vs Distance', fontsize=11, pad=10)
ax3.set_xlabel('Distance z (km)')
ax3.set_ylabel('Optical Power (µW)')
ax3.legend(fontsize=8, facecolor='#1a1d27', edgecolor='#333344', labelcolor='#cccccc')
ax3.grid(True, color='#2a2d3a', linewidth=0.5)
ax3.set_xlim(0, 50)

# ── Plot 4: Step vs Graded at alpha=1.0 ──
alpha_compare = 1.0
P_step = power_step_index(P0, alpha_compare, z)
P_graded = power_graded_index(P0, alpha_compare, z)

ax4.plot(z, P_step * 1000, color='#e24b4a', linewidth=2.5, linestyle='--',
         label='Step-Index (α=1.0 dB/km)')
ax4.plot(z, P_graded * 1000, color='#1d9e75', linewidth=2.5,
         label='Graded-Index (α=1.0 dB/km)')

# fill between to highlight the difference
ax4.fill_between(z, P_step * 1000, P_graded * 1000,
                 color='#1d9e75', alpha=0.12, label='GI advantage region')

ax4.set_title('Step-Index vs Graded-Index  —  α = 1.0 dB/km', fontsize=11, pad=10)
ax4.set_xlabel('Distance z (km)')
ax4.set_ylabel('Optical Power (µW)')
ax4.legend(fontsize=8, facecolor='#1a1d27', edgecolor='#333344', labelcolor='#cccccc')
ax4.grid(True, color='#2a2d3a', linewidth=0.5)
ax4.set_xlim(0, 50)

# ─────────────────────────────────────────
#  SUPER TITLE
# ─────────────────────────────────────────
fig.suptitle(
    'Optical Fiber Attenuation Simulation\nEffect of attenuation coefficient α on signal power',
    fontsize=13, color='#eeeeee', y=0.98
)

plt.savefig('optical_fiber_simulation.png',
            dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.show()
print("Simulation complete. Plot saved.")

# ─────────────────────────────────────────
#  PRINT SUMMARY TABLE
# ─────────────────────────────────────────
print("\n── Power Remaining at key distances (Step-Index) ──")
print(f"{'Alpha (dB/km)':<16} {'10 km':<12} {'25 km':<12} {'50 km':<12}")
print("─" * 52)
for alpha in alphas:
    p10  = power_step_index(P0, alpha, 10)
    p25  = power_step_index(P0, alpha, 25)
    p50  = power_step_index(P0, alpha, 50)
    print(f"{alpha:<16} {p10*100:>8.4f}%   {p25*100:>8.4f}%   {p50*100:>8.6f}%")

print("\n── Power Remaining at key distances (Graded-Index) ──")
print(f"{'Alpha (dB/km)':<16} {'10 km':<12} {'25 km':<12} {'50 km':<12}")
print("─" * 52)
for alpha in alphas:
    p10  = power_graded_index(P0, alpha, 10)
    p25  = power_graded_index(P0, alpha, 25)
    p50  = power_graded_index(P0, alpha, 50)
    print(f"{alpha:<16} {p10*100:>8.4f}%   {p25*100:>8.4f}%   {p50*100:>8.6f}%")