import numpy as np
import matplotlib.pyplot as plt

# --- 1. Configuration ---
L = 2000      # Length of fiber (microns)
a = 50        # Core radius (microns)
n1 = 1.5      # Center refractive index
delta = 0.01  # Index difference
c = 3e8       # Speed of light (m/s)
steps = 5000  
dx = L / steps
x_vals = np.linspace(0, L, steps)

def simulate_with_analysis(mode='step'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={'width_ratios': [3, 1]})
    angles = np.linspace(1, 12, 10) # 10 rays entering at different angles
    arrival_times = []

    for angle in angles:
        y = np.zeros(steps)
        theta = np.radians(angle)
        curr_y = 0
        curr_slope = np.tan(theta)
        total_time = 0
        
        for i in range(steps):
            y[i] = curr_y
            
            # Distance traveled in this step (Hypotenuse)
            dy = curr_slope * dx
            ds = np.sqrt(dx**2 + dy**2)
            
            # Local refractive index
            if mode == 'step':
                n_local = n1
                curr_y += dy
                if abs(curr_y) >= a:
                    curr_slope = -curr_slope
            else: # Graded
                n_local = n1 * (1 - delta * (curr_y/a)**2)
                acceleration = -(2 * n1 * delta / a**2) * curr_y
                curr_slope += acceleration * dx
                curr_y += curr_slope * dx
            
            # Time taken for this step (in picoseconds for visibility)
            total_time += (n_local * ds) / (c * 1e-6) # Scale units
            
        arrival_times.append(total_time)
        ax1.plot(x_vals, y, alpha=0.7)

    # --- Plot 1: Ray Tracing ---
    ax1.axhline(y=a, color='k', linestyle='--')
    ax1.axhline(y=-a, color='k', linestyle='--')
    ax1.set_title(f'{mode.capitalize()}-Index: Ray Propagation')
    ax1.set_ylabel('Core Position (y)')
    ax1.set_xlabel('Distance (x)')

    # --- Plot 2: Pulse Spread (The "Proof") ---
    # We plot the arrival times. A narrow peak = low loss/dispersion.
    ax2.hist(arrival_times, bins=10, color='orange', edgecolor='black')
    ax2.set_title('Signal Pulse at Exit')
    ax2.set_xlabel('Time of Arrival (ps)')
    ax2.set_ylabel('Intensity')
    
    # Calculate Spread (Dispersion)
    spread = max(arrival_times) - min(arrival_times)
    ax2.annotate(f'Pulse Spread:\n{spread:.4f} ps', xy=(0.5, 0.8), xycoords='axes fraction', ha='center', color='red', weight='bold')

    plt.tight_layout()
    plt.show()

# Run the comparison
simulate_with_analysis(mode='step')
simulate_with_analysis(mode='graded')