import numpy as np
import matplotlib.pyplot as plt

# Constants
G = 6.67430e-11       # gravitational constant (m^3 kg^-1 s^-2)
M_earth = 5.972e24    # mass of Earth (kg)
R_earth = 6.371e6     # radius of Earth (m)
atm_limit = R_earth + 100000  # atmosphere ends ~100 km above surface

# Micrometeoroid initial conditions (can be varied)
mass = 0.005        # in kg (1g)
speed = 1200       # m/s (typical entry speed)
angle_deg = 40      # degrees with horizontal
angle_rad = np.radians(angle_deg)

# Position and velocity vectors
x, y = 0, atm_limit  # start at top of atmosphere
vx = speed * np.cos(angle_rad)
vy = -speed * np.sin(angle_rad)

# Simulation time setup
dt = 0.1           # time step in seconds
t_max = 2000       # max simulation time (s)
positions_x = []
positions_y = []

# Gravity-only simulation (no drag in this version)
for _ in range(int(t_max / dt)):
    r = np.sqrt(x**2 + y**2)
    if r <= R_earth:
        print("🌍 Impacted Earth!")
        break

    # Gravitational acceleration
    a = -G * M_earth / r**2
    ax = a * x / r
    ay = a * y / r

    # Update velocities
    vx += ax * dt
    vy += ay * dt

    # Update positions
    x += vx * dt
    y += vy * dt

    positions_x.append(x / 1000)  # convert to km
    positions_y.append(y / 1000)

# Plot the trajectory
plt.figure(figsize=(8, 6))
plt.plot(positions_x, positions_y,color='pink',label="Micrometeoroid Path")
circle = plt.Circle((0, 0), R_earth / 1000, color='blue', alpha=0.5, label="Earth")
plt.gca().add_patch(circle)

plt.xlabel("x (km)")
plt.ylabel("y (km)")
plt.title("Micrometeoroid Entry Simulation")
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()
