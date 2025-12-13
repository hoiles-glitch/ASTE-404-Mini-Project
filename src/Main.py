import numpy as np
import matplotlib.pyplot as plt

# Numerical parameters
L = 0.01
Nx = 100
dx = L / (Nx - 1)

dt = 1e-4
Nt = 14000


# Material properties
k = 65.0
alpha = 1.7e-5
hg = 1.5e4
T0g = 2500


# Pulse schedule parameters
pulse_on = 0.15
pulse_period = 0.20


# Initial condition
T = np.ones(Nx) * 300.0
Tnew = np.copy(T)
T_history = np.zeros((Nt, Nx))

# To store the pulse schedule at each timestep
pulse_trace = np.zeros(Nt)


# FTCS time marching
for n in range(Nt):
    t = n * dt

    firing = (t % pulse_period) < pulse_on
    pulse_trace[n] = 1.0 if firing else 0.0

    # Left boundary (x=0) ghost node 
    if firing:
        # convection BC while on
        T_ghost = T[1] + (2 * dx * hg / k) * (T0g - T[0])
    else:
        # adiabatic while off
        T_ghost = T[1]

    # FTCS update at left boundary
    Tnew[0] = T[0] + alpha * dt * (T[1] - 2*T[0] + T_ghost) / dx**2

    # Interior points 
    for i in range(1, Nx-1):
        Tnew[i] = T[i] + alpha * dt * (T[i+1] - 2*T[i] + T[i-1]) / dx**2

    #  Right boundary (adiabatic) 
    Tnew[-1] = Tnew[-2]

    # update for next timestep
    T[:] = Tnew[:]
    T_history[n, :] = T[:]


# Temperature Contour Plot
x = np.linspace(0, L, Nx)
time = np.arange(Nt) * dt

plt.figure(figsize=(10,5))
plt.pcolormesh(x, time, T_history, shading='auto', cmap='inferno')
plt.colorbar(label="Temperature [K]")
plt.xlabel("Wall Depth [m]")
plt.ylabel("Time [s]")
plt.title("Temperature Evolution — FTCS Contour Plot")
plt.tight_layout()
plt.show()


# Square Pulse Time Trace
plt.figure(figsize=(10,3))
plt.step(time, pulse_trace, where='post', linewidth=2)
plt.ylim(-0.2, 1.2)
plt.xlabel("Time [s]")
plt.ylabel("Pulse")
plt.title("Rocket Pulse Schedule")
plt.grid(True)
plt.tight_layout()
plt.show()
