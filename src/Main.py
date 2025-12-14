import numpy as np
import matplotlib.pyplot as plt

# Numerical parameters
L = 0.01
Nx = 100
dx = L / (Nx - 1)
# L is wall thickness may make thicker in future, Nx is number of nodes, dx is spatial step

dt = 1e-4
Nt = 15000
# dt is time step, Nt is number of time steps,
# both will chnage when I make orbital period longer

# Material properties
k = 65.0
alpha = 1.7e-5
# k is thermal conductivity, and alpha is the thermal diffusivity
# I'm keeping these constant from the homework

# Solar radiation parameters 
G_sun = 1370.0            
alpha_s = 0.7             
epsilon = 0.8             
sigma = 5.670374419e-8    
# G_sun is solar irridance alpha_s is absorptivity,
# epsilon is emissivity, sigma is the Stefan-Boltzmann constant

# Pulse schedule parameters
pulse_on = 0.15
pulse_period = 0.20
# pulse period will eventually change to match an orbit


# Initial condition
T = np.ones(Nx) * 300.0
# set the initial temperature to 300 K everywhere will probably be less later

Tnew = np.copy(T)
T_history = np.zeros((Nt, Nx))
# update the temnperature field and store it

# To store the pulse schedule at each timestep
pulse_trace = np.zeros(Nt)
# storing the orbital period



# FTCS time marching
for n in range(Nt):
    t = n * dt

    sunlit = (t % pulse_period) < pulse_on
    pulse_trace[n] = 1.0 if sunlit else 0.0

    # Incoming absorbed solar heat flux
    if sunlit:
        q_in = alpha_s * G_sun
    else:
        q_in = 0.0

    # Outgoing thermal radiation from the surface
    q_rad = epsilon * sigma * T[0]**4

    # Ghost node temperature from radiative BC
    T_ghost = T[1] + (2 * dx / k) * (q_in - q_rad)

    # FTCS update at surface node
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
plt.title("Temperature Evolution — Solar Radiation FTCS Contour Plot")
plt.tight_layout()
plt.show()


# The following trace will change to be a sine-like wave

# Square Pulse Time Trace
plt.figure(figsize=(10,3))
plt.step(time, pulse_trace, where='post', linewidth=2)
plt.ylim(-0.2, 1.2)
plt.xlabel("Time [s]")
plt.ylabel("Sunlight")
plt.title("Oribital Eclispse Schedule")
plt.grid(True)
plt.tight_layout()
plt.show()
