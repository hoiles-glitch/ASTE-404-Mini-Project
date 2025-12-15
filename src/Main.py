import numpy as np
import matplotlib.pyplot as plt

# Numerical parameters
L = 1
Nx = 200
dx = L / (Nx - 1)
# L is wall thickness, Nx is number of nodes, dx is spatial step

dt = 0.01
T_orbit = 5400.0
t_final = 10 * T_orbit
Nt = int(t_final / dt)
# dt is a larger timestep as test is much longer, T_orbit is period of one 90 min orbit in seconds,
# t_final is 3 orbits, Nt is the total number of steps

n_skip = 100
Nt_store = Nt // n_skip
# n_skips creates steps of 100 data points at a time, Nt_store stores data every 100 steps


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

# Orbital Parameters
T_orbit = 5400.0          
omega = 2.0 * np.pi / T_orbit
# T_orbit is the orbital period of 90 min, and omega is the angular frequency


# Initial condition
T = np.ones(Nx) * 300.0
# set the initial temperature to 300 K everywhere will probably be less later

Tnew = np.copy(T)
T_history = np.zeros((Nt_store, Nx))
# update the temnperature field and store it

solar_trace = np.zeros(Nt_store)
# storing the orbital period history (cos(phi))

# coefficient for time marching
c = alpha * dt / dx**2
store_idx = 0
# c is presetting the coefficient for each time step skip,
# store_idx stets the count of time jumps to 0 as a secondary time step for skips

# FTCS time marching
for n in range(Nt):
    t = n * dt

    # Orbital Sun Incidence Angle
    phi = omega * t
    phi_0 = np.pi
    mu = np.cos(phi+phi_0)
    mu_eff = max(mu, 0.0)
    # phi is the sun incidence angle (rad), mu is the cosine of incidence, mu_eff is the effective angle of the sun
    

    # Absorbed solar heat flux
    q_in = alpha_s * G_sun * mu_eff

    # Outgoing thermal radiation from the surface
    q_rad = epsilon * sigma * T[0]**4

    # Ghost node temperature from radiative BC
    T_ghost = T[1] + (2 * dx / k) * (q_in - q_rad)

    # FTCS update at surface node
    Tnew[0] = T[0] + c * (T[1] - 2*T[0] + T_ghost)

    # Interior points
    Tnew[1:-1] = T[1:-1] + c * (T[2:] - 2*T[1:-1] + T[:-2])

    #  Right boundary (adiabatic) 
    Tnew[-1] = Tnew[-2]

    # update for next timestep
    T[:] = Tnew[:]


    # Store for each time skip
    if n % n_skip == 0:
        T_history[store_idx, :] = T
        solar_trace[store_idx] = mu_eff
        store_idx += 1


# Temperature Contour Plot
x = np.linspace(0, L, Nx)
time = np.linspace(0, t_final, Nt_store)


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
plt.step(time, solar_trace, where='post', linewidth=2)
plt.xlabel("Time [s]")
plt.ylabel("Sunlight")
plt.title("Oribital Eclispse Schedule")
plt.grid(False)
plt.tight_layout()
plt.show()
