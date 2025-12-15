import numpy as np
import matplotlib.pyplot as plt

# Numerical parameters
L = 0.1
Nx = 100
dx = L / (Nx - 1)
# L is wall thickness may make thicker in future, Nx is number of nodes, dx is spatial step

dt = 1e-3
Nt = 108000000
# dt is time step, Nt is number of time steps,
# both will change when I make orbital period longer

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
T_history = np.zeros((Nt, Nx))
# update the temnperature field and store it

solar_trace = np.zeros(Nt)
# storing the orbital period history (cos(phi))


# FTCS time marching
for n in range(Nt):
    t = n * dt

    # Orbital Sun Incidence Angle
    phi = omega * t
    mu = np.cos(phi)
    mu_eff = max(mu, 0.0)
    # phi is the sun incidence angle (rad), mu is the cosine of incidence, mu_eff is the effective angle of the sun
    
    solar_trace[n] = mu_eff
    # the solar trace of solar flux

    # Absorbed solar heat flux
    q_in = alpha_s * G_sun * mu_eff

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
plt.step(time, solar_trace, where='post', linewidth=2)
plt.ylim(-0.2, 1.2)
plt.xlabel("Time [s]")
plt.ylabel("Sunlight")
plt.title("Oribital Eclispse Schedule")
plt.grid(True)
plt.tight_layout()
plt.show()
