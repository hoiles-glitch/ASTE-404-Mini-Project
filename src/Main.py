import numpy as np
import matplotlib.pyplot as plt

# Numerical parameters
L = 1
Nx = 200
dx = L / (Nx - 1)
# L is wall thickness (m), Nx is number of nodes, dx is spatial step

dt = 0.01
T_orbit = 5400.0
t_final = 16 * T_orbit
Nt = int(t_final / dt)
# dt is a larger timestep as test is much longer, T_orbit is period of one 90 min orbit in seconds,
# t_final is 16 orbits or 1 day, Nt is the total number of steps

n_skip = 100
Nt_store = Nt // n_skip
# n_skips creates steps of 100 data points at a time, Nt_store stores data every 100 steps

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Material properties
k = 65.0
alpha = 1.7e-5
# k is thermal conductivity, and alpha is the thermal diffusivity
# I'm keeping these constant from the homework

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Solar radiation parameters 
G_sun = 1370.0                         
sigma = 5.670374419e-8    
# G_sun is solar flux, sigma is the Stefan-Boltzmann constant

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Different coating materials
materials = {
    "Optical Solar Reflector":        {"alpha_s": 0.07, "epsilon": 0.80},
    "White Paint":                   {"alpha_s": 0.22, "epsilon": 0.85},
    "Black Paint":                   {"alpha_s": 0.97, "epsilon": 0.84},
    "Aluminized Kapton":             {"alpha_s": 0.38, "epsilon": 0.67},
    "Metallic":                      {"alpha_s": 0.13, "epsilon": 0.04},
    "MLI (White Beta Cloth)":         {"alpha_s": 0.45, "epsilon": 0.04},
    "MLI (Aluminized Beta Cloth)":    {"alpha_s": 0.37, "epsilon": 0.04},
    "MLI (Tedlar Reinforced)":        {"alpha_s": 0.30, "epsilon": 0.04},
    "MLI (Teflon-backed)":            {"alpha_s": 0.10, "epsilon": 0.04},
}
# epsilon is emissivity, alpha_s is absorptivity,

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Orbital Parameters          
omega = 2.0 * np.pi / T_orbit
phi_0 = np.pi
# T_orbit is the orbital period of 90 min, and omega is the angular frequency,
# phi_0 starts the orbit from the far side of earth

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# coefficient for time marching
c = alpha * dt / dx**2
# c is presetting the coefficient for each time step skip,

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Determine x and time for Plots
x = np.linspace(0, L, Nx)
time = np.linspace(0, t_final, Nt_store)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# compute solar trace
solar_trace = np.maximum(np.cos(omega * time + phi_0), 0.0) * G_sun

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def run_simulation(alpha_s, epsilon):
    # Initial condition
    T = np.ones(Nx) * 300.0
    # set the initial temperature to 300 K everywhere

    Tnew = np.copy(T)
    T_history = np.zeros((Nt_store, Nx))
    # update the temnperature field and store it

    store_idx = 0
    # store_idx stets the count of time jumps to 0 as a secondary time step for skips

    # FTCS time marching
    for n in range(Nt):
        t = n * dt

        # Orbital Sun Incidence Angle
        phi = omega * t
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

    return T_history

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Orbital Time Trace
plt.figure(figsize=(10,3))
plt.step(time, solar_trace, where='post', linewidth=2)
plt.xlabel("Time [s]")
plt.ylabel("Sunlight")
plt.title("Oribital Eclispse Schedule")
plt.grid(False)
plt.tight_layout()
plt.show()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Temperature contour plot
for name, props in materials.items():

    print(f"Running simulation for: {name}")

    T_hist = run_simulation(props["alpha_s"], props["epsilon"])
    
    plt.figure(figsize=(10,5))
    plt.pcolormesh(x, time, T_hist, shading='auto', cmap='inferno')
    plt.colorbar(label="Temperature [K]")
    plt.xlabel("Wall Depth [m]")
    plt.ylabel("Time [s]")
    plt.title("Temperature Evolution — Solar Radiation FTCS Contour Plot for {name} coating")
    plt.tight_layout()
    plt.show()

