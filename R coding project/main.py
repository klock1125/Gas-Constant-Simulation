import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# variables
wall = 10 # (meters)
#n = int(input("Number of particles: ")) # number of particle
n = 100
moles = n / (6.022 * 10**23) # number of moles of particles
#t = float(input("Temperature (K): ")) # temperature (K)
t = 300
#molar_mass = float(input("Molar mass (kg): ")) # molar mass (kg)
molar_mass = 0.032
m = molar_mass / (6.022 * 10**23) # mass of particle (kg)

volume = wall ** 3 * 1000 # (liters)


# constants
KB = 1.380649 * 10**-23

# plot
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_xlim(0, wall), ax.set_ylim(0, wall), ax.set_zlim(0, wall)

plt.figtext(0.02, 0.98, "Temperature (K): " + str(t), va="top", ha="left")
plt.figtext(0.02, 0.94, "Number of particles: " + str(n), va="top", ha="left")
plt.figtext(0.02, 0.90, "Volume (L): " + str(format(volume, ".2e")), va="top", ha="left")
plt.figtext(0.02, 0.84, "Molar mass (kg): " + str(molar_mass), va="top", ha="left")

pressure_display = plt.figtext(0.02, 0.06, "Average pressure: ", va="bottom", ha="left")
R_display = plt.figtext(0.02, 0.02, "Calculated gas constant: ", va="bottom", ha="left")

# initializing particle positions
rng = np.random.default_rng()
pos = rng.uniform(low=0, high=wall, size=(n,3))
points = ax.scatter(pos[:, 0], pos[:, 1], pos[:,2])

# initializing velocities from maxwell-boltzman distr.
sd = (KB * t / m) ** 0.5 # standard deviation
vel = rng.normal(0, sd, size=(n,3))

# zeroing momentum
avg_vel = np.array([np.sum(vel[:,0]), np.sum(vel[:,1]), np.sum(vel[:,2])]) / n
for i in range(len(vel)):
    vel[i] = vel[i] - avg_vel

# temperature rescaling
mean_square_v = np.sum(vel ** 2) / n
curr_temp = (m * mean_square_v) / (3 * KB)
vel = vel * (t / curr_temp) ** 0.5

interval_count = 0
pressure_sum = 0
#pressure_display = ax.text(0, 0, 0, "", ha="right")

def update(frame):
    global pos, vel, pressure_sum, interval_count
    impulse = 0 # reset impulse
    
    dt = ani.event_source.interval * 0.001 # seconds per update interval

    pos = pos + vel * dt # updates position
    points._offsets3d = (pos[:, 0], pos[:, 1], pos[:,2]) # updates plot

    # wall collisions
    for x, v in zip(pos, vel): # x: position, v: velocity
        for axis in range(3):
            while x[axis] >= wall or x[axis] <= 0:
                if x[axis] >= wall:
                    impulse += abs(2 * m * v[axis])
                    x[axis] = wall - (x[axis] - wall)
                    v[axis] *= -1
                elif x[axis] <= 0:
                    impulse += abs(2 * m * v[axis])
                    x[axis] = -x[axis]
                    v[axis] *= -1

    curr_temp = (m * np.sum(vel ** 2) / n) / (3 * KB)
    
    pressure_pasc = impulse / ((6 * wall ** 2) * dt) # (pascal)
    pressure_sum += pressure_pasc / 101325 # (atm)

    #print(r)

    interval_count += 1

    average_pressure = pressure_sum / interval_count
    r = average_pressure * volume / (moles * curr_temp)


    pressure_display.set_text("Average pressure: " + str(format(average_pressure, ".2e")))
    R_display.set_text("Calculated R: " + str(format(r, ".4")))
    return points,


ani = animation.FuncAnimation(fig=fig, func=update, frames=40, interval=50)
plt.show()