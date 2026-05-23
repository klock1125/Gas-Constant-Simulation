import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button

KB = 1.380649 * 10**-23
SPEED_ADJUSTMENT = 2500000 # increase to slow down simulation

# 3D plot (particle simulation)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.tick_params(labelsize=7)

pressure_display = plt.figtext(0.98, 0.98, "Average pressure (atm): ", va="top", ha="right")
R_display = plt.figtext(0.98, 0.94, "Calculated R (L atm mol\u207b\u00b9 K\u207b\u00b9): ", va="top", ha="right")

# number of particles
n_slider = Slider(ax=fig.add_axes([0.25, 0.18, 0.5, 0.025]), label='Number of particles', valmin = 5, valmax = 1000, valinit=300, valstep=1)
n = n_slider.val
# temperature
t_slider = Slider(ax=fig.add_axes([0.25, 0.13, 0.5, 0.025]), label='Temperature (K)', valmin = 5, valmax = 2000, valinit=298.15, valstep=0.01, valfmt='%1.2f')
t = t_slider.val
# volume
v_slider = Slider(ax=fig.add_axes([0.25, 0.08, 0.5, 0.025]), label='Volume (L)', valmin = 0.01, valmax = 10, valinit=1, valstep=0.01, valfmt='%1.2f')
volume = v_slider.val
# molar mass
mm_slider = Slider(ax=fig.add_axes([0.25, 0.03, 0.5, 0.025]), label='Molar mass (g)', valmin = 1, valmax = 200, valinit = 32, valstep=0.01, valfmt='%1.2f')
molar_mass = mm_slider.val * SPEED_ADJUSTMENT

#color
red_slider = Slider(ax=fig.add_axes([0.04, 0.8, 0.01, 0.15]), label="R", valmin = 0, valmax = 255, valinit = 31, orientation="vertical", valstep=1, handle_style={'size': 6} )
red_slider.label.set_size(8)
red_slider.valtext.set_size(7)
red_slider.valtext.set_position((0.5, -0.1))

green_slider = Slider(ax=fig.add_axes([0.08, 0.8, 0.01, 0.15]), label="G", valmin = 0, valmax = 255, valinit = 119, orientation="vertical", valstep=1, handle_style={'size': 6})
green_slider.label.set_size(8)
green_slider.valtext.set_size(7)
green_slider.valtext.set_position((0.5, -0.1))

blue_slider = Slider(ax=fig.add_axes([0.12, 0.8, 0.01, 0.15]), label="B", valmin = 0, valmax = 255, valinit = 180, orientation="vertical", valstep=1, handle_style={'size': 6})
blue_slider.label.set_size(8)
blue_slider.valtext.set_size(7)
blue_slider.valtext.set_position((0.5, -0.1))

# initializing particle positions
def initialize(val):
    global points, pos, vel, n, t, m, volume, wall, pressure_sum, interval_count
    ax.clear()
    
    # variables
    n = n_slider.val # number of particles
    t = t_slider.val # temperature (Kelvin)
    molar_mass = mm_slider.val * SPEED_ADJUSTMENT # molar mass (g/mol)
    m = (molar_mass / 1000) / (6.022 * 10**23) # mass of particle (kg)
    volume = v_slider.val # volume (Liters)
    wall = (volume / 1000) ** (1/3) # length of axes (meters)

    color = (red_slider.val / 255, green_slider.val / 255, blue_slider.val / 255) # RGB


    # plot particles
    rng = np.random.default_rng()
    pos = rng.uniform(low=0, high=wall, size=(n,3)) # pos: position
    points = ax.scatter(pos[:, 0], pos[:, 1], pos[:,2], c=[color])
    ax.set_xlim(0, wall), ax.set_ylim(0, wall), ax.set_zlim(0, wall)

    # initializing velocities from maxwell-boltzman distr.
    sd = (KB * t / m) ** 0.5 # standard deviation
    vel = rng.normal(0, sd, size=(n,3)) # vel: velocity

    # zeroing momentum
    avg_vel = np.array([np.sum(vel[:,0]), np.sum(vel[:,1]), np.sum(vel[:,2])]) / n
    vel -= avg_vel

    # temperature rescaling
    mean_square_v = np.sum(vel ** 2) / n
    curr_temp = (m * mean_square_v) / (3 * KB)
    vel = vel * (t / curr_temp) ** 0.5

    interval_count = 0
    pressure_sum = 0


def change_color(val):
    global points

    color = (red_slider.val / 255, green_slider.val / 255, blue_slider.val / 255)
    points.set_facecolor([color])
    
def update(frame):
    global points, pos, vel, n, moles, pressure_sum, interval_count, average_pressure
    interval_count += 1

    # local variables
    impulse = 0 # reset impulse
    moles = n / (6.022 * 10**23) # number of moles of particles
    dt = ani.event_source.interval * 0.001 # seconds per update interval
    curr_temp = (m * np.sum(vel ** 2) / n) / (3 * KB)

    # update position
    pos = pos + vel * dt
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

    # calculating pressure 
    pressure_sum += (impulse / ((6 * wall ** 2) * dt)) / 101325 # 1 atm = 101325 pascal
    average_pressure = pressure_sum / interval_count # average pressure of all intervals that have occurred

    # calculating R
    r = average_pressure * volume / (moles * curr_temp)

    pressure_display.set_text("Average pressure (atm): " + str(format(average_pressure, ".2e")))
    R_display.set_text("Calculated R (L atm mol\u207b\u00b9 K\u207b\u00b9): " + str(format(r, ".4")))
    return points,

# 2D graph
slope_display = plt.figtext(0.97, 0.75, "Slope: ", va="top", ha="right")
graph_ax = fig.add_subplot(122)
graph_ax.set_title("LSRL Graph: R = PV / nT")
graph_ax.set_xlabel("nT")
graph_ax.set_ylabel("PV")
plot_button = Button(fig.add_axes([0.9, 0.05, 0.1, 0.075]), 'plot')
line, = graph_ax.plot([], [], color='red', label='LSRL')
x = np.array([]) # nT
y = np.array([]) # PV
# plotting the data points and LSRL
def plot_point(val):
    global x, y
    x = np.append(x, moles * t)
    y = np.append(y, average_pressure * volume)
    graph_ax.scatter(moles * t, average_pressure * volume, c='black')

    if len(x) > 1 and len(y) > 1:
        m, b = np.polyfit(x, y, 1)
        line.set_ydata(m*(x) + b)
        line.set_xdata(x)
        slope_display.set_text("Slope: " + str(format(m, ".4")))



# layout
fig.tight_layout(pad=4.0)
fig.set_size_inches(4, 3)
plt.subplots_adjust(bottom=0.25, wspace=0.5, hspace=0.5)
ax.set_position([0.05, 0.25, 0.7, 0.7])
graph_ax.set_position([0.67, 0.5, 0.2, 0.3])


# running
initialize(n)
n_slider.on_changed(initialize)
t_slider.on_changed(initialize)
v_slider.on_changed(initialize)
mm_slider.on_changed(initialize)
red_slider.on_changed(change_color)
green_slider.on_changed(change_color)
blue_slider.on_changed(change_color)
ani = animation.FuncAnimation(fig=fig, func=update, frames=40, interval=10)
plot_button.on_clicked(plot_point)

plt.show()