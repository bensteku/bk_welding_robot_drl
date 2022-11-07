import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
"""
def reward_func(x,y):
    dist = np.sqrt(((4-x)**2 + (4-y)**2))
    y_mask = y > 8
    y_mask2 = y < 0
    x_mask = x > 8
    x_mask2 = x < 0
    all_mask = y_mask | x_mask | x_mask2 | y_mask2
    y_mask3 = y > 4
    y_mask4 = y < 4.5
    x_mask3 = x > 4
    x_mask4 = x < 4.5
    target_mask = y_mask3 & y_mask4 & x_mask3 & x_mask4
    dist[target_mask] = 10
    dist[all_mask] = -30
    return -dist

X = np.linspace(-1,9,1000)
Y = np.linspace(-1,9,1000)
x, y = np.meshgrid(X, Y)
z = reward_func(x, y)


plt.contourf(x, y, z, levels=80)

plt.colorbar()
plt.xlim(-1,9)
plt.ylim(-1,9)
plt.show()
"""

target = np.array([4,4])

def reward(x,y, c1, c2, n):
    dis = np.sqrt(((target[0]-x)**2 + (target[1]-y)**2))

    reward1 = 0.5 * dis * dis

    reward2 = (1/(1+dis))**n

    return -c1* reward1 - c2* reward2

init_c1 = 500
init_c2 = 15
init_n = 35

X = np.linspace(3.5,4.5,1000)
Y = np.linspace(3.5,4.5,1000)
x, y = np.meshgrid(X, Y)
z = reward(x, y, init_c1, init_c2, init_n)

fig, ax = plt.subplots()
plot = ax.contourf(x, y, z, levels=80)

fig.subplots_adjust(left=0.25, bottom=0.25)

# Make a horizontal slider to control the frequency.
axfreq = fig.add_axes([0.25, 0.1, 0.65, 0.03])
freq_slider = Slider(
    ax=axfreq,
    label='c1',
    valmin=0.0,
    valmax=3000,
    valinit=init_c1,
)

axfreq = fig.add_axes([0.25, 0.01, 0.65, 0.03])
n_slider = Slider(
    ax=axfreq,
    label='n',
    valmin=0.0,
    valmax=150,
    valinit=init_n,
)

axamp = fig.add_axes([0.1, 0.25, 0.0225, 0.63])
amp_slider = Slider(
    ax=axamp,
    label="c2",
    valmin=0,
    valmax=3000,
    valinit=init_c2,
    orientation="vertical"
)

cax = make_axes_locatable(ax).append_axes("right", size="5%", pad="2%")

def update(val):
    ax.clear()
    cax.clear()
    z = reward(x, y, freq_slider.val, amp_slider.val, n_slider.val)
    pp = ax.contourf(x,y,z,levels=5)
    plt.colorbar(pp, cax=cax)


freq_slider.on_changed(update)
amp_slider.on_changed(update)
n_slider.on_changed(update)


plt.show()