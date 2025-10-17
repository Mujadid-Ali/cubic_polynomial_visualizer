"""
polynomial_visualizer.py

A small, standalone Python 3D visualizer to explore how each coefficient of a cubic
polynomial changes the elevation of a centerline. Use the interactive sliders to
adjust every coefficient and see the effect in real time.

Cubic polynomial (generic):
    z(s) = a + b*s + c*s^2 + d*s^3
    
# Real-life applications of cubic polynomials include:
# 1. Road Design: In ASAM OpenDRIVE, cubic polynomials are used to define the elevation of roads, ensuring smooth transitions and proper drainage.
# 2. Animation: Cubic splines are often used in computer graphics to create smooth curves for character movements and object paths.
# 3. Robotics: Cubic equations help in trajectory planning for robotic arms, allowing for smooth and efficient movements.
# 4. Structural Engineering: Cubic equations model the deflection of beams under load, aiding in the design of safe structures.
# 5. Physics: In projectile motion, cubic equations can describe the trajectory of objects under the influence of gravity and air resistance.
"""

import os
import numpy as np
import matplotlib

# Try to force a GUI backend if none specified (most systems have TkAgg).
# If you prefer Qt and have it installed, change 'TkAgg' -> 'Qt5Agg'.
if os.environ.get("MPLBACKEND") is None:
    try:
        matplotlib.use("TkAgg")
    except Exception:
        # fallback silently if not possible
        pass

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D

# ---------- geometry & helpers ----------
def cubic_poly(s, a, b, c, d):
    """Cubic polynomial z(s)."""
    return a + b*s + c*s**2 + d*s**3

def cubic_poly_derivative(s, b, c, d):
    """dz/ds = b + 2c s + 3d s^2"""
    return b + 2.0*c*s + 3.0*d*s**2

def build_centerline(s, a, b, c, d, lateral_func=None):
    """
    Returns centerline arrays Xc, Yc, Zc and local left-unit vectors Lx,Ly,Lz.
    lateral_func(s) -> (y, y') can be provided for horizontal curvature; default = zero.
    """
    Xc = s.copy()                      # longitudinal along X
    if lateral_func is None:
        Yc = np.zeros_like(s)
        dYds = np.zeros_like(s)
    else:
        Yc, dYds = lateral_func(s)

    Zc = cubic_poly(s, a, b, c, d)
    dZds = cubic_poly_derivative(s, b, c, d)

    # Tangent vector T = (dX/ds, dY/ds, dZ/ds). dX/ds = 1
    Tx = np.ones_like(s)
    Ty = dYds
    Tz = dZds
    Tnorm = np.sqrt(Tx*Tx + Ty*Ty + Tz*Tz)
    Tx /= Tnorm; Ty /= Tnorm; Tz /= Tnorm

    # Compute a local "left" direction L such that L is roughly horizontal (perp to tangent)
    # Using up vector U = (0,0,1), left = U x T
    Ux, Uy, Uz = 0.0, 0.0, 1.0
    Lx = Uy * Tz - Uz * Ty      # = 0*Tz - 1*Ty = -Ty
    Ly = Uz * Tx - Ux * Tz      # = 1*Tx - 0*Tz = Tx
    Lz = Ux * Ty - Uy * Tx      # = 0*Ty - 0*Tx = 0
    Lnorm = np.sqrt(Lx*Lx + Ly*Ly + Lz*Lz)
    # protect against degenerate case (T parallel to up)
    small = 1e-8
    Lnorm = np.where(Lnorm < small, 1.0, Lnorm)
    Lx /= Lnorm; Ly /= Lnorm; Lz /= Lnorm

    return Xc, Yc, Zc, Lx, Ly, Lz

def make_road_mesh(Xc, Yc, Zc, Lx, Ly, Lz, width):
    """Create 2xN mesh arrays for plot_surface (2 rows: left edge, right edge)."""
    half = width / 2.0
    left_x  = Xc +  half * Lx
    left_y  = Yc +  half * Ly
    left_z  = Zc +  half * Lz

    right_x = Xc -  half * Lx
    right_y = Yc -  half * Ly
    right_z = Zc -  half * Lz

    # stack into 2xN arrays (rows = left/right)
    X = np.vstack((left_x, right_x))
    Y = np.vstack((left_y, right_y))
    Z = np.vstack((left_z, right_z))
    return X, Y, Z

# ---------- parameters ----------
s = np.linspace(0.0, 100.0, 600)   # distance along road
width = 6.0                        # road width (meters)

# initial coefficients (reasonable defaults)
a0, b0, c0, d0 = 0.0, 0.02, 0.001, -1e-5

# ---------- create figure ----------
plt.style.use('seaborn-v0_8-darkgrid')
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(left=0.10, bottom=0.32, right=0.98, top=0.95)

ax.set_title("ASAM OpenDRIVE — 3D Road Visualizer (adjust a,b,c,d)")
ax.set_xlabel("X (s / longitudinal)")
ax.set_ylabel("Y (lateral)")
ax.set_zlabel("Z (elevation)")

# compute initial mesh
Xc, Yc, Zc, Lx, Ly, Lz = build_centerline(s, a0, b0, c0, d0)
Xmesh, Ymesh, Zmesh = make_road_mesh(Xc, Yc, Zc, Lx, Ly, Lz, width)

# initial surface + centerline + lane marking
road_surf = ax.plot_surface(Xmesh, Ymesh, Zmesh,
                            rstride=1, cstride=1, linewidth=0, antialiased=True,
                            shade=True, color=(0.33,0.33,0.33), alpha=0.95)

center_line = ax.plot(Xc, Yc, Zc, color='yellow', linewidth=2.2)[0]
lane_mark = ax.plot(Xc, Yc, Zc + 0.01, color='white', linewidth=1.0, linestyle='--')[0]

# set nice aspect / limits
ax.set_xlim(s.min(), s.max())
ax.set_ylim(-width*2, width*2)
zmin = float(np.min(Zc)) - 5.0
zmax = float(np.max(Zc)) + 5.0
ax.set_zlim(zmin, zmax)
ax.view_init(elev=20, azim=-60)

# ---------- sliders ----------
axcolor = 'lightgoldenrodyellow'
ax_a = plt.axes([0.12, 0.22, 0.78, 0.03], facecolor=axcolor)
ax_b = plt.axes([0.12, 0.17, 0.78, 0.03], facecolor=axcolor)
ax_c = plt.axes([0.12, 0.12, 0.78, 0.03], facecolor=axcolor)
ax_d = plt.axes([0.12, 0.07, 0.78, 0.03], facecolor=axcolor)

s_a = Slider(ax_a, 'a (base)', -50.0, 50.0, valinit=a0)
s_b = Slider(ax_b, 'b (slope)', -0.15, 0.15, valinit=b0)
s_c = Slider(ax_c, 'c (curvature)', -0.02, 0.02, valinit=c0)
s_d = Slider(ax_d, 'd (S-shape)', -0.0002, 0.0002, valinit=d0)

# references to current artists (we will reassign these)
# note: to modify from inner scope, use 'global' in update()
def update(val):
    global road_surf, center_line, lane_mark
    a = s_a.val; b = s_b.val; c = s_c.val; d = s_d.val

    # remove previous artists
    try:
        road_surf.remove()
    except Exception:
        pass
    try:
        center_line.remove()
    except Exception:
        pass
    try:
        lane_mark.remove()
    except Exception:
        pass

    # rebuild geometry
    Xc, Yc, Zc, Lx, Ly, Lz = build_centerline(s, a, b, c, d)
    Xmesh, Ymesh, Zmesh = make_road_mesh(Xc, Yc, Zc, Lx, Ly, Lz, width)

    # replot
    road_surf = ax.plot_surface(Xmesh, Ymesh, Zmesh,
                                rstride=1, cstride=1, linewidth=0, antialiased=True,
                                shade=True, color=(0.33,0.33,0.33), alpha=0.95)

    center_line = ax.plot(Xc, Yc, Zc, color='yellow', linewidth=2.2)[0]
    lane_mark = ax.plot(Xc, Yc, Zc + 0.01, color='white', linewidth=1.0, linestyle='--')[0]

    # update z-limits dynamically so uphill/downhill remain visible
    zmin = float(np.min(Zc)) - 6.0
    zmax = float(np.max(Zc)) + 6.0
    ax.set_zlim(zmin, zmax)

    fig.canvas.draw_idle()

for sld in (s_a, s_b, s_c, s_d):
    sld.on_changed(update)

# Reset button
reset_ax = plt.axes([0.82, 0.015, 0.12, 0.045])
button = Button(reset_ax, 'Reset', color=axcolor, hovercolor='0.95')

def reset(event):
    s_a.reset(); s_b.reset(); s_c.reset(); s_d.reset()
button.on_clicked(reset)

# ---------- run ----------
if __name__ == "__main__":
    print("Backend in use:", matplotlib.get_backend())
    print("If sliders don't respond, run this script from a system terminal (not an inline notebook)")
    plt.show()
