import numpy as np
import matplotlib.pyplot as plt

def plot_droplet_shape(r_vals, z_vals, title="Pendant Drop Shape", ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 8))
    ax.plot(r_vals, z_vals, 'b-', linewidth=2)
    ax.plot(-r_vals, z_vals, 'b-', linewidth=2)
    ax.set_xlabel('r')
    ax.set_ylabel('z')
    ax.set_title(title)
    ax.axis('equal')
    ax.grid(True)
    return ax

def plot_laplace_pressure(z_vals, P_L, Bo=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(z_vals, P_L, 'b-', linewidth=2)
    ax.grid(True)
    return ax
