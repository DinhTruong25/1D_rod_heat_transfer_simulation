import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
from matplotlib.collections import LineCollection
from matplotlib import cm
from heat_core import setup_matrices, compute_next_u

def run_plot(L, alpha, f, back_callback, title_suffix=''):
    nx = 101
    dx = L / (nx - 1)
    x = np.linspace(0, L, nx)
    u = f(x)
    sigma = 0.5
    dt = sigma * dx**2 / alpha
    r = alpha * dt / dx**2
    M_imp, M_exp = setup_matrices(nx, r)
    
    # Plot
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_xlim(0, L)
    global_min = np.min(u)
    global_max = np.max(u)
    ax.set_ylim(global_min - 1, global_max + 1)
    ax.set_xlabel('Position (x)')
    ax.set_ylabel('Temperature (u(x,t))')
    ax.set_title('Heat Distribution Over Time' + title_suffix)
    
    # Gradient-colored line
    norm = plt.Normalize(global_min, global_max)
    cmap = cm.jet
    points = np.array([x, u]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=3)
    lc.set_array(u)
    ax.add_collection(lc)
    
    # Colorbar
    cb = fig.colorbar(lc, ax=ax, pad=0.1)
    cb.set_label('Temperature')
    
    # Back Button
    btn_ax = plt.axes([0.84, 0.02, 0.13, 0.05], facecolor='#37474F')
    back_btn = Button(btn_ax, 'Back to Input', color='#37474F', hovercolor='#455A64')
    for spine in btn_ax.spines.values():
        spine.set_color('#78909C')
        spine.set_linewidth(1.5)
        spine.set_capstyle('round')
    back_btn.label.set_color('white')
    back_btn.label.set_fontsize(9)
    back_btn.label.set_fontweight('bold')
    
    def on_back_click(event):
        plt.close(fig)
        back_callback()
    
    back_btn.on_clicked(on_back_click)
    
    # Animation
    steps_per_frame = 10
    num_frames = 200
    
    def update(frame):
        nonlocal u
        for _ in range(steps_per_frame):
            u = compute_next_u(u, M_imp, M_exp)
        points = np.array([x, u]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc.set_segments(segments)
        lc.set_array(u)
        return lc,
    
    ani = FuncAnimation(fig, update, frames=num_frames, interval=50, blit=True)
    plt.tight_layout()
    plt.show()


def run_dual_plots(L1, alpha1, f1, metal1, L2, alpha2, f2, metal2, back_callback):
    """Run two simulations side-by-side in the same window"""
    
    # Setup for left simulation
    nx1 = 101
    dx1 = L1 / (nx1 - 1)
    x1 = np.linspace(0, L1, nx1)
    u1 = f1(x1)
    sigma1 = 0.5
    dt1 = sigma1 * dx1**2 / alpha1
    r1 = alpha1 * dt1 / dx1**2
    M_imp1, M_exp1 = setup_matrices(nx1, r1)
    
    # Setup for right simulation
    nx2 = 101
    dx2 = L2 / (nx2 - 1)
    x2 = np.linspace(0, L2, nx2)
    u2 = f2(x2)
    sigma2 = 0.5
    dt2 = sigma2 * dx2**2 / alpha2
    r2 = alpha2 * dt2 / dx2**2
    M_imp2, M_exp2 = setup_matrices(nx2, r2)
    
    # Create single figure with 2 subplots side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle('Heat Distribution Comparison', fontsize=14, fontweight='bold')
    
    # Setup left subplot
    ax1.set_xlim(0, L1)
    global_min1 = np.min(u1)
    global_max1 = np.max(u1)
    ax1.set_ylim(global_min1 - 1, global_max1 + 1)
    ax1.set_xlabel('Position (x)', fontsize=10)
    ax1.set_ylabel('Temperature (u(x,t))', fontsize=10)
    ax1.set_title(f'{metal1.capitalize()} (α={alpha1:.2e} m²/s)', fontsize=11, fontweight='bold')
    
    norm1 = plt.Normalize(global_min1, global_max1)
    cmap1 = cm.jet
    points1 = np.array([x1, u1]).T.reshape(-1, 1, 2)
    segments1 = np.concatenate([points1[:-1], points1[1:]], axis=1)
    lc1 = LineCollection(segments1, cmap=cmap1, norm=norm1, linewidth=3)
    lc1.set_array(u1)
    ax1.add_collection(lc1)
    
    # Colorbar for left plot
    cb1 = fig.colorbar(lc1, ax=ax1, pad=0.02, fraction=0.046)
    cb1.set_label('Temperature', fontsize=9)
    
    # Setup right subplot
    ax2.set_xlim(0, L2)
    global_min2 = np.min(u2)
    global_max2 = np.max(u2)
    ax2.set_ylim(global_min2 - 1, global_max2 + 1)
    ax2.set_xlabel('Position (x)', fontsize=10)
    ax2.set_ylabel('Temperature (u(x,t))', fontsize=10)
    ax2.set_title(f'{metal2.capitalize()} (α={alpha2:.2e} m²/s)', fontsize=11, fontweight='bold')
    
    norm2 = plt.Normalize(global_min2, global_max2)
    cmap2 = cm.jet
    points2 = np.array([x2, u2]).T.reshape(-1, 1, 2)
    segments2 = np.concatenate([points2[:-1], points2[1:]], axis=1)
    lc2 = LineCollection(segments2, cmap=cmap2, norm=norm2, linewidth=3)
    lc2.set_array(u2)
    ax2.add_collection(lc2)
    
    # Colorbar for right plot
    cb2 = fig.colorbar(lc2, ax=ax2, pad=0.02, fraction=0.046)
    cb2.set_label('Temperature', fontsize=9)
    
    # Back button (centered at bottom)
    btn_ax = fig.add_axes([0.44, 0.02, 0.12, 0.04], facecolor='#37474F')
    back_btn = Button(btn_ax, 'Back to Input', color='#37474F', hovercolor='#455A64')
    for spine in btn_ax.spines.values():
        spine.set_color('#78909C')
        spine.set_linewidth(1.5)
    back_btn.label.set_color('white')
    back_btn.label.set_fontsize(10)
    back_btn.label.set_fontweight('bold')
    
    def on_back_click(event):
        plt.close(fig)
        back_callback()
    
    back_btn.on_clicked(on_back_click)
    
    # Animation for both plots
    steps_per_frame = 10
    num_frames = 200
    
    def update(frame):
        nonlocal u1, u2
        
        # Update left simulation
        for _ in range(steps_per_frame):
            u1 = compute_next_u(u1, M_imp1, M_exp1)
        points1 = np.array([x1, u1]).T.reshape(-1, 1, 2)
        segments1 = np.concatenate([points1[:-1], points1[1:]], axis=1)
        lc1.set_segments(segments1)
        lc1.set_array(u1)
        
        # Update right simulation
        for _ in range(steps_per_frame):
            u2 = compute_next_u(u2, M_imp2, M_exp2)
        points2 = np.array([x2, u2]).T.reshape(-1, 1, 2)
        segments2 = np.concatenate([points2[:-1], points2[1:]], axis=1)
        lc2.set_segments(segments2)
        lc2.set_array(u2)
        
        return lc1, lc2
    
    ani = FuncAnimation(fig, update, frames=num_frames, interval=50, blit=True)
    
    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    plt.show()