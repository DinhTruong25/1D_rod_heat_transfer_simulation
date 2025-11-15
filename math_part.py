import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
from matplotlib.collections import LineCollection
from matplotlib import cm
from sympy import sympify, symbols, lambdify
import tkinter as tk
from tkinter import ttk, messagebox

# ----------------------------------------------------------------------
# Thermal diffusivities (alpha) for common metals (m²/s)
metals = {
    'copper': 1.17e-4,
    'aluminum': 9.7e-5,
    'iron': 2.3e-5,
    'steel': 1.4e-5,
    'gold': 1.27e-4,
    'silver': 1.65e-4
}

# ----------------------------------------------------------------------
gui_root = None
gui_widgets = {}
current_fig = None

# ----------------------------------------------------------------------
def run_simulation(L, alpha, f):
    """Core simulation + animation + back button."""
    global current_fig

    nx = 101
    dx = L / (nx - 1)
    x = np.linspace(0, L, nx)
    u = f(x)

    sigma = 0.5
    dt = sigma * dx**2 / alpha
    r = alpha * dt / dx**2

    # Implicit matrix
    M_imp = np.zeros((nx, nx))
    for i in range(1, nx-1):
        M_imp[i, i-1] = -r / 2
        M_imp[i, i]   = 1 + r
        M_imp[i, i+1] = -r / 2
    M_imp[0, 0] = 1 + r
    M_imp[0, 1] = -r
    M_imp[nx-1, nx-1] = 1 + r
    M_imp[nx-1, nx-2] = -r

    # Explicit matrix
    M_exp = np.zeros((nx, nx))
    for i in range(1, nx-1):
        M_exp[i, i-1] = r / 2
        M_exp[i, i]   = 1 - r
        M_exp[i, i+1] = r / 2
    M_exp[0, 0] = 1 - r
    M_exp[0, 1] = r
    M_exp[nx-1, nx-1] = 1 - r
    M_exp[nx-1, nx-2] = r

    # Plot
    fig, ax = plt.subplots(figsize=(9, 5))
    current_fig = fig
    ax.set_xlim(0, L)
    global_min = np.min(u)
    global_max = np.max(u)
    ax.set_ylim(global_min - 1, global_max + 1)
    ax.set_xlabel('Position (x)')
    ax.set_ylabel('Temperature (u(x,t))')
    ax.set_title('Heat Distribution Over Time')

    # Gradient-colored line
    norm = plt.Normalize(global_min, global_max)
    cmap = cm.jet  # Blue (low) to red (high)

    points = np.array([x, u]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=3)
    lc.set_array(u)
    ax.add_collection(lc)

    # # Colorbar
    # cb = fig.colorbar(lc, ax=ax, pad=0.1)
    # cb.set_label('Temperature')

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
        gui_root.deiconify()

    back_btn.on_clicked(on_back_click)

    # Animation
    steps_per_frame = 10
    num_frames = 200

    def update(frame):
        nonlocal u
        for _ in range(steps_per_frame):
            u = np.linalg.solve(M_imp, M_exp @ u)
        points = np.array([x, u]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc.set_segments(segments)
        lc.set_array(u)
        return lc,

    ani = FuncAnimation(fig, update, frames=num_frames, interval=50, blit=True)
    plt.tight_layout()
    plt.show()

# ----------------------------------------------------------------------
def start_simulation():
    try:
        L = float(gui_widgets['length'].get())
        if L <= 0:
            raise ValueError("Length must be positive.")

        metal = gui_widgets['metal'].get().lower()
        alpha = metals.get(metal)
        if alpha is None:
            raise ValueError("Please select a metal.")

        init_expr = gui_widgets['expr'].get()
        x_sym = symbols('x')
        expr = sympify(init_expr)
        f = lambdify(x_sym, expr, modules='numpy')

        gui_root.withdraw()
        run_simulation(L, alpha, f)

    except ValueError as ve:
        messagebox.showerror("Input Error", str(ve))
    except Exception as e:
        messagebox.showerror("Error",
            f"Invalid expression: {e}\n\nExample: sin(pi * x / L)")

# ----------------------------------------------------------------------
def create_rounded_rectangle(canvas, x1, y1, x2, y2, radius=20, **kwargs):
    points = [x1 + radius, y1,
              x2 - radius, y1,
              x2, y1,
              x2, y1 + radius,
              x2, y2 - radius,
              x2, y2,
              x2 - radius, y2,
              x1 + radius, y2,
              x1, y2,
              x1, y2 - radius,
              x1, y1 + radius,
              x1, y1]
    return canvas.create_polygon(points, **kwargs, smooth=True, splinesteps=20)

# ----------------------------------------------------------------------
def create_gui():
    """Modern dark-themed UI with rounded button."""
    global gui_root, gui_widgets

    gui_root = tk.Tk()
    gui_root.title("Heat Transfer Simulator")
    gui_root.geometry("600x500")
    gui_root.configure(bg='#212121')
    gui_root.resizable(False, False)

    # === Style ===
    style = ttk.Style()
    style.theme_use('clam')

    # Configure dark theme
    style.configure('TLabel', background='#212121', foreground='#FFFFFF', font=('Helvetica', 11))
    style.configure('TButton', font=('Helvetica', 11, 'bold'))
    style.configure('TCombobox', fieldbackground='#424242', background='#424242', foreground='#FFFFFF')
    style.map('TCombobox', fieldbackground=[('readonly', '#424242')])
    style.map('TCombobox', selectbackground=[('readonly', '#4FC3F7')])

    # === Title ===
    title = tk.Label(gui_root, text="Heat Transfer Simulator", bg='#212121', fg='#4FC3F7',
                     font=('Helvetica', 16, 'bold'))
    title.pack(pady=(25, 15))

    # === Rod Length ===
    tk.Label(gui_root, text="Rod Length (meters):").pack(pady=(15, 5))
    entry_length = tk.Entry(gui_root, width=20, font=('Consolas', 11), bg='#424242', fg='#FFFFFF',
                            insertbackground='#FFFFFF', relief='flat', bd=5)
    entry_length.pack(pady=5)
    entry_length.insert(0, "1.0")
    gui_widgets['length'] = entry_length

    # === Metal Type ===
    tk.Label(gui_root, text="Metal Type:").pack(pady=(15, 5))
    combo_metal = ttk.Combobox(gui_root, values=list(metals.keys()), state="readonly", width=18)
    combo_metal.pack(pady=5)
    combo_metal.set("iron")
    gui_widgets['metal'] = combo_metal

    # === Initial Temp ===
    tk.Label(gui_root, text="Initial Temperature f(x):").pack(pady=(15, 5))
    entry_expr = tk.Entry(gui_root, width=40, font=('Consolas', 11), bg='#424242', fg='#FFFFFF',
                          insertbackground='#FFFFFF', relief='flat', bd=5)
    entry_expr.pack(pady=5)
    entry_expr.insert(0, "sin(pi * x / 0.5)")
    gui_widgets['expr'] = entry_expr

    # === Start Button (Rounded Rectangle) ===
    start_btn = tk.Canvas(gui_root, width=200, height=50, bg='#212121', highlightthickness=0)
    start_btn.pack(pady=25)

    # Draw rounded rectangle
    btn_rect = create_rounded_rectangle(start_btn, 10, 5, 190, 45, radius=20, fill='#4FC3F7')

    # Button text
    btn_text = start_btn.create_text(100, 25, text="Start Simulation", fill='white',
                                     font=('Helvetica', 12, 'bold'))

    def on_enter(e):
        start_btn.itemconfig(btn_rect, fill='#29B6F6')
    def on_leave(e):
        start_btn.itemconfig(btn_rect, fill='#4FC3F7')
    def on_click(e):
        start_simulation()

    start_btn.tag_bind(btn_rect, '<Enter>', on_enter)
    start_btn.tag_bind(btn_rect, '<Leave>', on_leave)
    start_btn.tag_bind(btn_rect, '<Button-1>', on_click)
    start_btn.tag_bind(btn_text, '<Enter>', on_enter)
    start_btn.tag_bind(btn_text, '<Leave>', on_leave)
    start_btn.tag_bind(btn_text, '<Button-1>', on_click)

    # === Close ===
    gui_root.protocol("WM_DELETE_WINDOW", gui_root.quit)
    gui_root.mainloop()

# ----------------------------------------------------------------------
if __name__ == "__main__":
    create_gui()