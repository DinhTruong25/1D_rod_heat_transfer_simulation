import tkinter as tk
from tkinter import ttk, messagebox
from heat_core import metals, parse_initial_condition
from heat_plot import run_plot, run_dual_plots

gui_root = None
gui_widgets = {}

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
        f = parse_initial_condition(init_expr, L)
        gui_root.withdraw()
        run_plot(L, alpha, f, back_callback=gui_root.deiconify)
    except ValueError as ve:
        messagebox.showerror("Input Error", str(ve))
    except Exception as e:
        messagebox.showerror("Error",
            f"Invalid expression: {e}\n\nExample: sin(pi * x / L)")

def start_dual_simulation():
    try:
        # Left simulation
        L1 = float(gui_widgets['length1'].get())
        if L1 <= 0:
            raise ValueError("Left rod length must be positive.")
        metal1 = gui_widgets['metal1'].get().lower()
        alpha1 = metals.get(metal1)
        if alpha1 is None:
            raise ValueError("Please select a metal for left rod.")
        init_expr1 = gui_widgets['expr1'].get()
        f1 = parse_initial_condition(init_expr1, L1)

        # Right simulation
        L2 = float(gui_widgets['length2'].get())
        if L2 <= 0:
            raise ValueError("Right rod length must be positive.")
        metal2 = gui_widgets['metal2'].get().lower()
        alpha2 = metals.get(metal2)
        if alpha2 is None:
            raise ValueError("Please select a metal for right rod.")
        init_expr2 = gui_widgets['expr2'].get()
        f2 = parse_initial_condition(init_expr2, L2)

        gui_root.withdraw()

        # Run two plots side-by-side simultaneously
        run_dual_plots(
            L1, alpha1, f1, metal1,
            L2, alpha2, f2, metal2,
            back_callback=gui_root.deiconify
        )

    except ValueError as ve:
        messagebox.showerror("Input Error", str(ve))
    except Exception as e:
        messagebox.showerror("Error",
            f"Invalid expression: {e}\n\nExample: sin(pi * x / L)")

def show_single_mode():
    gui_widgets['single_frame'].pack(pady=20)
    gui_widgets['dual_frame'].pack_forget()
    gui_widgets['start_single_btn'].pack(pady=25)
    gui_widgets['start_dual_btn'].pack_forget()

def show_dual_mode():
    gui_widgets['single_frame'].pack_forget()
    gui_widgets['dual_frame'].pack(pady=20)
    gui_widgets['start_single_btn'].pack_forget()
    gui_widgets['start_dual_btn'].pack(pady=25)

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

def create_gui():
    global gui_root
    gui_root = tk.Tk()
    gui_root.title("Heat Transfer Simulator")
    gui_root.geometry("600x680")
    gui_root.configure(bg='#212121')
    gui_root.resizable(False, False)

    # === Style ===
    style = ttk.Style()
    style.theme_use('clam')
    style.configure('TLabel', background='#212121', foreground='#FFFFFF', font=('Helvetica', 11))
    style.configure('TButton', font=('Helvetica', 11, 'bold'))
    style.configure('TCombobox', fieldbackground='#424242', background='#424242', foreground='#FFFFFF')
    style.map('TCombobox', fieldbackground=[('readonly', '#424242')])
    style.map('TCombobox', selectbackground=[('readonly', '#4FC3F7')])

    # === Title ===
    title = tk.Label(gui_root, text="Heat Transfer Simulator", bg='#212121', fg='#4FC3F7',
                     font=('Helvetica', 16, 'bold'))
    title.pack(pady=(25, 10))

    # === Mode Selection ===
    mode_frame = tk.Frame(gui_root, bg='#212121')
    mode_frame.pack(pady=10)
    tk.Label(mode_frame, text="Mode:", bg='#212121', fg='#FFFFFF', font=('Helvetica', 11, 'bold')).pack(side='left')
    btn_single = tk.Button(mode_frame, text="Single View", bg='#424242', fg='#FFFFFF', relief='flat', command=show_single_mode)
    btn_single.pack(side='left', padx=10)
    btn_dual = tk.Button(mode_frame, text="Dual-View Comparison", bg='#424242', fg='#FFFFFF', relief='flat', command=show_dual_mode)
    btn_dual.pack(side='left', padx=10)

    # === Single Mode Frame ===
    single_frame = tk.Frame(gui_root, bg='#212121')
    tk.Label(single_frame, text="Rod Length (meters):").pack(pady=(15, 5))
    entry_length = tk.Entry(single_frame, width=20, font=('Consolas', 11), bg='#424242', fg='#FFFFFF',
                            insertbackground='#FFFFFF', relief='flat', bd=5)
    entry_length.pack(pady=5)
    entry_length.insert(0, "1.0")
    gui_widgets['length'] = entry_length

    tk.Label(single_frame, text="Metal Type:").pack(pady=(15, 5))
    combo_metal = ttk.Combobox(single_frame, values=list(metals.keys()), state="readonly", width=18)
    combo_metal.pack(pady=5)
    combo_metal.set("iron")
    gui_widgets['metal'] = combo_metal

    tk.Label(single_frame, text="Initial Temperature f(x):").pack(pady=(15, 5))
    entry_expr = tk.Entry(single_frame, width=40, font=('Consolas', 11), bg='#424242', fg='#FFFFFF',
                          insertbackground='#FFFFFF', relief='flat', bd=5)
    entry_expr.pack(pady=5)
    entry_expr.insert(0, "sin(pi * x / 0.5)")
    gui_widgets['expr'] = entry_expr

    gui_widgets['single_frame'] = single_frame

    # === Dual Mode Frame ===
    dual_frame = tk.Frame(gui_root, bg='#212121')

    # Left side
    left_frame = tk.LabelFrame(dual_frame, text=" Left Simulation ", bg='#212121', fg='#4FC3F7', font=('Helvetica', 10, 'bold'))
    left_frame.pack(side='left', padx=15, pady=10)

    tk.Label(left_frame, text="Length (m):").pack(pady=(10,5))
    e1 = tk.Entry(left_frame, width=15, font=('Consolas', 10), bg='#424242', fg='#FFFFFF', insertbackground='#FFFFFF')
    e1.pack(pady=5); e1.insert(0, "1.0")
    gui_widgets['length1'] = e1

    tk.Label(left_frame, text="Metal:").pack(pady=(10,5))
    c1 = ttk.Combobox(left_frame, values=list(metals.keys()), state="readonly", width=13)
    c1.pack(pady=5); c1.set("iron")
    gui_widgets['metal1'] = c1

    tk.Label(left_frame, text="f(x):").pack(pady=(10,5))
    ex1 = tk.Entry(left_frame, width=25, font=('Consolas', 10), bg='#424242', fg='#FFFFFF', insertbackground='#FFFFFF')
    ex1.pack(pady=5); ex1.insert(0, "sin(pi * x / 0.5)")
    gui_widgets['expr1'] = ex1

    # Right side
    right_frame = tk.LabelFrame(dual_frame, text=" Right Simulation ", bg='#212121', fg='#4FC3F7', font=('Helvetica', 10, 'bold'))
    right_frame.pack(side='right', padx=15, pady=10)

    tk.Label(right_frame, text="Length (m):").pack(pady=(10,5))
    e2 = tk.Entry(right_frame, width=15, font=('Consolas', 10), bg='#424242', fg='#FFFFFF', insertbackground='#FFFFFF')
    e2.pack(pady=5); e2.insert(0, "1.0")
    gui_widgets['length2'] = e2

    tk.Label(right_frame, text="Metal:").pack(pady=(10,5))
    c2 = ttk.Combobox(right_frame, values=list(metals.keys()), state="readonly", width=13)
    c2.pack(pady=5); c2.set("copper")
    gui_widgets['metal2'] = c2

    tk.Label(right_frame, text="f(x):").pack(pady=(10,5))
    ex2 = tk.Entry(right_frame, width=25, font=('Consolas', 10), bg='#424242', fg='#FFFFFF', insertbackground='#FFFFFF')
    ex2.pack(pady=5); ex2.insert(0, "sin(pi * x / 0.5)")
    gui_widgets['expr2'] = ex2

    gui_widgets['dual_frame'] = dual_frame

    # === Start Buttons ===
    start_single_btn = tk.Canvas(gui_root, width=200, height=50, bg='#212121', highlightthickness=0)
    btn_rect1 = create_rounded_rectangle(start_single_btn, 10, 5, 190, 45, radius=20, fill='#4FC3F7')
    btn_text1 = start_single_btn.create_text(100, 25, text="Start Simulation", fill='white',
                                             font=('Helvetica', 12, 'bold'))

    def on_enter1(e): start_single_btn.itemconfig(btn_rect1, fill='#29B6F6')
    def on_leave1(e): start_single_btn.itemconfig(btn_rect1, fill='#4FC3F7')
    def on_click1(e): start_simulation()

    start_single_btn.tag_bind(btn_rect1, '<Enter>', on_enter1)
    start_single_btn.tag_bind(btn_rect1, '<Leave>', on_leave1)
    start_single_btn.tag_bind(btn_rect1, '<Button-1>', on_click1)
    start_single_btn.tag_bind(btn_text1, '<Enter>', on_enter1)
    start_single_btn.tag_bind(btn_text1, '<Leave>', on_leave1)
    start_single_btn.tag_bind(btn_text1, '<Button-1>', on_click1)
    gui_widgets['start_single_btn'] = start_single_btn

    start_dual_btn = tk.Canvas(gui_root, width=200, height=50, bg='#212121', highlightthickness=0)
    btn_rect2 = create_rounded_rectangle(start_dual_btn, 10, 5, 190, 45, radius=20, fill='#4FC3F7')
    btn_text2 = start_dual_btn.create_text(100, 25, text="Start Dual Comparison", fill='white',
                                           font=('Helvetica', 12, 'bold'))

    def on_enter2(e): start_dual_btn.itemconfig(btn_rect2, fill='#29B6F6')
    def on_leave2(e): start_dual_btn.itemconfig(btn_rect2, fill='#4FC3F7')
    def on_click2(e): start_dual_simulation()

    start_dual_btn.tag_bind(btn_rect2, '<Enter>', on_enter2)
    start_dual_btn.tag_bind(btn_rect2, '<Leave>', on_leave2)
    start_dual_btn.tag_bind(btn_rect2, '<Button-1>', on_click2)
    start_dual_btn.tag_bind(btn_text2, '<Enter>', on_enter2)
    start_dual_btn.tag_bind(btn_text2, '<Leave>', on_leave2)
    start_dual_btn.tag_bind(btn_text2, '<Button-1>', on_click2)
    gui_widgets['start_dual_btn'] = start_dual_btn

    # === Default: Show Single Mode ===
    show_single_mode()

    # === Close ===
    gui_root.protocol("WM_DELETE_WINDOW", gui_root.quit)
    gui_root.mainloop()