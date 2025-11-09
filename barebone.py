import numpy as np
import sympy as sp
import scipy.sparse as spmat
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def get_diffusivity(metal_type):
    table = {
        'aluminum': 9.7e-5,
        'copper':   1.11e-4,
        'iron':     2.3e-5,
    }
    mt = metal_type.lower()
    if mt not in table:
        raise ValueError(f"Unknown metal type '{metal_type}'")
    return table[mt]

def build_CN_matrices(N, alpha, dt, h):
    r = alpha * dt / (h*h)
    # internal nodes: N
    mainA = (1 + r) * np.ones(N)
    mainA[0] = 1 + r / 2
    mainA[-1] = 1 + r / 2
    offA  = (-r/2)   * np.ones(N-1)
    A = spmat.diags([offA, mainA, offA],
                    offsets=[-1, 0, +1],
                    shape=(N, N),
                    format='csc')
    mainB = (1 - r) * np.ones(N)
    mainB[0] = 1 - r / 2
    mainB[-1] = 1 - r / 2
    offB  = (r/2)   * np.ones(N-1)
    B = spmat.diags([offB, mainB, offB],
                    offsets=[-1, 0, +1],
                    shape=(N, N),
                    format='csc')
    return A, B

def simulate_CN_neumann(L, metal, expr_str, N, dt=0.05, tol=1e-6, t_max=100.0, save_every=1):
    alpha = get_diffusivity(metal)
    # full grid: include boundaries
    x_full = np.linspace(0, L, N+2)
    h = x_full[1] - x_full[0]
    x_int = x_full[1:-1]
    # parse initial cond
    x_sym = sp.symbols('x')
    L_sym = sp.symbols('L')
    expr = sp.sympify(expr_str, locals={'pi': sp.pi, 'L': L_sym})
    expr = expr.subs(L_sym, L)
    u0_func = sp.lambdify(x_sym, expr, 'numpy')
    u_full = u0_func(x_full)
    # Neumann BC: zero-flux → u[0] = u[1], u[-1] = u[-2]
    u_full[0]  = u_full[1]
    u_full[-1] = u_full[-2]
    u_int = u_full[1:-1].copy()  # internal nodes
    
    A, B = build_CN_matrices(N, alpha, dt, h)
    solver = spla.factorized(A)  # Factorize once for speedup
    
    times = [0.0]
    snapshots = [u_full.copy()]
    
    t = 0.0
    step = 0
    
    while True:
        rhs = B.dot(u_int)
        u_int_new = solver(rhs)
        # rebuild full
        u_full[1:-1] = u_int_new
        u_full[0]  = u_full[1]
        u_full[-1] = u_full[-2]
        
        # check equilibrium criterion
        diff = np.max(np.abs(u_int_new - u_int))
        
        t += dt
        step += 1
        if step % save_every == 0:
            times.append(t)
            snapshots.append(u_full.copy())
        
        if diff < tol:
            if step % save_every != 0:  # Ensure final snapshot is saved
                times.append(t)
                snapshots.append(u_full.copy())
            print(f"Equilibrium reached at t = {t:.2f} (max diff = {diff:.2e})")
            break
        if t >= t_max:
            if step % save_every != 0:
                times.append(t)
                snapshots.append(u_full.copy())
            print(f"Maximum time reached t = {t:.2f} without full equilibrium (max diff = {diff:.2e})")
            break
        
        u_int = u_int_new
    
    return x_full, np.array(times), snapshots

def animate_solution(x, times, snapshots):
    fig, ax = plt.subplots()
    # Initial plot
    line, = ax.plot(x, snapshots[0], lw=2)
    ax.set_xlabel('x')
    ax.set_ylabel('u(x,t)')
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(np.min(snapshots), np.max(snapshots))
    # Use ax.text instead of title for reliable updating with blit=True
    time_text = ax.text(0.02, 0.95, f"t = {times[0]:.2f} s", 
                        transform=ax.transAxes, va='top', ha='left')

    def update(i):
        line.set_ydata(snapshots[i])
        time_text.set_text(f"t = {times[i]:.2f} s")
        return line, time_text

    ani = FuncAnimation(
        fig,
        update,
        frames=len(times),
        interval=100,
        blit=True,  # Enabled for better performance
        repeat=False  # Prevent looping/resetting
    )
    plt.show()


if __name__ == '__main__':
    L     = float(input("Length of rod L (m): "))
    metal = input("Metal type (aluminum, copper, iron): ")
    expr  = input("Initial u(x,0) expression in x (e.g., sin(pi*x/L)): ")
    
    # Dynamically scale N with L for consistent resolution (fixes endpoint issues)
    N      = max(100, int(100 * L))  # Targets h ≈ 0.01; adjust multiplier if needed
    dt     = 0.1   # Unchanged
    tol    = 1e-6   # Slightly increased for speedup (was 1e-6; decrease for more precision)
    t_max  = 50000.0  # Increase if equilibrium takes longer for large L
    save_every = 100  # Save snapshots every 100 steps for speedup
    
    x, times, snaps = simulate_CN_neumann(L, metal, expr, N, dt=dt, tol=tol, t_max=t_max, save_every=save_every)
    animate_solution(x, times, snaps)