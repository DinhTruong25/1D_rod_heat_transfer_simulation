import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sympy import sympify, symbols, lambdify

# Dictionary of thermal diffusivities (alpha) for common metals (m²/s)
metals = {
    'copper': 1.17e-4,
    'aluminum': 9.7e-5,
    'iron': 2.3e-5,
    'steel': 1.4e-5,
    'gold': 1.27e-4,
    'silver': 1.65e-4
}

# Get user inputs
L = float(input("Enter the length of the rod (in meters): "))
metal = input("Enter the type of metal (silver, gold, steel, iron, copper, aluminum): ").lower()
alpha = metals.get(metal, None)
if alpha is None:
    print("Unknown metal. Using default alpha for copper.")
    alpha = 1.17e-4

init_expr = input("Enter the initial temperature distribution as a function of x (e.g., sin(x) or x**2): ")

# Parse the symbolic expression safely
x_sym = symbols('x')
try:
    expr = sympify(init_expr)
    f = lambdify(x_sym, expr, modules='numpy')
except Exception as e:
    print(f"Error parsing expression: {e}. Using default sin(pi * x / L).")
    f = lambdify(x_sym, sympify('sin(pi * x / L)'), modules='numpy')

# Simulation parameters
nx = 101  # Number of spatial points (increase for higher resolution)
dx = L / (nx - 1)
x = np.linspace(0, L, nx)

# Initial condition
u = f(x)

# Time step (chosen for accuracy; Crank-Nicolson is stable for larger dt)
sigma = 0.5  # CFL-like number for accuracy
dt = sigma * dx**2 / alpha
r = alpha * dt / dx**2

# Build implicit matrix (left side: I + (r/2) * Laplacian, adjusted for Neumann BC)
M_imp = np.zeros((nx, nx))
for i in range(1, nx-1):
    M_imp[i, i-1] = -r / 2
    M_imp[i, i] = 1 + r
    M_imp[i, i+1] = -r / 2
# Boundaries (Neumann: adjusted for zero flux)
M_imp[0, 0] = 1 + r
M_imp[0, 1] = -r
M_imp[nx-1, nx-1] = 1 + r
M_imp[nx-1, nx-2] = -r

# Build explicit matrix (right side: I - (r/2) * Laplacian, adjusted for Neumann BC)
M_exp = np.zeros((nx, nx))
for i in range(1, nx-1):
    M_exp[i, i-1] = r / 2
    M_exp[i, i] = 1 - r
    M_exp[i, i+1] = r / 2
# Boundaries
M_exp[0, 0] = 1 - r
M_exp[0, 1] = r
M_exp[nx-1, nx-1] = 1 - r
M_exp[nx-1, nx-2] = r

# Set up plot for animation
fig, ax = plt.subplots()
ax.set_xlim(0, L)
ax.set_ylim(np.min(u) - 1, np.max(u) + 1)
ax.set_xlabel('Position (x)')
ax.set_ylabel('Temperature (u)')
ax.set_title('Heat Distribution Over Time')
line, = ax.plot(x, u)

# Animation parameters
steps_per_frame = 10  # Computations per animation frame (to speed up visualization)
num_frames = 200  # Total frames (adjust for longer/shorter simulation)

def update(frame):
    global u
    for _ in range(steps_per_frame):
        # Solve for next time step: u = M_imp^{-1} * (M_exp * u)
        u = np.linalg.solve(M_imp, M_exp @ u)
    line.set_ydata(u)
    return line,

ani = FuncAnimation(fig, update, frames=num_frames, interval=50, blit=True)
plt.show()