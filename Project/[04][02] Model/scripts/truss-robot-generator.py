import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the edge lengths
AB = 2.0
AC = 1.0
AD = 1.0
BC = 1.0
BD = 1.0
CD = 2.0

# # randomly generated edge lengths
# AB = np.random.rand()+1
# AC = np.random.rand()+1
# AD = np.random.rand()+1
# BC = np.random.rand()+1
# BD = np.random.rand()+1
# CD = np.random.rand()+1

# Initial guess for the coordinates
# B and C coordinates lie on the ground plane (z = 0), D is in 3D space
initial_guess = [
    0.0, 0.0, 0.0,       # A
    np.random.random()+1, np.random.random()+1, 0.0,       # B
    0.5, np.sqrt(0.75), 0.0, # C
    0.5, 0.5, np.sqrt(0.75)  # D
]

# Define the system of equations based on the edge lengths
def equations(coords):
    A = np.array([0.0, 0.0, 0.0])
    B = np.array([coords[0], coords[1], 0])
    C = np.array([coords[2], coords[3], 0])
    D = np.array([coords[4], coords[5], coords[6]])
    
    
    return [
        np.linalg.norm(B - A) - AB,
        np.linalg.norm(C - A) - AC,
        np.linalg.norm(D - A) - AD,
        np.linalg.norm(C - B) - BC,
        np.linalg.norm(D - B) - BD,
        np.linalg.norm(D - C) - CD,
    ]

# Define bounds for each coordinate
lower_bounds = [0,0,0,-np.inf, -np.inf, 0, 0, 0, 0, -np.inf, -np.inf, 0.0]
upper_bounds = [1,1,1,np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]

# Solve the system of equations with bounds
result = least_squares(equations, initial_guess, bounds=(lower_bounds, upper_bounds))
coords = result.x


# Extract the coordinates of each vertex
A = np.array([0.0, 0.0, 0.0])
B = np.array([coords[0], coords[1], 0])
C = np.array([coords[2], coords[3], 0])
D = np.array([coords[4], coords[5], coords[6]])

# Print the length of each edge
print("Length of edges:")
print(f"AB: {np.linalg.norm(B - A)}")
print(f"AC: {np.linalg.norm(C - A)}")
print(f"AD: {np.linalg.norm(D - A)}")
print(f"BC: {np.linalg.norm(C - B)}")
print(f"BD: {np.linalg.norm(D - B)}")
print(f"CD: {np.linalg.norm(D - C)}")


print("Coordinates of vertices:")
print(f"A: {A}")
print(f"B: {B}")
print(f"C: {C}")
print(f"D: {D}")

# Plotting the vertices and edges in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the vertices
ax.scatter(*A, color='k', label='A')
ax.scatter(*B, color='r', label='B')
ax.scatter(*C, color='g', label='C')
ax.scatter(*D, color='b', label='D')

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Vertices in 3D')

# Add legend
ax.legend()

# plot the edges
edges = [
    (A, B), (A, C), (A, D),
    (B, C), (B, D),
    (C, D),
]

for edge in edges:
    ax.plot(*zip(*edge), color='gray')

# Show the plot
plt.show()