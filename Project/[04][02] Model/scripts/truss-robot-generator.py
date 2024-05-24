import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Select topology
N = 4 # Number of nodes
M = 6 # Number of edges
connectivity = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]) # tetrahedron

# Member lengths
L = np.random.rand(M) + 1
print(L)

C = np.zeros((M, N))
for i, (a, b) in enumerate(connectivity):
    C[i, a] = 1
    C[i, b] = -1


# Initial guess for the coordinates
# B and C coordinates lie on the ground plane (z = 0), D is in 3D space
initial_guess = [
    0.0, 0.0, 0.0,  # A
    np.random.random(), np.random.random(), 0.0,  # B
    np.random.random(), np.random.random(), 0.0,  # C
    *[np.random.random() for _ in range(M - 3)] # D, E, F, ...
]

# Define the system of equations based on the edge lengths
def equations(coords):
    P = coords.reshape(-1, 3)
    # print(P)
    D = np.dot(C, P)

    return [
        (np.linalg.norm(D[i])-(L[i])) for i in range(M)
    ]


# Define bounds for each coordinate
lower_bounds = [0,0,0,-np.inf, -np.inf, 0, 0, 0, 0, -np.inf, -np.inf, 0.0]
upper_bounds = [1e-9,1e-9,1e-9,np.inf, np.inf, 1e-9, np.inf, np.inf, 1e-9, np.inf, np.inf, np.inf]

# Solve the system of equations with bounds
result = least_squares(equations, initial_guess, bounds=(lower_bounds, upper_bounds))
coords = result.x


print(coords)
           


# Extract the coordinates of each vertex
P = coords.reshape(-1, 3)
A = P[0]
B = P[1]
C = P[2]
D = P[3]


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