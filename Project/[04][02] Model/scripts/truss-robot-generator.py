import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Select topology
N = 5 # Number of nodes 4
M = 9 # Number of edges 6 
connectivity = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]) # tetrahedron
connectivity = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3], [1, 4], [2, 4], [3, 4]]) # pyramid
# connectivity = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3], [2, 4], [4, 5], [5, 2], [6, 2], [6,4],[6,5]]) # pyramid

# Member lengths
L = np.random.rand(M) + 1
L = np.array([1 for i in range(1, M+1)])
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
    *[np.random.random() for _ in range(3 * (N - 3))]
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
A_lower = [0, 0, 0]
A_upper = [1e-9,1e-9,1e-9]
B_lower = [-np.inf, -np.inf, 0.0]
B_upper = [np.inf, np.inf, 1e-9]
C_lower = [-np.inf, -np.inf, 0.0]
C_upper = [np.inf, np.inf, 1e-9]

lower_bounds = A_lower + B_lower + C_lower + [-np.inf, -np.inf, 0.0]* (N - 3)
upper_bounds = A_upper + B_upper + C_upper + [np.inf, np.inf, np.inf]* (N - 3)
print(lower_bounds)
print(upper_bounds)
print(initial_guess)
# Solve the system of equations with bounds
result = least_squares(equations, initial_guess, bounds=(lower_bounds, upper_bounds))
coords = result.x

print(coords)
           
# Extract the coordinates of each vertex
P = coords.reshape(-1, 3)
for i, p in enumerate(P):
    print(f"{chr(65 + i)}: {p}")



# Print the length of each edge
print("Length of edges:")
for i, (a, b) in enumerate(connectivity):
    print(f"{chr(65 + a)}-{chr(65 + b)}: {np.linalg.norm(P[a] - P[b])}")


# Plotting the vertices and edges in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the vertices
for i, p in enumerate(P):
    ax.scatter(*p, label=chr(65 + i))


# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Vertices in 3D')

# Add legend
ax.legend()

# plot the edges
edges = [(P[a], P[b]) for a, b in connectivity]

for edge in edges:
    ax.plot(*zip(*edge), color='gray')

# Show the plot
plt.show()