import numpy as np
import networkx as nx
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Select topology
model_name = "2-tetrahedron"
N = 5 # Number of nodes 4
M = 9 # Number of edges 6 
# connectivity = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]) # tetrahedron
# connectivity = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3], [3,4]]) # tetrahedron
connectivity = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]) # pyramid
# connectivity = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3], [2, 4], [4, 5], [5, 2], [6, 2], [6,4],[6,5]]) # pyramid


# Create an empty directed graph ########################################################################################
tree = nx.DiGraph()

# Extract the first column (first element of each pair)
first_elements = connectivity[:, 0]

# Find the maximum value in the first column
depth = np.max(first_elements)

# Add the first level of nodes
for i in range(0, depth+1):
    for j in range(0, len(connectivity)):
        if connectivity[j][0] == i:
            tree.add_node(f"{i}-{connectivity[j][1]}", pos=(i, 0))
            if i-1 >= 0:
                tree.add_edge(f"{i}-{connectivity[j][1]}", f"{i-1}-{i}")
            else:
                tree.add_edge(f"{i}-{connectivity[j][1]}", "0")

# Draw the tree
pos = nx.spring_layout(tree)  # positions for all nodes
nx.draw(tree, pos, with_labels=True, node_size=700)
plt.show()

###########################################################################################################################


# Member lengths
# L = np.random.rand(M) + 1
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
    *[np.random.random() for _ in range(3 * (N - 4))],
    np.random.random(), np.random.random(), 2,  # C
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

# Move the coordinates such that center of mass is at the origin
center_of_mass = np.mean(coords.reshape(-1, 3), axis=0)

print(center_of_mass)
center_of_mass[2] = 0
coords = coords.reshape(-1, 3) - center_of_mass

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

# limits
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([ 0, 2])


# Show the plot
plt.show()

## Create MuJoCo model from the coordinates



# Define the XML template
xml_template = """
<mujoco model="%s">
  <compiler angle="radian" meshdir="assets" autolimits="true"/>
  <option gravity="0 0 -9.81" timestep="0.002" />

  <option cone="elliptic" impratio="10"/>

  <asset>
    <material name="metal" rgba="0.58 0.58 0.58 1"/>
    <material name="gray" rgba="0.4627 0.4627 0.4627 1"/>
    <material name="red" rgba="0.9 0.1 0.1 1"/>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1=".1 .2 .3" rgb2=".2 .3 .4"/>
    <material name="grid" texture="grid" texrepeat="1 1" texuniform="true" reflectance=".2"/>
  </asset>

  <worldbody>
  %s
  </worldbody>
"""




# Define the body template
body_template = """
<body name="[%s-0]" pos="%f %f %f">
    %s
    <geom type="sphere" name="(%f)" pos="0 0 0" size="0.1" material="red"contype="1" mass="10"/>
    <geom type="cylinder" pos="%f %f %f" euler="%f %f %f" size="%f %f" material="metal"contype="1"/>
    %s
    <joint name="Pivot-%s" type="ball" pos="0 0 0" axis="%f %f %f" damping="0.9"/>
    <body name="[%f-1]" pos="0 0 0">
        <geom type="cylinder" pos="%f %f %f" euler="%f %f %f" size="%f %f" material="gray" contype="1"/>
        <geom type="sphere" name="(%f)" pos="%f %f %f" size="0.1" material="red" contype="1" mass="10"/>
        <joint name="Linear-%f" type="slide" axis="%f %f %f" range="0 2"/>
    </body>
</body>
"""

# Generate the body elements
body_elements = ""

for i, p in enumerate(P):
    if i == 0:
        body_elements += body_template % (chr(65 + i), p[0], p[1], p[2], "", i, 0, 0, 0, 0, 0, 0, 0.1, 0.1, "", chr(65 + i), 0, 0, 1, 0, 0, 0, 0.1, 0.1, chr(65 + i), 0, 0, 0, 0)
    else:
        body_elements += body_template % (chr(65 + i), p[0], p[1], p[2], "", i, 0, 0, 0, 0, 0, 0, 0.1, 0.1, "", chr(65 + i), 0, 0, 1, 0, 0, 0, 0.1, 0.1, chr(65 + i), i, 0, 0, 0, 0, 0, 0.1, 0.1, chr(65 + i), 0, 0, 0, 0)
    
# Generate the XML model
xml_model = xml_template % (model_name, body_elements)


print(xml_model)


