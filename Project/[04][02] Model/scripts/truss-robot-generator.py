import numpy as np
import math
import networkx as nx
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Select topology
model_name = "tetrahedron"  # "2-tetrahedron"
N = 3 # Number of nodes 4
M = 3 # Number of edges 6 
connectivity = np.array([[0, 1], [0, 2], [1,2]]) # tetrahedron
# connectivity = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]) # tetrahedron
# connectivity = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]) # pyramid
# connectivity = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3], [2, 4], [4, 5], [5, 2], [6, 2], [6,4],[6,5]]) # pyramid

# Create a graph from the connectivity
G = nx.Graph()
G.add_edges_from(connectivity)

# Create a tree ########################################################################################
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
                tree.add_edge(f"{i}-{connectivity[j][1]}", "0-0")

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
    *[np.random.random() for _ in range(3 * (N - 3))],
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
    print(f"{chr(65 + a)}-{chr(65 + b)}: {np.linalg.norm(P[b] - P[a])}")


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

print(P[0][0])

# member template xml
member_template = """
    <body name="[%s-0]" pos="%f %f %f" >
    <geom type="cylinder" pos="%f %f %f" axisangle="%f %f %f %f" size="0.05 0.95" material="metal" contype="1"/>
    </body>
"""

# <!--<joint name="Passive-%s" type="ball" pos="%f %f %f" axis="%f %f %f" damping=".9"/>-->
#     <body name="[%s-1]" pos="%f %f %f" >
#         <geom type="cylinder" pos="%f %f %f" euler="%f %f %f " size="0.04 0.95" material="gray" contype="1"/>
#         <joint name="Linear-%s" type="slide" axis="%f %f %f" range="0 2"/>
#     </body>
# create members from the tree
members = ""
def calculate_euler_angles(point_a, point_b):
  """
  Calculates the Euler angles (roll, pitch, yaw) to transform a cylinder direction
  from point A (pointing towards z-axis) to point B.

  Args:
      point_a (tuple): 3D coordinates of point A (x, y, z).
      point_b (tuple): 3D coordinates of point B (x, y, z).

  Returns:
      tuple: Euler angles (roll, pitch, yaw) in degrees.
  """

  # Calculate direction vector from point A to point B
  direction_vector = (point_b[0] - point_a[0],
                      point_b[1] - point_a[1],
                      point_b[2] - point_a[2])

  # Normalize the direction vector
  magnitude = math.sqrt(sum(x**2 for x in direction_vector))
  if magnitude < 1e-6:
    raise ValueError("Points A and B are too close together")
  direction_vector = tuple(v / magnitude for v in direction_vector)

  # Calculate yaw (rotation around z-axis)
  yaw = math.atan2(direction_vector[0], direction_vector[1]) * 180 / math.pi

  # Calculate pitch (rotation around x-axis after yaw)
  # Handle special case where direction vector is parallel to x-axis (pitch = 0 or 180)
  if abs(direction_vector[2]) < 1e-6:
    pitch = 0
  else:
    # Project direction vector onto xz-plane
    projected_vector = (direction_vector[0], 0, direction_vector[2])

    # Calculate pitch using dot product and magnitude
    pitch = math.atan2(projected_vector[2] * direction_vector[0],
                       projected_vector[0] * math.sqrt(direction_vector[0]**2 + direction_vector[2]**2)) * 180 / math.pi

  # Calculate roll (rotation around y-axis after yaw and pitch)
  # Handle special case where direction vector is parallel to y-axis (roll indeterminate)
  if abs(direction_vector[0]) < 1e-6 and abs(direction_vector[2]) < 1e-6:
    roll = 0  # Roll is indeterminate in this case
  else:
    # Rotate direction vector by yaw and pitch
    rotated_vector = (direction_vector[0] * math.cos(yaw) - direction_vector[1] * math.sin(yaw),
                      direction_vector[0] * math.sin(yaw) + direction_vector[1] * math.cos(yaw),
                      direction_vector[2])
    rotated_vector = (rotated_vector[0], rotated_vector[2] * math.cos(pitch) - rotated_vector[1] * math.sin(pitch),
                      rotated_vector[2] * math.sin(pitch) + rotated_vector[1] * math.cos(pitch))

    # Calculate roll using cross product and magnitude
    roll = math.atan2(rotated_vector[1], rotated_vector[0]) * 180 / math.pi

  return roll, pitch, yaw

def get_axis_angle(point_a, point_b):
  """
  Calculates the axis and angle to rotate a cylinder's z-axis towards point_b.

  Args:
      point_a: A numpy array representing point a (3D coordinates).
      point_b: A numpy array representing point b (3D coordinates).

  Returns:
      axis: A numpy array representing the rotation axis.
      angle: The angle of rotation in radians.
  """

  # Direction vector from point a to point b
  direction = point_b - point_a

  # Check if points are coincident
  if np.allclose(direction, np.zeros(3)):
    raise ValueError("Points a and b are coincident. Rotation is undefined.")

  # Normalize direction vector
  direction /= np.linalg.norm(direction)

  # Desired z-axis (cylinder's direction)
  z_axis = np.array([0, 0, 1])

  # Calculate the axis of rotation
  axis = np.cross(z_axis, direction)

  # Handle potential numerical issue with near-zero axis
  if np.allclose(axis, np.zeros(3)):
    # If points are almost aligned, use a small arbitrary axis
    axis = np.array([1, 0, 0])

  # Calculate the angle of rotation (using arccosine)
  angle = np.arccos(np.dot(z_axis, direction))

  return axis, angle

for edge in tree.edges:
    print(edge)
    a, b = edge
    a = a.split("-")
    b = b.split("-")
    print(a, b)
    a = int(a[1])
    b = int(b[1])
    print(a, b)
    print("A: ", P[a])
    print("B: ", P[b])
    print(np.linalg.norm(P[a] - P[b]))
    print(P[a][0], P[a][1], P[a][2], P[b][0], P[b][1], P[b][2])
    print(np.arctan2(P[b][1] - P[a][1], P[b][0] - P[a][0]))
    print(np.arccos((P[b][2] - P[a][2])/np.linalg.norm(P[a] - P[b])))
    print(np.arctan2(P[b][2] - P[a][2], np.linalg.norm(P[a] - P[b])))
    axis, angle = get_axis_angle(P[a], P[b])
    # roll, pitch, yaw = calculate_euler_angles(P[b], P[a])
    members += member_template % (edge[0], 2*P[a][0], 2*P[a][1], 2*P[a][2], (P[b][0] - P[a][0])/2, (P[b][1] - P[a][1])/2, (P[b][2] - P[a][2])/2, axis[0], axis[1], axis[2], angle)
       
print(members)
