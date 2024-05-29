#%╭─╮╭─╮╭─╮╱╱╱╱╱╱╱╱╱╭╮╱╭┬─╮╭─╮╭─╮┬─╮┌─╮╭─╮╱╱╱╱╭─╮┬─╮┌─╱╱╱╱╱╱╱╱╱┌─╮╭─╮╱╱╱╱╭─╮╭─╮╭─╮┬─╮┌─╮╭─╮╱╱╱╱╱╱╱╱╱╭┬─╮┬─╮
#  └─╯┴─╯╰┴─╯╰─────────╯╱╰─╯┴──╯╰─╯─╯┴─╯╰┴─╯╰──╯╰─╯─╯──────────╯┴─╯╰┴─╯╰─╯─╯┴─╯╰┴─╯╰─────────╯╱╰─╯┴──╯╰─|
# TRUSS ROBOT GENERATOR

import numpy as np
import copy
import math
import networkx as nx
from scipy.optimize import least_squares, minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from hierarcy_pos import hierarchy_pos

# Select topology
# minimum is a triangle
model_name = "4-tet-0"  # "tetrahedron"
# connectivity = np.array([[0, 1], [0, 2], [1,2]]) # triangle
# connectivity = np.array([[0, 1], [1, 2], [2, 3], [3,4]]) # 4-line
# connectivity = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]) # 1-tet
# connectivity = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [1, 4], [2, 3], [2, 4]]) # 1-tet-1
# connectivity = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]) # 2-tet
# connectivity = np.array([[0, 1], [0, 2], [0, 5], [1, 2], [1, 5], [2, 3], [2, 4], [2, 5], [3, 4], [3, 5], [4, 5]]) # 2-tet-1
# connectivity = np.array([[0, 1], [0, 2], [0, 5], [1, 2], [1, 3], [1, 4], [2, 3], [2, 5], [3, 4], [4, 5]]) # cupola
# connectivity = np.array([[0, 1], [0, 2], [0, 3], [0, 4],[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]) # 2-tet-E
# connectivity = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [1, 4], [1, 5], [2, 3], [2, 4], [2, 5], [3, 4], [4, 5]]) # 3-tet
# connectivity = np.array([[0, 1], [0, 2], [0, 3], [0,5], [1,2], [1, 3], [1, 4], [2, 3], [2, 4], [2, 5], [3, 4], [3, 5]]) # 3-tet
connectivity = np.array([[0, 1], [0, 2], [0, 3], [0, 5], [1, 2], [1, 4], [1, 5], [2, 3], [2, 4], [2, 5], [3, 4], [3, 5], [4, 5]]) # 4-tet-0
# connectivity = np.array([[0, 1], [0, 2], [0, 4], [0,5], [1,2], [1, 3], [1, 5], [2, 3], [2, 4], [3, 4], [3, 5], [4, 5]]) # octahedron
# connectivity = np.array([[0, 1], [0, 2], [0, 4], [0,5],[0,6], [1,2], [1, 3], [1, 5], [2, 3], [2, 4],[2,6], [3, 4], [3, 5], [4, 5],[4,6],[5,6], ]) # 4-octahedron


# connectivity = np.array([[0, 1], [0, 3], [0, 7], [1, 2], [1, 6], [2, 3], [2, 5], [3, 4], [4, 5], [4, 7], [5, 6], [6, 7]]) # cube
# connectivity = np.array([[0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [1, 2], [1, 3], [1, 4], [1, 6], [2, 3], [2, 5], [2, 6], [3, 4], [4, 5], [4, 6]]) # 3.5-tetrahedron

# needs to be ordered such that each node number is connected to the next number in line (1 should have a connection with 2)
# otherwise the tree will not be created correctly


N = np.max(connectivity) + 1
M = len(connectivity)

# Create a graph from the connectivity
G = nx.Graph()
G.add_edges_from(connectivity)

pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_size=700,node_color=(0.9,0.2,0.2), font_size=15, font_color='white', arrows=True, width=1.4)
plt.show()

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

# reverse directions
tree = tree.reverse()

# # Use pydot for better hierarchical layout
# # pos = nx.nx_pydot.graphviz_layout(tree, prog='dot')  # Requires pydot

# # Draw the tree
# # pos = nx.spring_layout(tree)  # positions for all nodes
# def hierarchy_pos(G, root=None, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
#     """
#     If there is a cycle that is reachable from root, then this will see infinite recursion.
#     G: the graph (must be a tree)
#     root: the root node of the current branch
#     width: horizontal space allocated for this branch - avoids overlap with other branches
#     vert_gap: vertical gap between levels of hierarchy
#     vert_loc: vertical location of the root node
#     xcenter: horizontal location of the root node
#     """

#     def _hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None, parsed=[]):
#         if pos is None:
#             pos = {root: (xcenter, vert_loc)}
#         else:
#             pos[root] = (xcenter, vert_loc)
#         parsed.append(root)
#         neighbors = list(G.neighbors(root))
#         if not isinstance(G, nx.DiGraph) and parent is not None:
#             neighbors.remove(parent)  # avoid revisiting the parent node in undirected graphs
#         if len(neighbors) != 0:
#             dx = width / len(neighbors)  # space allocated for each subtree
#             nextx = xcenter - width / 2 - dx / 2
#             for neighbor in neighbors:
#                 nextx += dx
#                 pos = _hierarchy_pos(G, neighbor, width=dx, vert_gap=vert_gap, vert_loc=vert_loc-vert_gap, xcenter=nextx, pos=pos, parent=root, parsed=parsed)
#         return pos

#     return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)


# # Get the hierarchical position of nodes
# pos = hierarchy_pos(tree, root="0-0")

# nx.draw(tree, pos, with_labels=True, node_size=600, node_color='red', font_size=11, font_color='white', arrows=True, width=2, font_weight='bold')
# Draw the tree
pos = hierarchy_pos(tree, root="0-0", width = 2*math.pi, xcenter=0)
new_pos = {u:(r*math.cos(theta),r*math.sin(theta)) for u, (theta, r) in pos.items()}
# nx.draw(G, pos=new_pos, node_size = 50)

# pos = nx.spring_layout(tree)  # positions for all nodes
nx.draw(tree, new_pos, with_labels=True, node_size=700, node_color=(0.9,0.2,0.2), font_size=13, font_color='white', arrows=True, width=1.4)
plt.show()

###########################################################################################################################


# Member lengths
# L = np.random.rand(M) + 1
L = np.array([1.25 for i in range(1, M+1)])
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

    residuals = [
        ((np.linalg.norm(D[i]) - L[i])+0.1/(np.sum(np.abs(coords)))) for i in range(M)
    ]
    
    # Adding the penalty as additional residuals
    # penalty = 0.1 / (1+np.abs(coords))# Scale the penalty appropriately
    
    return residuals

# Define bounds for each coordinate
A_lower = [0, 0, 0]
A_upper = [1e-9,1e-9,1e-9]
B_lower = [-np.inf, -np.inf, 0.0]
B_upper = [np.inf, np.inf, 1e-9]
C_lower = [-np.inf, -np.inf, 0.0]
C_upper = [np.inf, np.inf, 1e-9]

lower_bounds = A_lower + B_lower + C_lower + [-np.inf, -np.inf, 0.0]* (N - 3)
upper_bounds = A_upper + B_upper + C_upper + [np.inf, np.inf, np.inf]* (N - 3)
# print(lower_bounds)
# print(upper_bounds)
# print(initial_guess)
# Solve the system of equations with bounds

result = least_squares(equations, initial_guess, bounds=(lower_bounds, upper_bounds))
coords = result.x


# Move the coordinates such that center of mass is at the origin
center_of_mass = np.mean(coords.reshape(-1, 3), axis=0)

# print(center_of_mass)
center_of_mass[2] = 0
coords = coords.reshape(-1, 3) - center_of_mass

# print(coords)
           
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


# member template xml
member_template = """
    <body name="[%s-0]" pos="%f %f %f" >
    %s
    %s
    <geom type="cylinder" pos="%f %f %f" axisangle="%f %f %f %f" size="0.025 0.45" material="metal" contype="1"/>
    <body name="[%s-1]" pos="%f %f %f" >
        <geom type="cylinder" pos="%f %f %f" axisangle="%f %f %f %f" size="0.02 0.5" material="gray" contype="1"/>
        <joint name="Linear-%s" type="slide" axis="%f %f %f" range="%f %f"/>
        %s
        %s
    </body>
    </body>
    """

# <!--<joint name="Passive-%s" type="ball" pos="%f %f %f" axis="%f %f %f" damping=".9"/>-->

# create members from the tree
# members = ""

def rotate_vector_90_around_z(vector):
    x, y, z = vector
    return (y, -x, z)

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

def direction_vector(point1, point2):
  """
  Calculates the direction vector from point1 to point2.

  Args:
      point1: A list containing 3D coordinates (x1, y1, z1).
      point2: A list containing 3D coordinates (x2, y2, z2).

  Returns:
      A list containing the direction vector (dx, dy, dz).
  """
  dx = point2[0] - point1[0]
  dy = point2[1] - point1[1]
  dz = point2[2] - point1[2]
  return [dx, dy, dz]

def unit_direction_vector(point1, point2):
  """
  Calculates the unit direction vector from point1 to point2.

  Args:
      point1: A list containing 3D coordinates (x1, y1, z1).
      point2: A list containing 3D coordinates (x2, y2, z2).

  Returns:
      A list containing the unit direction vector.
  """
  direction = direction_vector(point1, point2)
  magnitude = np.sqrt(sum(component**2 for component in direction))
  if magnitude > 0:  # Avoid division by zero
    return [component / magnitude for component in direction]
  else:
    return direction  # Points coincide, direction vector is [0, 0, 0]


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def rotate_vector(vector, axis, theta):
    """
    Rotate a vector around the given axis by theta radians.
    """
    rotation_matrix_ = rotation_matrix(axis, theta)
    return np.dot(rotation_matrix_, vector)


members = {}
# # create a members structure, also storing name information
# for i in range(1, M):
#     members[i] = ""

# print("members",members)

# this will store the nodes that are placed in the xml file
nodes = {}


i = 0

for edge in tree.edges:
    i += 1
    print(edge)
    a, b = edge
    a = a.split("-")
    b = b.split("-")
    a = int(a[1])
    b = int(b[1])
    print(a, b)
    print("A: ", P[a])
    print("B: ", P[b])
    # print(np.linalg.norm(P[a] - P[b]))
    axis, angle = get_axis_angle(P[a], P[b])
    axis = [axis[0], axis[1], axis[2]]
    # print("Axis: ", axis)
    unit_dir = unit_direction_vector(P[b], P[a])
    # print("Unit direction: ", unit_dir)
    normal_dir = lambda unit_dir, axis:np.cross(unit_dir, axis)  # C'mon guys its 2024
    # print("Normal: ", normal_dir(unit_dir, axis))

    L_dir = rotate_vector(unit_dir, normal_dir(unit_dir, axis), np.pi)



    # orth_axis = rotate_vector_90_around_z(axis)
    
    # roll, pitch, yaw = calculate_euler_angles(P[b], P[a])

    P_offset = -(P[a] - P[b])/(np.linalg.norm(P[b] - P[a]))*0.50
    L_offset = (P[b] - P[a])/(np.linalg.norm(P[b] - P[a]))*0.50
    Passives = (P[a] - P[b])/(np.linalg.norm(P[b] - P[a]))
    Passives = [0 ,0, 0 ]

    passive_template = """<joint name="Passive%s" type="ball" pos="%f %f %f" axis="0 1 0" damping=".9"/>"""

    # node_template = """<geom type="sphere" name="(%s)" pos="%f %f %f" size="0.05" material="red"contype="1" mass="10"/>"""

    # if a not in nodes:
    #     nodes[a] = a
    #     node_1_add = node_template % (a, 0, 0, 0)
    # else:
    #     node_1_add = ""
    # if b not in nodes:
    #     nodes[b] = b
    #     node_2_add = node_template % (b, 0, 0, 0)
    


   
    if edge[1] == "0-1":
        passives = ""
    else:
        passives = passive_template % (edge[1], Passives[0], Passives[1], Passives[2])



    # members[i] = member_template % (edge[1], P[a][0], P[a][1], P[a][2], "%s",passives, nodes[a], P_offset[0], P_offset[1], P_offset[2], axis[0], axis[1], axis[2], angle, edge[1], L_offset[0], L_offset[1], L_offset[2], 0, 0, 0, axis[0], axis[1], axis[2], angle, edge[1], L_dir[0], L_dir[1], L_dir[2],0,0 ,"%s",  "%s")
# print(members)
# print(members[1])
# print(members[2])
# print(members[3])


# hmm = members[1] % (members[2]%(""))

# hmm = hmm +members[3] % ""

# print(hmm)
# print(type(members[1]))
# print(type(members[2]))
# print("test")



# generate a precessor list using the tree, recursively
def get_predecessors_to_root(G, start_node):
    predecessors = []
    current_node = start_node
    
    while True:
        # Get the predecessors of the current node
        pred = list(G.predecessors(current_node))
        
        if not pred:  # If there are no predecessors, we have reached the root
            break
        
        # Assume there is only one predecessor (if there could be more, handle accordingly)
        current_node = pred[0]
        predecessors.append(current_node)

    return predecessors

def generate_member(node):
    correct = [0,0,0]
    if node == "0-0":
       return ""
    a = node.split('-')[0]
    b = node.split('-')[1]
    a = int(a)
    b = int(b)
    print(a, b)
    print("A: ", P[a])
    print("B: ", P[b])
    # print(np.linalg.norm(P[a] - P[b]))
    axis, angle = get_axis_angle(P[a], P[b])
    axis = [axis[0], axis[1], axis[2]]
    # print("Axis: ", axis)
    unit_dir = unit_direction_vector(P[b], P[a])
    # print("Unit direction: ", unit_dir)
    normal_dir = lambda unit_dir, axis:np.cross(unit_dir, axis)  # C'mon guys its 2024
    # print("Normal: ", normal_dir(unit_dir, axis))
    L_dir = rotate_vector(unit_dir, normal_dir(unit_dir, axis), np.pi)

    length = np.linalg.norm(P[b] - P[a])

    P_offset = -(P[a] - P[b])/(np.linalg.norm(P[b] - P[a]))*0.50
    L_offset = (P[b] - P[a])/(np.linalg.norm(P[b] - P[a]))*0.50
    L_offset2 = (P[a] - P[b])/(np.linalg.norm(P[a] - P[b]))*(1-np.linalg.norm(P[a] - P[b]))
    N_offset = -(P[a] - P[b])/(np.linalg.norm(P[a] - P[b]))*(np.linalg.norm(P[a] - P[b])-0.5)
    Passives = (P[a] - P[b])/(np.linalg.norm(P[b] - P[a]))
    Passives = [0 ,0, 0 ]

    prev = list(tree.predecessors(node))
    print("prev")
    print(prev)
    prev_a, prev_b = prev[0].split("-")
    prev_a = int(prev_a)
    prev_b = int(prev_b)

    if not (prev_a == 0 and prev_b == 0):
        correcting_factor = (P[prev_b] - P[prev_a])/(np.linalg.norm(P[prev_b] - P[prev_a]))*0.50
    else:
        correcting_factor = [0,0,0]

    correct = P[prev_a].copy() + correcting_factor.copy()


    passive_template = """<joint name="Passive%s" type="ball" pos="%f %f %f" axis="0 1 0" damping=".9"/>"""

    node_template = """<geom type="sphere" name="(%s)" pos="%f %f %f" size="0.05" material="red"contype="1" mass="10"/>"""

    if a not in nodes:
        nodes[a] = a
        node_1_add = node_template % (a, 0, 0, 0)
    else:
        node_1_add = ""
    if b not in nodes:
        nodes[b] = b
        node_2_add = node_template % (b, N_offset[0], N_offset[1], N_offset[2])
    else:
        node_2_add = ""
    

   
    if node == "0-1":
        passives = ""
    else:
        passives = passive_template % (node, Passives[0], Passives[1], Passives[2])

    range_start = (1-np.linalg.norm(P[a] - P[b]))
    range_end = range_start +0.95


#     correct = [0,0,0]
#     prev = list(tree.predecessors(node))
#     print("prev")
#     print(prev)
#     prev_a, prev_b = prev[0].split("-")
#     prev_a = int(prev_a)
#     prev_b = int(prev_b)
#     if a == 0 and b == 0:
#         correct = [0,0,0]
#     elif prev_a == 0 and prev_b == 0:
#         correct = [0,0,0]
#     else:
#         correct = P[prev_a]
#         correct += (P[prev_b] - P[prev_a])/(np.linalg.norm(P[prev_b] - P[prev_a]))*0.50
#         correct = [0,0,0]


    
#     print("correct")
#     print(correct)



#     print("member_template")
#     print(member_template % (node, P[a][0]-correct[0], P[a][1]-correct[1], P[a][2]-correct[2],node, Passives[0], Passives[1], Passives[2], P_offset[0], P_offset[1], P_offset[2], axis[0], axis[1], axis[2], angle, node, L_offset[0], L_offset[1], L_offset[2], 0, 0, 0, axis[0], axis[1], axis[2], angle, node, L_dir[0], L_dir[1], L_dir[2], "")
# )
    return member_template % (node, P[a][0]-correct[0], P[a][1]-correct[1], P[a][2]-correct[2], node_1_add, passives, P_offset[0], P_offset[1], P_offset[2], axis[0], axis[1], axis[2], angle, node, L_offset[0], L_offset[1], L_offset[2], L_offset2[0], L_offset2[1], L_offset2[2], axis[0], axis[1], axis[2], angle, node, L_dir[0], L_dir[1], L_dir[2],range_start, range_end ,node_2_add,   "%s")

def generate_member2(node):
    correct = [0,0,0]

    # print("node")
    # print(node)
    if node == "0-0":
       return ""
    a = node.split('-')[0]
    b = node.split('-')[1]
    # print(a, b)
    a = int(a)
    b = int(b)
    # print("A: ", P[a])
    # print("B: ", P[b])
    # print(np.linalg.norm(P[a] - P[b]))
    axis, angle = get_axis_angle(P[a], P[b])
    axis = [axis[0], axis[1], axis[2]]
    # print("Axis: ", axis)
    unit_dir = unit_direction_vector(P[b], P[a])
    # print("Unit direction: ", unit_dir)
    normal_dir = lambda unit_dir, axis:np.cross(unit_dir, axis)  # C'mon guys its 2024
    # print("Normal: ", normal_dir(unit_dir, axis))
    L_dir = rotate_vector(unit_dir, normal_dir(unit_dir, axis), np.pi)

    P_offset = -(P[a] - P[b])/(np.linalg.norm(P[b] - P[a]))*0.50
    L_offset = (P[b] - P[a])/(np.linalg.norm(P[b] - P[a]))*0.50
    L_offset2 = (P[a] - P[b])/(np.linalg.norm(P[a] - P[b]))*(1-np.linalg.norm(P[a] - P[b]))
    N_offset = -(P[a] - P[b])/(np.linalg.norm(P[a] - P[b]))*(np.linalg.norm(P[a] - P[b])-0.5)
    Passives = (P[a] - P[b])/(np.linalg.norm(P[b] - P[a]))
    Passives = [0 ,0, 0 ]

    prev = list(tree.predecessors(node))
    print("prev")
    print(prev)
    prev_a, prev_b = prev[0].split("-")
    prev_a = int(prev_a)
    prev_b = int(prev_b)

    if not (prev_a == 0 and prev_b == 0):
        correcting_factor = (P[prev_b] - P[prev_a])/(np.linalg.norm(P[prev_b] - P[prev_a]))*0.50
    else:
        correcting_factor = [0,0,0]

    correct = P[prev_a].copy() + correcting_factor.copy()







    passive_template = """<joint name="Passive%s" type="ball" pos="%f %f %f" axis="0 1 0" damping=".9"/>"""

    node_template = """<geom type="sphere" name="(%s)" pos="%f %f %f" size="0.05" material="red"contype="1" mass="10"/>"""

    if a not in nodes:
        nodes[a] = a
        node_1_add = node_template % (a, 0, 0, 0)
    else:
        node_1_add = ""
    if b not in nodes:
        nodes[b] = b
        node_2_add = node_template % (b, N_offset[0], N_offset[1], N_offset[2])
    else:
        node_2_add = ""
   
    if node == "0-1":
        passives = ""
    else:
        passives = passive_template % (node, Passives[0], Passives[1], Passives[2])

    range_start = (1-np.linalg.norm(P[a] - P[b]))
    range_end = range_start +0.95
#     correct = [0,0,0]
#     prev = list(tree.predecessors(node))
#     print("prev")
#     print(prev)
#     prev_a, prev_b = prev[0].split("-")
#     prev_a = int(prev_a)
#     prev_b = int(prev_b)
#     if a == 0 and b == 0:
#         correct = [0,0,0]
#     elif prev_a == 0 and prev_b == 0:
#         correct = [0,0,0]
#     else:
#         correct = P[prev_a]
#         correct += (P[prev_b] - P[prev_a])/(np.linalg.norm(P[prev_b] - P[prev_a]))*0.50
#         correct = [0,0,0]


    
#     print("correct")
#     print(correct)



#     print("member_template")
#     print(member_template % (node, P[a][0]-correct[0], P[a][1]-correct[1], P[a][2]-correct[2],node, Passives[0], Passives[1], Passives[2], P_offset[0], P_offset[1], P_offset[2], axis[0], axis[1], axis[2], angle, node, L_offset[0], L_offset[1], L_offset[2], 0, 0, 0, axis[0], axis[1], axis[2], angle, node, L_dir[0], L_dir[1], L_dir[2], "")
# )
    return member_template % (node, P[a][0]-correct[0], P[a][1]-correct[1], P[a][2]-correct[2], node_1_add, passives, P_offset[0], P_offset[1], P_offset[2], axis[0], axis[1], axis[2], angle, node, L_offset[0], L_offset[1], L_offset[2], L_offset2[0], L_offset2[1], L_offset2[2], axis[0], axis[1], axis[2], angle, node, L_dir[0], L_dir[1], L_dir[2],range_start, range_end ,node_2_add,   "")
   

# Traverse the tree and combine members
def combine_members(tree, node):
    children = list(tree.successors(node))
    # print("node")
    # print(node)
    # print("children")
    # print(children)
    
    if not children:
        return generate_member2(node)
    combined_member = ""
    for child in children:
        combined_member += combine_members(tree, child)
    # print("star")
    last = generate_member(node)
    if last == "":
        return combined_member
    return last % combined_member

# Find the root node (assuming single root)
root = [node for node in tree.nodes if not list(tree.predecessors(node))][0]
# print(root)
final_member = combine_members(tree, root)

print(final_member)

# create equality constraints
# find leaf nodes, for all leaf nodes, write the equality constraint
equality_template = """
<connect name="kinematic_link_%s" active="true" body1="[%s-1]" body2="[%s-1]" anchor=" %f %f %f" />"""
leaf_count = 0
i = 0
for nodes in tree.nodes():
    if not list(tree.successors(nodes)):
        leaf_count += 1

leaves = np.zeros((leaf_count, 2))
leaffo = {}

for nodes in tree.nodes():
    if not list(tree.successors(nodes)):
        leaves[i] = [nodes.split("-")[0], nodes.split("-")[1]]
        leaffo[i] = nodes
        i += 1

print(leaves)

def find_all_possible_connections(tree, node):
    connections = []
    for nodes in tree.nodes():
        if nodes.split("-")[1] == node.split("-")[1] and nodes != node:
            connections.append(nodes)
    return connections
print("start")
print(find_all_possible_connections(tree, "2-3"))

short_list = []

def find_node(tree, leaf):
    for nodes in tree.nodes():
        if nodes.split("-")[1] == leaf.split("-")[1] and nodes != leaf:
            short_list.append(nodes)
            return nodes
        
# Anchor is calculated with respect to the body1

def calculate_anchor(P1, P2):
    return -(P1-P2)/np.linalg.norm(P1-P2)*(np.linalg.norm(P1 - P2)-0.5)

equality = ""
l = 0
# for leaf in leaves:
#     equality += equality_template % (l, leaffo[l], find_node(tree, leaffo[l]), calculate_anchor(P[int(leaf[0])],P[int(leaf[1])])[0], calculate_anchor(P[int(leaf[0])],P[int(leaf[1])])[1], calculate_anchor(P[int(leaf[0])],P[int(leaf[1])])[2])
#     l += 1

for leaf in leaves:
    print("star")
    print(leaf)
    print(leaffo[l])
    p = 0
    for nodes in find_all_possible_connections(tree, leaffo[l]):
        print("nodes")
        print(nodes)
        equality += equality_template % ( ("%s-%s"%(l,p)), leaffo[l], nodes, calculate_anchor(P[int(leaf[0])],P[int(leaf[1])])[0], calculate_anchor(P[int(leaf[0])],P[int(leaf[1])])[1], calculate_anchor(P[int(leaf[0])],P[int(leaf[1])])[2])
        p += 1
    l += 1
print(l)
print(p)

# create actuators from members
actuator_template = """
<intvelocity name="Member-%s" joint="Linear-%s" kp="18000"  kv="10000" inheritrange="1" />"""

actuator = ""

for edge in tree.edges:
    a, b = edge
    a = a.split("-")[1]
    b = b.split("-")[1]
    print(a, b)
    actuator += actuator_template % ((a+"-"+b),(a+"-"+b))







header = """<mujoco model="%s">
  <compiler angle="radian" meshdir="assets" autolimits="true"/>
  <option gravity="0 0 -9.81" timestep="0.002" integrator="implicit"/>

  <option cone="elliptic" impratio="10"/>

  <asset>
    <material name="metal" rgba="0.58 0.58 0.58 1"/>
    <material name="gray" rgba="0.4627 0.4627 0.4627 1"/>
    <material name="red" rgba="0.9 0.1 0.1 1"/>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1=".1 .2 .3" rgb2=".2 .3 .4"/>
    <material name="grid" texture="grid" texrepeat="1 1" texuniform="true" reflectance=".2"/>
  </asset>

  <worldbody>
  <body name="ground" pos="0 0 0">
  <freejoint/>
"""

final_XML = header %(model_name) +final_member + "</body> </worldbody>"+ " <equality> "+ equality + "</equality>"+ "<actuator> \n"+ actuator+"</actuator>\n"+"</mujoco>"
print(final_XML)


    





