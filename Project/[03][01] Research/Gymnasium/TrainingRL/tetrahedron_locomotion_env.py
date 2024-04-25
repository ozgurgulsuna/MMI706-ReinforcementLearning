__credits__ = ["Gulsuna-Ozgur"]
__license__ = "GPL"
__version__ = "1.0.0"

from typing import Dict, Tuple, Union

import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

import os



DEFAULT_CAMERA_CONFIG = {
    "distance": 8.0,
}

# This is the class that defines the MuJoCo environment, it can be modified with the directions in the comments.
class TetrahedronLocomotionEnv(MujocoEnv, utils.EzPickle):
    # description of the environment.
    r"""Tetrahedron Truss Robot Environment.
    ### Description ###
    This environment consist of a tetrahedron truss robot where the each edge (**member**) of the tetrahedron is a linear actuator and the vertices (**nodes**) are the passive joints. The robot is controlled by changing the length of the linear actuators either dynamically or by changing the position of the center of mass of the robot. The goal of the robot is to reach the target position. 

    ### Notes ###
    Problem parameters:
    - n : number of nodes
    - m : number of members
    - $l_i$ : length of the ith member ($i \in \{1,2,\cdots,m\}$)
    - $(x_i, y_i, z_i)$ : coordinates of the ith node ($i \in \{1,2,\cdots,n\}$)
    - $(x_{cm}, y_{cm}, z_{cm})$ : coordinates of the center of mass
    - $(x_{target}, y_{target}, z_{target})$ : coordinates of the target position

    ### Action Space ###
    The action space is a ```Box(0,2,(6,), float32)```. An action is a vector of 6 elements where each element is the length of the member. The length of the member is between 0 and 2 meters.

    |Num |  Action  | Min | Max |    Name     | Joint |    Unit    |
    |----|----------|-----|-----|-------------|-------|------------|
    |0   | position | 0   | 2   | [Member_1]  | slide | lenght (m) |
    |1   | position | 0   | 2   | [Member_2]  | slide | lenght (m) |
    |2   | position | 0   | 2   | [Member_3]  | slide | lenght (m) |
    |3   | position | 0   | 2   | [Member_4]  | slide | lenght (m) |
    |4   | position | 0   | 2   | [Member_5]  | slide | lenght (m) |
    |5   | position | 0   | 2   | [Member_6]  | slide | lenght (m) |

    __notes:__ might need to change the position control to force control, thus we can have force as an input and lenght as an observation.

    ### Observation Space ###
    Observations capture the positional values of the center of mass and the respective time derivative aka velocity of the center of mass. Additionally, the positional values of the nodes are also captured.
    The observation space is a ```Box(-inf, inf, (18,), float64)```. We might need to alter the observation space to have a better learning performance.


    |:-----:|:----------------:|:-----:|:-----:|:---------:|:-------:|:---------:|
    |**Num**| **Observation**  |**Min**|**Max**|  **Name** |**Joint**| **Unit**  |
    |   0   |  x-coord of CoM  |  -inf |  inf  |  coord_x  |  free   |   (m)     |
    |   1   |  y-coord of CoM  |  -inf |  inf  |  coord_y  |  free   |   (m)     |
    |   2   |  z-coord of CoM  |  -inf |  inf  |  coord_z  |  free   |   (m)     |
    |   3   |  x-orien of CoM  |  -inf |  inf  |  orient_x |  free   |   (rad)   |
    |   4   |  y-orien of CoM  |  -inf |  inf  |  orient_y |  free   |   (rad)   |
    |   5   |  z-orien of CoM  |  -inf |  inf  |  orient_z |  free   |   (rad)   |
    |   6   |  x-vel of CoM    |  -inf |  inf  |  vel_x    |  free   |   (m/s)   |
    |   7   |  y-vel of CoM    |  -inf |  inf  |  vel_y    |  free   |   (m/s)   |
    |   8   |  z-vel of CoM    |  -inf |  inf  |  vel_z    |  free   |   (m/s)   |
    |   9   |  x-ang.vel CoM   |  -inf |  inf  |  omega_x  |  free   |   (rad/s) |
    |  10   |  y-ang.vel CoM   |  -inf |  inf  |  omega_y  |  free   |   (rad/s) |
    |  11   |  z-ang.vel CoM   |  -inf |  inf  |  omega_z  |  free   |   (rad/s) |
    |  12   |  length of [M_1] |  -inf |  inf  |  length_1 |  slide  |   (m)     |
    |  13   |  length of [M_2] |  -inf |  inf  |  length_2 |  slide  |   (m)     |
    |  14   |  length of [M_3] |  -inf |  inf  |  length_3 |  slide  |   (m)     |
    |  15   |  length of [M_4] |  -inf |  inf  |  length_4 |  slide  |   (m)     |
    |  16   |  length of [M_5] |  -inf |  inf  |  length_5 |  slide  |   (m)     |
    |  17   |  length of [M_6] |  -inf |  inf  |  length_6 |  slide  |   (m)     |

    - add node positions
    - add ground nodes
    - add member orientation

    __notes:__ for the dynamic movement, linear forces rather than positions are controlled.  
    __notes:__ might want to minimize the lengths at the end, or final configuration enforces that.

    ### Rewards ###


    * *```healthy_reward```* : The robot is healthy and the target position is not reached.
    * *```forward_reward```* : A reward of moving forward in the x-direction which is measured as (x-coordinate before action - x-coordinate after action)/dt. dt is the time between actions and is dependent on the ```frame_skip``` parameter (default is 5), where the frametime is 0.01 - making the default dt = 5 * 0.01 = 0.05. This reward would be positive if the robot moves forward (in positive x direction). 
    * *```ctrl_cost```* : A penalty for the control effort. The control effort is the sum of the squares of the action values multiplied by the ctrl_cost_weight parameter (default is 0.05).
    * *```size_cost```* : A penalty for the overall robot size. The size cost is the sum of the squares of the lengths of the members multiplied by the size_cost_weight parameter (default is 0.0001).


    In the further implementation, the reward function might be changed to a more complex one.
    An example is the negative of the distance between the center of mass and the target position. The reward function is defined as:

    $$ r = -\sqrt{(x_{cm} - x_{target})^2 + (y_{cm} - y_{target})^2 + (z_{cm} - z_{target})^2} $$

    another cost can be: 
    * *```contact_cost```* : A penalty for the contact forces. The contact forces are the sum of the squares of the forces on the robot multiplied by the ```contact_cost_weight``` parameter (default is 0.0001).

    ### Episode Termination ###
    The truss robot is said to be unhealthy if any of the following conditions are met:
    - Any of the state space values are NaN.
    - The z-coordinate of the center of mass i outside the range of $[0.1, 2]$ meters. (inflated)

    The episode is terminated if the robot is unhealthy or the robot reaches the target position. trunckated at 1000 steps.

    """

    metadata = {
    "render_modes": ["human", "rgb_array", "depth_array"],
    "render_fps": 100,
    }

    # initial configuration of the environment with the default parameters.
    def __init__(
        self,
        xml_file: str = os.path.abspath("assets/scene.xml"),
        frame_skip: int = 5,
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        forward_reward_weight: float = 10,
        ctrl_cost_weight: float = 0.0000,
        healthy_reward: float = 1.0,
        main_body: Union[int, str] = 1,
        size_cost_weight: float = 0.0001,
        terminate_when_unhealthy: bool = True,
        healthy_z_range: Tuple[float, float] = (0.0, 5.0), # only in planar surface
        reset_noise_scale: float = 0.1,
        episode_horizon: int = 500,
        step_number: int = 0,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            default_camera_config,
            forward_reward_weight,
            ctrl_cost_weight,
            healthy_reward,
            main_body,
            size_cost_weight,
            terminate_when_unhealthy,
            healthy_z_range,
            reset_noise_scale,
            episode_horizon,
            step_number,
            **kwargs,
        )

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight

        self._healthy_reward = healthy_reward
        self._size_cost_weight = size_cost_weight
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range

        self._main_body = main_body

        self._reset_noise_scale = reset_noise_scale

        self._episode_horizon = episode_horizon
        self._step_number = step_number

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=None, # observation space is defined in the reset function
            default_camera_config=default_camera_config,
            **kwargs,
        )

        self.metadata = {
            "render_modes": ["human", "rgb_array", "depth_array"],
            "video_fps": int(np.round(1.0 / self.dt)),
        }

        # OBSERVATION SPACE

        obs_size = 3+3+6+12; # 3 for CoM position, 3 for CoM linear velocity, 6 for member lengths, 12 for node positions

        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_size,),
            dtype=np.float64,
        )

        self.observation_structure = {
            "com": 3,
            "comvel": 3,
            "member_lengths": 6,
            "node_positions": 12,
        }

    @property
    def healthy_reward(self):
        return self.is_healthy * self._healthy_reward
    
    def control_cost(self,action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    @property
    def is_healthy(self):
        # check if the robot is healthy
        state = self.state_vector()
        CoM_pos = self.data.subtree_com[0].copy()
        min_z, max_z = self._healthy_z_range
        is_healthy = np.isfinite(state).all() and min_z <= CoM_pos[2] <= max_z
        #print("state",state)
        #print("is healthy",is_healthy)
        return is_healthy  
    
    def step(self, action):
        # step function of the environment
        position_before = self.data.subtree_com[0].copy()
        self.do_simulation(action, self.frame_skip)
        self._step_number +=1
        print("step",self._step_number)
        
        position_after = self.data.subtree_com[0].copy()

        velocity = (position_after - position_before) / self.dt
        x_velocity, y_velocity, z_velocity = velocity

        observation = self._get_obs()
        reward, reward_info = self._get_reward(x_velocity, action)
        terminated = (not self.is_healthy) and self._terminate_when_unhealthy
        truncated = self._step_number > self._episode_horizon
        print("truncated",truncated)
        print("terminated",terminated)

        # info is incorrect
        info = {
            "x_position": self.data.subtree_com[0],
            "y_position": self.data.subtree_com[1],
            "z_position": self.data.subtree_com[2],
            "distance_from_origin": np.linalg.norm(self.data.subtree_com[:2], ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            **reward_info,
        }

        if self.render_mode == "human":
            self.render()

	# step must return five values: obs, reward, terminated, truncated, info.
        return observation, reward, terminated, truncated, info
    
    def _get_reward(self, x_velocity:float, action):
        # reward function
        forward_reward = self._forward_reward_weight * x_velocity
        healthy_reward = self._healthy_reward
        total_reward = forward_reward + healthy_reward

        ctrl_cost = self.control_cost(action)
        total_cost = ctrl_cost

        reward = total_reward - total_cost
        
        #print("forward_reward", forward_reward)
        #print("healthy_reward", healthy_reward)
        #print("total_reward", total_reward)
        #print("ctrl_cost", ctrl_cost)
        #print("total_cost", total_cost)
        #print("reward", reward)
        
        
        reward_info = {
            "forward_reward": forward_reward,
            "healthy_reward": healthy_reward,
            "ctrl_cost": -ctrl_cost,
        }

        return reward, reward_info
    
    def _get_obs(self):
        position = self.data.subtree_com[0] # we will have the first row
        velocity = self.data.subtree_com[0] # keep it same for now
        member_lengths = self.data.actuator_length[:6]
        node_positions = np.concatenate((np.array(self.data.geom("(1)").xpos[:3]),   # not sure if .xpos is correct
                                         np.array(self.data.geom("(2)").xpos[:3]),
                                         np.array(self.data.geom("(3)").xpos[:3]),
                                         np.array(self.data.geom("(6)").xpos[:3])), axis=0)

        print("position",position)
        print("velocity",velocity)
        print("member_lengths",member_lengths)
        print("node_pos",node_positions)
        
        #return position
        return np.concatenate((position, velocity, member_lengths, node_positions), axis=0)
    
    def reset_model(self):
        self._step_number = 0
	# randomizing the initial configuration should be done in a different way for us
        #noise_low = -self._reset_noise_scale
        #noise_high = self._reset_noise_scale
        
        #qpos = self.init_qpos + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq)
        #qvel = self.init_qvel + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nv)
        

        qpos = self.init_qpos 
        qvel = self.init_qvel
        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation
    
    def _get_reset_info(self):
        return {
            "x_position": self.data.subtree_com[0],
            "y_position": self.data.subtree_com[1],
            "z_position": self.data.subtree_com[2],
            "distance_from_origin": np.linalg.norm(self.data.subtree_com[:2], ord=2),
        }
    





        # NOTE: we did not had a way to calculate center of mass here, solve it.

        


