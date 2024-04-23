#### training a reinforcement learning agent with ####
# MuJoCo and Gymnasium #

### MuJoCo ###
MuJoCo is a physics engine for detailed, efficient rigid body simulations with contacts. It has a Python API that allows for easy integration with machine learning libraries such as TensorFlow, PyTorch, and others.

### Gymnasium ### 
Gymnasium is a project that provides an API for all single agent reinforcement learning environments, and includes implementations of common environments: cartpole, pendulum, mountain-car, mujoco, atari, and more.

https://gymnasium.farama.org/environments/mujoco/

It is needed to be a new MuJoCo environment to be added to Gymnasium with our robot model defined in the XML file.


### How to create Gymnasium enviroment from your MuJoCo model? ###
How to create a new environment in Gymnasium from a MuJoCo model and train a reinforcement learning agent using the Gymnasium shell and algorithms from the stable-baselines library.

In this example we will need python packages [MuJoCo](https://github.com/google-deepmind/mujoco), [Gymnasium](https://gymnasium.farama.org/), and [Stable-baselines](https://stable-baselines.readthedocs.io/en/master/).

```bash
pip install mujoco==3.1.4
pip install gymnasium==0.29.1
pip install stable-baselines==2.3.0
```




First start with a similar environment to the one you want to create. For example, if you want to create a new environment for a robot, you can start with the `ant-v5.py` environment in `mujoco`. 


[Note that the subtree center of mass for the world body is the center of mass of the entire model.](https://mujoco.readthedocs.io/en/stable/XMLreference.html)



### Resources ###
[https://www.youtube.com/watch?v=OqvXHi_QtT0&ab_channel=JohnnyCode]  





