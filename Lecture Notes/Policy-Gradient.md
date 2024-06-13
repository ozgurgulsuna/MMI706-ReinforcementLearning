###### Policy Gradient ######

## Policy Based Reinforcement Learning ##
Previously, we approximated the value function or action-value function.

$$V(s) \approx V^\pi(s, w)$$
$$Q(s, a) \approx Q^\pi(s, a, w)$$

The policy was derived from the value function, this time we will directly approximate the policy.

$$\pi(a|s) \approx \mathbb{P}[A_t = a | S_t = s, \theta_t = \theta]$$

In the context of model-free reinforcement learning.


- **Value Based RL**:
    - Learn the value function and derive the policy from the value function.
    - policy is implicit in the value function. ($\epsilon-greedy$)

- **Policy Based RL**:
    - No value function.
    - Learn the policy directly.

> Why we prefer policy-based methods over value-based methods?

The value based methods store the value of each state or state-action pair. The policy based methods store the policy directly, which is more efficient for high-dimensional or continuous action spaces.

However, they typically converge to a local optimum, and evaluating the policy is typically inefficient with high variance.

### Policy Objective Functions ###

The objective is with given policy $\pi_\theta$ and parameter $\theta$ to maximize the expected return. (find the best $\theta$)

In episodic environments, the start value can be used

$$J(\theta) = \mathbb{E}_\pi[G_0 | \pi_\theta]$$

In continuing environments, the average value can be used

$$J_{av}(\theta) = \sum_{s \in S} d(s) \sum_{a \in A} \pi_\theta(a|s) Q^\pi(s, a)$$

where $d(s)$ is the stationary distribution of states.

since the aim is to find $\theta$ that maximizes the objective function, we can use optimization methods like gradient ascent.

### Policy Gradient ###

Let $J(\theta)$ be the objective function that we want to maximize. We can update the weights $\theta$ using the gradient of the objective function.

$$ \Delta \theta = \alpha \nabla J(\theta)$$

where $\alpha$ is the learning rate, or step size.

$$\nabla J(\theta) = \begin{bmatrix} \frac{\partial J(\theta)}{\partial \theta_1} \\ \frac{\partial J(\theta)}{\partial \theta_2} \\ \vdots \\ \frac{\partial J(\theta)}{\partial \theta_n} \end{bmatrix}$$

Finite difference methods can be used to estimate the gradient. Basically, we can estimate the gradient by changing the parameter $\theta$ a little bit and observing the change in the objective function. Sometimes also called the _numerical gradient_ or _perturb and observe_.


### Actor-Critic Policy Gradient ###

Monte Carlo methods can be used to estimate the gradient, but it has high variance.

The actor-critic method uses a critic to estimate the value function and an actor to learn the policy.

$$ Q_w(s, a) \approx Q^\pi(s, a)$$

- **Critic**: The critic updates the action-value function parameters $w$ to minimize the error.
- **Actor**: The actor updates the policy parameters $\theta$, in the direction suggested by the critic.

The critic solves policy evaluation problem, and estimates the value function.

The actor updates policy parameters, which is considered as the policy improvement step.

The bias can be avoided by using an appropriate function approximator.


**Eligibility Traces** 
Eligibility traces are a key concept in reinforcement learning that help bridge the gap between Monte Carlo methods and Temporal Difference (TD) learning. They allow for more efficient learning by combining the strengths of both approaches. When used with policy gradients, eligibility traces can help improve the learning process in policy-based reinforcement learning algorithms.

**Eligibility Traces with Policy Gradient**
When combining eligibility traces with policy gradient methods, the objective is to adjust the policy parameters in a way that takes into account the temporal structure of the problem. The key idea is to use eligibility traces to assign credit to actions based on their contribution to future rewards. This can be particularly useful in environments with delayed rewards.

- Implementation :
Eligibility Traces Update: At each time step, update the eligibility traces for the actions taken. This typically involves decaying the traces over time and adding new traces for the current actions.

- Policy Parameter Update: Use the eligibility traces to adjust the policy parameters. The adjustment is based on the gradient of the policy with respect to the parameters, scaled by the eligibility traces.

-----
#MMI706 - [[Reinforcement Learning]] at [[METU]]