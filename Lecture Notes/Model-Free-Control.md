###### Model-Free Control ######
In model-free control, we do not have a model of the environment. The model can be unknown, incomplete, or computationally expensive. We directly learn the policy or the action-value function.

Prediction is the process of estimating the value function, while control is the process of finding the optimal policy. The objective function for control is the 
action-value function:

$$Q(s, a) = \mathbb{E}[G_t | S_t = s, A_t = a]$$

**On-Policy Control**:On-policy control methods learn the policy that they are following. The policy is updated based on the action-value function.

**Off-Policy Control**: Off-policy control methods learn the policy that is different from the policy that they are following. The target policy is updated based on the action-value function. The behavior policy is the policy that is followed.

_Policy evaluation_ is the process of estimating the value function for a given policy. _Policy improvement_ is the process of finding a better policy based on the value function.

- Greedy Policy Improvement over Value Function:
  - $\pi'(s) = argmax_a \mathcal{R}_s^a + \mathcal{P}_{ss'}^a V(s')$
  - requires a model of the environment

- Greedy Policy Improvement over Action-Value Function:
    - $\pi'(s) = argmax_a Q(s, a)$
    - does not require a model of the environment
    - uses the action-value function

### SARSA ###
SARSA is an on-policy control method. It is a model-free method that learns the action-value function.

$$ Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]$$

Every time step, the policy evaluation is done with SARSA, and the policy improvement is done with the $\epsilon$-greedy policy.

**n-step SARSA**: The update is done with n steps. The return is calculated with n steps.

$$ q_t^{(n)} = R_{t+1} + \gamma R_{t+2} + \dots + \gamma^{n-1} R_{t+n} + \gamma^n Q(S_{t+n}, A_{t+n})$$

$$ Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [q_t^{(n)} - Q(S_t, A_t)]$$

**Forward View SARSA**: The return is calculated with all the steps.

$$ q_t^{\lambda} = (1 - \lambda) \sum_{n=1}^{\infty} \lambda^{n-1} q_t^{(n)}$$

$$ Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [q_t^{\lambda} - Q(S_t, A_t)]$$

### Q-Learning ###
Q-learning is an off-policy control method. It is a model-free method that learns the action-value function.

The next action is selected based on the behavior policy. The update is done based on the target policy.

$$ Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [R_{t+1} + \gamma max_a Q(S_{t+1}, a) - Q(S_t, A_t)]$$

Both behavior and target policies are improved. Target policy is improved with the greedy policy improvement while the behavior policy is improved with the $\epsilon$-greedy policy.





-----
#MMI706 - [[Reinforcement Learning]] at [[METU]]