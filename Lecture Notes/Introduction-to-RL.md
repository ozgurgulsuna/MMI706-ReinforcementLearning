###### Introduction  ######

### Introduction to Reinforcement Learning ###
- **Reinforcement Learning (RL)** is a type of machine learning technique that enables an agent to learn in an interactive environment by trial and error using feedback from its own actions and experiences.

- What distinguishes RL from other machine learning techniques,
    - RL does not require labeled data.
    - RL learns from the consequences of its actions.
    - RL learns from the feedback it receives from the environment, which is delayed and not immediate.
    - RL learns to make a sequence of decisions to achieve a long-term goal. (sequental, non independent and identically distributed data)


**Reinforcement Learning Terminology:**
- **Agent**: The learner or decision-maker that interacts with the environment.
- **Environment**: The external system with which the agent interacts.
- **State (s)**: A representation of the environment at a given time.
- **Action (a)**: A decision taken by the agent to transition from one state to another.
- **Reward (r)**: A scalar feedback signal from the environment to the agent.
- **Policy (π)**: A strategy or a rule that the agent follows to select actions.
- **Value Function (V)**: The expected cumulative reward of following a policy from a given state. A prediction of future rewards. Used to evaluate the goodness of states.

> **Definition**: All goals can be framed as the maximization of the expected cumulative reward.

Goal is to select actions to maximize the total future reward. Actions may affect not only the immediate reward but also the future rewards. The agent learns to achieve a balance between immediate and future rewards.

> **Definition**: The **history** is the sequence of observations, actions, rewards, and states.
> - **History**: $H_t = O_1, R_1, A_1, O_2, R_2, A_2, ..., O_t, R_t, A_t$

> **Definition**: The **state** is a function of the history.
> - **State**: $S_t = f(H_t)$

**Full observability:** If the agent's sensors give it access to the complete state of the environment, then the environment is said to be fully observable. $O_t = S_t^a = S_t^e$


**Partial observability:** If the agent's sensors give it access to only a partial state of the environment, then the environment is said to be partially observable.

A **model** predicts what the environment will do next. It is a simulation of the environment. The model can be used for planning and learning.

**Transitions:** $\mathcal{P}$ predicts the next state.

**Rewards:** $\mathcal{R}$ predicts the next immediate reward.

### Categorizing RL Agents ###
- **Value-based RL**: The agent learns a value function that estimates how good it is to be in a particular state.
    - No policy is explicitly learned.
    - Value Function: $V(s)$

- **Policy-based RL**: The agent learns a policy that directly maps states to actions.
    - No value function is explicitly learned.
    - Policy: $\pi(s) \rightarrow a$

- **Actor-critic RL**: The agent learns both a policy and a value function.
    - Policy: $\pi(s) \rightarrow a$
    - Value Function: $V(s)$

- **Model-based RL**: The agent learns a model of the environment.
    - The model is used for planning and learning.
    - Transitions: $\mathcal{P}$
    - Rewards: $\mathcal{R}$
    - Model: $\mathcal{M} = \{\mathcal{P}, \mathcal{R}\}$

- **Model-free RL**: The agent learns directly from the environment without a model.
    - No model is learned.
    - No planning is done.
    - Value Function: $V(s)$
    - Policy: $\pi(s) \rightarrow a$

### Exploration vs. Exploitation ###
- **Exploration**: The agent explores the environment to find the best actions.
- **Exploitation**: The agent exploits the known information to maximize the reward.

### Prediction vs. Control ###
- **Prediction**: Given a policy, compute the value function.
- **Control**: Find the optimal policy.


-----
#MMI706 - [[Reinforcement Learning]] at [[METU]]