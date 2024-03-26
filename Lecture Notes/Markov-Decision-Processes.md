###### Markov Decision Processes ######

### Markov Process ###

*Markow process* formally describes an environment for reinforcement learning, where the environment is fully observable. The environment's response to an agent's action is probabilistic, and it is described by a set of probabilities. The agent and the environment interact at each of a sequence of discrete time steps. The agent selects an action, and the environment responds by presenting the agent with a reward and the next state. The environment's response at time t depends only on the state and action at time t. The process is described by a 5-tuple, (S, A, P, R, γ), where:

- S is a finite set of states,
- A is a finite set of actions,
- P is a state transition probability matrix,
- R is a reward function, R(s, a, s'),
- γ is a discount factor, γ ∈ [0, 1].

"The future is independent of the past given the present." This is the Markov property.
> **Definition**: A state S(t) is Markov if and only if,  
$\mathbb{P}[S_{t+1} | S_t] = \mathbb{P}[S_{t+1} | S_1, S_2, ..., S_t]$

The state captures all the relevant information from the history. Once the state is known, the history may be thrown away. The state is a sufficient statistic of the future.

### _State Transition Probability Matrix_ ###

For a Markov state $S$, and a next state $S'$, the state transition probability matrix is defined as the probability of transitioning from state $S$ to state $S'$.

$$\mathbb{P}[S_{t+1} = s' | S_t = s] = P_{ss'}$$

The state transition probability matrix, P, is a square matrix of size |S| x |S|, where |S| is the number of states. The element P(s, s') is the probability of transitioning from state s to state s'.

$$P = \begin{bmatrix}
    P_{11} & P_{12} & \cdots & P_{1n} \\
    P_{21} & P_{22} & \cdots & P_{2n} \\
    \vdots & \vdots & \ddots & \vdots \\
    P_{n1} & P_{n2} & \cdots & P_{nn}
\end{bmatrix}$$

Each row of the probability matrix sums to 1, which means that the sum of the probabilites from a single state to all other possible states is 1.

$$\sum_{s'= 1}^{n} P_{ss'} = 1$$

A Markow process is a memoryless random process, where the next state depends only on the current state and not on the sequence of events that preceded it.

> **Definition**: A Markov process (or Markov chain) is a tuple $(S, P)$, where S is a finite set of states, and P is a state transition probability matrix.

**Example**: Consider a simple weather model with 4 states: sunny, cloudy, rainy, and foggy. The state transition probability matrix is given by:


 $$P = \begin{array}{c|cccc}
    & \text{sun} & \text{cloud} & \text{rain} & \text{fog} \\
    \hline
    \text{sun} & 0.8 & 0.1 & 0.1 & 0 \\
    \text{cloud} & 0.2 & 0.6 & 0.1 & 0.1 \\
    \text{rain} & 0.1 & 0.1 & 0.7 & 0.1 \\
    \text{fog} & 0 & 0 & 0 & 1
\end{array}$$

### Markov Reward Process ###

A Markov reward process is a Markov chain with values. It is a tuple $(S, P, R, \gamma)$, where:

- S is a finite set of states,
- P is a state transition probability matrix,
- R is a reward function, $R_s = \mathbb{E}[R_{t+1} | S_t = s]$
- $\gamma$ is a discount factor, $\gamma \in [0, 1]$.

The reward function R(s, s') defines the immediate reward received after transitioning from state s to state s'. The discount factor $\gamma$ determines the present value of future rewards.

> **Definition**: A Markov reward process is a tuple $(S, P, R, \gamma)$, where S is a finite set of states, P is a state transition probability matrix, R is a reward function, and $\gamma$ is a discount factor.
> $$R_s = \mathbb{E}[R_{t+1} | S_t = s]$$

> **Definition**: The return $G_t$ is the total discounted reward from time-step t.
> $$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ... = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

Discount factor is the present value of future rewards. It is a value between 0 and 1. The purpose of discounting is to make the sum of rewards finite.

- gamma = 0: the agent is short-sighted "myopic" and only considers immediate rewards.
- gamma = 1: the agent is far-sighted and considers future rewards with equal weight.

Why do we use discount factor ?

- to make reward finite / mathematically well-defined
- to have a stable solution / to avoid infinite rewards in cyclic environments
- to not rely on future rewards due to the uncertainty of the future
- immidiate reward more valuable since it is not delayed
- it is how animals behave in nature

### Value Function ###
The value function $v(s)$ gives the long-term value of state s under policy $\pi$. It is the expected return starting from state s, and then following policy $\pi$.

> **Definition**: The state value function $v(s)$ of a Markov reward process is the expected return starting from state s, and then following policy $\pi$.
> $$v(s) = \mathbb{E}[G_t | S_t = s]$$

The value function is the expected return starting from state s, and then following policy $\pi$. It is the expected return when starting from state s and following policy $\pi$ thereafter.

### Bellman Equation ###
The Bellman equation is a fundamental equation in dynamic programming. It decomposes the value function into two parts: immediate reward and discounted value of the next state.

> **Definition**: The Bellman equation for the state value function $v(s)$ is given by:
> $$v(s) = \mathbb{E}[R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ... | S_t = s]$$
> $$v(s) = \mathbb{E}[R_{t+1} + \gamma (R_{t+2} + \gamma R_{t+3} + ...) | S_t = s]$$
> $$v(s) = \mathbb{E}[R_{t+1} + \gamma G_{t+1} | S_t = s]$$
> $$v(s) = \mathbb{E}[R_{t+1} + \gamma v(S_{t+1}) | S_t = s]$$





















-----
#MMI706 - [[Reinforcement Learning]] at [[METU]]