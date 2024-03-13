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







-----
#MMI706 - [[Reinforcement Learning]] at [[METU]]