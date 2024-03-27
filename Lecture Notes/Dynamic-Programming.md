###### Dynamic Programming ######

### Dynamic Programming ###

**Dynamic**: Sequental or temporal component.  
**Programming**: Optimizing a "program" or a sequence of decisions.

Dynamic programming is a method for solving complex problems by breaking them down into simpler subproblems. It is applicable to problems exhibiting the properties of overlapping subproblems and optimal substructure.

- Overlapping subproblems: The problem can be broken down into subproblems which are reused several times.
- Optimal substructure: The problem can be solved by combining optimal solutions to its subproblems / recursive subproblems.

Markov processes satisfy the both properties. The Bellman equation is a recursive equation that defines the value of a state as the sum of the immediate reward and the discounted value of the next state. The value function stores and reuses the solutions to the subproblems.

- Dynamic programming assumes full knowledge of the MDP.
- It is used for planning in an MDP.
- For prediction:
    - Input: MDP (S, A, P, R, γ)
    - or: MRP (S, P, R, γ)
    - Output: value function V(s) for all states s ∈ S
- For control:
    - Input: MDP (S, A, P, R, γ)
    - Output: optimal policy π*(s) for all states s ∈ S
- It is computationally expensive for large state spaces.

--------------------------------------------------------------------------------

### Policy Evaluation ###

**Problem**: Given a policy π, compute the value function V(s) for all states s ∈ S.
**Solution**: Iterative application of the Bellman expectation equation.

- start with an initial value function V0(s) for all states s ∈ S
- update the value function using the Bellman expectation
    - at each iteration k, 
    - update V(s) for all states s ∈ S
    - until convergence

**Policy Evaluation Algorithm in Small Gridworld Example**:

- Undiscounted episodic MDP (γ = 1)
- 4x4 gridworld
- 4 actions: up, down, left, right
- stochastic policy: 0.25 for each action
- reward: -1 for each step
- terminal states: A and B with 0 reward
- actions that would take the agent out of the grid leave the state unchanged

$$\begin{array}{|c|c|c|c|}
\hline
  & 1 & 2 & 3 \\
\hline
4 & 5 & 6 & 7 \\
\hline
8 & 9 & 10 & 11 \\
\hline
12 & 13 & 14 &  \\
\hline
\end{array}$$

- Initial value function V0(s) = 0 for all states s ∈ S

Iteration 0 -> 
$$\begin{array}{|c|c|c|c|}
\hline
  0  &  0  &  0  &  0  \\
\hline
  0  &  0  &  0  &  0  \\
\hline
  0  &  0  &  0  &  0  \\
\hline
  0  &  0  &  0  &  0  \\
\hline
\end{array}$$


Iteration 1 -> 
$$\begin{array}{|c|c|c|c|}
\hline
0.0 & -1.00 & -1.00 & -1.00 \\
\hline
-1.00 & -1.00 & -1.00 & -1.00 \\
\hline
-1.00 & -1.00 & -1.00 & -1.00 \\
\hline
-1.00 & -1.00 & -1.00 & 0.00 \\
\hline
\end{array}$$


Iteration 2 -> 
$$\begin{array}{|c|c|c|c|}
\hline
0 & -1.75 & -2 & -2 \\
\hline
-1.75 & -2 & -2 & -2 \\
\hline
-2 & -2 & -2 & -1.75 \\
\hline
-2 & -2 & -1.75 & 0 \\
\hline
\end{array}$$


Iteration 3 -> 
$$\begin{array}{|c|c|c|c|}
\hline
0 & -2.44 & -2.75 & -2.75 \\
\hline
-2.44 & -2.75 & -2.75 & -2.44 \\
\hline
-2.75 & -2.75 & -2.44 & -2 \\
\hline
-2.75 & -2.44 & -2 & 0 \\
\hline
\end{array}$$


Iteration 100 -> 
$$\begin{array}{|c|c|c|c|}
\hline
0 & -14.0 & -20.0 & -22.0 \\
\hline
-14.0 & -18.0 & -20.0 & -20.0 \\
\hline
-20.0 & -20.0 & -18.0 & -14.0 \\
\hline
-22.0 & -20.0 & -14.0 & 0 \\
\hline
\end{array}$$

--------------------------------------------------------------------------------

### Policy Iteration ###

**Problem**: Given an MDP, find the optimal policy π*.  
**Solution**: Iterative application of policy evaluation and policy improvement.

Given policy π,
- **Evaluate** the policy to find the value function Vπ(s) for all states s ∈ S
- **Improve** the policy by acting greedily with respect to Vπ(s)















-----
#MMI706 - [[Reinforcement Learning]] at [[METU]]