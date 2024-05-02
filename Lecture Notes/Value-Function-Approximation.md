###### Large-Scale RL ######

## Value Function Approximation ##
It is more useful to solve problems with large state spaces 

- Backgammon: 10^20 states
- Chess: 10^40 states
- Heliocopter:  continuous state space
 
By now we had our value function as a table, but it is not possible to store all the states in a table. We need to approximate the value function.
- Every state s has an entry in V(s) (prediction)
- Every state-action pair s has an entry in Q(s,a) (control)

For large MDPs, we can't store all the states and/or actions in the memory. We need to approximate the value function.
It is also slow to learn the value of each state individually. We need to generalize the value function.


### Solution for Large MDPs ###
- Estimate the value function with a parameterized function  
$V(s, w) \approx V_{\pi}(s)$  
$Q(s, a, w) \approx Q_{\pi}(s, a)$

- Generalize from the training states to the unseen states
- Update the weights w to minimize the error using monte-carlo (MC) or temporal difference (TD) learning


### Function Approximation ###
There are many ways to approximate the value function. Some of them are:
- **Linear function approximation**
- **Neural networks**
- Decision trees
- Nearest neighbors
- Fourier basis functions
- Tile coding

we consider **differentiable** function approximators, because we can use gradient-based optimization methods to update the weights. Those are suitable for **non-stationary**, **non-i.i.d.** data.

Since we are updating the policy, the data is non-stationary. The data is also non-i.i.d. because the states are correlated.

## Gradient Descent ##
Let $J(w)$ be the objective function that we want to minimize. We can update the weights w using the gradient of the objective function.

Gradient of the objective function:  
$$\nabla J(w) = \begin{bmatrix} \frac{\partial J(w)}{\partial w_1} \\ \frac{\partial J(w)}{\partial w_2} \\ \vdots \\ \frac{\partial J(w)}{\partial w_n} \end{bmatrix}$$

To find the minimum of the objective function, we update the weights using the gradient:

$$\Delta w = -\frac{1}{2}\alpha \nabla J(w)$$

where $\alpha$ is the learning rate, or step size.

### Stochastic Gradient Descent ###
In stochastic gradient descent, we want to find a parameter vector w that minimizes the expected value of the loss function:

$$J(w) = E_{\pi}[(V_{\pi}(s) - V(s, w))^2]$$
    
where $V_{\pi}(s)$ is the true value of state s, and $V(s, w)$ is the predicted value of state s.

$$\Delta w = -\frac{1}{2}\alpha \nabla J(w)$$ 
$$ = \alpha E_{\pi}[(V_{\pi}(s) - V(s, w))\nabla V(s, w)]$$

We can't compute the expectation exactly, so we use a sample to estimate the expectation:

$$\Delta w = \alpha (V_{\pi}(s) - V(s, w))\nabla V(s, w)$$

### Feature Vector ###
We need to represent the state s as a feature vector x(s). The feature vector is a vector of real numbers that represent the state s. The feature vector is a function of the state s.

For example: 
- Distance to the goal, 
- Distance to the obstacles,
- Trends in the stock market
- Queen and king positions in chess

### Linear Function Approximation ###
The value function is a linear combination of the feature vector:

$$V(s, w) = x^T(s) w = \sum_{j=1}^{n} x_j(s) w_j$$

where w is the weight vector, and x(s) is the feature vector.

$$ J(w) = E_{\pi}[(V_{\pi}(s) - x^T(s) w)^2]$$

now the objective function is quadratic in w, so we can find the minimum of the objective function using the gradient:

$$\nabla V(s, w) = x(s)$$

$$\Delta w = \alpha (V_{\pi}(s) - x^T(s) w) x(s)$$

since the feature vector is incorporated in the update rule, the update rule favors the features that are more important for the value function.

### Incremental Prediction Algorithms ###
In practice we substitute the true value of the state with the return from the sample:

- Monte-Carlo (MC): $G_t$  
$\Delta w= \alpha(G_t - V(s_t, w)) \nabla V(s_t, w)$

- Temporal Difference (TD(0)): $R_{t+1} + \gamma V(s_{t+1}, w)$  
$\Delta w= \alpha(R_{t+1} + \gamma V(s_{t+1}, w) - V(s_t, w)) \nabla V(s_t, w)$

- TD($\lambda$): $\lambda$-return  
$\Delta w= \alpha(G_t^{\lambda} - V(s_t, w)) \nabla V(s_t, w)$

#### Monte-Carlo (MC) Approximation ####
Having a training set of states and returns, training data, we can update the weights using the gradient descent algorithm.

$$\langle S_1, G_1 \rangle, \langle S_2, G_2 \rangle, \langle S_3, G_3 \rangle, \dots$$

$$\Delta w = \alpha(G_t - V(S_t, w)) \nabla V(S_t, w)$$
$$  = \alpha(G_t - x^T(S_t) w) x(S_t)$$

#### Temporal Difference (TD) Approximation ####
The TD target is $R_{t+1} + \gamma V(S_{t+1}, w)$, it is a biased estimate of the return. Training data is the sequence of states and rewards.

$$\langle S_1, R_2 + \gamma V(S_2, w) \rangle, \langle S_2, R_3 + \gamma V(S_3, w) \rangle, \langle S_3, R_4 + \gamma V(S_4, w) \rangle, \dots$$

Linear function approximation with TD(0) update rule:

$$\Delta w = \alpha(R_{t+1} + \gamma V(S_{t+1}, w) - V(S_t, w)) \nabla V(S_t, w)$$
$$  = \alpha \delta_t x(S)$$

where $\delta_t = R_{t+1} + \gamma V(S_{t+1}, w) - V(S_t, w)$ is the TD error.

#### TD($\lambda$) Approximation ####
The TD($\lambda$) target is the $\lambda$-return, it is a biased estimate of the return. Training data is the sequence of states and rewards.

$$\langle S_1, G_1^{\lambda} \rangle, \langle S_2, G_2^{\lambda} \rangle, \langle S_3, G_3^{\lambda} \rangle, \dots$$

Forward linear function approximation with TD($\lambda$) update rule:

$$\Delta w = \alpha(G_t^{\lambda} - V(S_t, w)) \nabla V(S_t, w)$$

Backward linear function approximation with TD($\lambda$) update rule:

$$\delta_t = R_{t+1} + \gamma V(S_{t+1}, w) - V(S_t, w)$$
$$E_t = \gamma \lambda e_{t-1} + x(S_t)$$
$$\Delta w = \alpha \delta_t E_t$$

where $E_t$ is the eligibility trace.

-----
#MMI706 - [[Reinforcement Learning]] at [[METU]]