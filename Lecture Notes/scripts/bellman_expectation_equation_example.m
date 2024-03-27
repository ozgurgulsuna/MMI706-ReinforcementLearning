clc 
clear 
close all

%% State definitions (6 states)
% state1: C1
% state2: C2
% state3: C3
% state4: Pass
% state5: Pub
% state6: FB
% state7: Sleep

% Transition Probability Matrix
P = [ 0   0.5 0   0   0   0.5 0  ;
      0   0   0.8 0   0   0   0.2;
      0   0   0   0.6 0.4 0   0  ;
      0   0   0   0   0   0   1  ;
      0.2 0.4 0.4 0   0   0   0  ;
      0.1 0   0   0   0   0.9 0  ;
      0   0   0   0   0   0   1  ]

% Reward Vector
R = [ -2 -2 -2 10  1 -1 0 ]'

% Discount Factor
gamma = 0.90

% Analytical Solution
V_analytical = inv(eye(7) - gamma * P) * R

% Value Iteration
V = zeros(7,1);
V_new = zeros(7,1);
epsilon = 0.00001;
delta = 1;

while delta > epsilon
    for i = 1:7
        V_new(i) = R(i) + gamma * max(P(i,:) * V);
    end
    delta = max(abs(V_new - V));
    V = V_new;
end

V_iterative = V


%% State definitions (6 states)
% state1: study
% state2: study
% state3: study
% state4: facebook
% state5: sleep
% state6: pub

% Transition Probability Matrix
P = [ 0   0.5 0   0.5 0   0  ;
      0   0   0.5 0   0.5 0  ;
      0   0   0   0   0.5 0.5;
      0.5 0   0   0.5 0   0  ;
      0   0   0   0   1   0  ;
      0.2 0.4 0.4 0   0   0  ]

% Reward Vector
R = [ -3/2 -1 11/2 -1/2  0 0 ]'

% Discount Factor
gamma = 0.999

% Analytical Solution
V_analytical = inv(eye(6) - gamma * P) * R

% Value Iteration
V = zeros(6,1);
V_new = zeros(6,1);
epsilon = 0.0001;
delta = 1;

while delta > epsilon
    for i = 1:6
        V_new(i) = R(i) + gamma * max(P(i,:) * V);
    end
    delta = max(abs(V_new - V));
    V = V_new;
end

V_iterative = V

%% State definitions (5 states)
% state1: study
% state2: study
% state3: study
% state4: facebook
% state5: sleep

% Transition Probability Matrix
P = [ 0   0.5 0   0.5 0   ;
      0   0   0.5 0   0.5 ;
      0.2*0.5 0.5*0.4 0.5*0.4   0   0.5 ;
      0.5 0   0   0.5 0   ;
      0   0   0   0   1   ]

% Reward Vector
R = [ -3/2 -1 11/2 -1/2  0 ]'

% Discount Factor
gamma = 0.999

% Analytical Solution
V_analytical = inv(eye(5) - gamma * P) * R

% Value Iteration
V = zeros(5,1);
V_new = zeros(5,1);
epsilon = 0.0001;
delta = 1;

while delta > epsilon
    for i = 1:5
        V_new(i) = R(i) + gamma * max(P(i,:) * V);
    end
    delta = max(abs(V_new - V));
    V = V_new;
end

V_iterative = V
