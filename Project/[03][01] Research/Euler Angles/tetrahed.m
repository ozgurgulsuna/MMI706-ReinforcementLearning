clc
clear 
close all

%%

v_2 = [0 0.8165066 0 ];
v_1 = [0.5 0 0.2886751];

v_diff = v_2-v_1;

x = v_diff(1);
y =  v_diff(2);
z =  v_diff(3);

yaw = atan2(x,z);
padj = sqrt(x^2+z^2) ;
pitch = atan2(padj, y);

%%

sc_fac = 1.05
sc_fac = 0.95
sc_fac = 2
pre_vec = [-0.5 0.288675 0.8165]
%pre_vec = [0.5  0.86602   0]
% pre_vec = [ 0 -0.5774    0.8165]
vec= pre_vec.*sc_fac

%% 
member_lenght = 1;
node_size = 0.1;

member_offset = member_lenght-node_size/2

%% axis-angle to euler
x =-0.5 
y = -0.8660254
z =  0  
a = 0.615474
heading = atan2(y * sin(a)- x * z * (1 - cos(a)) , 1 - (y^2 + z^2 ) * (1 - cos(a)))
attitude = asin(x * y * (1 - cos(a)) + z * sin(a))
bank = atan2(x * sin(a)-y * z * (1 - cos(a)) , 1 - (x^2 + z^2) * (1 - cos(a)))

euler = [bank attitude heading]
