% Define number of members and iterations
numMembers = 6;
iterations = 10000;

% Initialize member data structures
members = cell(1, numMembers);
for i = 1:numMembers
  members{i}.length = randomLength();
  members{i}.p1 = rand(3, 1); % Random initial position of point 1 in 3D
  direction = randn(3, 1); % Random direction vector
  direction = direction / norm(direction); % Normalize to unit vector
  members{i}.p2 = members{i}.p1 + members{i}.length * direction; % Random initial position of point 2 in 3D
end

%%
% close all
% figure
% axis equal
% grid on

% 
% % Plot initial member positions
% hold on
% for i = 1:numMembers
%   plot3([members{i}.p1(1), members{i}.p2(1)], [members{i}.p1(2), members{i}.p2(2)], [members{i}.p1(3), members{i}.p2(3)], 'b-');
% end
% hold off;
% drawnow;

%%
% Simulation loop
for iter = 1:iterations
  % Initialize forces and torques
  netForces = cell(1, numMembers);
  netTorques = cell(1, numMembers);
  
  for i = 1:numMembers
    netForces{i} = zeros(3, 1);
    netTorques{i} = zeros(3, 1);
  end

  % Apply forces between member endpoints
for i = 1:numMembers
  if i == 1
    % Calculate forces
    force1 = attractionForce(members{i}.p2, members{2}.p1, 10);
    force2 = attractionForce(members{i}.p1, members{3}.p2, 10);
    
    % Update net forces for members
    netForces{i} = 0;

    % Calculate torques (assuming midpoint of member as pivot)
    midpoint = (members{i}.p1 + members{i}.p2) / 2;
    r1 = members{i}.p1 - midpoint;
    r2 = members{i}.p2 - midpoint;
    
    % Cross product in 3D
    torque1 = cross(r1, force1);
    torque2 = cross(r2, force2);
    
    netTorques{i} = 0;

    elseif i == 2
    % Calculate forces
    force1 = attractionForce(members{i}.p1, members{1}.p2, 10);
    force2 = attractionForce(members{i}.p2, members{3}.p1, 10);
    
    % Update net forces for members
    netForces{i} = force1 + force2;

    % Calculate torques (assuming midpoint of member as pivot)
    midpoint = (members{i}.p1 + members{i}.p2) / 2;
    r1 = members{i}.p1 - midpoint;
    r2 = members{i}.p2 - midpoint;
    
    % Cross product in 3D
    torque1 = cross(r1, force1);
    torque2 = cross(r2, force2);
    
    netTorques{i} = torque1 + torque2;

  elseif i == 3
    % Calculate forces
    force1 = attractionForce(members{i}.p1, members{2}.p2, 10);
    force2 = attractionForce(members{i}.p2, members{1}.p1, 10);
    
    % Update net forces for members
    netForces{i} = force1 + force2;

    % Calculate torques (assuming midpoint of member as pivot)
    midpoint = (members{i}.p1 + members{i}.p1) / 2;
    r1 = members{i}.p1 - midpoint;
    r2 = members{i}.p2 - midpoint;
    
    % Cross product in 3D
    torque1 = cross(r1, force1);
    torque2 = cross(r2, force2);
    
    netTorques{i} = torque1 + torque2;

      elseif i == 4
    % Calculate forces
    force1 = attractionForce(members{i}.p1, members{1}.p2, 10);
    force2 = attractionForce(members{i}.p2, members{5}.p2, 10);
    
    % Update net forces for members
    netForces{i} = force1 + force2;

    % Calculate torques (assuming midpoint of member as pivot)
    midpoint = (members{i}.p1 + members{i}.p2) / 2;
    r1 = members{i}.p1 - midpoint;
    r2 = members{i}.p2 - midpoint;
    
    % Cross product in 3D
    torque1 = cross(r1, force1);
    torque2 = cross(r2, force2);
    
    netTorques{i} = torque1 + torque2;

  elseif i == 5

    % Calculate forces
    force1 = attractionForce(members{i}.p1, members{2}.p2, 10);
    force2 = attractionForce(members{i}.p2, members{4}.p2, 10);
    
    % Update net forces for members
    netForces{i} = force1 + force2;

    % Calculate torques (assuming midpoint of member as pivot)
    midpoint = (members{i}.p1 + members{i}.p2) / 2;
    r1 = members{i}.p1 - midpoint;
    r2 = members{i}.p2 - midpoint;
    
    % Cross product in 3D
    torque1 = cross(r1, force1);
    torque2 = cross(r2, force2);
    
    netTorques{i} = torque1 + torque2;


  elseif i == 6

    % Calculate forces
    force1 = attractionForce(members{i}.p1, members{1}.p1, 10);
    force2 = attractionForce(members{i}.p2, members{4}.p2, 10);
    
    % Update net forces for members
    netForces{i} = force1 + force2;

    % Calculate torques (assuming midpoint of member as pivot)
    midpoint = (members{i}.p1 + members{i}.p2) / 2;
    r1 = members{i}.p1 - midpoint;
    r2 = members{i}.p2 - midpoint;
    
    % Cross product in 3D
    torque1 = cross(r1, force1);
    torque2 = cross(r2, force2);
    
    netTorques{i} = torque1 + torque2;
  end
end
%   
  % Update member positions based on net forces and torques (simplified)
  for i = 2:numMembers
    % Update positions
    members{i}.p1 = members{i}.p1 + netForces{i} * 0.01;
    members{i}.p2 = members{i}.p2 + netForces{i} * 0.01;

    % Apply rotational motion based on torque (simplified as small angular displacement)
    midpoint = (members{i}.p1 + members{i}.p2) / 2;
    angle = norm(netTorques{i}) * 0.01 / members{i}.length; % Small angle approximation

    % Rotate p1 and p2 around the midpoint
    rotationAxis = netTorques{i} / norm(netTorques{i}); % Normalized rotation axis
    members{i}.p1 = rotateAround(members{i}.p1, midpoint, angle, rotationAxis);
    members{i}.p2 = rotateAround(members{i}.p2, midpoint, angle, rotationAxis);
  end
end
  % Plot member positions (optional)
  hold on;
  for i = 1:numMembers
    plot3([members{i}.p1(1), members{i}.p2(1)], [members{i}.p1(2), members{i}.p2(2)], [members{i}.p1(3), members{i}.p2(3)], 'b-');
  end
  hold off;
  drawnow;


% Function to generate random length between 1 and 2
function length = randomLength()
  length = 1 + rand();
end

% Function to calculate distance between two points
function distance = pointDistance(p1, p2)
  distance = norm(p1 - p2);
end

% Function to calculate force between two points (linearly proportional to distance)
function force = attractionForce(p1, p2, k)
  distance = pointDistance(p1, p2);
  force = k * (distance) * (p2 - p1) / distance; % Force proportional to distance from equilibrium (1 unit)
end

% Function to rotate a point around another point in 3D
function pRotated = rotateAround(p, center, angle, axis)
  % Rodrigues' rotation formula
  k = axis / norm(axis); % Ensure it's a unit vector
  pRotated = p * cos(angle) + cross(k, p) * sin(angle) + k * dot(k, p) * (1 - cos(angle));
end