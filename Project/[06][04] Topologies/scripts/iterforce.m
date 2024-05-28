% Define number of members and iterations
numMembers = 3;
iterations = 1000;

% Initialize member data structures
members = cell(1, numMembers);
for i = 1:numMembers
  members{i}.length = randomLength();
  members{i}.p1 = rand(2, 1); % Random initial position of point 1
  members{i}.p2 = members{i}.p1 + members{i}.length * [cos(rand()*2*pi); sin(rand()*2*pi)]; % Random initial position of point 2
end
%%
close
for i = 1:numMembers
plot([members{i}.p1(1), members{i}.p2(1)], [members{i}.p1(2), members{i}.p2(2)], 'b-');
hold on;
end
hold off;
drawnow;

%%
% Simulation loop
for iter = 1:iterations
  % Initialize forces and torques
  netForces = cell(1, numMembers);
  netTorques = zeros(1, numMembers);
  
  for i = 1:numMembers
    netForces{i} = [0; 0];
  end

  % Apply forces between member endpoints
  for i = 1:numMembers
    j = mod(i, numMembers) + 1; % Next member in circular chain
    z = mod(i-2, numMembers) + 1; % Previous member in circular chain
    
    % Calculate forces
    force1 = attractionForce(members{i}.p1, members{z}.p2, 10);
    force2 = attractionForce(members{i}.p2, members{j}.p1, 10);
    
    % Update net forces for members
    netForces{i} =  force1+force2;

    % Calculate torques (assuming midpoint of member as pivot)
    midpoint = (members{i}.p1 + members{i}.p2) / 2;
    r1 = members{i}.p1 - midpoint;
    r2 = members{i}.p2 - midpoint;
    
    % Cross product in 2D (only z-component is needed)
    torque1 = r1(1) * force1(2) - r1(2) * force1(1);
    torque2 = r2(1) * force2(2) - r2(2) * force2(1);
    
    netTorques(i) =  torque1 + torque2;
  end
  
  % Update member positions based on net forces and torques (simplified)
  for i = 1:numMembers-1
    % Update positions
    members{i}.p1 = members{i}.p1 + netForces{i} * 0.01;
    members{i}.p2 = members{i}.p2 + netForces{i} * 0.01;


    % Apply rotational motion based on torque (simplified as small angular displacement)
    midpoint = (members{i}.p1 + members{i}.p2) / 2;
    angle = netTorques(i) * 0.01 / members{i}.length; % Small angle approximation

    % Rotate p1 and p2 around the midpoint
    members{i}.p1 = rotateAround(members{i}.p1, midpoint, angle);
    members{i}.p2 = rotateAround(members{i}.p2, midpoint, angle);


  end
    % Plot member positions (optional)
  % Plot points and lines for each member
  for i = 1:numMembers
    plot([members{i}.p1(1), members{i}.p2(1)], [members{i}.p1(2), members{i}.p2(2)], 'b-');
    hold on;
  end
  hold off;
  drawnow;

end



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

% Function to rotate a point around another point
function pRotated = rotateAround(p, center, angle)
  R = [cos(angle) -sin(angle); sin(angle) cos(angle)];
  pRotated = center + R * (p - center);
end