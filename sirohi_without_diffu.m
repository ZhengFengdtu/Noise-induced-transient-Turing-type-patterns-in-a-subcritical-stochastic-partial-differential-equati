clc;
clear;
close all
%%
% parameters
p.alpha = 1.95;
p.beta = 1;
p.gamma = 0.6;
p.delta = 0.1;
%%
tspan = [0 1000];
x0 = [0.23  ;  0.14];

options45 = odeset('RelTol',1.0e-6,'AbsTol',1.0e-6);
[T,X]=ode15s(@Sirohi,tspan,x0,options45,p);

%%
plot(T,X(:,1))
hold on
plot(T,X(:,2))
legend('prey','predator')
ylabel('Populations');
xlabel('Time T');

figure
plot(X(:,1),X(:,2))
hold on
ylabel('predator');
xlabel('prey');
skip = 15;  % It can be adjusted according to the length of the track so that the arrows are not too dense or too sparse
for k = 1:skip:(length(X(:,1)) - 1)
    % dx, dy represent the displacement vectors of adjacent points
    dx = X(k+1,1) - X(k,1);
    dy = X(k+1,2) - X(k,2);
    % The last parameter of quiver is 0, which means no additional scaling of the arrow length.
    quiver(X(k,1), X(k,2), dx, dy, 0, ...
        'MaxHeadSize', 2, 'Color', 'r', 'LineWidth', 1.2);
end
