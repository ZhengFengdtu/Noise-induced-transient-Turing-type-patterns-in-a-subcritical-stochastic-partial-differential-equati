clc
clear
close all
%%
% parameters

alpha = 1.95;
beta = 1;
gamma = 0.6;
delta = 0.1;
d1 = 1.0;
d2 = 10; 
%%
% f(u, v) & g(u, v)
f = @(u, v) u .* (1 - u) - alpha * u .* v ./ (u + v);
g = @(u, v) beta * u .* v ./ (u + v) - gamma * v - delta * v.^2;

% equlibirium (u*, v*)
[steady_state, ~, ~]= fsolve(@(x) equations(x, alpha, beta, gamma, delta), [0.5, 0.5]);%[0.2287, 0.1436]
u_star = steady_state(1);
v_star = steady_state(2);

fprintf('equlibirium: u* = %.4f, v* = %.4f\n', u_star, v_star);

% Jacobian matrix
J = jacobian(u_star, v_star, alpha, beta, gamma, delta);

% wave number range
k_values = linspace(0, 0.6, 100);

% eigenvalue of Turing bifurcation
eigenvalues = zeros(length(k_values), 2);
for i = 1:length(k_values)
    k = k_values(i);
    A = J - [d1 * k^2, 0; 0, d2 * k^2];
    eigenvalues(i, :) = real(eig(A))';  % real part
end

% the Eigenvalues ​​versus Wavenumber
figure;
plot(k_values, eigenvalues(:, 1), 'b', 'DisplayName', 'eigenvalue 1');
hold on;
plot(k_values, eigenvalues(:, 2), 'r', 'DisplayName', 'eigenvalue 2');
yline(0, '--k');
xlabel('wave number k');
ylabel('Real partial of Eigenvalue');
title('Eigenvalue Analysis of Turing Bifurcation');
legend('Eigenvalue 1', 'Eigenvalue 2', '\lambda = 0', 'Location', 'southwest');
grid on;
%%
% compute the critical d
a11 = J(1,1);
a12 = J(1,2);
a21 = J(2,1);
a22 = J(2,2);
syms d
eqn = d*a11+a22-2*sqrt(d)*sqrt(a11*a22-a12*a21);
solve(eqn==0,d)
%%
% Define a function to solve the uniform steady-state equation
function F = equations(vars, alpha, beta, gamma, delta)
    u = vars(1);
    v = vars(2);
    F(1) = u * (1 - u) - alpha * u * v / (u + v);
    F(2) = beta * u * v / (u + v) - gamma * v - delta * v^2;
end

% Define Jacobian matrix calculation
function J = jacobian(u, v, alpha, beta, gamma, delta)
    df_du = (alpha*u*v)/(u + v)^2 - (alpha*v)/(u + v) - 2*u + 1;
    df_dv = -alpha * u^2 / (u + v)^2;
    dg_du = beta * v^2 / (u + v)^2;
    dg_dv = beta * u / (u + v) - gamma - 2 * delta * v - (beta * u * v)/(u + v)^2; 
    J = [df_du, df_dv; dg_du, dg_dv];
end
