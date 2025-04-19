clc;
clear;
close all;
%%
alpha = 1.95;
beta = 1;
gamma = 0.6;
delta = 0.1;

T = 500;

ustar = 0.2505;
vstar = 0.1564;

d = 10;
%%  
sigma1 = 0.01;
sigma2 = 0.01;

base_seed = 16;
dx_ref = 0.1;
dt_ref = 0.002;
[u_ref, ~] = spdesolver(dx_ref, dt_ref, T, alpha, beta, gamma, delta, d, ustar, vstar, sigma1, sigma2, base_seed);
%%
dt_list = [0.004, 0.005, 0.008, 0.01];
dx_list = [0.2, 0.4, 0.8, 1];
%%
errors_space = zeros(size(dx_list));

for idx = 1:length(dx_list)
    dx = dx_list(idx);
    dt = dt_ref;

    [u_test, ~] = spdesolver(dx, dt, T, alpha, beta, gamma, delta, d, ustar, vstar, sigma1, sigma2, base_seed);

    factor = round(dx / dx_ref);

    u_ref_down = average_downsample(u_ref, factor);

    [nx, ny] = size(u_test);
    u_ref_down = u_ref_down(1:nx, 1:ny);

    errors_space(idx) = norm(u_test - u_ref_down, 'fro') / (nx^2);

    fprintf('dx = %.3f, dt = %.4f, error = %.6f\n', dx, dt, errors_space(idx));
end
%%
errors_time = zeros(size(dt_list));

for idx = 1:length(dt_list)
    dx = dx_ref;
    dt = dt_list(idx);

    [u_test_t, ~] = spdesolver(dx, dt, T, alpha, beta, gamma, delta, d, ustar, vstar, sigma1, sigma2, base_seed);

    errors_time(idx) = norm(u_test_t - u_ref, 'fro');

    fprintf('dx = %.3f, dt = %.4f, error = %.6f\n', dx, dt, errors_time(idx));
end

%%
log_dx = log(dx_list);
log_error = log(errors_space);
p = polyfit(log_dx, log_error, 1); 
slope = p(1);

figure;
loglog(dx_list, errors_space, '-o', 'LineWidth', 1.5);
xlabel('Spatial step size (\Deltax)');
ylabel('Error');
title('Spatial Convergence Rate');
grid on;

hold on;
fit_line = exp(polyval(p, log_dx));
loglog(dx_list, fit_line, '--r', 'LineWidth', 1.2);
legend('Numerical Error', sprintf('Fitted Slope = %.2f', slope), 'Location','northwest');

%%
figure;
loglog(dt_list, errors_time, '-o', 'LineWidth', 1.5);
xlabel('Temporal step size (\Deltat)');
ylabel('Error');
title('Temporal Convergence Rate');
grid on;
log_dt = log(dt_list);
log_error = log(errors_time);
p_t = polyfit(log_dt, log_error, 1);
slope_t = p_t(1);

hold on;
fit_line_t = exp(polyval(p_t, log_dt));
loglog(dt_list, fit_line_t, '--r', 'LineWidth', 1.2);
legend('Numerical Error', sprintf('Fitted Slope = %.2f', slope_t), 'Location','northwest');

%%
function u_coarse = average_downsample(u_fine, factor)
    [nx, ny] = size(u_fine);
    nx_coarse = floor(nx / factor);
    ny_coarse = floor(ny / factor);
    u_coarse = zeros(nx_coarse, ny_coarse);

    for i = 1:nx_coarse
        for j = 1:ny_coarse
            block = u_fine((i-1)*factor + 1:i*factor, (j-1)*factor + 1:j*factor);
            u_coarse(i,j) = mean(block(:));
        end
    end
end
