clc;
clear;
close all
%%
alpha = 1.95;
beta = 1;
gamma = 0.6;
delta = 0.1;
d1 = 1;
d2 = 7.2266;
% (35432750955137195816314505597201^(1/2)/3799727552863810 + 4262057868131079/3799727552863810)^2
%%
Lx = 100;
Ly = 100;
dx = 1;
dy = 1;
T = 1000;
dt = 0.01;
Nt = round(T/dt);
Nx = Lx/dx;
Ny = Ly/dy;
%%
sigma1 = 0.01;
sigma2 = 0.01;
%%
% initial condition
u = zeros(Nx, Ny, Nt);
v = zeros(Nx, Ny, Nt);    
meanval_u = 0.2505;
meanval_v = 0.1564;
var = 0.01;
seed = 42;
rng(seed);
u(:,:,1) = meanval_u + var * randn(Nx, Ny);
v(:,:,1) = meanval_v + var * randn(Nx, Ny);
initial_val_u = u(:,:,1);
initial_val_v = v(:,:,1);
%%
% def f(u, v) and g(u, v) as the local advection term
f = @(u,v) u.*(1-u) - alpha.*u.*v./(u+v);
g = @(u,v) beta.*u.*v./(u+v) - gamma.*v - delta.*v.^2;
dtD = dt/10;

rxu = d1 * dt / dx^2;
ryu = d1 * dt / dy^2;
rxv = d2 * dt / dx^2;
ryv = d2 * dt / dy^2;
seed = 16;
for n = 1: Nt-1
    % reaction (Huen method)
    u_pred = u(:, :, n) + (dt / 2) * f(u(:, :, n), v(:, :, n));
    v_pred = v(:, :, n) + (dt / 2) * g(u(:, :, n), v(:, :, n));
    u_half = u(:, :, n) + (dt / 4) * (f(u(:, :, n), v(:, :, n)) + f(u_pred, v_pred));
    v_half = v(:, :, n) + (dt / 4) * (g(u(:, :, n), v(:, :, n)) + g(u_pred, v_pred));
    
    % diffusion
    % Initialize u_new and v_new with previous time step values
    u_old = u_half; 
    v_old = v_half; 
    dW_u = sqrt(dx*dy*dt) * randn(Ny, Nx);  % noise for u
    dW_v = sqrt(dx*dy*dt) * randn(Ny, Nx);  % noise for v 
    for i = 2:Nx-1
        for j = 2:Ny-1
        % Perform the explicit half-step based on the old time for u
        u_half(i,j) = u_old(i,j) ...
            + 0.5 * rxu * ( u_old(i+1,j) - 2*u_old(i,j) + u_old(i-1,j) ) ...
            + 0.5 * ryu * ( u_old(i,j+1) - 2*u_old(i,j) + u_old(i,j-1) );

        noise_u = sigma1 * u_old(i,j) * dW_u(i,j);
        u_half(i,j) = u_half(i,j) + noise_u;

        v_half(i,j) = v_old(i,j) ...
            + 0.5 * rxv * ( v_old(i+1,j) - 2*v_old(i,j) + v_old(i-1,j) ) ...
            + 0.5 * ryv * ( v_old(i,j+1) - 2*v_old(i,j) + v_old(i,j-1) );

        noise_v = sigma2 * v_old(i,j) * dW_v(i,j);
        v_half(i,j) = v_half(i,j) + noise_v;

        end
    end
    % diffusion implicit using pointwise iteration
    u_new = u_half;  
    v_new = v_half;
    nIter = dt/dtD;
    for iter = 1:nIter
        for i = 2:Nx-1
            for j = 2:Ny-1
            % For u: perform Crank–Nicolson pointwise iteration update
            % Numerator: "half of the old time" + "half of the new time neighbors"
            % e.g. u_half(i,j) + 0.5 * rxu * (u_new(i+1,j) + u_new(i-1,j)) + ...
            % Denominator: 1 + (rxu + ryu)
            u_new(i,j) = ( u_half(i,j) ...
                         + 0.5 * rxu * ( u_new(i+1,j) + u_new(i-1,j) ) ...
                         + 0.5 * ryu * ( u_new(i,j+1) + u_new(i,j-1) ) ) ...
                         / (1 + rxu + ryu);

            % For v:
            v_new(i,j) = ( v_half(i,j) ...
                         + 0.5 * rxv * ( v_new(i+1,j) + v_new(i-1,j) ) ...
                         + 0.5 * ryv * ( v_new(i,j+1) + v_new(i,j-1) ) ) ...
                         / (1 + rxv + ryv);
            end
        end
    end

    % reaction
    u_new_pred = u_new + (dt / 2) * f(u_new, v_new);
    v_new_pred = v_new + (dt / 2) * g(u_new, v_new);
    u_new = u_new + (dt / 4) * (f(u_new, v_new) + f(u_new_pred, v_new_pred));
    v_new = v_new + (dt / 4) * (g(u_new, v_new) + g(u_new_pred, v_new_pred));

    u(:, :, n + 1) = u_new;
    v(:, :, n + 1) = v_new;

    % boundary 
    u(1, :, n + 1) = u(2, :, n + 1);
    u(end, :, n + 1) = u(end - 1, :, n + 1);
    u(:, 1, n + 1) = u(:, 2, n + 1);
    u(:, end, n + 1) = u(:, end - 1, n + 1);

    v(1, :, n + 1) = v(2, :, n + 1);
    v(end, :, n + 1) = v(end - 1, :, n + 1);
    v(:, 1, n + 1) = v(:, 2, n + 1);
    v(:, end, n + 1) = v(:, end - 1, n + 1);

    % 4 point at corner
    u(1, 1, n+1) = u(2, 2, n+1); 
    u(1, Ny, n+1) = u(2, Ny-1, n+1);
    u(Nx, 1, n+1) = u(Nx-1, 2, n+1);
    u(Nx, Ny, n+1) = u(Nx-1, Ny-1, n+1);
    v(1, 1, n+1) = v(2, 2, n+1); 
    v(1, Ny, n+1) = v(2, Ny-1, n+1);
    v(Nx, 1, n+1) = v(Nx-1, 2, n+1);
    v(Nx, Ny, n+1) = v(Nx-1, Ny-1, n+1);

    disp(n)
end
%%
figure
imagesc(u(:,:,end))
title('Final State of u')
xlabel('x direction')
ylabel('y direction')
colorbar
%%
dkx = 2*pi / Lx;
dky = 2*pi / Ly;
kx_shifted = (-Nx/2 : Nx/2-1)*dkx;
ky_shifted = (-Ny/2 : Ny/2-1)*dky;

uend = u(:,:,end);
umines = uend - meanval_u;
uf = fft2(umines);
U_shifted = fftshift(uf);
ua = abs(U_shifted);

figure
idx_x = find(kx_shifted >= -1 & kx_shifted <= 1);
idx_y = find(ky_shifted >= -1 & ky_shifted <= 1);
F_sub = ua(idx_y, idx_x);  
x_sub = kx_shifted(idx_x);
y_sub = ky_shifted(idx_y);
imagesc(x_sub, y_sub, F_sub)
axis xy; 
axis tight; 
colormap jet; 
xlabel('k_x'); ylabel('k_y');
title('Final State of u after FFT')
xlabel('x direction')
ylabel('y direction')
colorbar
%%
% ===== Radial Spectrum Plot: Averaging ua over frequency radius =====

% Step 1: Create 2D meshgrid of frequency space
[KX, KY] = meshgrid(kx_shifted, ky_shifted);
K_abs = sqrt(KX.^2 + KY.^2);  % radial frequency: sqrt(kx^2 + ky^2)

% Step 2: Define radial bins
nbins = 100;
k_min = 0;
k_max = max(K_abs(:));
r_edges = linspace(k_min, k_max, nbins+1);
r_centers = 0.5 * (r_edges(1:end-1) + r_edges(2:end));  % bin centers

% Step 3: Initialize output
radial_profile = zeros(nbins, 1);

% Step 4: Compute mean amplitude within each radial bin
for i = 1:nbins
    mask = (K_abs >= r_edges(i)) & (K_abs < r_edges(i+1));
    radial_profile(i) = mean(ua(mask).^2, 'all');  % Power spectrum
end

% Step 5: Plot radial spectrum
figure;
% plot(r_centers, radial_profile);
% loglog(r_centers, radial_profile);
semilogy(r_centers, radial_profile);
xlabel('$|\mathbf{k}| = \sqrt{k_x^2 + k_y^2}$', 'Interpreter', 'latex');
ylabel('$\mathrm{Power}\ |\hat{u}(\mathbf{k})|^2$', 'Interpreter', 'latex');
title('Power Spectrum of Final State');
grid on;

