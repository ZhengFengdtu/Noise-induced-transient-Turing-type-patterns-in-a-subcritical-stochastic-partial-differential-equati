function [u,v]=spdesolver(spacestep, dt, T, alpha, beta, gamma, delta, d, ustar, vstar, sigma1, sigma2, base_seed)
Lx = 100;
Ly = 100;
dx = spacestep;
dy = spacestep;
Nt = round(T/dt);
Nx = round(Lx/dx);
Ny = round(Ly/dy);

% initial condition
u = zeros(Nx, Ny);
v = zeros(Nx, Ny);    
meanval_u = ustar;
meanval_v = vstar;
var = 0.01;
seed = 42;
rng(seed);
u(:,:) = meanval_u + var * randn(Nx, Ny);
v(:,:) = meanval_v + var * randn(Nx, Ny);

f = @(u,v) u.*(1-u) - alpha.*u.*v./(u+v);
g = @(u,v) beta.*u.*v./(u+v) - gamma.*v - delta.*v.^2;
dtD = dt/10;

rxu = dt / dx^2;
ryu = dt / dy^2;
rxv = d * dt / dx^2;
ryv = d * dt / dy^2;

rng(base_seed);

    for n = 1: Nt-1
        % reaction (Huen method)
        u_pred = u(:, : ) + (dt / 2) * f(u(:, : ), v(:, : ));
        v_pred = v(:, : ) + (dt / 2) * g(u(:, : ), v(:, : ));
        u_half = u(:, : ) + (dt / 4) * (f(u(:, : ), v(:, : )) + f(u_pred, v_pred));
        v_half = v(:, : ) + (dt / 4) * (g(u(:, : ), v(:, : )) + g(u_pred, v_pred));

        % diffusion
        % Initialize u_new and v_new with previous time step values
        u_old = u_half; 
        v_old = v_half;
        dW_u = sqrt(dx*dy*dt) * randn(Ny, Nx);  % 对 u 的噪声
        dW_v = sqrt(dx*dy*dt) * randn(Ny, Nx);  % 对 v 的噪声 
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

        u(:, : ) = u_new;
        v(:, : ) = v_new;

        % boundary 
        u(1, : ) = u(2, : );
        u(end, : ) = u(end - 1, : );
        u(:, 1 ) = u(:, 2 );
        u(:, end ) = u(:, end - 1 );

        v(1, : ) = v(2, : );
        v(end, : ) = v(end - 1, : );
        v(:, 1 ) = v(:, 2 );
        v(:, end ) = v(:, end - 1 );

        % 4 point at corner
        u(1, 1 ) = u(2, 2 ); 
        u(1, Ny ) = u(2, Ny-1 );
        u(Nx, 1 ) = u(Nx-1, 2 );
        u(Nx, Ny ) = u(Nx-1, Ny-1 );
        v(1, 1 ) = v(2, 2 ); 
        v(1, Ny ) = v(2, Ny-1 );
        v(Nx, 1 ) = v(Nx-1, 2 );
        v(Nx, Ny ) = v(Nx-1, Ny-1 );


        disp(n)
    end

end
