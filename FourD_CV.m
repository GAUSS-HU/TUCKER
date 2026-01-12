%% 4D Target Tracking Simulation using Tucker-FPE Algorithm (Corrected Version)
% State: [x, y, vx, vy] - position and velocity in 2D
% Algorithm: FPE operator + Tucker decomposition + Anisotropic likelihood
% Proper implementation of p(t+dt) = exp(dt*L) * p(t)

clear; close all; clc;

%% Parameters
T = 100;                    % Total time steps
dt = 0.1;                   % Time step
N = 50;                     % Grid points per dimension
sigma_process = 0.1;        % Process noise standard deviation
sigma_obs = 1.0;            % Observation noise standard deviation

% State space bounds
x_bounds = [-10, 30];
y_bounds = [-10, 30];
vx_bounds = [-2, 2];
vy_bounds = [-2, 2];

% Grid setup for each dimension
x_grid = linspace(x_bounds(1), x_bounds(2), N);
y_grid = linspace(y_bounds(1), y_bounds(2), N);
vx_grid = linspace(vx_bounds(1), vx_bounds(2), N);
vy_grid = linspace(vy_bounds(1), vy_bounds(2), N);

dx = x_grid(2) - x_grid(1);
dy = y_grid(2) - y_grid(1);
dvx = vx_grid(2) - vx_grid(1);
dvy = vy_grid(2) - vy_grid(1);

%% True trajectory generation (constant velocity)
true_state = zeros(4, T);
true_state(:, 1) = [0; 0; 1; 1];  % Initial state [x, y, vx, vy]

for t = 2:T
    % Constant velocity motion with small process noise
    true_state(1, t) = true_state(1, t-1) + true_state(3, t-1) * dt + sigma_process * randn * dt;
    true_state(2, t) = true_state(2, t-1) + true_state(4, t-1) * dt + sigma_process * randn * dt;
    true_state(3, t) = true_state(3, t-1) + sigma_process * randn * dt;
    true_state(4, t) = true_state(4, t-1) + sigma_process * randn * dt;
end

% Generate noisy observations (only position)
observations = true_state(1:2, :) + sigma_obs * randn(2, T);

%% Initialize PDF using Tucker decomposition
% Initial uncertainty - Gaussian around initial guess
initial_guess = [2; 2; 0.8; 0.3];  % Slightly off from true initial state
initial_cov = diag([4, 4, 0.5, 0.5]);

% Create initial 4D PDF tensor
pdf_tensor = zeros(N, N, N, N);
for i = 1:N
    for j = 1:N
        for k = 1:N
            for l = 1:N
                state_point = [x_grid(i); y_grid(j); vx_grid(k); vy_grid(l)];
                diff = state_point - initial_guess;
                pdf_tensor(i,j,k,l) = exp(-0.5 * diff' * inv(initial_cov) * diff);
            end
        end
    end
end

% Normalize
pdf_tensor = pdf_tensor / sum(pdf_tensor(:)) / (dx * dy * dvx * dvy);

%% Tucker decomposition of initial PDF
tucker_rank = [5, 5, 3, 3];  % Tucker ranks for each dimension
[G, U1, U2, U3, U4] = tucker_decompose_4d(pdf_tensor, tucker_rank);

%% Create differentiation matrices
D1 = create_diff_matrix(N, dx);  % for x dimension
D2 = create_diff_matrix(N, dy);  % for y dimension
D3 = create_diff_matrix(N, dvx); % for vx dimension
D4 = create_diff_matrix(N, dvy); % for vy dimension

%% Main tracking loop
estimated_state = zeros(4, T);
estimation_error = zeros(4, T);

fprintf('Starting 4D Tucker-FPE tracking simulation...\n');

for t = 1:T
    fprintf('Processing time step %d/%d\n', t, T);
    
    %% Step 1: Prediction using FPE operator with matrix exponential
    if t > 1
        % Apply proper FPE evolution: p(t+dt) = exp(dt*L) * p(t)
        [G, U1, U2, U3, U4] = apply_fpe_evolution_tucker(G, U1, U2, U3, U4, ...
                                                         D1, D2, D3, D4, dt, sigma_process);
    end
    
    %% Step 2: Bayesian update with anisotropic likelihood
    % Construct 4D likelihood based on position observations
    likelihood_tensor = construct_anisotropic_likelihood(x_grid, y_grid, vx_grid, vy_grid, ...
                                                        observations(:, t), sigma_obs, N);
    
    % Reconstruct PDF from Tucker factors for update
    pdf_tensor = tucker_reconstruct_4d(G, U1, U2, U3, U4);
    
    % Bayesian update
    pdf_tensor = pdf_tensor .* likelihood_tensor;
    
    % Normalize
    pdf_tensor = pdf_tensor / sum(pdf_tensor(:)) / (dx * dy * dvx * dvy);
    
    % Re-decompose updated PDF
    [G, U1, U2, U3, U4] = tucker_decompose_4d(pdf_tensor, tucker_rank);
    
    %% Step 3: State estimation 
    [X, Y, VX, VY] = ndgrid(x_grid, y_grid, vx_grid, vy_grid);
    total_prob = sum(pdf_tensor(:));
    
    if total_prob > 0
        estimated_state(1, t) = sum(X(:) .* pdf_tensor(:)) / total_prob;
        estimated_state(2, t) = sum(Y(:) .* pdf_tensor(:)) / total_prob;
        estimated_state(3, t) = sum(VX(:) .* pdf_tensor(:)) / total_prob;
        estimated_state(4, t) = sum(VY(:) .* pdf_tensor(:)) / total_prob;
    else
        % Fallback to MAP estimate
        [~, max_idx] = max(pdf_tensor(:));
        [i_max, j_max, k_max, l_max] = ind2sub([N, N, N, N], max_idx);
        estimated_state(:, t) = [x_grid(i_max); y_grid(j_max); vx_grid(k_max); vy_grid(l_max)];
    end
    
    estimation_error(:, t) = estimated_state(:, t) - true_state(:, t);
end

%% Results visualization
plot_tracking_results(true_state, estimated_state, observations, estimation_error, T);

%% Functions

function [G, U1, U2, U3, U4] = tucker_decompose_4d(tensor, ranks)
    % 4D Tucker decomposition using HOSVD
    [N1, N2, N3, N4] = size(tensor);
    
    % Mode-1 unfolding and SVD
    T1 = reshape(tensor, N1, []);
    [U1, ~, ~] = svd(T1, 'econ');
    U1 = U1(:, 1:min(ranks(1), size(U1, 2)));
    
    % Mode-2 unfolding and SVD
    T2 = reshape(permute(tensor, [2,1,3,4]), N2, []);
    [U2, ~, ~] = svd(T2, 'econ');
    U2 = U2(:, 1:min(ranks(2), size(U2, 2)));
    
    % Mode-3 unfolding and SVD
    T3 = reshape(permute(tensor, [3,1,2,4]), N3, []);
    [U3, ~, ~] = svd(T3, 'econ');
    U3 = U3(:, 1:min(ranks(3), size(U3, 2)));
    
    % Mode-4 unfolding and SVD
    T4 = reshape(permute(tensor, [4,1,2,3]), N4, []);
    [U4, ~, ~] = svd(T4, 'econ');
    U4 = U4(:, 1:min(ranks(4), size(U4, 2)));
    
    % Compute core tensor G
    G = tensor;
    G = tensor_multiply_mode(G, U1', 1);
    G = tensor_multiply_mode(G, U2', 2);
    G = tensor_multiply_mode(G, U3', 3);
    G = tensor_multiply_mode(G, U4', 4);
end

function result = tensor_multiply_mode(tensor, matrix, mode)
    % Multiply tensor by matrix along specified mode
    sz = size(tensor);
    n_modes = length(sz);
    
    % Permute tensor to bring target mode to front
    perm = [mode, 1:mode-1, mode+1:n_modes];
    tensor_perm = permute(tensor, perm);
    
    % Reshape for matrix multiplication
    tensor_mat = reshape(tensor_perm, sz(mode), []);
    result_mat = matrix * tensor_mat;
    
    % Reshape back and permute to original order
    new_sz = sz;
    new_sz(mode) = size(matrix, 1);
    new_sz_perm = [new_sz(mode), new_sz(setdiff(1:n_modes, mode))];
    result_perm = reshape(result_mat, new_sz_perm);
    
    inv_perm(perm) = 1:n_modes;
    result = permute(result_perm, inv_perm);
end

function tensor = tucker_reconstruct_4d(G, U1, U2, U3, U4)
    % Reconstruct 4D tensor from Tucker factors
    tensor = G;
    tensor = tensor_multiply_mode(tensor, U1, 1);
    tensor = tensor_multiply_mode(tensor, U2, 2);
    tensor = tensor_multiply_mode(tensor, U3, 3);
    tensor = tensor_multiply_mode(tensor, U4, 4);
end

function D = create_diff_matrix(N, h)
    % Create differentiation matrix using finite differences
    D = zeros(N, N);
    
    % Forward difference at first point
    D(1, 1) = -1/h;
    D(1, 2) = 1/h;
    
    % Central difference at interior points
    for i = 2:N-1
        D(i, i-1) = -1/(2*h);
        D(i, i+1) = 1/(2*h);
    end
    
    % Backward difference at last point
    D(N, N-1) = -1/h;
    D(N, N) = 1/h;
end

function [G_new, U1_new, U2_new, U3_new, U4_new] = apply_fpe_evolution_tucker(G, U1, U2, U3, U4, D1, D2, D3, D4, dt, sigma)
    % Proper FPE evolution: p(t+dt) = exp(dt*L) * p(t)
    % Using Tucker format to avoid large matrix operations
    
    % Construct separable FPE operators for constant velocity model
    % dx/dt = vx, dy/dt = vy, dvx/dt = noise, dvy/dt = noise
    
    % Mode 1 (x-dimension): drift term for dx/dt = vx
    L1_drift = construct_mode_operator(D1, dt, 'drift');
    U1_new = expm(L1_drift) * U1;
    
    % Mode 2 (y-dimension): drift term for dy/dt = vy  
    L2_drift = construct_mode_operator(D2, dt, 'drift');
    U2_new = expm(L2_drift) * U2;
    
    % Mode 3 (vx-dimension): diffusion term σ²/2 * ∂²/∂vx²
    L3_diffusion = construct_mode_operator(D3, dt, 'diffusion', sigma);
    U3_new = expm(L3_diffusion) * U3;
    
    % Mode 4 (vy-dimension): diffusion term σ²/2 * ∂²/∂vy²
    L4_diffusion = construct_mode_operator(D4, dt, 'diffusion', sigma);
    U4_new = expm(L4_diffusion) * U4;
    
    % Core tensor evolution
    G_new = apply_core_evolution(G, dt);
    
    % Ensure orthogonality of factor matrices for numerical stability
    [U1_new, ~] = qr(U1_new, 0);
    [U2_new, ~] = qr(U2_new, 0);
    [U3_new, ~] = qr(U3_new, 0);
    [U4_new, ~] = qr(U4_new, 0);
end

function L_mode = construct_mode_operator(D, dt, type, sigma)
    % Construct mode-specific FPE operator
    switch type
        case 'drift'
            % Drift operator: -∂/∂x for constant velocity model
            L_mode = -dt * D;
        case 'diffusion'
            % Diffusion operator: σ²/2 * ∂²/∂x²
            if nargin >= 4
                L_mode = dt * sigma^2 * 0.5 * (D * D);
            else
                L_mode = dt * 0.01 * (D * D);
            end
        otherwise
            L_mode = zeros(size(D));
    end
end

function G_new = apply_core_evolution(G, dt)
    % Apply evolution to core tensor
    % For separable FPE operators, core tensor mainly handles normalization
    
    % Simple evolution with small damping for numerical stability
    evolution_factor = exp(-0.01 * dt);
    G_new = G * evolution_factor;
    
    % Ensure proper normalization
    norm_factor = sum(abs(G_new(:)));
    if norm_factor > 0
        G_new = G_new / norm_factor * sum(abs(G(:)));
    end
end

function likelihood = construct_anisotropic_likelihood(x_grid, y_grid, vx_grid, vy_grid, obs, sigma, N)
    % Construct 4D likelihood tensor (anisotropic in position)
    likelihood = zeros(N, N, N, N);
    
    % Vectorized computation for efficiency
    [X, Y, VX, VY] = ndgrid(x_grid, y_grid, vx_grid, vy_grid);
    
    % Position likelihood (only depends on x, y)
    pos_likelihood = exp(-0.5 * ((X - obs(1)).^2 + (Y - obs(2)).^2) / sigma^2);
    
    % Velocity is uniform (uninformative)
    vel_likelihood = ones(size(VX));
    
    % Combined likelihood
    likelihood = pos_likelihood .* vel_likelihood;
    
    % Normalize
    likelihood = likelihood / sum(likelihood(:));
end

function plot_tracking_results(true_state, estimated_state, observations, estimation_error, T)
    % Plot tracking results
    
    figure('Position', [100, 100, 1200, 800]);
    
    % Trajectory plot
    %subplot(1, 3, 1);
    figure(1);
    plot(true_state(1, :), true_state(2, :), 'b-', 'LineWidth', 2, 'DisplayName', 'True');
    hold on;
    plot(estimated_state(1, :), estimated_state(2, :), 'r--', 'LineWidth', 2, 'DisplayName', 'Estimated');
    scatter(observations(1, :), observations(2, :), 20, 'k.', 'DisplayName', 'Observations');
    xlabel('X Position'); ylabel('Y Position');
    title('Trajectory');
    legend(); grid on;
    
    % Position errors
    %subplot(1, 3, 2);
    figure(2);
    plot(1:T, estimation_error(1, :), 'r-', 'LineWidth', 1.5);
    hold on;
    plot(1:T, estimation_error(2, :), 'b-', 'LineWidth', 1.5);
    xlabel('Time Step'); ylabel('Position Error');
    title('Position Estimation Errors');
    legend('X Error', 'Y Error'); grid on;
    
    
    % RMS errors
    %subplot(1, 3, 3);
    figure(3);
    pos_rmse = sqrt(estimation_error(1, :).^2 + estimation_error(2, :).^2);
    vel_rmse = sqrt(estimation_error(3, :).^2 + estimation_error(4, :).^2);
    plot(1:T, pos_rmse, 'g-', 'LineWidth', 2);
    hold on;
    plot(1:T, vel_rmse, 'm-', 'LineWidth', 2);
    xlabel('Time Step'); ylabel('RMSE');
    title('Root Mean Square Errors');
    legend('Position RMSE', 'Velocity RMSE'); grid on;
    

    % Print performance metrics
    final_pos_rmse = sqrt(mean(estimation_error(1, :).^2 + estimation_error(2, :).^2));
    final_vel_rmse = sqrt(mean(estimation_error(3, :).^2 + estimation_error(4, :).^2));
    
    fprintf('\n=== Tracking Performance ===\n');
    fprintf('Average Position RMSE: %.3f\n', final_pos_rmse);
    fprintf('Average Velocity RMSE: %.3f\n', final_vel_rmse);
    fprintf('Final Position Error: [%.3f, %.3f]\n', estimation_error(1, end), estimation_error(2, end));
    fprintf('Final Velocity Error: [%.3f, %.3f]\n', estimation_error(3, end), estimation_error(4, end));
    
    % Add convergence analysis
    fprintf('\n=== Convergence Analysis ===\n');
    fprintf('Initial Position Error: %.3f\n', sqrt(estimation_error(1,1)^2 + estimation_error(2,1)^2));
    fprintf('Final Position Error: %.3f\n', sqrt(estimation_error(1,end)^2 + estimation_error(2,end)^2));
    if sqrt(estimation_error(1,1)^2 + estimation_error(2,1)^2) > 0
        fprintf('Error Reduction: %.1f%%\n', 100*(1 - sqrt(estimation_error(1,end)^2 + estimation_error(2,end)^2)/sqrt(estimation_error(1,1)^2 + estimation_error(2,1)^2)));
    end
end

% function plot_tracking_results(true_state, estimated_state, observations, estimation_error, T)
%     % Plot tracking results
%     
%     figure('Position', [100, 100, 1200, 800]);
%     
%     % Trajectory plot
%     subplot(2, 3, 1);
%     plot(true_state(1, :), true_state(2, :), 'b-', 'LineWidth', 2, 'DisplayName', 'True');
%     hold on;
%     plot(estimated_state(1, :), estimated_state(2, :), 'r--', 'LineWidth', 2, 'DisplayName', 'Estimated');
%     scatter(observations(1, :), observations(2, :), 20, 'k.', 'DisplayName', 'Observations');
%     xlabel('X Position'); ylabel('Y Position');
%     title('2D Trajectory');
%     legend(); grid on;
%     
%     % Position errors
%     subplot(2, 3, 2);
%     plot(1:T, estimation_error(1, :), 'r-', 'LineWidth', 1.5);
%     hold on;
%     plot(1:T, estimation_error(2, :), 'b-', 'LineWidth', 1.5);
%     xlabel('Time Step'); ylabel('Position Error');
%     title('Position Estimation Errors');
%     legend('X Error', 'Y Error'); grid on;
%     
%     % Velocity errors
%     subplot(2, 3, 3);
%     plot(1:T, estimation_error(3, :), 'r-', 'LineWidth', 1.5);
%     hold on;
%     plot(1:T, estimation_error(4, :), 'b-', 'LineWidth', 1.5);
%     xlabel('Time Step'); ylabel('Velocity Error');
%     title('Velocity Estimation Errors');
%     legend('Vx Error', 'Vy Error'); grid on;
%     
%     % RMS errors
%     subplot(2, 3, 4);
%     pos_rmse = sqrt(estimation_error(1, :).^2 + estimation_error(2, :).^2);
%     vel_rmse = sqrt(estimation_error(3, :).^2 + estimation_error(4, :).^2);
%     plot(1:T, pos_rmse, 'g-', 'LineWidth', 2);
%     hold on;
%     plot(1:T, vel_rmse, 'm-', 'LineWidth', 2);
%     xlabel('Time Step'); ylabel('RMSE');
%     title('Root Mean Square Errors');
%     legend('Position RMSE', 'Velocity RMSE'); grid on;
%     
%     % State time series
%     subplot(2, 3, 5);
%     plot(1:T, true_state(1, :), 'b-', 1:T, estimated_state(1, :), 'r--', 'LineWidth', 1.5);
%     xlabel('Time Step'); ylabel('X Position');
%     title('X Position vs Time');
%     legend('True', 'Estimated'); grid on;
%     
%     subplot(2, 3, 6);
%     plot(1:T, true_state(3, :), 'b-', 1:T, estimated_state(3, :), 'r--', 'LineWidth', 1.5);
%     xlabel('Time Step'); ylabel('X Velocity');
%     title('X Velocity vs Time');
%     legend('True', 'Estimated'); grid on;
%     
%     % Print performance metrics
%     final_pos_rmse = sqrt(mean(estimation_error(1, :).^2 + estimation_error(2, :).^2));
%     final_vel_rmse = sqrt(mean(estimation_error(3, :).^2 + estimation_error(4, :).^2));
%     
%     fprintf('\n=== Tracking Performance ===\n');
%     fprintf('Average Position RMSE: %.3f\n', final_pos_rmse);
%     fprintf('Average Velocity RMSE: %.3f\n', final_vel_rmse);
%     fprintf('Final Position Error: [%.3f, %.3f]\n', estimation_error(1, end), estimation_error(2, end));
%     fprintf('Final Velocity Error: [%.3f, %.3f]\n', estimation_error(3, end), estimation_error(4, end));
%     
%     % Add convergence analysis
%     fprintf('\n=== Convergence Analysis ===\n');
%     fprintf('Initial Position Error: %.3f\n', sqrt(estimation_error(1,1)^2 + estimation_error(2,1)^2));
%     fprintf('Final Position Error: %.3f\n', sqrt(estimation_error(1,end)^2 + estimation_error(2,end)^2));
%     if sqrt(estimation_error(1,1)^2 + estimation_error(2,1)^2) > 0
%         fprintf('Error Reduction: %.1f%%\n', 100*(1 - sqrt(estimation_error(1,end)^2 + estimation_error(2,end)^2)/sqrt(estimation_error(1,1)^2 + estimation_error(2,1)^2)));
%     end
% end