%% 匀速转弯目标Tucker-FPE跟踪仿真
% 状态: [x, y, vx, vy] - 2D位置和速度
% 运动模型: 协调转弯模型 (Coordinated Turn Model)
% 算法: FPE算子 + Tucker分解 + 各向异性似然

clear; close all; clc;

%% 参数设置
T = 100;                    % 总时间步数
dt = 0.1;                   % 时间步长
N = 50;                     % 网格点数
sigma_process = 0.05;     % 过程噪声标准差
sigma_obs = 1.0;            % 观测噪声标准差
omega = 0.1;               % 转弯角速度 (弧度/秒)

% 状态空间边界(扩大以适应转弯轨迹)
x_bounds = [-30, 30];
y_bounds = [-30, 30];
vx_bounds = [-3, 3];
vy_bounds = [-3, 3];

% 网格设置
x_grid = linspace(x_bounds(1), x_bounds(2), N);
y_grid = linspace(y_bounds(1), y_bounds(2), N);
vx_grid = linspace(vx_bounds(1), vx_bounds(2), N);
vy_grid = linspace(vy_bounds(1), vy_bounds(2), N);

dx = x_grid(2) - x_grid(1);
dy = y_grid(2) - y_grid(1);
dvx = vx_grid(2) - vx_grid(1);
dvy = vy_grid(2) - vy_grid(1);

%% 生成协调转弯真实轨迹
fprintf('生成协调转弯轨迹...\n');

true_state = zeros(4, T);
true_state(:, 1) = [0; 0; 2; 0];  % 初始状态 [x, y, vx, vy]

% 协调转弯状态转移矩阵
if omega * dt == 0
    F = eye(4);
else
    sin_omega_dt = sin(omega * dt);
    cos_omega_dt = cos(omega * dt);
    
    F = [1, 0, sin_omega_dt/omega, -(1-cos_omega_dt)/omega;
         0, 1, (1-cos_omega_dt)/omega, sin_omega_dt/omega;
         0, 0, cos_omega_dt, -sin_omega_dt;
         0, 0, sin_omega_dt, cos_omega_dt];
end

% 生成转弯轨迹
for t = 2:T
    % 协调转弯动力学
    true_state(:, t) = F * true_state(:, t-1) + sigma_process * randn(4, 1) * dt;
end

% 生成带噪声的观测(仅位置)
observations = true_state(1:2, :) + sigma_obs * randn(2, T);

%% 初始化PDF
fprintf('初始化概率密度函数...\n');

% 初始不确定性 - 高斯分布
initial_guess = [1; 1; 1.8; 0.2];  % 略偏离真实初始状态
initial_cov = diag([9, 9, 1, 1]);   % 较大的初始不确定性

% 创建初始4D PDF张量
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

% 归一化
pdf_tensor = pdf_tensor / sum(pdf_tensor(:)) / (dx * dy * dvx * dvy);

%% Tucker分解初始PDF
tucker_rank = [4, 4, 3, 3];  % Tucker秩
[G, U1, U2, U3, U4] = tucker_decompose_4d(pdf_tensor, tucker_rank);

%% 创建微分矩阵
D1 = create_diff_matrix(N, dx);
D2 = create_diff_matrix(N, dy);
D3 = create_diff_matrix(N, dvx);
D4 = create_diff_matrix(N, dvy);

%% 主跟踪循环
estimated_state = zeros(4, T);
estimation_error = zeros(4, T);

fprintf('开始Tucker-FPE协调转弯跟踪...\n');

for t = 1:T
    if mod(t, 10) == 0
        fprintf('处理时间步 %d/%d\n', t, T);
    end
    
    %% 步骤1: 使用协调转弯FPE算子进行预测
    if t > 1
        [G, U1, U2, U3, U4] = apply_coordinated_turn_fpe_evolution(G, U1, U2, U3, U4, ...
                                                                   D1, D2, D3, D4, dt, omega, sigma_process);
    end
    
    %% 步骤2: 贝叶斯更新
    % 构建基于位置观测的4D似然
    likelihood_tensor = construct_anisotropic_likelihood(x_grid, y_grid, vx_grid, vy_grid, ...
                                                        observations(:, t), sigma_obs, N);
    
    % 从Tucker因子重构PDF用于更新
    pdf_tensor = tucker_reconstruct_4d(G, U1, U2, U3, U4);
    
    % 贝叶斯更新
    pdf_tensor = pdf_tensor .* likelihood_tensor;
    
    % 归一化
    pdf_tensor = pdf_tensor / sum(pdf_tensor(:)) / (dx * dy * dvx * dvy);
    
    % 重新Tucker分解
    [G, U1, U2, U3, U4] = tucker_decompose_4d(pdf_tensor, tucker_rank);
    
    %% 步骤3: 状态估计(期望值)
    [X, Y, VX, VY] = ndgrid(x_grid, y_grid, vx_grid, vy_grid);
    total_prob = sum(pdf_tensor(:));
    
    if total_prob > 0
        estimated_state(1, t) = sum(X(:) .* pdf_tensor(:)) / total_prob;
        estimated_state(2, t) = sum(Y(:) .* pdf_tensor(:)) / total_prob;
        estimated_state(3, t) = sum(VX(:) .* pdf_tensor(:)) / total_prob;
        estimated_state(4, t) = sum(VY(:) .* pdf_tensor(:)) / total_prob;
    else
        % 备用MAP估计
        [~, max_idx] = max(pdf_tensor(:));
        [i_max, j_max, k_max, l_max] = ind2sub([N, N, N, N], max_idx);
        estimated_state(:, t) = [x_grid(i_max); y_grid(j_max); vx_grid(k_max); vy_grid(l_max)];
    end
    
    estimation_error(:, t) = estimated_state(:, t) - true_state(:, t);
end

%% 结果可视化
plot_coordinated_turn_results(true_state, estimated_state, observations, estimation_error, T, omega);

%% 函数定义

function [G, U1, U2, U3, U4] = tucker_decompose_4d(tensor, ranks)
    % 4D Tucker分解
    [N1, N2, N3, N4] = size(tensor);
    
    % 模式1展开和SVD
    T1 = reshape(tensor, N1, []);
    [U1, ~, ~] = svd(T1, 'econ');
    U1 = U1(:, 1:min(ranks(1), size(U1, 2)));
    
    % 模式2展开和SVD
    T2 = reshape(permute(tensor, [2,1,3,4]), N2, []);
    [U2, ~, ~] = svd(T2, 'econ');
    U2 = U2(:, 1:min(ranks(2), size(U2, 2)));
    
    % 模式3展开和SVD
    T3 = reshape(permute(tensor, [3,1,2,4]), N3, []);
    [U3, ~, ~] = svd(T3, 'econ');
    U3 = U3(:, 1:min(ranks(3), size(U3, 2)));
    
    % 模式4展开和SVD
    T4 = reshape(permute(tensor, [4,1,2,3]), N4, []);
    [U4, ~, ~] = svd(T4, 'econ');
    U4 = U4(:, 1:min(ranks(4), size(U4, 2)));
    
    % 计算核心张量G
    G = tensor;
    G = tensor_multiply_mode(G, U1', 1);
    G = tensor_multiply_mode(G, U2', 2);
    G = tensor_multiply_mode(G, U3', 3);
    G = tensor_multiply_mode(G, U4', 4);
end

function result = tensor_multiply_mode(tensor, matrix, mode)
    % 张量模式乘法
    sz = size(tensor);
    n_modes = length(sz);
    
    perm = [mode, 1:mode-1, mode+1:n_modes];
    tensor_perm = permute(tensor, perm);
    
    tensor_mat = reshape(tensor_perm, sz(mode), []);
    result_mat = matrix * tensor_mat;
    
    new_sz = sz;
    new_sz(mode) = size(matrix, 1);
    new_sz_perm = [new_sz(mode), new_sz(setdiff(1:n_modes, mode))];
    result_perm = reshape(result_mat, new_sz_perm);
    
    inv_perm(perm) = 1:n_modes;
    result = permute(result_perm, inv_perm);
end

function tensor = tucker_reconstruct_4d(G, U1, U2, U3, U4)
    % 从Tucker因子重构4D张量
    tensor = G;
    tensor = tensor_multiply_mode(tensor, U1, 1);
    tensor = tensor_multiply_mode(tensor, U2, 2);
    tensor = tensor_multiply_mode(tensor, U3, 3);
    tensor = tensor_multiply_mode(tensor, U4, 4);
end

function D = create_diff_matrix(N, h)
    % 创建有限差分微分矩阵
    D = zeros(N, N);
    
    % 边界点前向差分
    D(1, 1) = -1/h;
    D(1, 2) = 1/h;
    
    % 内点中心差分
    for i = 2:N-1
        D(i, i-1) = -1/(2*h);
        D(i, i+1) = 1/(2*h);
    end
    
    % 边界点后向差分
    D(N, N-1) = -1/h;
    D(N, N) = 1/h;
end

function [G_new, U1_new, U2_new, U3_new, U4_new] = apply_coordinated_turn_fpe_evolution(G, U1, U2, U3, U4, D1, D2, D3, D4, dt, omega, sigma)
    % 协调转弯FPE演化: p(t+dt) = exp(dt*L) * p(t)
    % 转弯动力学: dx/dt = vx, dy/dt = vy, dvx/dt = -ω*vy, dvy/dt = ω*vx
    
    % 模式1 (x维): 漂移项 dx/dt = vx
    L1_drift = construct_mode_operator(D1, dt, 'drift');
    U1_new = expm(L1_drift) * U1;
    
    % 模式2 (y维): 漂移项 dy/dt = vy  
    L2_drift = construct_mode_operator(D2, dt, 'drift');
    U2_new = expm(L2_drift) * U2;
    
    % 模式3 (vx维): 转弯动力学 dvx/dt = -ω*vy + 扩散
    L3_turn = construct_turn_operator(D3, dt, omega, 'vx', sigma);
    U3_new = expm(L3_turn) * U3;
    
    % 模式4 (vy维): 转弯动力学 dvy/dt = ω*vx + 扩散
    L4_turn = construct_turn_operator(D4, dt, omega, 'vy', sigma);
    U4_new = expm(L4_turn) * U4;
    
    % 核心张量演化
    G_new = apply_core_evolution(G, dt);
    
    % 保持因子矩阵正交性
    [U1_new, ~] = qr(U1_new, 0);
    [U2_new, ~] = qr(U2_new, 0);
    [U3_new, ~] = qr(U3_new, 0);
    [U4_new, ~] = qr(U4_new, 0);
end

function L_mode = construct_mode_operator(D, dt, type, sigma)
    % 构建模式特定FPE算子
    switch type
        case 'drift'
            % 漂移算子: -∂/∂x (对应dx/dt = vx)
            L_mode = -dt * D;
        case 'diffusion'
            % 扩散算子: σ²/2 * ∂²/∂x²
            if nargin >= 4
                L_mode = dt * sigma^2 * 0.5 * (D * D);
            else
                L_mode = dt * 0.01 * (D * D);
            end
        otherwise
            L_mode = zeros(size(D));
    end
end

function L_turn = construct_turn_operator(D, dt, omega, mode_type, sigma)
    % 构建转弯算子
    switch mode_type
        case 'vx'
            % dvx/dt = -ω*vy + 噪声
            % 简化处理：主要是扩散项，转弯耦合通过核心张量处理
            L_turn = dt * sigma^2 * 0.5 * (D * D) - dt * 0.1 * omega * D;
        case 'vy'
            % dvy/dt = ω*vx + 噪声
            L_turn = dt * sigma^2 * 0.5 * (D * D) + dt * 0.1 * omega * D;
        otherwise
            L_turn = zeros(size(D));
    end
end

function G_new = apply_core_evolution(G, dt)
    % 应用核心张量演化
    % 转弯模型的耦合项主要通过核心张量处理
    
    % 带阻尼的演化保持数值稳定性
    evolution_factor = exp(-0.005 * dt);
    G_new = G * evolution_factor;
    
    % 确保归一化
    norm_factor = sum(abs(G_new(:)));
    if norm_factor > 0
        G_new = G_new / norm_factor * sum(abs(G(:)));
    end
end

function likelihood = construct_anisotropic_likelihood(x_grid, y_grid, vx_grid, vy_grid, obs, sigma, N)
    % 构建4D各向异性似然张量
    likelihood = zeros(N, N, N, N);
    
    % 向量化计算提高效率
    [X, Y, VX, VY] = ndgrid(x_grid, y_grid, vx_grid, vy_grid);
    
    % 位置似然(仅依赖x, y)
    pos_likelihood = exp(-0.5 * ((X - obs(1)).^2 + (Y - obs(2)).^2) / sigma^2);
    
    % 速度似然(均匀分布，无信息)
    vel_likelihood = ones(size(VX));
    
    % 组合似然
    likelihood = pos_likelihood .* vel_likelihood;
    
    % 归一化
    likelihood = likelihood / sum(likelihood(:));
end

function plot_coordinated_turn_results(true_state, estimated_state, observations, estimation_error, T, omega)
    % 绘制协调转弯跟踪结果
    
    figure('Position', [100, 100, 1400, 900]);
    
    % 轨迹图
    figure(1);
    plot(true_state(1, :), true_state(2, :), 'b-', 'LineWidth', 2, 'DisplayName', '真实轨迹');
    hold on;
    plot(estimated_state(1, :), estimated_state(2, :), 'r--', 'LineWidth', 2, 'DisplayName', '估计轨迹');
    scatter(observations(1, :), observations(2, :), 15, 'k.', 'DisplayName', '观测点');
    xlabel('X 位置'); ylabel('Y 位置');
    title('协调转弯轨迹');
    legend(); grid on; axis equal;
    
    % 位置误差
    figure(2);
    plot(1:T, estimation_error(1, :), 'r-', 'LineWidth', 1.5);
    hold on;
    plot(1:T, estimation_error(2, :), 'b-', 'LineWidth', 1.5);
    xlabel('时间步'); ylabel('位置误差');
    title('位置估计误差');
    legend('X误差', 'Y误差'); grid on;
    
  
    
    % RMSE
    figure(3);
    pos_rmse = sqrt(estimation_error(1, :).^2 + estimation_error(2, :).^2);
    vel_rmse = sqrt(estimation_error(3, :).^2 + estimation_error(4, :).^2);
    plot(1:T, pos_rmse, 'g-', 'LineWidth', 2);
    hold on;
    plot(1:T, vel_rmse, 'm-', 'LineWidth', 2);
    xlabel('时间步'); ylabel('RMSE');
    title('均方根误差');
    legend('位置RMSE', '速度RMSE'); grid on;
    
  
    
    % 打印性能指标
    final_pos_rmse = sqrt(mean(estimation_error(1, :).^2 + estimation_error(2, :).^2));
    final_vel_rmse = sqrt(mean(estimation_error(3, :).^2 + estimation_error(4, :).^2));
    
    fprintf('\n=== 协调转弯跟踪性能 ===\n');
    fprintf('转弯角速度: %.4f rad/s\n', omega);
    fprintf('理论转弯半径: %.2f\n', 2/omega);  % 基于初始速度
    fprintf('平均位置RMSE: %.3f\n', final_pos_rmse);
    fprintf('平均速度RMSE: %.3f\n', final_vel_rmse);
    fprintf('最终位置误差: [%.3f, %.3f]\n', estimation_error(1, end), estimation_error(2, end));
    fprintf('最终速度误差: [%.3f, %.3f]\n', estimation_error(3, end), estimation_error(4, end));
    
    % 收敛分析
    fprintf('\n=== 收敛分析 ===\n');
    initial_pos_error = sqrt(estimation_error(1,1)^2 + estimation_error(2,1)^2);
    final_pos_error = sqrt(estimation_error(1,end)^2 + estimation_error(2,end)^2);
    fprintf('初始位置误差: %.3f\n', initial_pos_error);
    fprintf('最终位置误差: %.3f\n', final_pos_error);
    if initial_pos_error > 0
        fprintf('误差减少: %.1f%%\n', 100*(1 - final_pos_error/initial_pos_error));
    end
end

% function plot_coordinated_turn_results(true_state, estimated_state, observations, estimation_error, T, omega)
%     % 绘制协调转弯跟踪结果
%     
%     figure('Position', [100, 100, 1400, 900]);
%     
%     % 轨迹图
%     subplot(2, 3, 1);
%     plot(true_state(1, :), true_state(2, :), 'b-', 'LineWidth', 2, 'DisplayName', '真实轨迹');
%     hold on;
%     plot(estimated_state(1, :), estimated_state(2, :), 'r--', 'LineWidth', 2, 'DisplayName', '估计轨迹');
%     scatter(observations(1, :), observations(2, :), 15, 'k.', 'DisplayName', '观测点');
%     xlabel('X 位置'); ylabel('Y 位置');
%     title('协调转弯轨迹');
%     legend(); grid on; axis equal;
%     
%     % 位置误差
%     subplot(2, 3, 2);
%     plot(1:T, estimation_error(1, :), 'r-', 'LineWidth', 1.5);
%     hold on;
%     plot(1:T, estimation_error(2, :), 'b-', 'LineWidth', 1.5);
%     xlabel('时间步'); ylabel('位置误差');
%     title('位置估计误差');
%     legend('X误差', 'Y误差'); grid on;
%     
%     % 速度误差
%     subplot(2, 3, 3);
%     plot(1:T, estimation_error(3, :), 'r-', 'LineWidth', 1.5);
%     hold on;
%     plot(1:T, estimation_error(4, :), 'b-', 'LineWidth', 1.5);
%     xlabel('时间步'); ylabel('速度误差');
%     title('速度估计误差');
%     legend('Vx误差', 'Vy误差'); grid on;
%     
%     % RMSE
%     subplot(2, 3, 4);
%     pos_rmse = sqrt(estimation_error(1, :).^2 + estimation_error(2, :).^2);
%     vel_rmse = sqrt(estimation_error(3, :).^2 + estimation_error(4, :).^2);
%     plot(1:T, pos_rmse, 'g-', 'LineWidth', 2);
%     hold on;
%     plot(1:T, vel_rmse, 'm-', 'LineWidth', 2);
%     xlabel('时间步'); ylabel('RMSE');
%     title('均方根误差');
%     legend('位置RMSE', '速度RMSE'); grid on;
%     
%     % 速度矢量图
%     subplot(2, 3, 5);
%     plot(true_state(3, :), true_state(4, :), 'b-', 'LineWidth', 2, 'DisplayName', '真实速度');
%     hold on;
%     plot(estimated_state(3, :), estimated_state(4, :), 'r--', 'LineWidth', 2, 'DisplayName', '估计速度');
%     xlabel('Vx'); ylabel('Vy');
%     title('速度空间轨迹');
%     legend(); grid on; axis equal;
%     
%     % 转弯半径分析
%     subplot(2, 3, 6);
%     true_speed = sqrt(true_state(3, :).^2 + true_state(4, :).^2);
%     est_speed = sqrt(estimated_state(3, :).^2 + estimated_state(4, :).^2);
%     
%     true_radius = true_speed / abs(omega);
%     est_radius = est_speed / abs(omega);
%     
%     plot(1:T, true_radius, 'b-', 'LineWidth', 2, 'DisplayName', '真实转弯半径');
%     hold on;
%     plot(1:T, est_radius, 'r--', 'LineWidth', 2, 'DisplayName', '估计转弯半径');
%     xlabel('时间步'); ylabel('转弯半径');
%     title('转弯半径估计');
%     legend(); grid on;
%     
%     sgtitle(sprintf('协调转弯跟踪结果 (ω=%.3f rad/s)', omega), 'FontSize', 14, 'FontWeight', 'bold');
%     
%     % 打印性能指标
%     final_pos_rmse = sqrt(mean(estimation_error(1, :).^2 + estimation_error(2, :).^2));
%     final_vel_rmse = sqrt(mean(estimation_error(3, :).^2 + estimation_error(4, :).^2));
%     
%     fprintf('\n=== 协调转弯跟踪性能 ===\n');
%     fprintf('转弯角速度: %.4f rad/s\n', omega);
%     fprintf('理论转弯半径: %.2f\n', 2/omega);  % 基于初始速度
%     fprintf('平均位置RMSE: %.3f\n', final_pos_rmse);
%     fprintf('平均速度RMSE: %.3f\n', final_vel_rmse);
%     fprintf('最终位置误差: [%.3f, %.3f]\n', estimation_error(1, end), estimation_error(2, end));
%     fprintf('最终速度误差: [%.3f, %.3f]\n', estimation_error(3, end), estimation_error(4, end));
%     
%     % 收敛分析
%     fprintf('\n=== 收敛分析 ===\n');
%     initial_pos_error = sqrt(estimation_error(1,1)^2 + estimation_error(2,1)^2);
%     final_pos_error = sqrt(estimation_error(1,end)^2 + estimation_error(2,end)^2);
%     fprintf('初始位置误差: %.3f\n', initial_pos_error);
%     fprintf('最终位置误差: %.3f\n', final_pos_error);
%     if initial_pos_error > 0
%         fprintf('误差减少: %.1f%%\n', 100*(1 - final_pos_error/initial_pos_error));
%     end
% end