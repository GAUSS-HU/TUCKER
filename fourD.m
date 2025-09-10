clear; close all; clc;

%% 4D目标跟踪参数设置 (位置x,y和速度vx,vy)
N = 20;                    % 进一步减少网格数以提高稳定性
% 定义4个维度的范围 
x_range = [-10, 10];       % x位置范围
y_range = [-10, 10];       % y位置范围
vx_range = [-3, 3];        % x速度范围
vy_range = [-3, 3];        % y速度范围

x = linspace(x_range(1), x_range(2), N);
y = linspace(y_range(1), y_range(2), N);
vx = linspace(vx_range(1), vx_range(2), N);
vy = linspace(vy_range(1), vy_range(2), N);

dx = x(2) - x(1);
dy = y(2) - y(1);
dvx = vx(2) - vx(1);
dvy = vy(2) - vy(1);

% 创建4D网格
[X, Y, VX, VY] = ndgrid(x, y, vx, vy);

% 时间参数
dt = 0.2;                  % 减小时间步长
t_final = 10;              % 减少仿真时间
num_steps = t_final/dt;

% 目标初始状态
x_target_init = 0;         
y_target_init = 0;         
vx_target_init = 0.8;      % 减小初始速度
vy_target_init = 0.3;      

%% 4D动力学模型参数
ax = 0.05;                 % 减小加速度
ay = -0.02;                
sigma_pos = 0.2;           
sigma_vel = 0.15;          

%% 初始4D PDF: 4维高斯分布
sigma_x_init = 1.0;        
sigma_y_init = 1.0;
sigma_vx_init = 0.4;
sigma_vy_init = 0.4;

% 4D高斯分布
p_init = (1/(2*pi)^2) * (1/(sigma_x_init*sigma_y_init*sigma_vx_init*sigma_vy_init)) * ...
         exp(-((X-x_target_init).^2/(2*sigma_x_init^2) + ...
               (Y-y_target_init).^2/(2*sigma_y_init^2) + ...
               (VX-vx_target_init).^2/(2*sigma_vx_init^2) + ...
               (VY-vy_target_init).^2/(2*sigma_vy_init^2)));

% 数值稳定性预处理
min_prob = 1e-8;
p_init = max(p_init, min_prob);
p_init = p_init / sum(p_init(:));

% 添加少量正则化噪声以提高数值稳定性
regularization_noise = 1e-10;
p_init = p_init + regularization_noise * ones(size(p_init));
p_init = p_init / sum(p_init(:));

%% 优化的Tucker分解策略

% 自适应秩选择：从高秩开始，如果失败则降低秩
rank_options = {[12, 12, 12, 12], [10, 10, 10, 10], [8, 8, 8, 8], [6, 6, 6, 6], [5, 5, 5, 5]};
rank_approx = [];
tucker_success = false;

% 设置更鲁棒的Tucker分解选项
opts = struct();
opts.tol = 5e-4;           % 更宽松的容差
opts.maxiters = 30;        % 适中的迭代次数
opts.printitn = 0;         
opts.init = 'random';      % 使用随机初始化

fprintf('尝试不同的Tucker分解秩...\n');

for rank_idx = 1:length(rank_options)
    current_rank = rank_options{rank_idx};
    fprintf('尝试秩 [%d, %d, %d, %d]...', current_rank(1), current_rank(2), current_rank(3), current_rank(4));
    
    try
        p_tensor = tensor(p_init);
        tucker_init = tucker_als(p_tensor, current_rank, opts);
        
        % 验证分解质量
        p_reconstructed = double(ttm(ttm(ttm(ttm(tucker_init.core, tucker_init.U{1}, 1), ...
                                tucker_init.U{2}, 2), tucker_init.U{3}, 3), tucker_init.U{4}, 4));
        
        reconstruction_error = norm(p_init(:) - p_reconstructed(:)) / norm(p_init(:));
        
        if reconstruction_error < 0.1 && all(isfinite(tucker_init.core(:))) && ...
           all(cellfun(@(x) all(isfinite(x(:))), tucker_init.U))
            
            G_current = tucker_init.core;
            U1_current = tucker_init.U{1};
            U2_current = tucker_init.U{2};
            U3_current = tucker_init.U{3};
            U4_current = tucker_init.U{4};
            rank_approx = current_rank;
            tucker_success = true;
            
            fprintf(' 成功! 重构误差: %.6f\n', reconstruction_error);
            break;
        else
            fprintf(' 重构质量不佳 (误差: %.6f)\n', reconstruction_error);
        end
        
    catch ME
        fprintf(' 失败: %s\n', ME.message);
    end
end

if ~tucker_success
    error('所有Tucker分解尝试都失败了');
end

%% 构建Fokker-Planck算子 (周期性边界条件)

% 一阶导数矩阵 (周期性边界条件)
e = ones(N,1);

% x方向一阶导数矩阵
D1_x = spdiags([-e 0*e e], [-1 0 1], N, N) / (2*dx);
D1_x(1,end) = -1 / (2*dx);  
D1_x(end,1) = 1 / (2*dx);   

% y方向一阶导数矩阵
D1_y = spdiags([-e 0*e e], [-1 0 1], N, N) / (2*dy);
D1_y(1,end) = -1 / (2*dy);
D1_y(end,1) = 1 / (2*dy);

% vx方向一阶导数矩阵
D1_vx = spdiags([-e 0*e e], [-1 0 1], N, N) / (2*dvx);
D1_vx(1,end) = -1 / (2*dvx);
D1_vx(end,1) = 1 / (2*dvx);

% vy方向一阶导数矩阵
D1_vy = spdiags([-e 0*e e], [-1 0 1], N, N) / (2*dvy);
D1_vy(1,end) = -1 / (2*dvy);
D1_vy(end,1) = 1 / (2*dvy);

% 二阶导数矩阵 (周期性边界条件)
D2_vx = spdiags([e -2*e e], -1:1, N, N) / (dvx^2);
D2_vx(1,end) = 1 / (dvx^2);  
D2_vx(end,1) = 1 / (dvx^2);  

D2_vy = spdiags([e -2*e e], -1:1, N, N) / (dvy^2);
D2_vy(1,end) = 1 / (dvy^2);
D2_vy(end,1) = 1 / (dvy^2);

% 使用平均速度的简化Fokker-Planck算子
vx_mean = mean(vx);
vy_mean = mean(vy);

Lx = -vx_mean * D1_x;
Ly = -vy_mean * D1_y;
Lvx = -ax * D1_vx + (sigma_vel^2/2) * D2_vx;
Lvy = -ay * D1_vy + (sigma_vel^2/2) * D2_vy;

% 计算矩阵指数
expLx = expm(dt * Lx);
expLy = expm(dt * Ly);
expLvx = expm(dt * Lvx);
expLvy = expm(dt * Lvy);

%% 误差分析准备
time_array = (0:num_steps)*dt;
relative_error_array = zeros(num_steps+1,1);
tucker_fails = 0;
tucker_success_count = 0;

% 用于保存特定时间点的分布
p_update_5 = [];
p_update_10 = [];

%% 观测参数
obs_x = 3;    
obs_y = -1.5;   
sigma_obs_base = 0.8;  

%% 时间迭代
fprintf('\n开始4D目标跟踪仿真 (优化Tucker版本)...\n');
fprintf('使用Tucker分解秩: [%d, %d, %d, %d]\n', rank_approx(1), rank_approx(2), rank_approx(3), rank_approx(4));

for k = 1:num_steps
    if mod(k, 5) == 0
        fprintf('时间步: %d/%d, Tucker成功率: %d/%d\n', k, num_steps, tucker_success_count, k);
    end
    
    % 当前时间
    t_now = k * dt;
    
    %% 预测步骤: 应用Fokker-Planck算子
    try
        U1_current = expLx * U1_current;   
        U2_current = expLy * U2_current;   
        U3_current = expLvx * U3_current;  
        U4_current = expLvy * U4_current;  
        
        % 重建预测PDF
        p_pred = ttm(ttm(ttm(ttm(tensor(G_current), U1_current, 1), ...
                     U2_current, 2), U3_current, 3), U4_current, 4);
        p_pred_double = double(p_pred);
        
        % 检查重建结果
        if any(~isfinite(p_pred_double(:))) || any(p_pred_double(:) < 0)
            error('重建的PDF包含无效值');
        end
        
        p_pred_double = max(p_pred_double, min_prob);
        p_pred_full = p_pred_double / sum(p_pred_double(:));
        
    catch ME
        error('Tucker预测失败: %s', ME.message);
    end
    
    %% 贝叶斯更新
    % 计算真实目标位置
    x_target = x_target_init + vx_target_init * t_now + 0.5 * ax * t_now^2;
    y_target = y_target_init + vy_target_init * t_now + 0.5 * ay * t_now^2;
    
    % 计算观测距离
    obs_r = sqrt((x_target - obs_x)^2 + (y_target - obs_y)^2);
    
    % 构造4D似然函数
    distance_4d = sqrt((X - obs_x).^2 + (Y - obs_y).^2);
    sigma_obs = sigma_obs_base + 0.02 * distance_4d;  
    L_func = (1./(sqrt(2*pi)*sigma_obs)) .* exp(-((distance_4d - obs_r).^2) ./ (2 * sigma_obs.^2));
    
    % 温和的似然函数归一化
    L_func = L_func / sum(L_func(:));  % 使用求和归一化
    L_func = max(L_func, 0.001);       % 避免过小的似然值，但不要太大
    
    % 贝叶斯更新
    p_update_full = p_pred_full .* L_func;
    p_update_full = max(p_update_full, min_prob);
    p_update_full = p_update_full / sum(p_update_full(:));
    
    % 添加微量正则化以保持数值稳定性
    p_update_full = p_update_full + regularization_noise * ones(size(p_update_full));
    p_update_full = p_update_full / sum(p_update_full(:));
    
    % 保存特定时间点的分布
    if k == round(5/dt)
        p_update_5 = p_update_full;
    elseif k == round(10/dt)
        p_update_10 = p_update_full;
    end
    
    %% Tucker分解更新 (每3步进行一次)
    if mod(k, 3) == 0
        tucker_updated = false;
        
        % 尝试当前秩的Tucker分解
        try
            p_update_stable = max(p_update_full, min_prob);
            p_update_stable = p_update_stable / sum(p_update_stable(:));
            
            p_tensor_update = tensor(p_update_stable);
            
            % 使用更严格的选项进行更新
            opts_update = opts;
            opts_update.tol = 1e-3;  % 稍微严格一些
            
            tucker_update = tucker_als(p_tensor_update, rank_approx, opts_update);
            
            % 验证分解结果
            if any(~isfinite(tucker_update.core(:))) || ...
               any(cellfun(@(x) any(~isfinite(x(:))), tucker_update.U))
                error('Tucker分解结果包含无效值');
            end
            
            % 验证重构质量
            p_test = double(ttm(ttm(ttm(ttm(tucker_update.core, tucker_update.U{1}, 1), ...
                        tucker_update.U{2}, 2), tucker_update.U{3}, 3), tucker_update.U{4}, 4));
            
            reconstruction_error = norm(p_update_stable(:) - p_test(:)) / norm(p_update_stable(:));
            
            if reconstruction_error < 0.2  % 允许一定的重构误差
                G_current = tucker_update.core;
                U1_current = tucker_update.U{1};
                U2_current = tucker_update.U{2};
                U3_current = tucker_update.U{3};
                U4_current = tucker_update.U{4};
                tucker_updated = true;
                tucker_success_count = tucker_success_count + 1;
            else
                fprintf('警告: 重构误差过大 (%.4f)，保持前一分解\n', reconstruction_error);
            end
            
        catch ME
            tucker_fails = tucker_fails + 1;
            fprintf('警告: Tucker分解失败 (%s)，保持前一分解\n', ME.message);
        end
        
        % 如果当前秩失败太多次，尝试降低秩
        if ~tucker_updated && tucker_fails > 3
            fprintf('尝试降低Tucker分解秩...\n');
            for lower_rank_idx = 1:length(rank_options)
                if all(rank_options{lower_rank_idx} < rank_approx)
                    try
                        new_rank = rank_options{lower_rank_idx};
                        tucker_update = tucker_als(p_tensor_update, new_rank, opts_update);
                        
                        if all(isfinite(tucker_update.core(:))) && ...
                           all(cellfun(@(x) all(isfinite(x(:))), tucker_update.U))
                            
                            G_current = tucker_update.core;
                            U1_current = tucker_update.U{1};
                            U2_current = tucker_update.U{2};
                            U3_current = tucker_update.U{3};
                            U4_current = tucker_update.U{4};
                            rank_approx = new_rank;
                            tucker_fails = 0;  % 重置失败计数
                            fprintf('成功降低秩到 [%d, %d, %d, %d]\n', new_rank(1), new_rank(2), new_rank(3), new_rank(4));
                            break;
                        end
                    catch
                        continue;
                    end
                end
            end
        end
    end
    
    %% 计算相对误差
    if k > 1
        pred_norm = norm(p_pred_full(:));
        if pred_norm > 1e-10
            relative_error = norm(p_update_full(:) - p_pred_full(:)) / pred_norm;
            relative_error_array(k+1) = min(relative_error, 1.0);  
        else
            relative_error_array(k+1) = relative_error_array(k);
        end
    end
end

fprintf('仿真完成! Tucker成功率: %.1f%%\n', 100*tucker_success_count/num_steps);

%% 结果可视化

% 边际分布计算 - 投影到2D位置空间
if ~isempty(p_update_5)
    p_pos_5 = squeeze(sum(sum(p_update_5, 4), 3));
    p_pos_5 = p_pos_5 / sum(p_pos_5(:));
end

if ~isempty(p_update_10)
    p_pos_10 = squeeze(sum(sum(p_update_10, 4), 3));
    p_pos_10 = p_pos_10 / sum(p_pos_10(:));
end

% 绘制位置的边际分布
figure('Position', [100, 100, 1200, 400]);
if ~isempty(p_update_5)
    subplot(1,3,1)
    imagesc(x, y, p_pos_5'); axis xy; colorbar;
    xlabel('X Position (m)'); ylabel('Y Position (m)');
    title('Position PDF at t=5s');
    hold on;
    t_5 = 5;
    x_true_5 = x_target_init + vx_target_init * t_5 + 0.5 * ax * t_5^2;
    y_true_5 = y_target_init + vy_target_init * t_5 + 0.5 * ay * t_5^2;
    plot(x_true_5, y_true_5, 'r*', 'MarkerSize', 15, 'LineWidth', 3);
    plot(obs_x, obs_y, 'ks', 'MarkerSize', 10, 'LineWidth', 2);
    legend('Target (true)', 'Observer', 'Location', 'best');
end

if ~isempty(p_update_10)
    subplot(1,3,2)
    imagesc(x, y, p_pos_10'); axis xy; colorbar;
    xlabel('X Position (m)'); ylabel('Y Position (m)');
    title('Position PDF at t=10s');
    hold on;
    t_10 = 10;
    x_true_10 = x_target_init + vx_target_init * t_10 + 0.5 * ax * t_10^2;
    y_true_10 = y_target_init + vy_target_init * t_10 + 0.5 * ay * t_10^2;
    plot(x_true_10, y_true_10, 'r*', 'MarkerSize', 15, 'LineWidth', 3);
    plot(obs_x, obs_y, 'ks', 'MarkerSize', 10, 'LineWidth', 2);
    legend('Target (true)', 'Observer', 'Location', 'best');
end

% 绘制相对误差随时间的变化
subplot(1,3,3)
valid_errors = relative_error_array(relative_error_array > 0);
valid_times = time_array(1:length(valid_errors));
plot(valid_times, valid_errors, '-', 'LineWidth', 2);
xlabel('Time (s)'); ylabel('Relative Error');
title('Tracking Error over Time');
grid on;

% 绘制速度的边际分布
if ~isempty(p_update_10)
    figure('Position', [100, 600, 800, 300]);
    
    % vx的边际分布
    p_vx_10 = squeeze(sum(sum(sum(p_update_10, 4), 2), 1));
    p_vx_10 = p_vx_10 / sum(p_vx_10);
    
    subplot(1,2,1)
    plot(vx, p_vx_10, 'b-', 'LineWidth', 2);
    xlabel('X Velocity (m/s)'); ylabel('Probability Density');
    title('X Velocity PDF at t=10s');
    grid on;
    hold on;
    vx_true_10 = vx_target_init + ax * 10;
    plot(vx_true_10, 0, 'r*', 'MarkerSize', 15, 'LineWidth', 3);
    legend('Estimated PDF', 'True velocity', 'Location', 'best');
    
    % vy的边际分布
    p_vy_10 = squeeze(sum(sum(sum(p_update_10, 3), 2), 1));
    p_vy_10 = p_vy_10 / sum(p_vy_10);
    
    subplot(1,2,2)
    plot(vy, p_vy_10, 'b-', 'LineWidth', 2);
    xlabel('Y Velocity (m/s)'); ylabel('Probability Density');
    title('Y Velocity PDF at t=10s');
    grid on;
    hold on;
    vy_true_10 = vy_target_init + ay * 10;
    plot(vy_true_10, 0, 'r*', 'MarkerSize', 15, 'LineWidth', 3);
    legend('Estimated PDF', 'True velocity', 'Location', 'best');
end

%% 性能统计
fprintf('\n=== 4D目标跟踪仿真结果 (优化Tucker版本) ===\n');
fprintf('网格大小: %d^4 = %d 个网格点\n', N, N^4);
fprintf('最终Tucker分解秩: [%d, %d, %d, %d]\n', rank_approx(1), rank_approx(2), rank_approx(3), rank_approx(4));
fprintf('仿真时间: %.1f 秒\n', t_final);
fprintf('时间步数: %d\n', num_steps);
fprintf('Tucker分解成功次数: %d/%d (%.1f%%)\n', tucker_success_count, num_steps, 100*tucker_success_count/num_steps);
fprintf('Tucker分解失败次数: %d\n', tucker_fails);

if ~isempty(valid_errors)
    final_error = valid_errors(end);
    mean_error = mean(valid_errors);
    fprintf('最终跟踪误差: %.6f\n', final_error);
    fprintf('平均跟踪误差: %.6f\n', mean_error);
end

% 计算最终位置估计
if ~isempty(p_update_10)
    p_pos_final = squeeze(sum(sum(p_update_10, 4), 3));
    p_pos_final = p_pos_final / sum(p_pos_final(:));
    
    [X_pos, Y_pos] = meshgrid(x, y);
    x_est = sum(sum(X_pos .* p_pos_final')) * dx * dy;
    y_est = sum(sum(Y_pos .* p_pos_final')) * dx * dy;
    
    x_true_final = x_target_init + vx_target_init * t_final + 0.5 * ax * t_final^2;
    y_true_final = y_target_init + vy_target_init * t_final + 0.5 * ay * t_final^2;
    
    pos_error = sqrt((x_est - x_true_final)^2 + (y_est - y_true_final)^2);
    
    fprintf('最终位置估计: (%.2f, %.2f)\n', x_est, y_est);
    fprintf('真实最终位置: (%.2f, %.2f)\n', x_true_final, y_true_final);
    fprintf('位置估计误差: %.2f m\n', pos_error);
end

fprintf('=====================================\n');