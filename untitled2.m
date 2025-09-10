clear; close all; clc;

%% 参数设置
N = 80;                   % 每个维度网格数
x = linspace(-5,5,N);
y = linspace(-5,5,N);
dx = x(2)-x(1); dy = y(2)-y(1);
[X,Y] = meshgrid(x,y);

sigma2 = 1;             % 扩散系数
dt = 1;                 % 时间步长
t_final = 100;            % 模拟结束时间
num_steps = t_final/dt;

% 目标初始位置
x_target_init = 0;  % 初始 x 坐标
y_target_init = 0;  % 初始 y 坐标

%% 初始pdf: 二维高斯分布 N(0,1)×N(0,1)
p_init = (1/(2*pi))*exp(-(X.^2+Y.^2)/2);
p_init = p_init / sum(p_init(:));

%% Tucker分解初始pdf
p_tensor = tensor(p_init);
rank_approx = [20,20];
tucker_init = tucker_als(p_tensor, rank_approx);
G_init = tucker_init.core;
Ux_init = tucker_init.U{1};
Uy_init = tucker_init.U{2};

% 当前pdf的Tucker表示初始化为初始值
G_current = G_init;
Ux_current = Ux_init;
Uy_current = Uy_init;

% 初始化保存t=50时分布的变量
p_update_50 = [];

% 漂移项参数
mu_x = 0.2; mu_y = 0.2;

% 一阶导数矩阵 (周期性边界条件)
e = ones(N,1);
D1 = spdiags([-e 0*e e], [-1 0 1], N, N) / (2*dx); % 中心差分法
D1(1,end) = -1 / (2*dx);  % x_1 的左邻居是 x_N
D1(end,1) = 1 / (2*dx);   % x_N 的右邻居是 x_1

% 二阶导数矩阵 (周期性边界条件)
D2 = spdiags([e -2*e e], -1:1, N, N) / (dx^2); % 二阶中心差分
D2(1,end) = 1 / (dx^2);  % x_1 的左邻居是 x_N
D2(end,1) = 1 / (dx^2);  % x_N 的右邻居是 x_1

% Fokker-Planck 算子
Lx = -mu_x * D1 + (sigma2/2) * D2;  % x方向
Ly = -mu_y * D1 + (sigma2/2) * D2;  % y方向

% 计算矩阵指数的准确方法
expLx_half = expm((dt / 2) * Lx);
expLy = expm(dt * Ly);

%% 误差分析准备
time_array = (0:num_steps)*dt;
rmse_array = zeros(num_steps+1,1);
% 初始化 L2 误差数组
relative_error_array = zeros(num_steps+1,1);

% 定义完整的Fokker-Planck算子
L = kron(speye(N), Lx) + kron(Ly, speye(N)); % Lx 和 Ly 合并

% 计算矩阵指数演化: exp(dt * L)
expL = expm(dt * full(L));  % 使用完整的矩阵指数计算

% 初始化p_exact为初始分布的列向量形式
p_exact_vec = p_init(:);  % 将二维分布展平成列向量

p_exact_50 = [];
p_exact_100 = [];

%% 时间迭代
for k = 1:num_steps  
    % 当前时间
    t_now = k * dt;

    %% 预测步骤: 包含漂移和扩散项的 FPE
    % 使用准确的矩阵指数进行步进
    Ux_current = expLx_half * Ux_current;
    Uy_current = expLy * Uy_current;

    % 使用更新后的因子矩阵和核张量重建预测pdf
    p_pred = ttm(ttm(tensor(G_current), Ux_current, 1), Uy_current, 2);
    p_pred_double = double(p_pred);
    p_pred_double(p_pred_double < 0) = 0;  % 确保非负
    p_pred_full = p_pred_double / sum(p_pred_double(:));  % 归一化

    %% 贝叶斯更新
    % 观测平台位置
    obs_x = 2;  % 平台 x 坐标
    obs_y = -1; % 平台 y 坐标

    % 动态计算目标位置
    x_target = x_target_init + mu_x * t_now;  % 目标 x 坐标
    y_target = y_target_init + mu_y * t_now;  % 目标 y 坐标

    % 动态计算观测距离
    obs_r = sqrt((x_target - obs_x)^2 + (y_target - obs_y)^2);

    % 构造基于距离的高斯似然函数
    distance = sqrt((X - obs_x).^2 + (Y - obs_y).^2);  % 网格点到观测平台的距离
    sigma_obs = 1 + 0.01 * distance;  % 观测噪声标准差
    L_func = (1./(sqrt(2*pi)*sigma_obs)) * exp(-((distance - obs_r).^2) ./ (2 * sigma_obs^2));
    L_func = L_func / sum(L_func(:));  % 归一化

    % 进行贝叶斯更新
    p_update_full = p_pred_full .* L_func;  % 逐元素乘法

    % 检查负值并处理
    p_update_full(p_update_full < 0) = 0;

    % 归一化
    p_update_full = p_update_full / sum(p_update_full(:));  % 归一化

    % 保存t=50时的分布
    if k == 50
        p_update_50 = p_update_full;
    end

    %% 对更新后的pdf进行Tucker分解
    p_tensor_update = tensor(p_update_full);  % 将更新后的pdf转化为张量
    tucker_update = tucker_als(p_tensor_update, rank_approx);  % 重新进行Tucker分解
    G_current = tucker_update.core;  % 更新核张量
    Ux_current = tucker_update.U{1};  % 更新x方向的因子矩阵
    Uy_current = tucker_update.U{2};  % 更新y方向的因子矩阵

    % 通过矩阵指数计算p_exact在下一时间步的分布
    p_exact_vec = expL * p_exact_vec;

    % 恢复二维分布并归一化
    p_exact_full = reshape(p_exact_vec, [N, N]);
    p_exact_full = p_exact_full / sum(p_exact_full(:));  % 确保归一化

    % 更新 p_exact
    p_exact_full = p_exact_full .* L_func;  % 元素相乘
    p_exact_full = p_exact_full / sum(p_exact_full(:));  % 归一化

    % 保存特定时间点的真实分布
    if k == 50
        p_exact_50 = p_exact_full;
    elseif k == 100
        p_exact_100 = p_exact_full;
    end

    % 将更新后的p_exact转换为列向量，用于下一个时间步
    p_exact_vec = p_update_full(:);    

    % 计算 RMSE
    rmse = sqrt(mean((double(p_update_full(:)) - p_exact_full(:)).^2));
    rmse_array(k+1) = rmse;    

    %% 计算相对误差
    relative_error = norm(double(p_update_full(:)) - p_exact_full(:), 2) / norm(p_exact_full(:), 2);
    relative_error_array(k+1) = relative_error;
end

%% 绘制真实pdf对比
figure;
subplot(1,2,1)
imagesc(x, y, p_exact_50'); axis xy; colorbar;
xlabel('X(m)'); ylabel('Y(m)');
title('Exact PDF at t=50s');

subplot(1,2,2)
imagesc(x, y, p_exact_100'); axis xy; colorbar;
xlabel('X(m)'); ylabel('Y(m)');
title('Exact PDF at t=100s');

%% 绘制pdf对比
figure;
subplot(1,2,1)
imagesc(x, y, p_update_50'); axis xy; colorbar;
xlabel('X(m)'); ylabel('Y(m)');
title('');

p_update = double(p_pred).*L_func;
p_update = p_update / sum(p_update(:));
subplot(1,2,2)
imagesc(x,y,p_update'); axis xy; colorbar;
xlabel('X(m)'); ylabel('Y(m)');
title('');


%% 绘制误差随时间的变化
figure;
plot(time_array(2:end), rmse_array(2:end), '-','LineWidth',2);
xlabel('Time(s)'); ylabel('RMSE');
title('RMSE over Time');
grid on;

%% 绘制相对误差随时间的变化
figure;
plot(time_array(2:end), relative_error_array(2:end), '-', 'LineWidth', 2);
xlabel('Time(s)');
ylabel('Relative error');
title('Relative Error over Time');
grid on;