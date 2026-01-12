function benchmark_compare_kron_tucker_tt()
% 三法对比：Tucker-FPE(HOSVD) / TT-FPE / Kronecker-Strang(密集)
% 结构与流程一致：低秩(或密集)预测 → 似然更新 → 再分解(或直接替换)
% 这里只保留 CT 场景（匀速转弯）
%
% 输出：CT 的位置RMSE曲线（带放大图）+ 运行时柱图 + 总结表

clear; clc; close all; rng(0);

%% 公共参数 =========
T  = 100;          % 时间步
dt = 0.1;          % 步长
N  = 50;           % 每维网格点
sigma_process     = 0.1;   % 过程噪声(CV) - 现在不用了
sigma_obs         = 1.0;   % 观测噪声(仅似然)
omega             = 0.1;   % CT角速度
sigma_process_ct  = 0.05;  % 过程噪声(CT)

x_bounds = [-30,30]; y_bounds = [-30,30];
vx_bounds = [-3,3];  vy_bounds = [-3,3];

x_grid  = linspace(x_bounds(1), x_bounds(2), N);
y_grid  = linspace(y_bounds(1), y_bounds(2), N);
vx_grid = linspace(vx_bounds(1), vx_bounds(2), N);
vy_grid = linspace(vy_bounds(1), vy_bounds(2), N);

dx = x_grid(2)-x_grid(1); dy = y_grid(2)-y_grid(1);
dvx= vx_grid(2)-vx_grid(1); dvy = vy_grid(2)-vy_grid(1);

% 差分矩阵
D1 = create_diff_matrix(N, dx);
D2 = create_diff_matrix(N, dy);
D3 = create_diff_matrix(N, dvx);
D4 = create_diff_matrix(N, dvy);

% 初始PDF
initial_guess = [2;2;0.8;0.3];
initial_cov   = diag([4,4,0.5,0.5]);

pdf0 = build_initial_pdf_4d(x_grid,y_grid,vx_grid,vy_grid,initial_guess,initial_cov);
pdf0 = pdf0 / sum(pdf0(:)) / (dx*dy*dvx*dvy);

% Tucker：能量阈值秩截断
tucker_rank_max = [10,10,4,4];
energy_keep     = 0.995;

% TT 内秩
tt_ranks = [8,8,6];

%% 真实轨迹 + 观测 =========
% 只保留 CT 场景
[true_ct, obs_ct] = gen_true_obs_ct(T, dt, omega, sigma_process_ct, sigma_obs);

scenes = { ...
    struct('name','CT', 'true',true_ct, 'obs',obs_ct, ...
           'dyn','ct', 'omega',omega, 'sigma',sigma_process_ct) ...
};

%% 方法列表（避免与 MATLAB 内置 methods 冲突） =========
meths = { ...
    struct('name','Tucker-FPE(HOSVD)', 'tag','Tucker', 'runner', @(pdf)runner_tucker(pdf,tucker_rank_max,energy_keep)), ...
    struct('name','TT-FPE',            'tag','TT',     'runner', @(pdf)runner_tt(pdf,tt_ranks)), ...
    struct('name','Kron-Strang',       'tag','Kron',   'runner', @(pdf)runner_kron(pdf)) ...
};

%% 逐场景逐方法 =========
results = struct();
for s=1:numel(scenes)
    sc = scenes{s};
    fprintf('\n=== Scene: %s ===\n', sc.name);
    for m=1:numel(meths)
        meth = meths{m};
        fprintf('Method: %s ... ', meth.name);
        t_start = tic;
        [est, err, pos_rmse, vel_rmse] = run_scene_with_method( ...
            sc, pdf0, meth.runner, ...
            x_grid,y_grid,vx_grid,vy_grid, dx,dy,dvx,dvy, ...
            D1,D2,D3,D4, dt, sigma_obs);
        time_cost = toc(t_start);
        fprintf('done. time = %.2fs\n', time_cost);

        results.(sc.name).(meth.tag).est      = est;
        results.(sc.name).(meth.tag).err      = err;
        results.(sc.name).(meth.tag).pos_rmse = pos_rmse;
        results.(sc.name).(meth.tag).vel_rmse = vel_rmse;
        results.(sc.name).(meth.tag).time     = time_cost;
        results.(sc.name).(meth.tag).avg_time = time_cost/numel(pos_rmse);
    end
end

%% 绘图 =========
plot_compare_all(results, meths, scenes);

%% 文本表 =========
print_table(results, meths, scenes);

end % ===== 主函数结束 =====


%% ============ 单场景 + 单方法骨架 ============

function [est, err, pos_rmse, vel_rmse] = run_scene_with_method(sc, pdf0, runner, ...
    xg,yg,vxg,vyg, dx,dy,dvx,dvy, D1,D2,D3,D4, dt, sigma_obs_nom)

T = size(sc.true,2);
N = numel(xg);
[X, Y, VX, VY] = ndgrid(xg,yg,vxg,vyg);

S = runner(pdf0); % 分解结构体（或密集PDF）含 reconstruct/step/redecompose

est = zeros(4,T); err = zeros(4,T);

for t=1:T
    % 预测
    if t>1
        switch sc.dyn
            case 'cv', S = S.step_cv(S, D1,D2,D3,D4, dt, sc.sigma); %#ok<UNRCH>
            case 'ct', S = S.step_ct(S, D1,D2,D3,D4, dt, sc.omega, sc.sigma);
        end
    end

    % 似然退火：前 10 步放宽 25%，后续回到标称
    sigma_eff = sigma_obs_nom * (t<=10)*1.25 + sigma_obs_nom*(t>10);

    % 观测更新
    like = construct_anisotropic_likelihood(xg,yg,vxg,vyg, sc.obs(:,t), sigma_eff, N);
    pdf  = S.reconstruct(S) .* like;
    pdf  = pdf / sum(pdf(:)) / (dx*dy*dvx*dvy);

    % 再分解（投影回低秩/或直接替换）
    S = S.redecompose(S, pdf);

    % 估计
    tot = sum(pdf(:));
    if tot>0
        est(1,t) = sum(X(:).*pdf(:))/tot;
        est(2,t) = sum(Y(:).*pdf(:))/tot;
        est(3,t) = sum(VX(:).*pdf(:))/tot;
        est(4,t) = sum(VY(:).*pdf(:))/tot;
    else
        [~,idx]=max(pdf(:)); [i,j,k,l]=ind2sub([N,N,N,N],idx);
        est(:,t) = [xg(i); yg(j); vxg(k); vyg(l)];
    end

    err(:,t) = est(:,t) - sc.true(:,t);
end

pos_rmse = sqrt(err(1,:).^2 + err(2,:).^2);
vel_rmse = sqrt(err(3,:).^2 + err(4,:).^2);
end


%% ============ 方法1：Tucker-FPE(HOSVD) ============

function S = runner_tucker(T4, ranks_max, keep_energy)
[U1,U2,U3,U4,G] = tucker_hosvd_consistent_energy(T4, ranks_max, keep_energy);
S.type='tucker'; S.G=G; S.U1=U1; S.U2=U2; S.U3=U3; S.U4=U4;
S.reconstruct = @(S) ttm(S.G,{S.U1,S.U2,S.U3,S.U4},[1,2,3,4]);
S.redecompose = @(S,pdf) tucker_redecompose(S,pdf,ranks_max,keep_energy);
S.step_cv = @(S,D1,D2,D3,D4,dt,sig) tucker_step(S,D1,D2,D3,D4,dt,sig,0);
S.step_ct = @(S,D1,D2,D3,D4,dt,om,sig) tucker_step(S,D1,D2,D3,D4,dt,sig,om);
end

function S = tucker_redecompose(S, pdf, ranks_max, keep_energy)
[U1,U2,U3,U4,G] = tucker_hosvd_consistent_energy(pdf, ranks_max, keep_energy);
S.G=G; S.U1=U1; S.U2=U2; S.U3=U3; S.U4=U4;
end

function S = tucker_step(S, D1,D2,D3,D4, dt, sigma, omega)
% 结构不变：位置漂移 + 速度扩散 +（CT时）一阶耦合；仅调系数
L1 = -dt*D1; L2 = -dt*D2;
if omega==0
    L3 = dt*(sigma^2*0.5)*(D3*D3);
    L4 = dt*(sigma^2*0.5)*(D4*D4);
else
    % 自适应耦合强度（仍是一阶项）
    kappa = min(1.0, 0.5 + 2.0*omega*dt);   % 0.5~1.0
    L3 = dt*(sigma^2*0.5)*(D3*D3) - dt*kappa*omega*D3;
    L4 = dt*(sigma^2*0.5)*(D4*D4) + dt*kappa*omega*D4;
end
S.U1 = ortho(expm(L1)*S.U1);
S.U2 = ortho(expm(L2)*S.U2);
S.U3 = ortho(expm(L3)*S.U3);
S.U4 = ortho(expm(L4)*S.U4);
% 注意使用 .*，保持 4D 结构
S.G  = S.G .* exp(-0.01*dt*(omega==0) - 0.005*dt*(omega~=0));  % 轻阻尼
end


%% ============ 方法2：TT-FPE（TT-SVD） ============

function S = runner_tt(T4, tt_ranks)
[G1,G2,G3,G4] = tt_svd_4d(T4, tt_ranks);
S.type='tt';
S.G1=G1; S.G2=G2; S.G3=G3; S.G4=G4; S.ranks=tt_ranks;
S.reconstruct = @(S) tt_reconstruct_4d(S.G1,S.G2,S.G3,S.G4);
S.redecompose = @(S,pdf) tt_redecompose(S,pdf,S.ranks);
S.step_cv = @(S,D1,D2,D3,D4,dt,sig) tt_step(S,D1,D2,D3,D4,dt,sig,0);
S.step_ct = @(S,D1,D2,D3,D4,dt,om,sig) tt_step(S,D1,D2,D3,D4,dt,sig,om);
end

function S = tt_redecompose(S, pdf, tt_ranks)
[G1,G2,G3,G4] = tt_svd_4d(pdf, tt_ranks);
S.G1=G1; S.G2=G2; S.G3=G3; S.G4=G4;
end

function S = tt_step(S, D1,D2,D3,D4, dt, sigma, omega)
Lx = -dt*D1; Ly = -dt*D2;
Lvx= dt*(sigma^2*0.5)*(D3*D3) - dt*0.1*omega*D3;
Lvy= dt*(sigma^2*0.5)*(D4*D4) + dt*0.1*omega*D4;

% G1: [N r1]
S.G1 = expm(Lx) * S.G1;

% G2: [r1 N r2]，沿第2维乘 Ly
tmp = permute(S.G2,[2 1 3]);          % N x r1 x r2
tmp = reshape(tmp, size(tmp,1), []);  % N x (r1*r2)
tmp = expm(Ly) * tmp;
S.G2 = permute( reshape(tmp, [size(S.G2,2), size(S.G2,1), size(S.G2,3)]), [2 1 3] );

% G3: [r2 N r3]，沿第2维乘 Lvx
tmp = permute(S.G3,[2 1 3]);          % N x r2 x r3
tmp = reshape(tmp, size(tmp,1), []);  % N x (r2*r3)
tmp = expm(Lvx) * tmp;
S.G3 = permute( reshape(tmp, [size(S.G3,2), size(S.G3,1), size(S.G3,3)]), [2 1 3] );

% G4: [r3 N]
S.G4 = (expm(Lvy) * S.G4.').';         % 左乘
end


%% ============ 方法3：Kron-Strang（密集PDF） ============

function S = runner_kron(pdf_init)
S.type = 'kron';
S.P = pdf_init;   % 直接保存密集PDF
S.reconstruct = @(S) S.P;
S.redecompose = @(S,pdf) setfield(S,'P',pdf); %#ok<SFLD> 简单替换

S.step_cv = @(S,D1,D2,D3,D4,dt,sig) kron_step(S,D1,D2,D3,D4,dt,sig,0);
S.step_ct = @(S,D1,D2,D3,D4,dt,om,sig) kron_step(S,D1,D2,D3,D4,dt,om,sig);
end

function S = kron_step(S, D1,D2,D3,D4, dt, sigma, omega)
% 一维算子
A1 = -dt*D1; A2 = -dt*D2;
if omega==0
    A3 = dt*(sigma^2*0.5)*(D3*D3);
    A4 = dt*(sigma^2*0.5)*(D4*D4);
else
    A3 = dt*(sigma^2*0.5)*(D3*D3) - dt*0.1*omega*D3;
    A4 = dt*(sigma^2*0.5)*(D4*D4) + dt*0.1*omega*D4;
end

% 对称 Strang 分裂：按模态“左乘”实现
E1 = expm(A1/2); E2 = expm(A2/2);
E3 = expm(A3/2); E4 = expm(A4/2);

P = S.P;
P = tensor_multiply_mode(P, E1, 1);
P = tensor_multiply_mode(P, E2, 2);
P = tensor_multiply_mode(P, E3, 3);
P = tensor_multiply_mode(P, E4, 4);

% 中间整步
E1f = expm(A1 - A1/2);  % 等价于 expm(A1/2)
E2f = expm(A2 - A2/2);
E3f = expm(A3 - A3/2);
E4f = expm(A4 - A4/2);

P = tensor_multiply_mode(P, E1f, 1);
P = tensor_multiply_mode(P, E2f, 2);
P = tensor_multiply_mode(P, E3f, 3);
P = tensor_multiply_mode(P, E4f, 4);

% 回来半步
P = tensor_multiply_mode(P, E4, 4);
P = tensor_multiply_mode(P, E3, 3);
P = tensor_multiply_mode(P, E2, 2);
P = tensor_multiply_mode(P, E1, 1);

S.P = P;
end


%% ============ Tucker/TT 基础算子 ============

% HOSVD（能量阈值版）：返回 U1..U4 (N_i x r_i) 与核心 G (r1 x r2 x r3 x r4)
function [U1,U2,U3,U4,G] = tucker_hosvd_consistent_energy(T4, ranks_max, keep)
[N1,N2,N3,N4] = size(T4);
U1 = topU_energy( reshape(T4, N1, []),                 ranks_max(1), keep );
U2 = topU_energy( reshape(permute(T4,[2,1,3,4]),N2,[]),ranks_max(2), keep );
U3 = topU_energy( reshape(permute(T4,[3,1,2,4]),N3,[]),ranks_max(3), keep );
U4 = topU_energy( reshape(permute(T4,[4,1,2,3]),N4,[]),ranks_max(4), keep );
G  = ttm(T4, {U1',U2',U3',U4'}, [1,2,3,4]);            % core: r1 x r2 x r3 x r4
end

function X = topU_energy(M, rmax, keep)
[U,S,~] = svd(M,'econ');
s = diag(S); cs = cumsum(s.^2); tot = cs(end);
r = find(cs >= keep*tot, 1, 'first');
if isempty(r), r = 1; end
r = min(r, min(rmax,size(U,2)));
X = U(:,1:r);
end

% 4D TT-SVD：给定内部秩 [r1,r2,r3]，返回 G1..G4
function [G1,G2,G3,G4] = tt_svd_4d(T4, r)
[N1,N2,N3,N4] = size(T4);

% 1) N1 × (N2*N3*N4)
M1 = reshape(T4, N1, []);
[U,S,V] = svd(M1,'econ'); r1 = min(r(1), size(U,2));
U = U(:,1:r1); S = S(1:r1,1:r1); V = V(:,1:r1);
G1 = U;                  % N1 x r1
Trem = S*V.';            % r1 × (N2*N3*N4)

% 2) (r1*N2) × (N3*N4)
M2 = reshape(Trem, r1*N2, []);
[U,S,V] = svd(M2,'econ'); r2 = min(r(2), size(U,2));
U = U(:,1:r2); S = S(1:r2,1:r2); V = V(:,1:r2);
G2 = reshape(U, r1, N2, r2);
Trem = S*V.';            % r2 × (N3*N4)

% 3) (r2*N3) × N4
M3 = reshape(Trem, r2*N3, N4);
[U,S,V] = svd(M3,'econ'); r3 = min(r(3), size(U,2));
U = U(:,1:r3); S = S(1:r3,1:r3); V = V(:,1:r3);
G3 = reshape(U, r2, N3, r3);
G4 = (S*V.').';          % N4 x r3 -> 转 r3 x N4
G4 = G4.';
end

function T4 = tt_reconstruct_4d(G1,G2,G3,G4)
N1=size(G1,1); r1=size(G1,2);
r2=size(G2,3); N2=size(G2,2);
r3=size(G3,3); N3=size(G3,2);
N4=size(G4,2);

T12 = zeros(N1,N2,r2);
for i2=1:N2
    T12(:,i2,:) = G1 * squeeze(G2(:,i2,:));   % N1 x r2
end

T123 = zeros(N1,N2,N3,r3);
for i3=1:N3
    A = reshape(T12, N1*N2, r2) * squeeze(G3(:,i3,:));  % (N1*N2) x r3
    T123(:,:,i3,:) = reshape(A, N1,N2,1,r3);
end

T4 = zeros(N1,N2,N3,N4);
for i4=1:N4
    A = reshape(T123, N1*N2*N3, r3) * G4(:,i4);          % (N1*N2*N3) x 1
    T4(:,:,:,i4) = reshape(A, N1,N2,N3,1);
end
end


%% ============ 工具函数（ttm/张量乘/正交等） ============

function Y = ttm(X, Us, modes)
Y = X;
for k = 1:numel(modes)
    mode = modes(k); U = Us{k};
    Y = tensor_multiply_mode(Y, U, mode);
end
end

function Y = tensor_multiply_mode(X, U, mode)
% 模式乘法 Y = X ×_mode U
% 要求 size(U) = [new_dim, old_dim]；若写反了自动转置

sz = size(X);
if numel(sz) < mode
    sz = [sz, ones(1, mode - numel(sz))];
end
nm = numel(sz);

perm = [mode, 1:mode-1, mode+1:nm];
Xp = permute(X, perm);
old_dim = sz(mode);
Xp = reshape(Xp, old_dim, []);  % old_dim x prod(others)

if size(U,2) ~= old_dim
    if size(U,1) == old_dim
        U = U.';    % 自动转置
    else
        error('Dimension mismatch in tensor_multiply_mode: U(%d,%d) vs Xp(%d,%d)', ...
              size(U,1), size(U,2), size(Xp,1), size(Xp,2));
    end
end

Yp = U * Xp;  % new_dim x prod(others)

others = [1:mode-1, mode+1:nm];
new_sz = sz; new_sz(mode) = size(U,1);
Yp = reshape(Yp, [new_sz(mode), sz(others)]);
invperm = zeros(1,nm); invperm(perm) = 1:nm;
Y = ipermute(Yp, invperm);
end

function M = ortho(M), [M,~]=qr(M,0); end


%% ============ PDF / 似然 / 差分 / 轨迹生成 ============

function pdf = build_initial_pdf_4d(xg,yg,vxg,vyg, mu, Sigma)
N = [numel(xg), numel(yg), numel(vxg), numel(vyg)];
pdf = zeros(N);
Sinv = inv(Sigma);
for i=1:N(1)
for j=1:N(2)
for k=1:N(3)
for l=1:N(4)
    z = [xg(i); yg(j); vxg(k); vyg(l)];
    d = z - mu;
    pdf(i,j,k,l) = exp(-0.5 * (d.'*Sinv*d));
end
end
end
end
end

function like = construct_anisotropic_likelihood(xg,yg,vxg,vyg, obs, sigma, N)
[X,Y,~,~] = ndgrid(xg,yg,vxg,vyg);
posL = exp(-0.5*((X-obs(1)).^2 + (Y-obs(2)).^2)/sigma^2);
like = posL; like = like / sum(like(:));
end

function D = create_diff_matrix(N, h)
D = zeros(N,N);
D(1,1)=-1/h; D(1,2)=1/h;
for i=2:N-1
    D(i,i-1)=-1/(2*h); D(i,i+1)=1/(2*h);
end
D(N,N-1)=-1/h; D(N,N)=1/h;
end

function [true_state, obs] = gen_true_obs_cv(T, dt, sigma_proc, sigma_obs) %#ok<DEFNU>
% 保留 CV 生成函数以备以后使用（当前未调用）
true_state = zeros(4,T);
true_state(:,1) = [0;0; 1;1];
for t=2:T
    true_state(1,t)=true_state(1,t-1)+true_state(3,t-1)*dt + sigma_proc*randn*dt;
    true_state(2,t)=true_state(2,t-1)+true_state(4,t-1)*dt + sigma_proc*randn*dt;
    true_state(3,t)=true_state(3,t-1)+sigma_proc*randn*dt;
    true_state(4,t)=true_state(4,t-1)+sigma_proc*randn*dt;
end
obs = true_state(1:2,:) + sigma_obs*randn(2,T);
end

function [true_state, obs] = gen_true_obs_ct(T, dt, omega, sigma_proc, sigma_obs)
true_state = zeros(4,T);
true_state(:,1) = [0;0; 2;0];
if omega*dt == 0, F=eye(4); else
    s=sin(omega*dt); c=cos(omega*dt);
    F = [1,0, s/omega, -(1-c)/omega;
         0,1, (1-c)/omega, s/omega;
         0,0, c,-s;
         0,0, s, c];
end
for t=2:T
    true_state(:,t) = F*true_state(:,t-1) + sigma_proc*randn(4,1)*dt;
end
obs = true_state(1:2,:) + sigma_obs*randn(2,T);
end


%% ============ 绘图与表 ============

function plot_compare_all(results, meths, scenes) %#ok<INUSD>
% 只绘制 CT 场景：
% 1) CT 位置 RMSE 主图 + 放大窗口
% 2) 运行时（总时间）
% 3) 运行时（平均每步）

%% --- CT: Position RMSE + 放大区域 ---
figure(201); clf;

% 主坐标轴
axMain = axes('Position',[0.13 0.11 0.775 0.815]);
hold(axMain, 'on'); grid(axMain, 'on');

for m=1:numel(meths)
    tag = meths{m}.tag;
    plot(axMain, results.CT.(tag).pos_rmse, ...
        'LineWidth',1.8,'DisplayName',tag);
end

xlabel(axMain,'Time step');
ylabel(axMain,'Position RMSE');
title(axMain,'CT: Position RMSE');
legend(axMain,'Location','best');

% ---- 放大窗口（inset axes）----
axInset = axes('Position',[0.205357 0.266667 0.3375 0.35]);
hold(axInset, 'on'); grid(axInset,'on');

for m=1:numel(meths)
    tag = meths{m}.tag;
    plot(axInset, results.CT.(tag).pos_rmse, 'LineWidth',1.4);
end

%放大范围
xlim(axInset, [20 40]);     % 时间步 20~40
ylim(axInset, [-10 20]);    % RMSE -10~20
set(axInset,'Box','on');

%% --- 运行时（总时长） ---
figure(203); clf;
cats = categorical({meths{1}.tag, meths{2}.tag, meths{3}.tag});
cats = reordercats(cats, cellstr(cats));

bar([ results.CT.(meths{1}.tag).time; ...
      results.CT.(meths{2}.tag).time; ...
      results.CT.(meths{3}.tag).time ]);

set(gca,'XTickLabel',cellstr(cats));
grid on;
legend({'CT total(s)'},'Location','best');
ylabel('Seconds');
title('Runtime Comparison (CT only)');

%% --- 每步平均运行时 ---
figure(204); clf;

bar([ results.CT.(meths{1}.tag).avg_time; ...
      results.CT.(meths{2}.tag).avg_time; ...
      results.CT.(meths{3}.tag).avg_time ]);

set(gca,'XTickLabel',cellstr(cats));
grid on;
legend({'CT avg(s/step)'},'Location','best');
ylabel('Seconds/step');
title('Average Runtime per Step (CT only)');
end


function print_table(results, meths, scenes)
fprintf('\n===== Summary Table (CT only) =====\n');
fprintf('%-6s | %-8s | %-12s | %-12s | %-9s | %-10s\n', ...
    'Scene','Method','Mean Pos RMSE','Mean Vel RMSE','Total(s)','Avg/step');
fprintf(repmat('-',1,68)); fprintf('\n');
for s=1:numel(scenes)
    sc = scenes{s}.name;
    for m=1:numel(meths)
        tag = meths{m}.tag;
        mpos = mean(results.(sc).(tag).pos_rmse);
        mvel = mean(results.(sc).(tag).vel_rmse);
        tot  = results.(sc).(tag).time;
        avg  = results.(sc).(tag).avg_time;
        fprintf('%-6s | %-8s | %12.4f | %12.4f | %9.3f | %10.4f\n', ...
            sc, tag, mpos, mvel, tot, avg);
    end
end
end
