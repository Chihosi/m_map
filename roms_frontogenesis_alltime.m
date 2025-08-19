%% ================== ROMS Frontogenesis 全时序计算脚本 ==================
% 说明：
% - 使用 ROMS 网格自带的 pm, pn (分别≈ dξ/dx, dη/dy) 在 rho 点上计算水平梯度
% - u,v 先插值到 rho 点
% - 分别计算 temp 和 salt 的 frontogenesis（总公式 & 分解公式）
% - 针对所有时间步循环计算；结果按变量分别保存到单独 MAT 文件
% - 仅计算单一垂向层（可配置 depth_index）
% =======================================================================

clear; clc;

%% 1. 基本配置
nc_file    = 'roms_avg_00001_200m.nc';  % NetCDF 文件路径
depth_index = 1;                         % 垂向层索引（与原脚本一致，示例取第1层）
save_dir   = fullfile(pwd, 'frontogenesis_out');
if ~exist(save_dir, 'dir'); mkdir(save_dir); end

%% 2. 读取静态网格与时间
lon_rho = ncread(nc_file, 'lon_rho');
lat_rho = ncread(nc_file, 'lat_rho');
pm      = ncread(nc_file, 'pm');    % dξ/dx（约等于 1/Δx）
pn      = ncread(nc_file, 'pn');    % dη/dy（约等于 1/Δy）

% 时间（假定存在 ocean_time 变量；若名称不同请相应更改）
ocean_time = ncread(nc_file, 'ocean_time');
num_time_steps = numel(ocean_time);

% 预估 rho 网格尺寸并初始化输出数组大小
[num_x_rho, num_y_rho] = size(lon_rho);

%% 3. 为所有时间步预分配输出数组（rho 点，第三维为 time）
FT_total  = nan(num_x_rho, num_y_rho, num_time_steps, 'double');
FT_decomp = nan(num_x_rho, num_y_rho, num_time_steps, 'double');
FS_total  = nan(num_x_rho, num_y_rho, num_time_steps, 'double');
FS_decomp = nan(num_x_rho, num_y_rho, num_time_steps, 'double');

% 如需保存梯度/剪切等中间量，可解注以下预分配
% dTdx_all = nan(num_x_rho, num_y_rho, num_time_steps, 'double');
% dTdy_all = nan(num_x_rho, num_y_rho, num_time_steps, 'double');
% dSdx_all = nan(num_x_rho, num_y_rho, num_time_steps, 'double');
% dSdy_all = nan(num_x_rho, num_y_rho, num_time_steps, 'double');
% dudx_all = nan(num_x_rho, num_y_rho, num_time_steps, 'double');
% dudy_all = nan(num_x_rho, num_y_rho, num_time_steps, 'double');
% dvdx_all = nan(num_x_rho, num_y_rho, num_time_steps, 'double');
% dvdy_all = nan(num_x_rho, num_y_rho, num_time_steps, 'double');

fprintf('Start frontogenesis computation across %d time steps (depth_index=%d) ...\n', num_time_steps, depth_index);
tic;

%% 4. 时间循环（单层）
for t_idx = 1:num_time_steps
    % 2D 切片读取（仅读当前时间步与指定垂向层）
    u_slice    = ncread(nc_file, 'u',    [1 1 depth_index t_idx], [Inf Inf 1 1]); % [xi_u, eta_u]
    v_slice    = ncread(nc_file, 'v',    [1 1 depth_index t_idx], [Inf Inf 1 1]); % [xi_v, eta_v]
    temp_slice = ncread(nc_file, 'temp', [1 1 depth_index t_idx], [Inf Inf 1 1]); % [xi_rho, eta_rho]
    salt_slice = ncread(nc_file, 'salt', [1 1 depth_index t_idx], [Inf Inf 1 1]); % [xi_rho, eta_rho]

    % 4.1 u,v 插值到 rho 点并补齐边界（保持与 rho 尺寸一致）
    u_rho = 0.5 .* (u_slice(1:end-1, :) + u_slice(2:end, :));
    v_rho = 0.5 .* (v_slice(:, 1:end-1) + v_slice(:, 2:end));
    u_rho = pad_last_row(u_rho);
    v_rho = pad_last_col(v_rho);

    % 4.2 标量梯度（在 rho 点，采用 pm/pn 权重的差分并补边）
    [dTdx, dTdy] = compute_gradients_on_rho(temp_slice, pm, pn);
    [dSdx, dSdy] = compute_gradients_on_rho(salt_slice, pm, pn);

    % 4.3 速度梯度（在 rho 点）
    [dudx, dudy] = compute_gradients_on_rho(u_rho, pm, pn);
    dvdx_core    = diff(v_rho, 1, 1) .* pm(1:end-1, :);
    dvdy_core    = diff(v_rho, 1, 2) .* pn(:, 1:end-1);
    dvdx = pad_last_row(dvdx_core);
    dvdy = pad_last_col(dvdy_core);

    % 4.4 =============== Frontogenesis (温度) =================
    % --- (1) 总公式（与原脚本保持一致） ---
    advT = (u_rho .* dTdx + v_rho .* dTdy);
    FT_total(:, :, t_idx) = - (dTdx .* advT) - (dTdy .* advT);

    % --- (2) 分解公式（与原脚本保持一致） ---
    stretch_T = - (dudx .* dTdx.^2 + dvdy .* dTdy.^2);
    shear_T   = - (dudy .* dTdx .* dTdy + dvdx .* dTdx .* dTdy);
    FT_decomp(:, :, t_idx) = stretch_T + shear_T; % rot=0

    % 4.5 =============== Frontogenesis (盐度) =================
    advS = (u_rho .* dSdx + v_rho .* dSdy);
    FS_total(:, :, t_idx) = - (dSdx .* advS) - (dSdy .* advS);

    stretch_S = - (dudx .* dSdx.^2 + dvdy .* dSdy.^2);
    shear_S   = - (dudy .* dSdx .* dSdy + dvdx .* dSdx .* dSdy);
    FS_decomp(:, :, t_idx) = stretch_S + shear_S; % rot=0

    % 如需保存中间量，解注并赋值
    % dTdx_all(:, :, t_idx) = dTdx;  dTdy_all(:, :, t_idx) = dTdy;
    % dSdx_all(:, :, t_idx) = dSdx;  dSdy_all(:, :, t_idx) = dSdy;
    % dudx_all(:, :, t_idx) = dudx;  dudy_all(:, :, t_idx) = dudy;
    % dvdx_all(:, :, t_idx) = dvdx;  dvdy_all(:, :, t_idx) = dvdy;

    if mod(t_idx, max(1, floor(num_time_steps/10))) == 0 || t_idx == 1
        fprintf('  processed %d / %d time steps (%.1f%%)\n', t_idx, num_time_steps, 100*t_idx/num_time_steps);
    end
end

elapsed_sec = toc;
fprintf('All time steps finished in %.2f s. Saving outputs...\n', elapsed_sec);

%% 5. 保存结果到不同的 MAT 文件（按变量分别保存）
% 网格与时间信息
save(fullfile(save_dir, 'grid_time.mat'), 'lon_rho', 'lat_rho', 'pm', 'pn', 'ocean_time', 'depth_index', '-v7.3');

% Frontogenesis（温度与盐度）
save(fullfile(save_dir, 'FT_total.mat'),  'FT_total',  '-v7.3');
save(fullfile(save_dir, 'FT_decomp.mat'), 'FT_decomp', '-v7.3');
save(fullfile(save_dir, 'FS_total.mat'),  'FS_total',  '-v7.3');
save(fullfile(save_dir, 'FS_decomp.mat'), 'FS_decomp', '-v7.3');

% 如需保存中间量，解注如下语句
% save(fullfile(save_dir, 'gradients_temp.mat'), 'dTdx_all', 'dTdy_all', '-v7.3');
% save(fullfile(save_dir, 'gradients_salt.mat'), 'dSdx_all', 'dSdy_all', '-v7.3');
% save(fullfile(save_dir, 'velocity_gradients.mat'), 'dudx_all', 'dudy_all', 'dvdx_all', 'dvdy_all', '-v7.3');

fprintf('Saved to folder: %s\n', save_dir);

%% 6.（可选）简单预览首个时间步
% figure;
% subplot(2,2,1); pcolor(lon_rho,lat_rho,FT_total(:,:,1)); shading interp; colorbar; title('Temp Frontogenesis (总公式) t=1');
% subplot(2,2,2); pcolor(lon_rho,lat_rho,FT_decomp(:,:,1)); shading interp; colorbar; title('Temp Frontogenesis (分解公式) t=1');
% subplot(2,2,3); pcolor(lon_rho,lat_rho,FS_total(:,:,1)); shading interp; colorbar; title('Salt Frontogenesis (总公式) t=1');
% subplot(2,2,4); pcolor(lon_rho,lat_rho,FS_decomp(:,:,1)); shading interp; colorbar; title('Salt Frontogenesis (分解公式) t=1');

%% ======================= 本地函数 =======================
function [dFdx, dFdy] = compute_gradients_on_rho(F_rho, pm, pn)
% 在 rho 点上使用 pm/pn 计算水平梯度；采用一阶差分并对 pm/pn 做邻格平均
    % ξ 方向梯度（对应 x）
    dFdx_core = 0.5 .* ((F_rho(2:end, :) - F_rho(1:end-1, :)) .* pm(2:end, :) + ...
                        (F_rho(2:end, :) - F_rho(1:end-1, :)) .* pm(1:end-1, :));
    dFdx = pad_last_row(dFdx_core);

    % η 方向梯度（对应 y）
    dFdy_core = 0.5 .* ((F_rho(:, 2:end) - F_rho(:, 1:end-1)) .* pn(:, 2:end) + ...
                        (F_rho(:, 2:end) - F_rho(:, 1:end-1)) .* pn(:, 1:end-1));
    dFdy = pad_last_col(dFdy_core);
end

function A_padded = pad_last_row(A)
% 复制最后一行，以补齐到 rho 网格尺寸
    A_padded = [A; A(end, :)];
end

function A_padded = pad_last_col(A)
% 复制最后一列，以补齐到 rho 网格尺寸
    A_padded = [A, A(:, end)];
end

