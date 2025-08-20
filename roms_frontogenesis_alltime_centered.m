%% ================== ROMS Frontogenesis 全时序计算（中心差分版） ==================
% - 在 rho 点上使用中心差分 × 本格 pm/pn 计算梯度（边界一阶）
% - u,v 插值到 rho 点
% - 计算温度/盐度 frontogenesis：总公式（=分解公式）且带负号，旋转项为 0
% - 对所有时间步进行计算，结果分别按变量保存至 MAT 文件
% ============================================================================

clear; clc;

%% 1. 基本配置
nc_file     = 'roms_avg_00001_200m.nc';   % NetCDF 文件
depth_index = 1;                           % 垂向层索引（示例：第 1 层）
out_dir     = fullfile(pwd, 'frontogenesis_out_centered');
if ~exist(out_dir, 'dir'); mkdir(out_dir); end

%% 2. 读取网格与时间
lon_rho = ncread(nc_file, 'lon_rho');
lat_rho = ncread(nc_file, 'lat_rho');
pm      = ncread(nc_file, 'pm');   % ≈ 1/Δx
pn      = ncread(nc_file, 'pn');   % ≈ 1/Δy
ocean_time = ncread(nc_file, 'ocean_time');
nt = numel(ocean_time);

[Im, Jm] = size(lon_rho);  % 与 rho 网格一致

%% 3. 预分配输出（rho×rho×time）
FT_total_all    = nan(Im, Jm, nt, 'double');
FT_stretch_all  = nan(Im, Jm, nt, 'double');
FT_shear_all    = nan(Im, Jm, nt, 'double');
FT_rot_all      = nan(Im, Jm, nt, 'double');

FS_total_all    = nan(Im, Jm, nt, 'double');
FS_stretch_all  = nan(Im, Jm, nt, 'double');
FS_shear_all    = nan(Im, Jm, nt, 'double');
FS_rot_all      = nan(Im, Jm, nt, 'double');

% 可选：诊断量
zeta_all   = nan(Im, Jm, nt, 'double');
strain_all = nan(Im, Jm, nt, 'double');

fprintf('Computing centered-gradient frontogenesis for %d time steps (depth=%d) ...\n', nt, depth_index);
tic;

%% 4. 时间循环
for it = 1:nt
    % 4.1 读取单时刻单层切片
    u_now    = ncread(nc_file, 'u',    [1 1 depth_index it], [Inf Inf 1 1]); % [xi_u, eta_u]
    v_now    = ncread(nc_file, 'v',    [1 1 depth_index it], [Inf Inf 1 1]); % [xi_v, eta_v]
    temp_now = ncread(nc_file, 'temp', [1 1 depth_index it], [Inf Inf 1 1]); % [xi_rho, eta_rho]
    salt_now = ncread(nc_file, 'salt', [1 1 depth_index it], [Inf Inf 1 1]); % [xi_rho, eta_rho]

    % 4.2 u,v 插值到 rho 点，并复制补边
    u_rho = 0.5 .* (u_now(1:end-1, :) + u_now(2:end, :));
    v_rho = 0.5 .* (v_now(:, 1:end-1) + v_now(:, 2:end));
    u_rho = [u_rho; u_rho(end, :)];     % 复制最后一行
    v_rho = [v_rho, v_rho(:, end)];     % 复制最后一列

    % 4.3 梯度（中心差分 × 本格 pm/pn；边界一阶）
    [dTdx, dTdy] = grad_rho_centered(temp_now, pm, pn);
    [dSdx, dSdy] = grad_rho_centered(salt_now, pm, pn);
    [dudx, dudy] = grad_rho_centered(u_rho,    pm, pn);
    [dvdx, dvdy] = grad_rho_centered(v_rho,    pm, pn);

    % 4.4 相对涡度/应变率
    zeta = dvdx - dudy;                   % 相对涡度
    sn   = dudx - dvdy;                   % 法向应变
    ss   = dvdx + dudy;                   % 切向应变
    strain = sqrt(sn.^2 + ss.^2);

    % 4.5 Temperature Frontogenesis（单位：d|∇T|/dt）
    gradT = sqrt(dTdx.^2 + dTdy.^2);
    denomT = gradT + eps;

    % 正确的总公式（= 分解公式；带负号）
    FT_total   = - ( dTdx .* (dudx .* dTdx + dvdx .* dTdy) ...
                   + dTdy .* (dudy .* dTdx + dvdy .* dTdy) ) ./ denomT;

    FT_stretch = - (dTdx.^2 .* dudx + dTdy.^2 .* dvdy) ./ denomT;
    FT_shear   = - ((dTdx .* dTdy) .* (dudy + dvdx))   ./ denomT;
    FT_rot     = zeros(Im, Jm, 'like', FT_total);      % 旋转项不贡献

    % 4.6 Salinity Frontogenesis（单位：d|∇S|/dt）
    gradS = sqrt(dSdx.^2 + dSdy.^2);
    denomS = gradS + eps;

    FS_total   = - ( dSdx .* (dudx .* dSdx + dvdx .* dSdy) ...
                   + dSdy .* (dudy .* dSdx + dvdy .* dSdy) ) ./ denomS;

    FS_stretch = - (dSdx.^2 .* dudx + dSdy.^2 .* dvdy) ./ denomS;
    FS_shear   = - ((dSdx .* dSdy) .* (dudy + dvdx))   ./ denomS;
    FS_rot     = zeros(Im, Jm, 'like', FS_total);

    % 4.7 存储
    FT_total_all(:, :, it)   = FT_total;
    FT_stretch_all(:, :, it) = FT_stretch;
    FT_shear_all(:, :, it)   = FT_shear;
    FT_rot_all(:, :, it)     = FT_rot;

    FS_total_all(:, :, it)   = FS_total;
    FS_stretch_all(:, :, it) = FS_stretch;
    FS_shear_all(:, :, it)   = FS_shear;
    FS_rot_all(:, :, it)     = FS_rot;

    zeta_all(:, :, it)   = zeta;
    strain_all(:, :, it) = strain;

    if mod(it, max(1, floor(nt/10))) == 0 || it == 1
        fprintf('  processed %d / %d (%.1f%%)\n', it, nt, 100*it/nt);
    end
end

elapsed = toc;
fprintf('All done in %.2f s. Saving...\n', elapsed);

%% 5. 保存结果（分文件）
save(fullfile(out_dir, 'grid_time.mat'), 'lon_rho', 'lat_rho', 'pm', 'pn', 'ocean_time', 'depth_index', '-v7.3');

save(fullfile(out_dir, 'FT_total.mat'),   'FT_total_all',   '-v7.3');
save(fullfile(out_dir, 'FT_stretch.mat'), 'FT_stretch_all', '-v7.3');
save(fullfile(out_dir, 'FT_shear.mat'),   'FT_shear_all',   '-v7.3');
save(fullfile(out_dir, 'FT_rot.mat'),     'FT_rot_all',     '-v7.3');

save(fullfile(out_dir, 'FS_total.mat'),   'FS_total_all',   '-v7.3');
save(fullfile(out_dir, 'FS_stretch.mat'), 'FS_stretch_all', '-v7.3');
save(fullfile(out_dir, 'FS_shear.mat'),   'FS_shear_all',   '-v7.3');
save(fullfile(out_dir, 'FS_rot.mat'),     'FS_rot_all',     '-v7.3');

save(fullfile(out_dir, 'diagnostics.mat'), 'zeta_all', 'strain_all', '-v7.3');

fprintf('Saved results to: %s\n', out_dir);

%% ======================= 本地函数 =======================
function [dFdx, dFdy] = grad_rho_centered(F, pm, pn)
% 在 rho 点上使用中心差分 × 本格 pm/pn 计算水平梯度；边界使用一阶差分
    [Im, Jm] = size(F);
    dFdx = zeros(Im, Jm, 'like', F);
    dFdy = zeros(Im, Jm, 'like', F);

    % x (ξ) 方向
    if Im >= 3
        dFdx(2:Im-1, :) = 0.5 .* (F(3:Im, :) - F(1:Im-2, :)) .* pm(2:Im-1, :);
    end
    dFdx(1, :)  = (F(2, :)    - F(1, :))    .* pm(1, :);
    dFdx(Im, :) = (F(Im, :)   - F(Im-1, :)) .* pm(Im, :);

    % y (η) 方向
    if Jm >= 3
        dFdy(:, 2:Jm-1) = 0.5 .* (F(:, 3:Jm) - F(:, 1:Jm-2)) .* pn(:, 2:Jm-1);
    end
    dFdy(:, 1)  = (F(:, 2)    - F(:, 1))    .* pn(:, 1);
    dFdy(:, Jm) = (F(:, Jm)   - F(:, Jm-1)) .* pn(:, Jm);
end

