%% ================== ROMS Frontogenesis 全时序计算（中心差分版） ==================
% - 在 rho 点上使用中心差分 × 本格 pm/pn 计算梯度（边界一阶）
% - u,v 插值到 rho 点
% - 计算温度/盐度 frontogenesis：总公式（=分解公式）且带负号，旋转项为 0
% - 新增：计算密度（GSW：p=gsw_p_from_z，rho=gsw_rho），密度梯度与密度 frontogenesis
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

% 静态地形与 s-坐标参数（用于 z_rho 与压力计算）
h = ncread(nc_file, 'h');
hc = 0;  try, hc = double(ncread(nc_file, 'hc')); end
vt = 2;  try, vt = double(ncread(nc_file, 'Vtransform')); end
% s 坐标分布（若文件含 s_rho / Cs_r 则读取）
s_rho_levels = []; Cs_r_levels = [];
try, s_rho_levels = ncread(nc_file, 's_rho'); end
try, Cs_r_levels  = ncread(nc_file, 'Cs_r');  end

% GSW 工具箱检查
if ~(exist('gsw_p_from_z','file')==2 && exist('gsw_SA_from_SP','file')==2 && ...
      exist('gsw_CT_from_pt','file')==2 && exist('gsw_rho','file')==2)
    error('GSW toolbox not found on MATLAB path. Please add GSW Oceanographic Toolbox.');
end

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

% 密度锋与 frontogenesis（rho）
FR_total_all    = nan(Im, Jm, nt, 'double');
FR_stretch_all  = nan(Im, Jm, nt, 'double');
FR_shear_all    = nan(Im, Jm, nt, 'double');
FR_rot_all      = nan(Im, Jm, nt, 'double');
gradR_all       = nan(Im, Jm, nt, 'double');

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

    % 4.3b 密度（GSW）：pressure -> SA -> CT -> rho
    % 读取海表高度（zeta）并计算本层 z_rho
    zeta_now = ncread(nc_file, 'zeta', [1 1 it], [Inf Inf 1]);
    % 优先直接读取 z_rho 层；若不存在则用 s-坐标重构
    z_layer = [];
    try
        z_layer = ncread(nc_file, 'z_rho', [1 1 depth_index it], [Inf Inf 1 1]);
    catch
        % 无 z_rho 变量时，用 s-坐标参数重构
        if isempty(s_rho_levels) || isempty(Cs_r_levels)
            error('z_rho not found and s-coordinate parameters missing (s_rho/Cs_r).');
        end
        s  = double(s_rho_levels(depth_index));
        Cs = double(Cs_r_levels(depth_index));
        z_layer = compute_z_rho_layer(h, zeta_now, s, Cs, hc, vt);
    end

    % 压力（dbar）：z 向下为负
    p_layer = gsw_p_from_z(z_layer, lat_rho);

    % 绝对盐度（SA）与保守温度（CT），假设 temp 为位温（theta）
    SA = gsw_SA_from_SP(salt_now, p_layer, lon_rho, lat_rho);
    CT = gsw_CT_from_pt(SA, temp_now);
    rho_now = gsw_rho(SA, CT, p_layer);

    % 密度梯度与密度 frontogenesis
    [dRdx, dRdy] = grad_rho_centered(rho_now, pm, pn);
    gradR = sqrt(dRdx.^2 + dRdy.^2);
    denomR = gradR + eps;
    FR_total = - ( dRdx .* (dudx .* dRdx + dvdx .* dRdy) ...
                 + dRdy .* (dudy .* dRdx + dvdy .* dRdy) ) ./ denomR;
    FR_stretch = - (dRdx.^2 .* dudx + dRdy.^2 .* dvdy) ./ denomR;
    FR_shear   = - ((dRdx .* dRdy) .* (dudy + dvdx))   ./ denomR;
    FR_rot     = zeros(Im, Jm, 'like', FR_total);

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

    FR_total_all(:, :, it)   = FR_total;
    FR_stretch_all(:, :, it) = FR_stretch;
    FR_shear_all(:, :, it)   = FR_shear;
    FR_rot_all(:, :, it)     = FR_rot;
    gradR_all(:, :, it)      = gradR;

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

% 密度 frontogenesis 输出
save(fullfile(out_dir, 'FR_total.mat'),   'FR_total_all',   '-v7.3');
save(fullfile(out_dir, 'FR_stretch.mat'), 'FR_stretch_all', '-v7.3');
save(fullfile(out_dir, 'FR_shear.mat'),   'FR_shear_all',   '-v7.3');
save(fullfile(out_dir, 'FR_rot.mat'),     'FR_rot_all',     '-v7.3');
save(fullfile(out_dir, 'density_front_strength.mat'), 'gradR_all', '-v7.3');

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

function z = compute_z_rho_layer(h, zeta, s, Cs, hc, vt)
% 根据 ROMS s-坐标与海表高度重构单层 z_rho（m，向下为负）
    if vt == 1
        % Song & Haidvogel (1994) 变换
        S = (hc.*s + h.*Cs) ./ (hc + h);
        z = zeta + (zeta + h) .* S;
    else
        % Shchepetkin (2005) 变换（ROMS 新版，Vtransform=2）
        z0 = hc.*s + h.*Cs;
        z  = z0 + zeta .* (1 + z0 ./ h);
    end
end

