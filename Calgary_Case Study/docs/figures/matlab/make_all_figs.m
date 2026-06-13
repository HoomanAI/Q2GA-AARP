% make_all_figs.m
% Regenerates all Calgary case-study figures (white background) as MATLAB
% .fig files from the .mat data exported by src/make_figures.py and
% src/make_problem_figures.py.
%
% Usage: open MATLAB, cd to this folder, run:  make_all_figs

clear; clc;
data_dir = fullfile('..', 'mat');
out_dir  = fullfile('..', 'fig');
if ~exist(out_dir, 'dir'); mkdir(out_dir); end

algos  = {'GA','SA','QGA','Q2GA'};
colors = [0.122 0.467 0.706;   % GA
          1.000 0.498 0.055;   % SA
          0.173 0.627 0.173;   % QGA
          0.839 0.153 0.157];  % Q2GA
type_colors = [0.839 0.153 0.157;   % Type 1 critical
                1.000 0.498 0.055;  % Type 2 moderate
                0.173 0.627 0.173]; % Type 3 minor
type_labels = {'Critical (Type 1)','Moderate (Type 2)','Minor (Type 3)'};

%% 1. Convergence (shaded mean +/- std)
d = load(fullfile(data_dir, 'convergence.mat'));
f = figure('Color','w','Position',[100 100 800 550]);
hold on; box on; grid on;
h = gobjects(1,numel(algos));
for a = 1:numel(algos)
    alg = algos{a};
    gens = double(d.([alg '_gens']));
    m    = double(d.([alg '_mean']));
    sd   = double(d.([alg '_std']));
    fill([gens, fliplr(gens)], [m-sd, fliplr(m+sd)], colors(a,:), ...
         'FaceAlpha', 0.18, 'EdgeColor', 'none', 'HandleVisibility','off');
    h(a) = plot(gens, m, 'Color', colors(a,:), 'LineWidth', 2, 'DisplayName', alg);
end
xlabel('Generation'); ylabel('Best fitness (objective value)');
title('Convergence comparison -- Calgary case study (N=25 patients)');
legend(h, 'Location','best');
set(gca,'Color','w');
savefig(f, fullfile(out_dir, 'convergence.fig'));
close(f);

%% 2. Boxplot of final fitness
d = load(fullfile(data_dir, 'boxplot.mat'));
f = figure('Color','w','Position',[100 100 800 550]);
data = []; grp = [];
for a = 1:numel(algos)
    v = double(d.(algos{a}))(:);
    data = [data; v];
    grp  = [grp; repmat(a, numel(v), 1)];
end
boxplot(data, grp, 'Labels', algos, 'Colors', 'k');
h = findobj(gca,'Tag','Box');
for j = 1:length(h)
    patch(get(h(j),'XData'), get(h(j),'YData'), colors(numel(algos)-j+1,:), 'FaceAlpha', 0.5);
end
ylabel('Final best fitness');
title('Final-fitness distribution -- Calgary case study (10 seeds)');
set(gca,'Color','w');
savefig(f, fullfile(out_dir, 'boxplot.fig'));
close(f);

%% 3. Objective component bar chart
comp_labels = {'C1 (critical)','C2 (moderate)','C3 (minor)','Penalty-1','Penalty-2'};
d = load(fullfile(data_dir, 'objectives.mat'));
f = figure('Color','w','Position',[100 100 900 550]);
M = [];
for a = 1:numel(algos)
    M = [M; double(d.([algos{a} '_means']))];
end
b = bar(M', 'grouped');
for a = 1:numel(algos)
    b(a).FaceColor = colors(a,:);
end
set(gca, 'XTickLabel', comp_labels);
ylabel('Mean value');
title('Objective-component breakdown -- Calgary case study (mean over 10 seeds)');
legend(algos, 'Location','best');
set(gca,'Color','w');
savefig(f, fullfile(out_dir, 'objectives.fig'));
close(f);

%% 4. Runtime bar chart
d = load(fullfile(data_dir, 'runtime_bar.mat'));
f = figure('Color','w','Position',[100 100 800 550]);
means = double(d.mean_runtime);
stds  = double(d.std_runtime);
b = bar(means); hold on;
for a = 1:numel(algos)
    b.FaceColor = 'flat';
    b.CData(a,:) = colors(a,:);
end
errorbar(1:numel(algos), means, stds, 'k.', 'LineWidth', 1);
set(gca, 'XTickLabel', algos);
ylabel('Mean runtime (s)');
title('Runtime comparison -- Calgary case study (10 seeds)');
set(gca,'Color','w');
savefig(f, fullfile(out_dir, 'runtime_bar.fig'));
close(f);

%% 5. Instance layout (patients + hospitals)
d = load(fullfile(data_dir, 'instance_layout_xy.mat'));
f = figure('Color','w','Position',[100 100 800 700]);
hold on; box on; grid on; axis equal;
ptype = double(d.patient_type);
for t = 1:3
    mask = ptype == t;
    scatter(d.patient_x(mask), d.patient_y(mask), 60, type_colors(t,:), 'filled', ...
            'MarkerEdgeColor','k', 'DisplayName', type_labels{t});
end
scatter(d.hospital_x, d.hospital_y, 400, [1 0.84 0], 'p', 'filled', ...
        'MarkerEdgeColor','k', 'LineWidth',1.2, 'DisplayName','Hospitals');
xlabel('x (km, east of Calgary city centre)');
ylabel('y (km, north of Calgary city centre)');
title('Calgary case-study instance layout');
legend('Location','best');
set(gca,'Color','w');
savefig(f, fullfile(out_dir, 'instance_layout_xy.fig'));
close(f);

%% 6. Patient demographics
d = load(fullfile(data_dir, 'patient_demographics.mat'));
f = figure('Color','w','Position',[100 100 700 550]);
counts = double(d.counts);
b = bar(counts);
b.FaceColor = 'flat';
for t = 1:3
    b.CData(t,:) = type_colors(t,:);
end
set(gca, 'XTickLabel', type_labels);
ylabel('Number of patients');
title('Patient-type composition -- Calgary case study');
set(gca,'Color','w');
savefig(f, fullfile(out_dir, 'patient_demographics.fig'));
close(f);

%% 7. Service / drop-off time histograms
d = load(fullfile(data_dir, 'service_dropoff_times.mat'));
f = figure('Color','w','Position',[100 100 1100 500]);
subplot(1,2,1);
histogram(double(d.service_time), 10, 'FaceColor', [0.122 0.467 0.706], 'FaceAlpha', 0.8);
xlabel('Service time s_i (minutes)'); ylabel('Number of patients');
title('On-scene service-time distribution'); set(gca,'Color','w');
subplot(1,2,2);
histogram(double(d.dropoff_time), 10, 'FaceColor', [1.000 0.498 0.055], 'FaceAlpha', 0.8);
xlabel('Hospital drop-off time (minutes)'); ylabel('Number of patients');
title('Drop-off-time distribution (Type 1/2 only)'); set(gca,'Color','w');
savefig(f, fullfile(out_dir, 'service_dropoff_times.fig'));
close(f);

%% 8. Time-window thresholds
d = load(fullfile(data_dir, 'time_window_thresholds.mat'));
f = figure('Color','w','Position',[100 100 800 550]);
hold on; box on; grid on;
positions = [1 1.7 3 3.7];
groups = {double(d.a1_type1)(:), double(d.a2_type1)(:), double(d.a1_type2)(:), double(d.a2_type2)(:)};
data = []; grp = [];
for i = 1:4
    v = groups{i};
    data = [data; v];
    grp  = [grp; repmat(positions(i), numel(v), 1)];
end
boxplot(data, grp, 'Positions', positions, 'Widths', 0.6, 'Colors', 'k');
h = findobj(gca,'Tag','Box');
box_colors = [0.839 0.153 0.157; 0.122 0.467 0.706; 0.839 0.153 0.157; 0.122 0.467 0.706];
for j = 1:length(h)
    patch(get(h(j),'XData'), get(h(j),'YData'), box_colors(length(h)-j+1,:), 'FaceAlpha', 0.6);
end
set(gca, 'XTick', [1.35 4.05], 'XTickLabel', {'Type 1 (critical)','Type 2 (moderate)'});
ylabel('Threshold (minutes from dispatch)');
title('Semi-soft time-window thresholds -- Calgary case study');
set(gca,'Color','w');
savefig(f, fullfile(out_dir, 'time_window_thresholds.fig'));
close(f);

%% 9. Penalty function shape
d = load(fullfile(data_dir, 'penalty_function_shape.mat'));
f = figure('Color','w','Position',[100 100 850 550]);
hold on; box on; grid on;
arrival = double(d.arrival);
plot(arrival, double(d.penalty_type1), 'Color', type_colors(1,:), 'LineWidth', 2, ...
     'DisplayName', sprintf('Type 1 (critical), a1=%.1f, a2=%.1f', double(d.a1_type1), double(d.a2_type1)));
plot(arrival, double(d.penalty_type2), 'Color', type_colors(2,:), 'LineWidth', 2, ...
     'DisplayName', sprintf('Type 2 (moderate), a1=%.1f, a2=%.1f', double(d.a1_type2), double(d.a2_type2)));
xline(double(d.a1_type1), '--', 'Color', type_colors(1,:), 'Alpha', 0.5, 'HandleVisibility','off');
xline(double(d.a2_type1), ':',  'Color', type_colors(1,:), 'Alpha', 0.5, 'HandleVisibility','off');
xline(double(d.a1_type2), '--', 'Color', type_colors(2,:), 'Alpha', 0.5, 'HandleVisibility','off');
xline(double(d.a2_type2), ':',  'Color', type_colors(2,:), 'Alpha', 0.5, 'HandleVisibility','off');
xlabel('Patient arrival time a_i (minutes from dispatch)');
ylabel('Delay penalty');
title('Semi-soft time-window delay-penalty function -- Calgary case study');
legend('Location','best');
set(gca,'Color','w');
savefig(f, fullfile(out_dir, 'penalty_function_shape.fig'));
close(f);

%% 10. Goal-function weights
d = load(fullfile(data_dir, 'goal_function_weights.mat'));
f = figure('Color','w','Position',[100 100 700 550]);
weights = [double(d.w1) double(d.w2) double(d.w3)];
b = bar(weights);
b.FaceColor = 'flat';
b.CData = type_colors;
set(gca, 'XTickLabel', {'w1.C1 (critical)','w2.C2 (moderate)','w3.C3 (minor)'});
ylabel('Weight'); ylim([0 0.6]);
title({'Goal-function weighted-sum structure (Eq. 6-8)', ...
       'fitness = w1.C1 + w2.C2 + w3.C3 + penalty1 + penalty2 + infeasibility'});
set(gca,'Color','w');
savefig(f, fullfile(out_dir, 'goal_function_weights.fig'));
close(f);

%% 11. Disruption scenario
d = load(fullfile(data_dir, 'disruption_scenario.mat'));
f = figure('Color','w','Position',[100 100 1000 500]);
theta = double(d.theta);
n = numel(theta);
b = bar(0:n-1, theta);
b.FaceColor = 'flat';
for i = 1:n
    if theta(i) > 0
        b.CData(i,:) = [0.839 0.153 0.157];
    else
        b.CData(i,:) = [0.8 0.8 0.8];
    end
end
xlabel('Patient index'); ylabel('Realised disruption buffer \theta_i (minutes)');
title(sprintf('Realised disruption scenario -- Calgary case study (Gamma=%d of %d, MTTR=%g)', ...
      double(d.gamma_budget), n, double(d.mttr)));
set(gca,'Color','w');
savefig(f, fullfile(out_dir, 'disruption_scenario.fig'));
close(f);

%% 12. QoS vs Gamma
d = load(fullfile(data_dir, 'qos_vs_gamma.mat'));
f = figure('Color','w','Position',[100 100 1500 450]);
gamma = double(d.gamma_frac);

subplot(1,3,1); hold on; box on; grid on;
fm = double(d.fitness_mean); fs = double(d.fitness_std);
fill([gamma, fliplr(gamma)], [fm-fs, fliplr(fm+fs)], [0.122 0.467 0.706], 'FaceAlpha', 0.15, 'EdgeColor','none');
plot(gamma, fm, '-o', 'Color', [0.122 0.467 0.706], 'LineWidth', 2);
xline(0.20, '--', 'Color', [0.5 0.5 0.5]);
xlabel('Gamma (fraction of patients affected)'); ylabel('Mean best fitness (Q2GA)');
title('Overall fitness vs. Gamma'); set(gca,'Color','w');

subplot(1,3,2); hold on; box on; grid on;
pm = double(d.penalty_mean); ps = double(d.penalty_std);
fill([gamma, fliplr(gamma)], [pm-ps, fliplr(pm+ps)], [0.839 0.153 0.157], 'FaceAlpha', 0.15, 'EdgeColor','none');
plot(gamma, pm, '-o', 'Color', [0.839 0.153 0.157], 'LineWidth', 2);
xline(0.20, '--', 'Color', [0.5 0.5 0.5]);
xlabel('Gamma (fraction of patients affected)'); ylabel('Mean total delay penalty');
title('Service-quality penalty vs. Gamma'); set(gca,'Color','w');

subplot(1,3,3); hold on; box on; grid on;
plot(gamma, double(d.unserved_mean), '-o', 'Color', [0.580 0.404 0.741], 'LineWidth', 2);
xline(0.20, '--', 'Color', [0.5 0.5 0.5]);
xlabel('Gamma (fraction of patients affected)'); ylabel('Mean number of unserved patients');
title('Coverage failure vs. Gamma'); set(gca,'Color','w');

sgtitle('Impact of cyber-disruption breadth (Gamma) on quality of service -- Calgary case study (Q2GA, 5 seeds)');
savefig(f, fullfile(out_dir, 'qos_vs_gamma.fig'));
close(f);

%% 13. QoS vs MTTR (with availability axis)
d = load(fullfile(data_dir, 'qos_vs_mttr.mat'));
f = figure('Color','w','Position',[100 100 900 550]);
mttr = double(d.mttr);
fm = double(d.fitness_mean); fs = double(d.fitness_std);
pm = double(d.penalty_mean); ps = double(d.penalty_std);

yyaxis left;
hold on; box on; grid on;
fill([mttr, fliplr(mttr)], [fm-fs, fliplr(fm+fs)], [0.122 0.467 0.706], 'FaceAlpha', 0.15, 'EdgeColor','none');
plot(mttr, fm, '-o', 'Color', [0.122 0.467 0.706], 'LineWidth', 2, 'DisplayName','Mean best fitness (Q2GA)');
ylabel('Mean best fitness');
ax = gca; ax.YColor = [0.122 0.467 0.706];

yyaxis right;
fill([mttr, fliplr(mttr)], [pm-ps, fliplr(pm+ps)], [0.839 0.153 0.157], 'FaceAlpha', 0.12, 'EdgeColor','none');
plot(mttr, pm, '-s', 'Color', [0.839 0.153 0.157], 'LineWidth', 2, 'DisplayName','Mean total delay penalty');
ylabel('Mean total delay penalty (Penalty-1 + Penalty-2)');
ax.YColor = [0.839 0.153 0.157];

xlabel('MTTR (mean time to repair, minutes)');
title('Recovery time (MTTR) and resulting availability vs. quality of service -- Calgary case study (Q2GA, 5 seeds)');
legend('Location','best');
set(gca,'Color','w');
savefig(f, fullfile(out_dir, 'qos_vs_mttr.fig'));
close(f);

%% 14. Availability surface
d = load(fullfile(data_dir, 'availability_surface.mat'));
f = figure('Color','w','Position',[100 100 800 650]);
mttr_range = double(d.mttr_range);
mtbf_range = double(d.mtbf_range);
avail = double(d.availability);
[MTTR, MTBF] = meshgrid(mttr_range, mtbf_range);
contourf(MTTR, MTBF, avail, 20, 'LineColor','none');
colormap(redgreencmap_local());
cb = colorbar; cb.Label.String = 'System availability = MTBF / (MTBF + MTTR)';
hold on;
[C,h] = contour(MTTR, MTBF, avail, [0.7 0.8 0.9 0.95 0.98], 'k', 'LineWidth', 0.8);
clabel(C, h, 'FontSize', 8);
plot(double(d.default_mttr), double(d.default_mtbf), 'kx', 'MarkerSize', 14, 'LineWidth', 3, ...
     'DisplayName', sprintf('Calgary operating point (MTTR=%.0f, MTBF=%.0f)', double(d.default_mttr), double(d.default_mtbf)));
xlabel('MTTR (minutes)'); ylabel('MTBF (minutes)');
title('System availability as a function of MTTR and MTBF');
legend('Location','southeast');
set(gca,'Color','w');
savefig(f, fullfile(out_dir, 'availability_surface.fig'));
close(f);

%% 15. Cyber-risk degradation summary
d = load(fullfile(data_dir, 'cyber_risk_degradation_summary.mat'));
f = figure('Color','w','Position',[100 100 800 600]);
fitness_pct = double(d.fitness_pct);
penalty_pct = double(d.penalty_pct);
x = 1:numel(fitness_pct);
width = 0.35;
hold on; box on; grid on;
b1 = bar(x - width/2, fitness_pct, width, 'FaceColor', [0.122 0.467 0.706], 'DisplayName','Fitness degradation (%)');
b2 = bar(x + width/2, penalty_pct, width, 'FaceColor', [0.839 0.153 0.157], 'DisplayName','Total delay-penalty increase (%)');
yline(0, 'k', 'LineWidth', 0.8);
set(gca, 'XTick', x, 'XTickLabel', {'Gamma: 0.20 -> 0.50 N (wider attack surface)', 'MTTR: 10 -> 30 (slower recovery)'});
ylabel('Relative change vs. baseline (%)');
title('Quality-of-service degradation under worsening cyber-risk conditions -- Calgary case study (Q2GA, 5 seeds)');
legend('Location','best');
set(gca,'Color','w');
savefig(f, fullfile(out_dir, 'cyber_risk_degradation_summary.fig'));
close(f);

disp('All .fig files written to figures/fig');

%% Local helper: dependency-free red-yellow-green colormap
function cmap = redgreencmap_local()
    n = 256;
    half = floor(n/2);
    r1 = linspace(0.84, 1.0, half)';
    g1 = linspace(0.15, 1.0, half)';
    b1 = linspace(0.16, 0.0, half)';
    r2 = linspace(1.0, 0.0, n-half)';
    g2 = linspace(1.0, 0.6, n-half)';
    b2 = linspace(0.0, 0.0, n-half)';
    cmap = [r1 g1 b1; r2 g2 b2];
end
