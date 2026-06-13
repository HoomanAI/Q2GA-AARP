% make_all_figs.m
% Regenerates all LA wildfire case-study figures (white background) as
% MATLAB .fig files from the .mat data exported by src/make_figures.py and
% src/make_problem_figures.py.
%
% Usage: open MATLAB, cd to this folder, run:  make_all_figs

clear; clc;
data_dir = fullfile('..', 'mat');
out_dir  = fullfile('..', 'fig');
if ~exist(out_dir, 'dir'); mkdir(out_dir); end

algos  = {'GA','SA','QGA','Q2GA'};
scenarios = {'baseline','wildfire'};
scen_labels = {'Baseline (normal operations)','Wildfire (Palisades Fire active)'};
colors = [0.122 0.467 0.706;   % GA
          1.000 0.498 0.055;   % SA
          0.173 0.627 0.173;   % QGA
          0.839 0.153 0.157];  % Q2GA
type_colors = [0.839 0.153 0.157;   % Type 1 critical
                1.000 0.498 0.055;  % Type 2 moderate
                0.173 0.627 0.173]; % Type 3 minor
type_labels = {'Critical (Type 1)','Moderate (Type 2)','Minor (Type 3)'};

%% 1. Convergence (shaded mean +/- std), baseline vs wildfire
d = load(fullfile(data_dir, 'convergence.mat'));
f = figure('Color','w','Position',[80 80 1400 550]);
for s = 1:2
    subplot(1,2,s); hold on; box on; grid on;
    scen = scenarios{s};
    h = gobjects(1,numel(algos));
    for a = 1:numel(algos)
        alg = algos{a};
        gens = double(d.([scen '_' alg '_gens']));
        m    = double(d.([scen '_' alg '_mean']));
        sd   = double(d.([scen '_' alg '_std']));
        fill([gens, fliplr(gens)], [m-sd, fliplr(m+sd)], colors(a,:), ...
             'FaceAlpha', 0.18, 'EdgeColor', 'none', 'HandleVisibility','off');
        h(a) = plot(gens, m, 'Color', colors(a,:), 'LineWidth', 2, 'DisplayName', alg);
    end
    xlabel('Generation'); ylabel('Best fitness (objective value)');
    title(scen_labels{s});
    legend(h, 'Location','best');
    set(gca,'Color','w');
end
sgtitle('Convergence comparison -- LA wildfire case study (N=25 patients)');
savefig(f, fullfile(out_dir, 'convergence.fig'));
close(f);

%% 2. Boxplot of final fitness, baseline vs wildfire
d = load(fullfile(data_dir, 'boxplot.mat'));
f = figure('Color','w','Position',[80 80 1300 550]);
for s = 1:2
    subplot(1,2,s); hold on; box on; grid on;
    scen = scenarios{s};
    data = []; grp = [];
    for a = 1:numel(algos)
        v = double(d.([scen '_' algos{a}]))(:);
        data = [data; v];
        grp  = [grp; repmat(a, numel(v), 1)];
    end
    boxplot(data, grp, 'Labels', algos, 'Colors', 'k');
    h = findobj(gca,'Tag','Box');
    for j = 1:length(h)
        patch(get(h(j),'XData'), get(h(j),'YData'), colors(numel(algos)-j+1,:), 'FaceAlpha', 0.5);
    end
    ylabel('Final best fitness');
    title(scen_labels{s});
    set(gca,'Color','w');
end
sgtitle('Final-fitness distribution -- LA wildfire case study (10 seeds)');
savefig(f, fullfile(out_dir, 'boxplot.fig'));
close(f);

%% 3. Objective component bar chart, baseline vs wildfire
d = load(fullfile(data_dir, 'objectives.mat'));
comp_labels = {'C1 (critical)','C2 (moderate)','C3 (minor)','Penalty-1','Penalty-2'};
f = figure('Color','w','Position',[80 80 1400 550]);
for s = 1:2
    subplot(1,2,s);
    scen = scenarios{s};
    M = [];
    for a = 1:numel(algos)
        M = [M; double(d.([scen '_' algos{a} '_means']))];
    end
    b = bar(M', 'grouped');
    for a = 1:numel(algos)
        b(a).FaceColor = colors(a,:);
    end
    set(gca, 'XTickLabel', comp_labels);
    ylabel('Mean value');
    title(scen_labels{s});
    legend(algos, 'Location','best');
    set(gca,'Color','w');
end
sgtitle('Objective-component breakdown -- LA wildfire case study (mean over 10 seeds)');
savefig(f, fullfile(out_dir, 'objectives.fig'));
close(f);

%% 4. Runtime bar chart (baseline)
d = load(fullfile(data_dir, 'runtime_bar.mat'));
f = figure('Color','w','Position',[100 100 800 550]);
means = double(d.mean_runtime);
stds  = double(d.std_runtime);
b = bar(means); hold on;
b.FaceColor = 'flat';
for a = 1:numel(algos)
    b.CData(a,:) = colors(a,:);
end
errorbar(1:numel(algos), means, stds, 'k.', 'LineWidth', 1);
set(gca, 'XTickLabel', algos);
ylabel('Mean runtime (s)');
title('Runtime comparison -- LA wildfire case study (10 seeds, baseline scenario)');
set(gca,'Color','w');
savefig(f, fullfile(out_dir, 'runtime_bar.fig'));
close(f);

%% 5. Wildfire impact summary (fitness / penalty / unserved degradation)
d = load(fullfile(data_dir, 'wildfire_impact_summary.mat'));
f = figure('Color','w','Position',[80 80 1300 550]);
fitness_pct = double(d.fitness_pct);
penalty_pct = double(d.penalty_pct);
unserved_delta = double(d.unserved_delta);
x = 1:numel(algos);
width = 0.35;

subplot(1,2,1); hold on; box on; grid on;
b1 = bar(x - width/2, fitness_pct, width, 'FaceColor', [0.122 0.467 0.706], 'DisplayName','Fitness increase (%)');
b2 = bar(x + width/2, penalty_pct, width, 'FaceColor', [0.839 0.153 0.157], 'DisplayName','Total delay-penalty increase (%)');
yline(0, 'k', 'LineWidth', 0.8);
set(gca, 'XTick', x, 'XTickLabel', algos);
ylabel('Relative change, baseline -> wildfire (%)');
title('Fitness and delay-penalty degradation');
legend('Location','best'); set(gca,'Color','w');

subplot(1,2,2); hold on; box on; grid on;
b = bar(x, unserved_delta);
b.FaceColor = 'flat';
for a = 1:numel(algos)
    b.CData(a,:) = colors(a,:);
end
set(gca, 'XTick', x, 'XTickLabel', algos);
ylabel('Increase in mean # unserved patients');
title('Additional unserved patients due to wildfire');
set(gca,'Color','w');

sgtitle('Impact of the Palisades Fire network/road outage on quality of service -- LA case study (10 seeds)');
savefig(f, fullfile(out_dir, 'wildfire_impact_summary.fig'));
close(f);

%% 6. Instance layout (patients + hospitals + fire zone)
d = load(fullfile(data_dir, 'instance_layout_xy.mat'));
f = figure('Color','w','Position',[100 100 800 700]);
hold on; box on; grid on; axis equal;
ptype = double(d.patient_type);
zone_idx = double(d.wildfire_zone_idx) + 1; % 0-based -> 1-based
zone_mask = false(numel(ptype),1);
zone_mask(zone_idx) = true;

theta_circ = linspace(0, 2*pi, 100);
cxy = double(d.wildfire_zone_center_xy);
r = double(d.wildfire_zone_radius_km);
fill(cxy(1) + r*cos(theta_circ), cxy(2) + r*sin(theta_circ), [1 0.27 0], ...
     'FaceAlpha', 0.10, 'EdgeColor', [1 0.27 0], 'LineStyle','--', 'LineWidth', 2, ...
     'DisplayName', 'Palisades Fire zone (approx.)');

for t = 1:3
    mask = (ptype == t) & ~zone_mask;
    scatter(d.patient_x(mask), d.patient_y(mask), 60, type_colors(t,:), 'filled', ...
            'MarkerEdgeColor','k', 'DisplayName', type_labels{t});
end
for t = 1:3
    mask = (ptype == t) & zone_mask;
    if any(mask)
        scatter(d.patient_x(mask), d.patient_y(mask), 110, type_colors(t,:), 'filled', 's', ...
                'MarkerEdgeColor','k', 'LineWidth', 1.6, 'HandleVisibility','off');
    end
end
scatter(d.hospital_x, d.hospital_y, 400, [1 0.84 0], 'p', 'filled', ...
        'MarkerEdgeColor','k', 'LineWidth',1.2, 'DisplayName','Hospitals');
xlabel('x (km, east of reference point)');
ylabel('y (km, north of reference point)');
title('LA wildfire case-study instance layout (squares = Palisades Fire zone)');
legend('Location','best');
set(gca,'Color','w');
savefig(f, fullfile(out_dir, 'instance_layout_xy.fig'));
close(f);

%% 7. Patient demographics
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
title('Patient-type composition -- LA wildfire case study');
set(gca,'Color','w');
savefig(f, fullfile(out_dir, 'patient_demographics.fig'));
close(f);

%% 8. Service / drop-off time histograms
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

%% 9. Time-window thresholds
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
title('Semi-soft time-window thresholds -- LA wildfire case study');
set(gca,'Color','w');
savefig(f, fullfile(out_dir, 'time_window_thresholds.fig'));
close(f);

%% 10. Penalty function shape
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
title('Semi-soft time-window delay-penalty function -- LA wildfire case study');
legend('Location','best');
set(gca,'Color','w');
savefig(f, fullfile(out_dir, 'penalty_function_shape.fig'));
close(f);

%% 11. Goal-function weights
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

%% 12. Disruption scenario, baseline vs wildfire
d = load(fullfile(data_dir, 'disruption_scenario.mat'));
f = figure('Color','w','Position',[80 80 1400 500]);
zone_idx = double(d.wildfire_zone_idx) + 1;
for s = 1:2
    subplot(1,2,s); hold on; box on; grid on;
    if s == 1
        theta = double(d.theta_baseline); gam = double(d.gamma_baseline);
        mttr = double(d.mttr_baseline); mtbf = double(d.mtbf_baseline);
    else
        theta = double(d.theta_wildfire); gam = double(d.gamma_wildfire);
        mttr = double(d.mttr_wildfire); mtbf = double(d.mtbf_wildfire);
    end
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
    avail = mtbf / (mtbf + mttr);
    title(sprintf('%s: Gamma=%d/%d, MTTR=%.0f, MTBF=%.0f, avail=%.2f', ...
          scen_labels{s}, gam, n, mttr, mtbf, avail));
    set(gca,'Color','w');
end
sgtitle('Realised disruption scenario, baseline vs wildfire -- LA case study');
savefig(f, fullfile(out_dir, 'disruption_scenario.fig'));
close(f);

%% 13. Availability by zone
d = load(fullfile(data_dir, 'availability_by_zone.mat'));
f = figure('Color','w','Position',[80 80 1100 500]);
mttrs = double(d.mttr); mtbfs = double(d.mtbf); avails = double(d.availability);
labels = cellstr(d.labels);

subplot(1,2,1); hold on; box on; grid on;
x = 1:numel(labels); width = 0.35;
bar(x - width/2, mttrs, width, 'FaceColor', [0.839 0.153 0.157], 'DisplayName','MTTR (min)');
bar(x + width/2, mtbfs, width, 'FaceColor', [0.122 0.467 0.706], 'DisplayName','MTBF (min)');
set(gca, 'XTick', x, 'XTickLabel', labels);
ylabel('Minutes'); title('MTTR / MTBF by zone'); legend('Location','best'); set(gca,'Color','w');

subplot(1,2,2); hold on; box on; grid on;
b = bar(x, avails);
b.FaceColor = 'flat';
b.CData = [0.173 0.627 0.173; 1.000 0.498 0.055; 0.839 0.153 0.157];
set(gca, 'XTick', x, 'XTickLabel', labels);
ylim([0 1]);
ylabel('System availability = MTBF / (MTBF + MTTR)');
title('Resulting availability by zone'); set(gca,'Color','w');

sgtitle('Wildfire-induced network-outage impact on MTTR / MTBF / availability -- LA case study');
savefig(f, fullfile(out_dir, 'availability_by_zone.fig'));
close(f);

%% 14. Distance impact (road closures)
d = load(fullfile(data_dir, 'distance_impact.mat'));
f = figure('Color','w','Position',[100 100 1100 550]);
delta = double(d.delta);
n = numel(delta);
zone_idx = double(d.wildfire_zone_idx) + 1;
zone_mask = false(n,1); zone_mask(zone_idx) = true;
b = bar(0:n-1, delta);
b.FaceColor = 'flat';
for i = 1:n
    if zone_mask(i)
        b.CData(i,:) = [0.839 0.153 0.157];
    else
        b.CData(i,:) = [0.122 0.467 0.706];
    end
end
xlabel('Patient index (red = inside Palisades Fire zone)');
ylabel('Change in shortest distance to nearest hospital (km), wildfire - baseline');
title('Road-network impact of burn-zone closures on ambulance access -- LA wildfire case study');
set(gca,'Color','w');
savefig(f, fullfile(out_dir, 'distance_impact.fig'));
close(f);

disp('All .fig files written to figures/fig');
