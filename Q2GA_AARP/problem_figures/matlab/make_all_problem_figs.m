% make_all_problem_figs.m
% Regenerates all problem-level figures (white background) as MATLAB .fig
% files from the .mat data exported by src/make_problem_figures.py.
%
% Usage: open MATLAB, cd to this folder, run:  make_all_problem_figs

clear; clc;
data_dir = fullfile('..', 'mat');
out_dir  = fullfile('..', 'fig');
if ~exist(out_dir, 'dir'); mkdir(out_dir); end

sizes = {'small','medium','large'};
size_colors = [0.122 0.467 0.706;   % small
                1.000 0.498 0.055;   % medium
                0.173 0.627 0.173];  % large
type_colors = [0.839 0.153 0.157;   % Type 1 critical
                1.000 0.498 0.055;   % Type 2 moderate
                0.173 0.627 0.173]; % Type 3 minor
type_labels = {'Critical (Type 1)','Moderate (Type 2)','Minor (Type 3)'};

%% 1. Instance layouts
for s = 1:numel(sizes)
    sz = sizes{s};
    d = load(fullfile(data_dir, ['instance_layout_' sz '.mat']));
    f = figure('Color','w','Position',[100 100 700 700]);
    hold on; box on; grid on; axis equal;
    pt = double(d.patient_type);
    px = double(d.patient_x); py = double(d.patient_y);
    h = gobjects(1,4);
    for t = 1:3
        mask = pt == t;
        h(t) = scatter(px(mask), py(mask), 50, type_colors(t,:), 'filled', ...
                        'MarkerEdgeColor','k', 'DisplayName', type_labels{t});
    end
    h(4) = scatter(double(d.hospital_x), double(d.hospital_y), 350, [1 0.84 0], ...
                    'p', 'filled', 'MarkerEdgeColor','k', 'LineWidth',1, 'DisplayName','Hospitals');
    xlabel('x coordinate'); ylabel('y coordinate');
    title(['Problem instance layout — ' sz]);
    legend(h, 'Location','best');
    set(gca,'Color','w');
    savefig(f, fullfile(out_dir, ['instance_layout_' sz '.fig']));
    close(f);
end

%% 2. Patient demographics
d = load(fullfile(data_dir, 'patient_demographics.mat'));
f = figure('Color','w','Position',[100 100 800 550]);
M = [double(d.type1_counts); double(d.type2_counts); double(d.type3_counts)];
b = bar(M', 'grouped');
for t = 1:3
    b(t).FaceColor = type_colors(t,:);
end
set(gca,'XTickLabel', sizes);
ylabel('Number of patients');
title('Patient-type composition by instance size');
legend(type_labels, 'Location','best');
set(gca,'Color','w');
savefig(f, fullfile(out_dir, 'patient_demographics.fig'));
close(f);

%% 3. Vehicle fleet composition
d = load(fullfile(data_dir, 'vehicle_fleet_composition.mat'));
f = figure('Color','w','Position',[100 100 800 550]);
M = [double(d.class_0_counts); double(d.class_1_counts); double(d.class_2_counts)];
b = bar(M', 'grouped');
class_colors = [0.122 0.467 0.706; 0.580 0.404 0.741; 0.173 0.627 0.173];
for c = 1:3
    b(c).FaceColor = class_colors(c,:);
end
set(gca,'XTickLabel', sizes);
ylabel('Number of vehicles');
title('Autonomous-ambulance fleet composition by instance size');
legend({'Class A','Class B','Class C'}, 'Location','best');
set(gca,'Color','w');
savefig(f, fullfile(out_dir, 'vehicle_fleet_composition.fig'));
close(f);

%% 4. Service / drop-off time distributions
d = load(fullfile(data_dir, 'service_dropoff_times.mat'));
f = figure('Color','w','Position',[100 100 1100 500]);
subplot(1,2,1); hold on; box on; grid on;
for s = 1:numel(sizes)
    histogram(double(d.(['service_time_' sizes{s}])), 10, 'FaceColor', size_colors(s,:), ...
              'FaceAlpha', 0.5, 'DisplayName', sizes{s});
end
xlabel('Service time s_i (time units)'); ylabel('Number of patients');
title('On-scene service-time distribution'); legend('Location','best');
set(gca,'Color','w');

subplot(1,2,2); hold on; box on; grid on;
for s = 1:numel(sizes)
    histogram(double(d.(['dropoff_time_' sizes{s}])), 10, 'FaceColor', size_colors(s,:), ...
              'FaceAlpha', 0.5, 'DisplayName', sizes{s});
end
xlabel('Hospital drop-off time (time units)'); ylabel('Number of patients');
title('Drop-off-time distribution (Type 1/2 only)'); legend('Location','best');
set(gca,'Color','w');
savefig(f, fullfile(out_dir, 'service_dropoff_times.fig'));
close(f);

%% 5. Time-window thresholds
d = load(fullfile(data_dir, 'time_window_thresholds.mat'));
f = figure('Color','w','Position',[100 100 1100 500]);
for tIdx = 1:2
    subplot(1,2,tIdx); hold on; box on; grid on;
    data = []; grp = [];
    for s = 1:numel(sizes)
        v1 = double(d.(['a1_type' num2str(tIdx) '_' sizes{s}]))(:);
        v2 = double(d.(['a2_type' num2str(tIdx) '_' sizes{s}]))(:);
        data = [data; v1; v2];
        grp = [grp; repmat((s-1)*2+1, numel(v1), 1); repmat((s-1)*2+2, numel(v2), 1)];
    end
    boxplot(data, grp);
    set(gca, 'XTickLabel', {'small a1','small a2','medium a1','medium a2','large a1','large a2'});
    ylabel('Threshold value (time units)');
    if tIdx == 1
        title('Time-window thresholds — Type 1 (critical)');
    else
        title('Time-window thresholds — Type 2 (moderate)');
    end
    set(gca,'Color','w');
end
savefig(f, fullfile(out_dir, 'time_window_thresholds.fig'));
close(f);

%% 6. Penalty function shape
d = load(fullfile(data_dir, 'penalty_function_shape.mat'));
f = figure('Color','w','Position',[100 100 800 550]);
hold on; box on; grid on;
plot(double(d.arrival), double(d.penalty_type1), 'Color', type_colors(1,:), 'LineWidth', 2, ...
     'DisplayName', 'Type 1 (critical)');
plot(double(d.arrival), double(d.penalty_type2), 'Color', type_colors(2,:), 'LineWidth', 2, ...
     'DisplayName', 'Type 2 (moderate)');
xline(double(d.a1_type1), '--', 'Color', type_colors(1,:));
xline(double(d.a2_type1), ':', 'Color', type_colors(1,:));
xline(double(d.a1_type2), '--', 'Color', type_colors(2,:));
xline(double(d.a2_type2), ':', 'Color', type_colors(2,:));
xlabel('Patient arrival time a_i (time units)'); ylabel('Delay penalty');
title('Semi-soft time-window delay-penalty function (Eq. 1-2)');
legend('Location','best');
set(gca,'Color','w');
savefig(f, fullfile(out_dir, 'penalty_function_shape.fig'));
close(f);

%% 7. MTTR / MTBF availability
d = load(fullfile(data_dir, 'mttr_mtbf_availability.mat'));
f = figure('Color','w','Position',[100 100 1100 500]);
subplot(1,2,1); hold on; box on; grid on;
mtbf_vals = [30 60 90 120];
for i = 1:numel(mtbf_vals)
    plot(double(d.mttr_range), double(d.(['avail_vs_mttr_mtbf' num2str(mtbf_vals(i))])), ...
         'LineWidth', 2, 'DisplayName', ['MTBF=' num2str(mtbf_vals(i))]);
end
xlabel('MTTR (time units)'); ylabel('Availability = MTBF/(MTBF+MTTR)');
title('System availability vs. MTTR'); legend('Location','best');
set(gca,'Color','w');

subplot(1,2,2); hold on; box on; grid on;
mttr_vals = [4 8 16 24];
for i = 1:numel(mttr_vals)
    plot(double(d.mtbf_range), double(d.(['avail_vs_mtbf_mttr' num2str(mttr_vals(i))])), ...
         'LineWidth', 2, 'DisplayName', ['MTTR=' num2str(mttr_vals(i))]);
end
xlabel('MTBF (time units)'); ylabel('Availability = MTBF/(MTBF+MTTR)');
title('System availability vs. MTBF'); legend('Location','best');
set(gca,'Color','w');
savefig(f, fullfile(out_dir, 'mttr_mtbf_availability.fig'));
close(f);

%% 8. MTTR vs theta sensitivity
d = load(fullfile(data_dir, 'mttr_theta_sensitivity.mat'));
f = figure('Color','w','Position',[100 100 800 550]);
hold on; box on; grid on;
mttr = double(d.mttr_values); m = double(d.theta_mean); sd = double(d.theta_std);
fill([mttr fliplr(mttr)], [m-sd fliplr(m+sd)], [0.839 0.153 0.157], ...
     'FaceAlpha', 0.15, 'EdgeColor','none', 'HandleVisibility','off');
plot(mttr, m, '-o', 'Color', [0.839 0.153 0.157], 'LineWidth', 2, 'DisplayName','Mean theta (affected patients)');
xlabel('MTTR (time units)'); ylabel('Realised disruption buffer \theta_i (time units)');
title('Disruption-buffer magnitude vs. MTTR (medium instance)');
legend('Location','best');
set(gca,'Color','w');
savefig(f, fullfile(out_dir, 'mttr_theta_sensitivity.fig'));
close(f);

%% 9. Gamma budget sensitivity
d = load(fullfile(data_dir, 'gamma_budget_sensitivity.mat'));
f = figure('Color','w','Position',[100 100 1100 500]);
subplot(1,2,1); hold on; box on; grid on;
plot(double(d.gamma_frac), double(d.n_affected), '-o', 'Color', [0.122 0.467 0.706], 'LineWidth',2);
xlabel('Disruption budget Gamma (fraction of patients)'); ylabel('Number of affected patients');
title('Disruption coverage vs. Gamma');
set(gca,'Color','w');

subplot(1,2,2); hold on; box on; grid on;
plot(double(d.gamma_frac), double(d.total_theta), '-o', 'Color', [0.839 0.153 0.157], 'LineWidth',2, 'DisplayName','Total theta');
plot(double(d.gamma_frac), double(d.mean_theta), '-s', 'Color', [1.000 0.498 0.055], 'LineWidth',2, 'DisplayName','Mean theta (affected)');
xlabel('Disruption budget Gamma (fraction of patients)'); ylabel('Disruption buffer theta (time units)');
title('Disruption severity vs. Gamma'); legend('Location','best');
set(gca,'Color','w');
savefig(f, fullfile(out_dir, 'gamma_budget_sensitivity.fig'));
close(f);

%% 10. Disruption scenario per instance
for s = 1:numel(sizes)
    sz = sizes{s};
    d = load(fullfile(data_dir, ['disruption_scenario_' sz '.mat']));
    f = figure('Color','w','Position',[100 100 900 500]);
    theta = double(d.theta);
    bar(theta, 'FaceColor', [0.839 0.153 0.157]);
    xlabel('Patient index'); ylabel('Realised disruption buffer \theta_i (time units)');
    title(['Realised disruption scenario — ' sz ' (Gamma=' num2str(d.gamma_budget) ...
           ', MTTR=' num2str(d.mttr) ')']);
    set(gca,'Color','w');
    savefig(f, fullfile(out_dir, ['disruption_scenario_' sz '.fig']));
    close(f);
end

%% 11. Goal-function weights
d = load(fullfile(data_dir, 'goal_function_weights.mat'));
f = figure('Color','w','Position',[100 100 700 550]);
weights = [double(d.w1) double(d.w2) double(d.w3)];
b = bar(weights);
b.FaceColor = 'flat';
b.CData = type_colors;
set(gca, 'XTickLabel', {'w1.C1','w2.C2','w3.C3'});
ylabel('Weight'); ylim([0 0.6]);
title({'Goal-function weighted-sum structure (Eq. 6-8)', ...
       'fitness = w1.C1 + w2.C2 + w3.C3 + penalty1 + penalty2 + infeasibility'});
set(gca,'Color','w');
savefig(f, fullfile(out_dir, 'goal_function_weights.fig'));
close(f);

%% 12. Objective component breakdown
comp_labels = {'w1*C1','w2*C2','w3*C3','penalty1','penalty2','infeasibility'};
comp_colors = [0.122 0.467 0.706; 1.000 0.498 0.055; 0.173 0.627 0.173; ...
                0.839 0.153 0.157; 0.580 0.404 0.741; 0.498 0.498 0.498];
for s = 1:numel(sizes)
    sz = sizes{s};
    d = load(fullfile(data_dir, ['objective_components_' sz '.mat']));
    f = figure('Color','w','Position',[100 100 800 550]);
    vals = double(d.values);
    b = bar(vals);
    b.FaceColor = 'flat';
    b.CData = comp_colors;
    set(gca, 'XTickLabel', comp_labels);
    ylabel('Contribution to fitness');
    title(['Goal-function component breakdown — ' sz ...
           ' (fitness=' num2str(d.fitness, '%.2f') ', unserved=' num2str(d.n_unserved) ')']);
    set(gca,'Color','w');
    savefig(f, fullfile(out_dir, ['objective_components_' sz '.fig']));
    close(f);
end

%% 13. QoS vs. availability
d = load(fullfile(data_dir, 'qos_vs_availability.mat'));
f = figure('Color','w','Position',[100 100 1500 500]);
avail = double(d.availability);

subplot(1,3,1); hold on; box on; grid on;
fm = double(d.fitness_mean); fs = double(d.fitness_std);
fill([avail fliplr(avail)], [fm-fs fliplr(fm+fs)], [0.122 0.467 0.706], 'FaceAlpha',0.15,'EdgeColor','none','HandleVisibility','off');
plot(avail, fm, '-o', 'Color', [0.122 0.467 0.706], 'LineWidth', 2);
set(gca,'XDir','reverse');
xlabel('System availability = MTBF/(MTBF+MTTR)'); ylabel('Mean best fitness');
title('Overall fitness vs. availability'); set(gca,'Color','w');

subplot(1,3,2); hold on; box on; grid on;
pm = double(d.penalty_mean); ps = double(d.penalty_std);
fill([avail fliplr(avail)], [pm-ps fliplr(pm+ps)], [0.839 0.153 0.157], 'FaceAlpha',0.15,'EdgeColor','none','HandleVisibility','off');
plot(avail, pm, '-o', 'Color', [0.839 0.153 0.157], 'LineWidth', 2);
set(gca,'XDir','reverse');
xlabel('System availability = MTBF/(MTBF+MTTR)'); ylabel('Mean total delay penalty');
title('Service-quality penalty vs. availability'); set(gca,'Color','w');

subplot(1,3,3); hold on; box on; grid on;
plot(avail, double(d.unserved_mean), '-o', 'Color', [0.580 0.404 0.741], 'LineWidth', 2);
set(gca,'XDir','reverse');
xlabel('System availability = MTBF/(MTBF+MTTR)'); ylabel('Mean unserved patients');
title('Coverage failure vs. availability'); set(gca,'Color','w');

sgtitle('Impact of system availability (MTTR sweep, MTBF=60) on quality of service');
savefig(f, fullfile(out_dir, 'qos_vs_availability.fig'));
close(f);

%% 14. QoS vs. Gamma
d = load(fullfile(data_dir, 'qos_vs_gamma.mat'));
f = figure('Color','w','Position',[100 100 1500 500]);
gamma = double(d.gamma_frac);

subplot(1,3,1); hold on; box on; grid on;
fm = double(d.fitness_mean); fs = double(d.fitness_std);
fill([gamma fliplr(gamma)], [fm-fs fliplr(fm+fs)], [0.122 0.467 0.706], 'FaceAlpha',0.15,'EdgeColor','none','HandleVisibility','off');
plot(gamma, fm, '-o', 'Color', [0.122 0.467 0.706], 'LineWidth', 2);
xline(0.20, '--', 'Color', [0.5 0.5 0.5]);
xlabel('Disruption budget Gamma (fraction of patients)'); ylabel('Mean best fitness');
title('Overall fitness vs. Gamma'); set(gca,'Color','w');

subplot(1,3,2); hold on; box on; grid on;
pm = double(d.penalty_mean); ps = double(d.penalty_std);
fill([gamma fliplr(gamma)], [pm-ps fliplr(pm+ps)], [0.839 0.153 0.157], 'FaceAlpha',0.15,'EdgeColor','none','HandleVisibility','off');
plot(gamma, pm, '-o', 'Color', [0.839 0.153 0.157], 'LineWidth', 2);
xline(0.20, '--', 'Color', [0.5 0.5 0.5]);
xlabel('Disruption budget Gamma (fraction of patients)'); ylabel('Mean total delay penalty');
title('Service-quality penalty vs. Gamma'); set(gca,'Color','w');

subplot(1,3,3); hold on; box on; grid on;
plot(gamma, double(d.unserved_mean), '-o', 'Color', [0.580 0.404 0.741], 'LineWidth', 2);
xline(0.20, '--', 'Color', [0.5 0.5 0.5]);
xlabel('Disruption budget Gamma (fraction of patients)'); ylabel('Mean unserved patients');
title('Coverage failure vs. Gamma'); set(gca,'Color','w');

sgtitle('Impact of cyber-disruption breadth (Gamma) on quality of service');
savefig(f, fullfile(out_dir, 'qos_vs_gamma.fig'));
close(f);

%% 15. QoS vs. MTTR with availability twin axis
d = load(fullfile(data_dir, 'qos_vs_mttr.mat'));
f = figure('Color','w','Position',[100 100 900 550]);
mttr = double(d.mttr); fm = double(d.fitness_mean); fs = double(d.fitness_std);
pm = double(d.penalty_mean); ps = double(d.penalty_std);

yyaxis left
hold on; box on; grid on;
fill([mttr fliplr(mttr)], [fm-fs fliplr(fm+fs)], [0.122 0.467 0.706], 'FaceAlpha',0.15,'EdgeColor','none','HandleVisibility','off');
plot(mttr, fm, '-o', 'Color', [0.122 0.467 0.706], 'LineWidth', 2, 'DisplayName','Mean best fitness');
ylabel('Mean best fitness');

yyaxis right
fill([mttr fliplr(mttr)], [pm-ps fliplr(pm+ps)], [0.839 0.153 0.157], 'FaceAlpha',0.12,'EdgeColor','none','HandleVisibility','off');
plot(mttr, pm, '-s', 'Color', [0.839 0.153 0.157], 'LineWidth', 2, 'DisplayName','Mean total delay penalty');
ylabel('Mean total delay penalty');

xlabel('MTTR (mean time to repair, time units)');
title('Recovery time (MTTR) vs. quality of service (medium instance)');
legend('Location','best');
set(gca,'Color','w');
savefig(f, fullfile(out_dir, 'qos_vs_mttr.fig'));
close(f);

%% 16. Availability(MTTR, MTBF) contour surface
d = load(fullfile(data_dir, 'availability_surface.mat'));
f = figure('Color','w','Position',[100 100 800 650]);
[MTTR, MTBF] = meshgrid(double(d.mttr_range), double(d.mtbf_range));
contourf(MTTR, MTBF, double(d.availability), 20, 'LineColor','none');
colormap(flipud(redgreencmap_local()));
cb = colorbar; cb.Label.String = 'System availability = MTBF/(MTBF+MTTR)';
hold on;
contour(MTTR, MTBF, double(d.availability), [0.7 0.8 0.9 0.95 0.98], 'k', 'ShowText','on');
plot(double(d.default_mttr), double(d.default_mtbf), 'kx', 'MarkerSize', 14, 'LineWidth', 3);
xlabel('MTTR (time units)'); ylabel('MTBF (time units)');
title('System availability as a function of MTTR and MTBF');
set(gca,'Color','w');
savefig(f, fullfile(out_dir, 'availability_surface.fig'));
close(f);

%% 17. Cyber-risk QoS degradation summary
d = load(fullfile(data_dir, 'cyber_risk_degradation_summary.mat'));
f = figure('Color','w','Position',[100 100 800 600]);
fitp = double(d.fitness_pct); penp = double(d.penalty_pct);
b = bar([fitp; penp]');
b(1).FaceColor = [0.122 0.467 0.706];
b(2).FaceColor = [0.839 0.153 0.157];
set(gca, 'XTickLabel', {'Gamma: 0.20->0.50N','MTTR: 8->24'});
ylabel('Relative change vs. baseline (%)');
title('Quality-of-service degradation under worsening cyber-risk conditions');
legend({'Fitness degradation (%)','Total delay-penalty increase (%)'}, 'Location','best');
set(gca,'Color','w');
savefig(f, fullfile(out_dir, 'cyber_risk_degradation_summary.fig'));
close(f);

disp('All .fig files written to problem_figures/fig');

function cmap = redgreencmap_local()
    % Simple red-yellow-green colormap fallback (avoids toolbox dependency)
    n = 64;
    r = [linspace(0.84,1,n/2), linspace(1,0.0,n/2)];
    g = [linspace(0.15,1,n/2), linspace(1,0.55,n/2)];
    b = [linspace(0.16,0.4,n/2), linspace(0.4,0.0,n/2)];
    cmap = [r' g' b'];
end
