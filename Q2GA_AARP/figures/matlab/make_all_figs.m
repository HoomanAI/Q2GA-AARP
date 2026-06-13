% make_all_figs.m
% Regenerates all figures (white background) as MATLAB .fig files from the
% .mat data exported by src/make_figures.py.
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
sizes  = {'small','medium','large'};

%% 1. Shaded convergence plots
for s = 1:numel(sizes)
    sz = sizes{s};
    d = load(fullfile(data_dir, ['conv_' sz '.mat']));
    f = figure('Color','w','Position',[100 100 800 550]);
    hold on; box on; grid on;
    h = gobjects(1,numel(algos));
    for a = 1:numel(algos)
        alg = algos{a};
        gens = double(d.([alg '_gens']));
        m    = double(d.([alg '_mean']));
        sd   = double(d.([alg '_std']));
        fillX = [gens, fliplr(gens)];
        fillY = [m-sd, fliplr(m+sd)];
        fill(fillX, fillY, colors(a,:), 'FaceAlpha', 0.18, 'EdgeColor', 'none', ...
             'HandleVisibility','off');
        h(a) = plot(gens, m, 'Color', colors(a,:), 'LineWidth', 2, 'DisplayName', alg);
    end
    xlabel('Generation'); ylabel('Best fitness (objective value)');
    title(['Convergence comparison — ' sz ' instance']);
    legend(h, 'Location','best');
    set(gca,'Color','w');
    savefig(f, fullfile(out_dir, ['conv_' sz '.fig']));
    close(f);
end

%% 2. Boxplots of final fitness
for s = 1:numel(sizes)
    sz = sizes{s};
    d = load(fullfile(data_dir, ['box_' sz '.mat']));
    f = figure('Color','w','Position',[100 100 800 550]);
    data = [];
    grp  = [];
    for a = 1:numel(algos)
        v = double(d.(algos{a}))(:);
        data = [data; v];
        grp  = [grp; repmat(a, numel(v), 1)];
    end
    boxplot(data, grp, 'Labels', algos, 'Colors', 'k');
    h = findobj(gca,'Tag','Box');
    for j = 1:length(h)
        patch(get(h(j),'XData'), get(h(j),'YData'), colors(numel(algos)-j+1,:), ...
              'FaceAlpha', 0.5);
    end
    ylabel('Final best fitness');
    title(['Final solution quality — ' sz ' instance']);
    set(gca,'Color','w');
    savefig(f, fullfile(out_dir, ['box_' sz '.fig']));
    close(f);
end

%% 3. Objective component bar charts
comp_labels = {'C1 (critical)','C2 (moderate)','C3 (minor)','Penalty-1','Penalty-2'};
for s = 1:numel(sizes)
    sz = sizes{s};
    d = load(fullfile(data_dir, ['objectives_' sz '.mat']));
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
    title(['Objective components of best solutions — ' sz ' instance']);
    legend(algos, 'Location','best');
    set(gca,'Color','w');
    savefig(f, fullfile(out_dir, ['objectives_' sz '.fig']));
    close(f);
end

%% 4. Runtime bar chart
d = load(fullfile(data_dir, 'runtime_bar.mat'));
f = figure('Color','w','Position',[100 100 800 550]);
M = [];
for a = 1:numel(algos)
    M = [M; double(d.([algos{a} '_means']))];
end
b = bar(M', 'grouped');
for a = 1:numel(algos)
    b(a).FaceColor = colors(a,:);
end
set(gca, 'XTickLabel', sizes);
ylabel('Mean runtime (s)');
title('Mean runtime per algorithm and instance size');
legend(algos, 'Location','best');
set(gca,'Color','w');
savefig(f, fullfile(out_dir, 'runtime_bar.fig'));
close(f);

disp('All .fig files written to figures/fig');
