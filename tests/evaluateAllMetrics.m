function [results, names] = evaluateAllMetrics(im_vis, im_ir, im_fused)
%% CONVERT IMAGES GRAYSCALE FOR SOME METRICS
g_im_fused = rgb2gray(im_fused);
g_im_vis = rgb2gray(im_vis);
g_im_ir = rgb2gray(im_ir);

%% Output Init
results = [];
names = [''];

%% Metric Calculations
metric_num = 1;

% Mutual Information Metric
resultMI_thalis = metricMI(im_vis, im_ir, im_fused, 3);
names{metric_num} = 'MI_thalis';
results(metric_num) = resultMI_thalis;
metric_num = metric_num + 1;

resultMI_orig = metricMI(im_vis, im_ir, im_fused, 1);
names{metric_num} = 'MI_orig';
results(metric_num) = resultMI_orig;
metric_num = metric_num + 1;


% Wang Method
resultWang = metricWang(im_vis, im_ir, im_fused);
names{metric_num} = 'Wang';
results(metric_num) = resultWang;
metric_num = metric_num + 1;

% Xydeas Method
resultXydeas = metricXydeas(g_im_vis, g_im_ir, g_im_fused);
names{metric_num} = 'Xydeas-Q';
results(metric_num) = resultXydeas;
metric_num = metric_num + 1;

resultXydeas = metricXydeas_L(g_im_vis, g_im_ir, g_im_fused);
names{metric_num} = 'Xydeas-L';
results(metric_num) = resultXydeas;
metric_num = metric_num + 1;

resultXydeas = metricXydeas_N(g_im_vis, g_im_ir, g_im_fused);
names{metric_num} = 'Xydeas-N';
results(metric_num) = resultXydeas;
metric_num = metric_num + 1;


% PWW Method
resultPww = metricPWW(im_vis, im_ir, im_fused);
names{metric_num} = 'PWW';
results(metric_num) = resultPww;
metric_num = metric_num + 1;

%Zheng Method
resultZheng = metricZheng(im_vis, im_ir, im_fused);
names{metric_num} = 'Zheng';
results(metric_num) = resultZheng;
metric_num = metric_num + 1;

%Zhao Method
resultZhao = metricZhao(g_im_vis, g_im_ir, g_im_fused);
names{metric_num} = 'Zhao';
results(metric_num) = resultZhao;
metric_num = metric_num + 1;

%Peilla Method
resultPeilla = metricPeilla(g_im_vis, g_im_ir, g_im_fused, 3);
names{metric_num} = 'Peilla';
results(metric_num) = resultPeilla;
metric_num = metric_num + 1;

%Cvejic Method
% resultCvejic1 = metricCvejic(im_vis, im_ir, im_fused, 1);
% names{metric_num} = 'Cvejic-1';
% results(metric_num) = resultCvejic1;
% metric_num = metric_num + 1;

resultCvejic2 = metricCvejic(g_im_vis, g_im_ir, g_im_fused, 2);
names{metric_num} = 'Cvejic-2';
results(metric_num) = resultCvejic2;
metric_num = metric_num + 1;

%Yang Method
resultYang = metricYang(g_im_vis, g_im_ir, g_im_fused);
names{metric_num} = 'Yang';
results(metric_num) = resultYang;
metric_num = metric_num + 1;

%Chen Method
resultChen = metricChen(g_im_vis, g_im_ir, g_im_fused);
names{metric_num} = 'Chen';
results(metric_num) = resultChen;
metric_num = metric_num + 1;

%ChenBlum Method
resultChenBlum = metricChenBlum(g_im_vis, g_im_ir, g_im_fused);
names{metric_num} = 'ChenBlum';
results(metric_num) = resultChenBlum;
metric_num = metric_num + 1;

%Hossny Method
% resultHossny = metricHossny(g_im_vis, g_im_ir, g_im_fused);


end