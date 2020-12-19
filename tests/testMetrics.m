%% TEST EACH METRIC
% created by: Ferhat Can ATAMAN
% date: 12/19/20

close all, clear all, clc;

addpath('./imageFusionMetrics-master/matlabPyrTools')
addpath('./imageFusionMetrics-master/')

DEBUG = false;

%% Load input and fuse images

FUSE_IMAGE_PATH = '/home/ferhatcan/Desktop/PytorchProjects/pyFusionSR/Outputs/HSV+IR/';
INPUT_IMAGE_PATH = '/media/ferhatcan/common/Image_Datasets/VIFB-master/input/';

image_name = 'carWhite';

im_fused = imread(strcat(FUSE_IMAGE_PATH, image_name, '_proposed.jpg'));
im_ir = imread(strcat(INPUT_IMAGE_PATH, 'IR/', image_name, '.jpg'));
im_vis = imread(strcat(INPUT_IMAGE_PATH, 'VI/', image_name, '.jpg'));

%% DEBUG IMAGES
if DEBUG
   figure, imshow(im_fused)
   title('Fused Image Result');
   figure, imshow(im_vis)
   title('Visible Image Result');
   figure, imshow(im_ir)
   title('IR Image Result')
end

%% CONVERT IMAGES GRAYSCALE FOR SOME METRICS
g_im_fused = rgb2gray(im_fused);
g_im_vis = rgb2gray(im_vis);
g_im_ir = rgb2gray(im_ir);

%% DEBUG IMAGES
if DEBUG
   figure, imshow(g_im_fused)
   title('Fused Image Result');
   figure, imshow(g_im_vis)
   title('Visible Image Result');
   figure, imshow(g_im_ir)
   title('IR Image Result')
end

%% Metric Calculations
if DEBUG
    % Mutual Information Metric
    resultMI_thalis = metricMI(im_vis, im_ir, im_fused, 3);
    resultMI_orig = metricMI(im_vis, im_ir, im_fused, 1);

    % Wang Method
    resultWang = metricWang(im_vis, im_ir, im_fused);

    % Xydeas Method
    resultXydeas = metricXydeas(im_vis, im_ir, im_fused);

    % PWW Method
    resultPww = metricPWW(im_vis, im_ir, im_fused);

    %Zheng Method
    resultZheng = metricZheng(im_vis, im_ir, im_fused);

    %Zhao Method
    resultZhao = metricZhao(g_im_vis, g_im_ir, g_im_fused);

    %Peilla Method
    resultPeilla = metricPeilla(im_vis, im_ir, im_fused, 3);

    %Cvejic Method
    resultCvejic1 = metricCvejic(im_vis, im_ir, im_fused, 1);
    resultCvejic2 = metricCvejic(im_vis, im_ir, im_fused, 2);

    %Yang Method
    resultYang = metricYang(im_vis, im_ir, im_fused);

    %Chen Method
    resultChen = metricChen(g_im_vis, g_im_ir, g_im_fused);

    %ChenBlum Method
    resultChenBlum = metricChenBlum(g_im_vis, g_im_ir, g_im_fused);

    %Hossny Method
    % resultHossny = metricHossny(g_im_vis, g_im_ir, g_im_fused);
end

%% Test evaluateAllMetrics function
[results, names] = evaluateAllMetrics(im_vis, im_ir, im_fused);

