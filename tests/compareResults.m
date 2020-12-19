%% Evaluate all images in the input file
% created by: Ferhat Can ATAMAN
% date: 12/19/20

close all, clear all, clc;

addpath('./imageFusionMetrics-master/matlabPyrTools')
addpath('./imageFusionMetrics-master/')

DEBUG = false;

%% Load input and fuse images

FUSE_IMAGE_PATH = '/home/ferhatcan/Desktop/PytorchProjects/pyFusionSR/Outputs/HSV+IR/';
INPUT_IMAGE_PATH = '/media/ferhatcan/common/Image_Datasets/VIFB-master/input/';

fuse_images = dir(fullfile(FUSE_IMAGE_PATH, '*_proposed.jpg'));
vis_images = dir(fullfile(INPUT_IMAGE_PATH, 'VI/*.jpg'));
ir_images = dir(fullfile(INPUT_IMAGE_PATH, 'IR/*.jpg'));

%% Calculate scores for each pairs

total_results = zeros(size(fuse_images, 1), 12);

for i = 1:size(ir_images, 1)
    disp(ir_images(i).name);
    disp(vis_images(i).name);
    
    fusedImageName = strcat(ir_images(i).name(1:end-4), '_proposed.jpg');
    fuse_index = find(contains({fuse_images.name}, fusedImageName));
    
    disp(fuse_images(fuse_index).name);
    
    
    im_fused = imread(fullfile(fuse_images(fuse_index).folder, fuse_images(fuse_index).name));
    im_ir = imread(fullfile(ir_images(i).folder, ir_images(i).name));
    im_vis = imread(fullfile(vis_images(i).folder, vis_images(i).name));
    
    if length(size(im_vis)) == 2
       im_vis = cat(3, im_vis, im_vis, im_vis); 
    end
    if length(size(im_ir)) == 2
       im_ir = cat(3, im_ir, im_ir, im_ir); 
    end
    if length(size(im_fused)) == 2
       im_fused = cat(3, im_fused, im_fused, im_fused); 
    end
    
    [curr_results, names] = evaluateAllMetrics(im_vis, im_ir, im_fused);
    
    total_results(i, :) = curr_results;
    
end
