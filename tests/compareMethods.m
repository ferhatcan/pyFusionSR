%% Obtain results for each method
% created by: Ferhat Can ATAMAN
% date: 12/19/20

close all, clear all, clc;

addpath('./imageFusionMetrics-master/matlabPyrTools')
addpath('./imageFusionMetrics-master/')

%% Load input images

INPUT_IMAGE_PATH = '../../../Image_Datasets/VIFB-master/input/';

vis_images = dir(fullfile(INPUT_IMAGE_PATH, 'VI/*.jpg'));
ir_images = dir(fullfile(INPUT_IMAGE_PATH, 'IR/*.jpg'));

%% Find method names

FUSE_IMAGE_PATH = '../../../Image_Datasets/VIFB-master/fused_images/';

fuse_images = dir(fullfile(FUSE_IMAGE_PATH, '*.jpg'));
image_name = ir_images(1).name(1:end-4);
relevant_images = {fuse_images(contains({fuse_images.name}, strcat(image_name, '_'))).name};
method_names = {};
for i=1:size(relevant_images, 2)
   name = relevant_images{i};
   method_names{i} = name(size(image_name, 2) + 2 : end -4); 
end

%% Calculate metrics for each method

total_result = zeros(size(method_names, 2), 12);

for i = 1:size(method_names, 2)
   disp('-------------------');
   fprintf(method_names{i});
   relevant_fuse_images = fuse_images(contains({fuse_images.name}, strcat(method_names(i), '.jpg')));  
   [curr_result, names] = evaluateAllPairs(relevant_fuse_images, ir_images, vis_images, method_names(i));
   total_result(i, :) = sum(curr_result) / size(total_results, 1);
   disp('FINISHED....')
   disp('');
end

%% Save Results (run time is too much)
save('compare_method_results.mat', 'total_result', 'names', 'method_names');

