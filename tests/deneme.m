fuse_images = dir(fullfile('/media/ferhatcan/common/Image_Datasets/VIFB-master/fused_images/', '*.jpg'));
image_name = ir_images(1).name(1:end-4);
relevant_images = {fuse_images(contains({fuse_images.name}, strcat(image_name, '_'))).name};
method_names = {};
for i=1:size(tmp2, 2)
   name = tmp2{i};
   method_names{i} = name(size(image_name, 2) + 2 : end -4); 
end

disp('Worked');