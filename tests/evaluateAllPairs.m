function [total_results, names] = evaluateAllPairs(fuse_images, ir_images, vis_images, method_name)
% @todo 12 is a mgic number should be parametrized
% total_results = zeros(size(fuse_images, 1), 12);

for i = 1:size(ir_images, 1)
%     disp(ir_images(i).name);
%     disp(vis_images(i).name);
    
    fusedImageName = strcat(ir_images(i).name(1:end-4), '_', method_name, '.jpg');
    fuse_index = find(contains({fuse_images.name}, fusedImageName));
    
%     disp(fuse_images(fuse_index).name);
    
    
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

end