function ResizeAndSaveTrainData(multi_imageSizes, max_allow_size, images, gt_resizes, strel_rads, data_aug_path_fn, anno_de_path_fn, num_classes)
disp(gt_resizes)
disp(strel_rads)

for k = 1 : length(strel_rads)
    strels{k} = strel('disk', strel_rads(k));
end

for size_ind = 1 : length(multi_imageSizes)
    img_scaled_size = max_allow_size + 32 * multi_imageSizes(size_ind)
    if img_scaled_size(1) > max_allow_size(1)
        imageSize = max_allow_size;
    else
        imageSize = img_scaled_size;
    end
    
    resize_name = ['sz' num2str(img_scaled_size(1)) '-cr' num2str(imageSize(1))];
	disp(resize_name);
    
    for i = 1 : numel(images)
        img_i =  images(i);
        for ri = 1 : length(strel_rads)
        	[rgb_path, anno_path, filename] = data_aug_path_fn(img_i, img_scaled_size(1), imageSize(1));
        	anno = imread(anno_path);
        	anno_ = mod(single(anno) + 1, 256); % 1: backgorund 0: don't care
            anno_save_path = anno_de_path_fn(img_i, img_scaled_size(1), imageSize(1), gt_resizes(ri), strel_rads(ri));
            
            if exist(anno_save_path, 'file')
                continue;
            end
            %fprintf('-');
            [mask_r, mask_dr, mask_er] = ResizeDilateErode(anno_, gt_resizes(ri), strels{ri}, num_classes);
            save(anno_save_path, 'mask_r', 'mask_dr', 'mask_er');
        end
        
        if mod(i,1000) == 1
            fprintf('%d/%d ... \n', i, numel(images));
            disp(anno_save_path);
        end
    end
end

function [mask_d, mask_e] = BoundMask(labels, se, N)
labels(labels == 0) = 1; %convert don't care to background
mask_d = -ones(size(labels,1), size(labels, 2), N);
mask_e = -ones(size(labels,1), size(labels, 2), N);
for ci = 1 : N
    label = (labels == ci);
    if sum(label(:)) == 0
        continue;
    end
    mask_d(:, :, ci) = double(imdilate(label, se));
    mask_e(:, :, ci) = double(imerode(label, se));
end
mask_d(mask_d == 0) = -1;
mask_e(mask_e == 0) = -1;

function [mask_r, mask_dr, mask_er] = ResizeDilateErode(anno, gt_resize, strel_rads, num_classes)
mask_r = imresize(anno, 1/gt_resize, 'nearest');
[mask_d, mask_e] = BoundMask(anno, strel_rads, num_classes);
mask_dr = imresize(mask_d, 1/gt_resize, 'nearest');
mask_er = imresize(mask_e, 1/gt_resize, 'nearest');

if 0
    figure; imagesc(anno);
    for i = 1 :size(mask_dr,3)
        figure; imagesc(mask_dr(:,:,i));
        figure; imagesc(mask_er(:,:,i));
        pause;
    end
end
