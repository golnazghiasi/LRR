% This file is from Matconvnet library (https://github.com/vlfeat/matconvnet)
% and has minor modifications.
function y = getBatch(imdb, images, gt_resizes, batch_image_size_ind, data_path_fn, data_aug_path_fn, anno_de_path_fn, varargin)
% GET_BATCH  Load, preprocess, and pack images for CNN evaluation

opts.imageSize = [384, 384];
opts.imageSize_multi = [0];
opts.rgbMean = [] ;
opts.randFlip = true;
opts.useGpu = false ;
opts.type = 'train';
opts.strelRad = [];
opts = vl_argparse(opts, varargin);

opts.rgbMean = reshape(single(opts.rgbMean), [1 1 3]) ;

debug = false;
if debug
    addpath ~/export_fig/;
    save_dir = 'models/check_data/';
    if ~exist(save_dir)
        mkdir(save_dir)
    end
end

if isfield(imdb, 'coco_data')
    cocoData{1} = imdb.coco_data{1};
    cocoData{2} = imdb.coco_data{2};
    cocoDataType = {'train2014', 'val2014'};
end

strel_rads = cell(1, 0);
for k = 1 : length(opts.strelRad)
    strel_rads{k} = strel('disk', opts.strelRad(k));
end

if opts.imageSize(1) == 0
    % Uses original image size (just makes it multiplicate of 32).
    assert(numel(images) == 1);
    img_sz = imdb.images.size(:, images(1));
    img_sz = round(img_sz([2 1]) / 32) * 32;
    img_scaled_size = img_sz;
else
    img_scaled_size = opts.imageSize + 32 * opts.imageSize_multi(batch_image_size_ind);
    if img_scaled_size(1) > opts.imageSize(1)
        img_sz = opts.imageSize;
    else
        img_sz = img_scaled_size;
    end
end

% Space for images and labels
ims = zeros(img_sz(1), img_sz(2), 3, numel(images), 'single');
labels = zeros(img_sz(1), img_sz(2), 1, numel(images), 'single');
masks_d = cell(1, length(gt_resizes));
masks_e = cell(1, length(gt_resizes));
labels_ds = cell(1, length(gt_resizes));
for ri = 1 : length(gt_resizes)
    masks_d{ri} = zeros(img_sz(1)/gt_resizes(ri), img_sz(2)/gt_resizes(ri), imdb.num_classes, numel(images));
    masks_e{ri} = zeros(img_sz(1)/gt_resizes(ri), img_sz(2)/gt_resizes(ri), imdb.num_classes, numel(images));
    labels_ds{ri} = zeros(img_sz(1)/gt_resizes(ri), img_sz(2)/gt_resizes(ri), 1, numel(images));
end

for i = 1 : numel(images)
    
    if opts.imageSize(1) == 0
        [rgb_path, anno_path, filename] = data_path_fn(images(i));
        rgb = imread(rgb_path);
        anno = imread(anno_path);
        sz = imdb.images.size(:, images(1));
        assert(size(rgb,1) == sz(2) && size(rgb,2) == sz(1));
        [rgb, anno] = resizeMult32(rgb, anno);
    else
        [rgb_path, anno_path, filename] = data_aug_path_fn(images(i), img_scaled_size(1), img_sz(1));
        rgb = imread(rgb_path);
        anno = imread(anno_path);
    end
    
    if size(rgb,3) == 1
        rgb = cat(3, rgb, rgb, rgb) ;
    end
    
    flip_it = false;
    if opts.randFlip && ~debug
        if rand > 0.5
            flip_it = true;
            rgb = fliplr(rgb);
            anno = fliplr(anno);
        end
    end
    
    anno_ = mod(single(anno) + 1, 256); % 1: backgorund 0: don't care
    ims(:, :, :, i) = bsxfun(@minus, single(rgb), opts.rgbMean);
    labels(:, :, 1, i) = anno_;
    
    for ri = 1 : length(gt_resizes)
        if length(strel_rads) < ri
            labels_ds{ri}(:, :, 1, i) = imresize(anno_, 1 / gt_resizes(ri), 'nearest');
            continue;
        end
        
        anno_de_save_path = anno_de_path_fn(images(i), img_scaled_size(1), img_sz(1), gt_resizes(ri), opts.strelRad(ri));
        if exist(anno_de_save_path, 'file') && opts.imageSize(1) ~=0
            load(anno_de_save_path, 'mask_r', 'mask_dr', 'mask_er');
            if flip_it
                mask_r = fliplr(mask_r);
                mask_dr = fliplr(mask_dr);
                mask_er = fliplr(mask_er);
            end
            
        else
            %disp(anno_de_save_path);
            mask_r = imresize(anno_, 1/gt_resizes(ri), 'nearest');
            if length(strel_rads) >= ri
                [mask_d, mask_e] = BoundMask(anno_, strel_rads{ri}, imdb.num_classes);
                mask_dr = imresize(mask_d, 1/gt_resizes(ri), 'nearest');
                mask_er = imresize(mask_e, 1/gt_resizes(ri), 'nearest');
            end
        end
        labels_ds{ri}(:, :, 1, i) = mask_r;
        if length(strel_rads) >= ri
            masks_d{ri}(:,:,:,i) = mask_dr;
            masks_e{ri}(:,:,:,i) = mask_er;
        end
    end
    
    if debug
        VisImgAnno(rgb_path, anno_path, filename, imdb, images(i), i, ims, labels, labels_ds, ...
            masks_d, masks_e, img_scaled_size, gt_resizes, save_dir, opts);
    end
end

% -------------------------------------------------------------------------
% Set ups input for the net
% -------------------------------------------------------------------------
if opts.useGpu
    ims = gpuArray(ims) ;
end
y = {'input', ims} ;
if strcmp(opts.type,'test') || isempty(gt_resizes) || (numel(gt_resizes) == 1 && gt_resizes == 1)
    y{end+1} = 'label';
    if opts.useGpu
        y{end+1} = gpuArray(labels);
    else
        y{end+1} = labels;
    end
end
if ~(numel(gt_resizes) == 1 && gt_resizes == 1)
    for ri = 1 : length(gt_resizes)
        y{end + 1} = ['label' num2str(gt_resizes(ri))];
        if opts.useGpu
            y{end + 1} = gpuArray(labels_ds{ri});
        else
            y{end + 1} = labels_ds{ri};
        end
    end
end
for k = 1 : length(opts.strelRad)
    y{end + 1} = ['dil_gt_' num2str(gt_resizes(k))];
    if opts.useGpu
        y{end + 1} = gpuArray(masks_d{k});
    else
        y{end + 1} = masks_d{k};
    end
    y{end + 1} = ['ero_gt_' num2str(gt_resizes(k))];
    if opts.useGpu
        y{end + 1} = gpuArray(masks_e{k});
    else
        y{end + 1} = masks_e{k};
    end
end

% -------------------------------------------------------------------------
function cmap = PascalLabelColors()
% -------------------------------------------------------------------------
N = 21;
cmap = zeros(N, 3);
for i = 1 : N
    id = i - 1; r = 0; g = 0; b = 0;
    for j=0:7
        r = bitor(r, bitshift(bitget(id,1),7 - j));
        g = bitor(g, bitshift(bitget(id,2),7 - j));
        b = bitor(b, bitshift(bitget(id,3),7 - j));
        id = bitshift(id,-3);
    end
    cmap(i,1)=r; cmap(i,2)=g; cmap(i,3)=b;
end
cmap(end + 1, :) = [255 255 255];
cmap = cmap / 255;



% -------------------------------------------------------------------------
function cmap = PascalLabelColorsDNC0()
% -------------------------------------------------------------------------
% Colormap for the case that background is 1 and don't care is zero
cmap = PascalLabelColors();
cmap = [cmap(end, :) ; cmap(1 : end - 1, :)];


function im_ = padAndCrop(im, v, sz)
im_ = ones(sz(1), sz(2), size(im, 3)) * v;

br_ = 1; br = 1;
if(sz(1)>size(im,1))
    br_ = floor((sz(1) - size(im, 1))/2) + 1;
    lr = size(im, 1);
else
    br = floor((size(im,1) - sz(1))/2) + 1;
    lr = sz(1);
end

bc_ = 1; bc = 1;
if(sz(2)>size(im,2))
    bc_ = floor((sz(2) - size(im, 2))/2) + 1;
    lc = size(im, 2);
else
    bc = floor((size(im,2) - sz(2))/2) + 1;
    lc = sz(2);
end

im_(br_: br_ + lr - 1, bc_ : bc_ + lc - 1, :) = im(br : br + lr - 1, bc : bc + lc - 1, :);


function [mask_d, mask_e] = BoundMask(labels, se, N)
labels(labels == 0) = 1; % Converts don't care to background.
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

function VisImgAnno(rgb_path, anno_path, filename, imdb, img_i, i, ims, labels, labels_ds, ...
    masks_d, masks_e, img_scaled_size, gt_resizes, save_dir, opts)
cmap = PascalLabelColors() ;
cmap_dnc0 = PascalLabelColorsDNC0() ;

rgb_orig = imread(rgb_path);
anno_orig = imread(anno_path);
image_name = [filename '-t' num2str(imdb.images.data_type(img_i)) '-set' num2str(imdb.images.set(img_i))];
im_path = fullfile(save_dir, [image_name '-img.png']);
figure(1); clf; image(rgb_orig); axis 'equal';
export_fig(im_path, '-r300');
im_path = fullfile(save_dir, [image_name '-lb.png']);
figure(2); clf; image(uint8(anno_orig)) ; axis image ; colormap(cmap) ;
export_fig(im_path, '-r300');

im_path = fullfile(save_dir, [image_name '-imgr.png']);
I = bsxfun(@plus, ims(:, :, :, i), opts.rgbMean);
I = uint8(I);
figure(3); clf; image(I); axis 'equal'; title(img_scaled_size(1));
export_fig(im_path, '-r300');

im_path = fullfile(save_dir, [image_name '-lpr.png']);
figure(4); clf; image(uint8(labels(:, :, :, i))); axis image ; colormap(cmap_dnc0) ;
l = labels(:, :, :, i);
title(img_scaled_size(1));
export_fig(im_path, '-r300');

for ri = 1 : length(gt_resizes)
    im_path = fullfile(save_dir, [image_name 'ri' num2str(gt_resizes(ri)) '-l.png']);
    figure(4); clf; image(uint8(labels_ds{ri}(:,:,:,i))); axis image ; colormap(cmap_dnc0) ;
    title(['r ' num2str(gt_resizes(ri))]);
    export_fig(im_path, '-r300');
end

for ri = 1 : length(masks_d)
    for ci = 1 : imdb.num_classes
        maski = squeeze(masks_d{ri}(:, :, ci, i));
        if all(maski(:)<0)
            continue;
        end
        im_path = fullfile(save_dir, [image_name '-m-' num2str(ci) '-ri' num2str(ri) '-d.png']);
        figure(5); clf; imagesc(maski); axis image; colorbar;
        title(['cl-' num2str(ci) '-ri-' num2str(ri) '-dil']);
        export_fig(im_path, '-r300');
        
        maski = squeeze(masks_e{ri}(:, :, ci, i));
        im_path = fullfile(save_dir, [image_name '-m-' num2str(ci) '-ri' num2str(ri) '-e.png']);
        figure(5); clf; imagesc(maski); axis image; colorbar;
        title(['cl-' num2str(ci) '-ri-' num2str(ri) '-ero']);
        export_fig(im_path, '-r300');
    end
end

% -------------------------------------------------------------------------
function [im_o, anno_o] = resizeMult32(im, anno)
% -------------------------------------------------------------------------
im_size = [size(im, 1), size(im, 2)];
net_input_size = round(im_size / 32)*32;
resize_size = min(net_input_size./im_size - eps) * im_size;
im = imresize(im, resize_size, 'bicubic') ;
anno = imresize(anno, resize_size, 'nearest') ;

im_o = zeros(net_input_size(1), net_input_size(2), size(im, 3), 'single');
anno_o = zeros(net_input_size(1), net_input_size(2), size(anno, 3), 'single');
[rinds, cinds] = subInds(im, net_input_size);
im_o(rinds, cinds, :) = im;
anno_o(rinds, cinds, :) = anno;

% -------------------------------------------------------------------------
function [rinds, cinds] = subInds(im, s)
% -------------------------------------------------------------------------
assert(size(im,1) <= s(1));
assert(size(im,2) <= s(2));
rb = 1 + ceil((s(1) - size(im,1))/2);
cb = 1 + ceil((s(2) - size(im,2))/2);
rinds = rb:rb-1+size(im,1);
cinds = cb:cb-1+size(im,2);

