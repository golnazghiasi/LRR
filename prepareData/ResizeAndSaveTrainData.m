function ResizeAndSaveTrainData(multi_imageSizes, max_allow_size, imdb, images, data_path_fn, data_aug_path_fn)
resize_method = 'bicubic';
cmap = labelColors();

fprintf('\nPre-computing and saving scale augmented data (if they are not saved already) ...\n');

for size_ind = 1 : length(multi_imageSizes)
    resizeImageSize = max_allow_size + 32 * multi_imageSizes(size_ind);
    if resizeImageSize(1) > max_allow_size(1)
        imageSize = max_allow_size;
    else
        imageSize = resizeImageSize;
    end
    
    for i = 1 : numel(images)
        if mod(i,500) == 1
            fprintf('%d/%d\n', i, numel(images));
        end
        img_i =  images(i);
        [rgb_path, anno_path, file_name] = data_path_fn(img_i);
        [rgb_aug_path, anno_aug_path] = data_aug_path_fn(img_i, resizeImageSize(1), imageSize(1));
        if i == 1
            [st ms] = mkdir(fileparts(rgb_aug_path)) ;
            [st ms] = mkdir(fileparts(anno_aug_path)) ;
        end
        
        if exist(anno_aug_path, 'file')
            continue;
        end
        fprintf('.')
        
        I = imread(rgb_path);
        resize_img = min(resizeImageSize(1)/size(I,1), resizeImageSize(1)/size(I,2));
        I = imresize(I, resize_img, resize_method);
        %assert(max(size(I,1),size(I,2))==resizeImageSize(1));
        
        I = padAndCrop(I, 128, imageSize);
        %assert(size(I,1)==imageSize(1));
        %assert(size(I,2)==imageSize(1));
        imwrite(I/255, rgb_aug_path);
        
        anno = imread(anno_path);
        anno = imresize(anno, resize_img, 'nearest');
        
        anno = padAndCrop(anno, 255, imageSize);
        
        imwrite(uint8(anno), cmap, anno_aug_path, 'png');
        
        if mod(i,500) == 1
            disp(anno_aug_path);
            disp(rgb_aug_path);
        end
    end
    
    
end



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


function cmap = labelColors()
% -------------------------------------------------------------------------
N=21;
cmap = zeros(N,3);
for i=1:N
    id = i-1; r=0;g=0;b=0;
    for j=0:7
        r = bitor(r, bitshift(bitget(id,1),7 - j));
        g = bitor(g, bitshift(bitget(id,2),7 - j));
        b = bitor(b, bitshift(bitget(id,3),7 - j));
        id = bitshift(id,-3);
    end
    cmap(i,1)=r; cmap(i,2)=g; cmap(i,3)=b;
end
cmap = cmap / 255;

