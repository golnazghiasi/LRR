function patches = ClassPatches(imdb, get_anno_rgb_path_fn, train, patch_size, class_ind, class_patches_path)

try
    load(class_patches_path);
catch
    
    debug = false;
    close all;
    
    max_entry = 10000;
    ind = 0; %index of training example
    cnt = 0; %index of patchs
    patches = zeros(max_entry, patch_size * patch_size);
    
    while(1)
        ind = ind + 1;
        if(ind == length(train) + 1)
            ind = 1;
        end
        if rem(ind, 1000) == 1
            fprintf('class ind %d :: %d/%d\t', class_ind, ind, numel(train));
        	fprintf('number of collected patches: %d\n', cnt);
        end
        [rgb_path, anno_path, file_name] = get_anno_rgb_path_fn(train(ind));
        
        % Loads an image and its gt segmentation.
        rgb = imread(rgb_path);
        anno = imread(anno_path);
        lb = single(mod(anno + 1, 256)); % 0: don't care
        
        if debug
            figure(1); clf; image(rgb); axis('equal');
            figure(2); clf; imagesc(lb); axis('equal'); colorbar;
        end
        
        for i = 1 : 100
            str = randi(size(rgb,1)-patch_size-1);
            stc = randi(size(rgb,2)-patch_size-1);
            sub_rgb = rgb(str:str+patch_size-1, stc:stc+patch_size-1, :);
            sub_lb = lb(str:str+patch_size-1, stc:stc+patch_size-1,:);
            
            if debug
                figure(11); clf; image(sub_rgb); axis('equal');
                figure(12); clf; imagesc(sub_lb); axis('equal'); colorbar;
            end
            
            if(sum(sub_lb(:)==class_ind)<0.02*numel(sub_lb))
                continue;
            end
            
            d = zeros(patch_size, patch_size);
            d(sub_lb == class_ind) = 1;
            
            cnt = cnt + 1;
            patches(cnt, :) = d(:)';
            
            if debug
                figure(13); clf; imagesc(d); axis('equal'); colorbar;
                pause;
            end
        end
        if cnt>=max_entry
            break;
        end
    end
    
    save(class_patches_path, 'patches');
end
