function ComputeIniBases(imdb, get_anno_rgb_path_fn, train, patch_size, opts)
fprintf('\nComputing deconvolutional bases ... \n');

if ~exist(opts.iniBasesDir, 'dir')
    mkdir(opts.iniBasesDir);
end

f = zeros(patch_size, patch_size, 1, opts.num_classes * opts.num_basis);
max_basis = 200;
for class_ind = 1 : opts.num_classes
    class_bases_path = fullfile(opts.iniBasesDir, ['bases_' num2str(max_basis) 'class' num2str(class_ind)  '_ps' num2str(patch_size) '.mat']);
    disp(class_bases_path);
    try
        load(class_bases_path, 'T');
    catch
        class_patches_path = fullfile(opts.iniBasesDir, ['class_patches_' num2str(class_ind) '.mat']);
        patches = ClassPatches(imdb, get_anno_rgb_path_fn, train, patch_size, class_ind, class_patches_path);
        
        m = mean(patches,1);
        D = patches - repmat(m, size(patches,1),1);
        
        [U,S,V] = svds(D, max_basis);              % compute svd
        T = sqrt(S(1:max_basis,1:max_basis))*V';   % bases for data
        
        % Visualizes first 16
        %for i = 1:16
        %    I1 = T(i,:); I1 = reshape(I1,patch_size,patch_size);
        %    subplot(4,4,i); imagesc(I1); colorbar; axis 'equal';
        %end
        %export_fig(fullfile(opts.iniBasesDir, ['bases_' num2str(max_basis) 'class' num2str(class_ind) '_ps' num2str(patch_size)]));
        save(class_bases_path, 'T');
    end
    
    st = (class_ind - 1) * opts.num_basis;
    for i = 1 : opts.num_basis
        t = reshape(T(i, :), [32 32]);
        f(:, :, 1, st + i) = t;
    end
end

save(opts.bases_add, 'f');

class_names = imdb.classes.name;
%assert(length(class_names) == opts.num_classes);

if 0
	addpath '~/export_fig'
	% Visualizes bases
    vis_dir = fullfile(opts.iniBasesDir, 'vis');
    if ~exist(vis_dir, 'dir')
        mkdir(vis_dir);
    end
    for i = 1 : opts.num_classes
        fprintf('%d/%d\n', i, opts.num_classes);
        clf;
        for ind = 1 : opts.num_basis
            subplot(6, 5, ind); imagesc(squeeze(f(:,:,:,(i-1)*opts.num_basis + ind))); axis 'equal'; axis 'off';
        end
        title(class_names{i});
        export_fig(fullfile(vis_dir, ['f-' num2str(i) '-' class_names{i} '.jpg']),'-r300');
    end
end

