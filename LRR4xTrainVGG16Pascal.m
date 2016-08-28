function LRR4xTrainVGG16Pascal(varargin)

path_to_matconvnet = '../matconvnet-1.0-beta20/';
fprintf('path to matconvnet library: %s\n', path_to_matconvnet);
run(fullfile(path_to_matconvnet, 'matlab/vl_setupnn.m'));
addpath(fullfile(path_to_matconvnet, 'examples'));
addpath prepareData;
addpath deconvBases;
addpath util;
addpath modelInitialization;
%addpath ~/export_fig/;

rng(0);

% Experiment and data paths
opts.modelsDir = 'models';
opts.expDir = fullfile(opts.modelsDir, 'LRR4x-VGG16-pascal-train/');
opts.iniBasesDir = fullfile(opts.expDir, 'deconvBases');
opts.vocEdition = '11';
opts.dataDir = ['data/voc' opts.vocEdition];
opts.pascalDataDir = ['data/voc' opts.vocEdition];
opts.archiveDir = 'data/archives';
opts.image_set = 2;
opts.imdbPath = fullfile(opts.expDir, ['imdb' '.mat']) ;
opts.imdbPascalPath = fullfile(opts.expDir, 'imdb_pascal.mat');
opts.imdbStatsPath = fullfile(opts.expDir, 'imdbStats.mat') ;
opts.vocAdditionalSegmentations = true;
opts.vocAdditionalSegmentationsMergeMode = 2 ;

% Options for computing deconvolution bases
opts.patch_size = 32;
opts.num_basis = 10;

% reconstruction factor for all levels.
opts.rec_upsample = 4;
% resizing factor for training intermediate and final objectvies.
opts.gt_resizes = [32, 16, 8, 4] / opts.rec_upsample;
% Options for adding dilation and erossion objectives
opts.dilate_erode_seg = [1 0 0 0];
% Element size for eroding and dilating ground-truth labels for Dilation
% and erosion objectives.
opts.strelRad = 32;
% Path to save precomputed augmented data and ground-truth labels for
% dilation and erosion objectives.
opts.save_augment_data_dir = 'data/resized_data';
% Whether to pre-compute and save dilated and eroded ground-truth labels. If true, the
% preparation may take a long time. But, the training will be speed up.
opts.preComputeSaveDilateErodeGT = false;

% training options (SGD)
opts.train.batchSize = 20;
% For training the full model, we may require to change numSubBatches to 2.
opts.train.numSubBatches = 1;
opts.train.continue = true ;
opts.train.gpus = [2];

% -------------------------------------------------------------------------
% Setups data fetching options for training batches
bopts.useGpu = numel(opts.train.gpus) > 0 ;
bopts.imageSize = [384, 384];
bopts.randFlip = true;
bopts.type = 'train';
% scale augmentation, all images of a batch will be resized to
% bopts.imageSize + bopts.imageSize_multi(randi) * 32
bopts.imageSize_multi = [-3 : 1 : 10];
opts.train.train_imageSize_multi = bopts.imageSize_multi;
train_batchsize =  opts.train.batchSize;

% -------------------------------------------------------------------------
% Setups data fetching options for validation (resizes images to [384 384]).
bopts_test(1) = bopts;
bopts_test(1).randFlip = false;
bopts_test(1).imageSize_multi = [0];
bopts_test(1).type = 'test';
val_batchsize(1) = 20;

% -------------------------------------------------------------------------
% Setups data fetching options for validation (original size images).
bopts_test(2) = bopts;
bopts_test(2).imageSize = [0 0]; %original size
bopts_test(2).randFlip = false;
bopts_test(2).imageSize_multi = [0];
bopts_test(2).type = 'test';
val_batchsize(2) = 1;

% -------------------------------------------------------------------------
% Setup data
% -------------------------------------------------------------------------
if exist([opts.imdbPath ])
    imdb = load(opts.imdbPath) ;
else
    try
        imdb = load(opts.imdbPascalPath);
    catch
        imdb = vocSetup('dataDir', opts.dataDir, ...
            'archiveDir', opts.archiveDir, ...
            'edition', opts.vocEdition, ...
            'includeTest', false, ...
            'includeSegmentation', true, ...
            'includeDetection', false) ;
        if opts.vocAdditionalSegmentations
            imdb = vocSetupAdditionalSegmentations(imdb, 'dataDir', ...
                opts.dataDir, 'archiveDir', opts.archiveDir);
        end
        if ~exist(opts.expDir, 'dir')
            mkdir(opts.expDir);
        end
        imdb.classes.name = ['background' imdb.classes.name];
        imdb.classes.id = [0 imdb.classes.id];
        imdb.num_classes = length(imdb.classes.name);
        save(opts.imdbPascalPath, '-struct', 'imdb') ;
    end
    
    imdb.images.set(imdb.images.segmentation == 0 ) = 0;
    imdb.images.data_type = ones(1, length(imdb.images.id));
    imdb.images.data_type_names = {'pascal'};
    
    if exist(opts.imdbStatsPath, 'file')
        stats = load(opts.imdbStatsPath) ;
    else
        stats = getDatasetStatistics(imdb) ;
        save(opts.imdbStatsPath, '-struct', 'stats') ;
    end
    
    imdb = setPascalPaths(imdb, opts.save_augment_data_dir);
    data_path_fn = getDataPathsWrapper(imdb);
    data_aug_path_fn = getAugDataPathsWrapper(imdb);
    anno_de_path_fn = getDEAnnoPathsWrapper(imdb);
    
    % -------------------------------------------------------------------------
    % Saves scale augmented examples for speed up.
    ResizeAndSaveTrainData(bopts.imageSize_multi, bopts.imageSize, imdb, ...
        find(imdb.images.set==1) , data_path_fn, data_aug_path_fn);
    ResizeAndSaveTrainData(bopts_test(1).imageSize_multi, bopts_test(1).imageSize, ...
        imdb, find(imdb.images.set==2), data_path_fn, data_aug_path_fn);
    
    % -------------------------------------------------------------------------
    % Pre-computes and saves dilation and erosion of ground-truth lables.
    if opts.preComputeSaveDilateErodeGT
        disp('Pre-computing and saving dilation and erosion of ground-truth annotation (this may take a long time) ...');
        PrepareDilatedErrodedGT(bopts.imageSize_multi, bopts.imageSize, find(imdb.images.set==1), ...
            opts.gt_resizes, opts.strelRad, data_aug_path_fn, anno_de_path_fn, imdb.num_classes);
        disp('Done pre-computing and saving dilation and erosion of ground-truth annotation for training data.');
        PrepareDilatedErrodedGT(bopts_test(1).imageSize_multi, bopts_test(1).imageSize, find(imdb.images.set==2), ...
            opts.gt_resizes, opts.strelRad, data_aug_path_fn, anno_de_path_fn, imdb.num_classes);
        disp('Done pre-computing and saving dilation and erosion of ground-truth annotation for validation data.');
    end
    save(opts.imdbPath, '-struct', 'imdb') ;
end
opts.num_classes = imdb.num_classes;
stats = load(opts.imdbStatsPath) ;

train = find(imdb.images.set == 1);
val = find(imdb.images.set == 2);
fprintf('num train: %d, num val: %d\n', numel(train), numel(val));

% function handles for getting data paths.
data_path_fn = getDataPathsWrapper(imdb);
data_aug_path_fn = getAugDataPathsWrapper(imdb);
anno_de_path_fn = getDEAnnoPathsWrapper(imdb);

% -------------------------------------------------------------------------
% Computes deconvolution bases
% -------------------------------------------------------------------------
opts.bases_add = [opts.iniBasesDir num2str(opts.num_basis) '_fs' num2str(opts.patch_size) '.mat'];
if ~exist(opts.bases_add, 'file')
    get_img_paths_fn = getDataPathsWrapper(imdb);
    train_pascal = find(imdb.images.set == 1 & imdb.images.data_type == 1);
    ComputeIniBases(imdb, get_img_paths_fn, train_pascal, opts.patch_size, opts);
end

% -------------------------------------------------------------------------
% for debuging
% -------------------------------------------------------------------------
if 0
    train_inds = randperm(length(train));
    train = train(train_inds(1:100));
    val = val(1:50);
    fprintf('all: num train %d, num val %d\n', numel(train), numel(val));
end

% ---------------------------------------------------------------------
% Sets up the model
% ---------------------------------------------------------------------
sub_models = {'32x', '16x', '8x', '4x'};
add_maskings = [0 1 1 1];
sig = {'', '-dp-', '-dp-', '-dp-'};
vars_to_upsample={'x37', 'x30', 'x23', 'x16'}; % for VGG-16

lrs = 0.001 * [1, 1, 1, 1/16];
num_train_epochs = [65 0 0 16];
div_lr_by_2_at = {[51 61],[],[],[]};
upsample_fac = [32, 16, 8, 4];
learning_rate_for_new_vars = [0.1, 0.1/2, 0.1/8, 0.1/512];
% Size of convolution kernels for computing coeficients of deconvolution.
neigh_size = 5;

opts.sub_model = '';
opts.add_masking = 0;
for i = 1 : length(sub_models)
    Opts(i) = opts;
    Opts(i).sub_model = sub_models{i};
    Opts(i).add_masking = add_maskings(i);
    Opts(i).train.expDir = fullfile(opts.expDir, [sub_models{i} sig{i} num2str(add_maskings(i))]) ;
    Opts(i).gt_resizes = opts.gt_resizes(1 : i);
    Opts(i).strelRad = opts.strelRad(1 : min(length(opts.strelRad), i));
    Opts(i).train.numEpochs = num_train_epochs(i);
    Opts(i).train.learningRate = ones(1, num_train_epochs(i)) * lrs(i);
    for k = 1 : length(div_lr_by_2_at{i})
        Opts(i).train.learningRate(div_lr_by_2_at{i}(k) : end) = Opts(i).train.learningRate(div_lr_by_2_at{i}(k) : end) / 2;
    end
end

bopts.rgbMean = stats.rgbMean ;
bopts_test(1).rgbMean = bopts.rgbMean;
bopts_test(2).rgbMean = bopts.rgbMean;

derOutputs = cell(1, 0);
for step = 1 : length(sub_models)
    opts = Opts(step);
    fprintf('training submodel: %s\n', opts.sub_model);
    
    bopts_test(1).strelRad = opts.strelRad;
    bopts_test(2).strelRad = opts.strelRad;
    bopts.strelRad = opts.strelRad;
    
    % -------------------------------------------------------------------------
    % Constructs model
    if step == 1
        net = LRRInitializeFromVGG16(opts);
        net.meta.normalization.rgbMean = stats.rgbMean ;
        net.meta.classes = imdb.classes.name ;
    end
    
    upsample_2x_pre_layer = upsample_fac(step) * 2 / opts.rec_upsample > 1;
    net = AddSubModel(net, upsample_fac(step), upsample_fac(step)/opts.rec_upsample, vars_to_upsample{step}, ...
        opts.rec_upsample*2, opts.num_basis, neigh_size, learning_rate_for_new_vars(step), upsample_2x_pre_layer, opts);
    
    derOutputs{end+1} = ['objective_' num2str(upsample_fac(step)) 'x'];
    derOutputs{end+1} = 1;
    opts.train.derOutputs = derOutputs;
    
    if opts.dilate_erode_seg(step)
        net = AddDilationErosionObjectives(net, upsample_fac(step), opts.rec_upsample, vars_to_upsample{step}, ...
            opts.rec_upsample*2, opts.num_basis, neigh_size, learning_rate_for_new_vars(step), opts);
        derOutputs{end+1} = ['obj_dil_seg' num2str(upsample_fac(step)) 'x'];
        derOutputs{end+1} = 1;
        derOutputs{end+1} = ['obj_ero_seg' num2str(upsample_fac(step)) 'x'];
        derOutputs{end+1} = 1;
        opts.train.derOutputs = derOutputs;
    end
    
    if(opts.add_masking)
        net = LRRAddMasking(net, upsample_fac(step), upsample_fac(step)/opts.rec_upsample, upsample_2x_pre_layer);
    end
    
    if opts.train.numEpochs == 0
        continue;
    end
    
    if ~exist(opts.train.expDir, 'dir')
        mkdir(opts.train.expDir);
    end
    
    if 1
        % Saves visualization of the model.
        net_ = net.saveobj() ;
        save_dot_path = fullfile(opts.train.expDir, ['model-vis.dot']);
        model2dot(net_, save_dot_path, 'inputs', {'input', [224, 224, 3, 10]});
        %'label1', [224, 224, 10], 'label2', [112, 112, 10], ...
        %'label4', [64, 64, 10], 'label8', [32, 32, 10]});
        save_png_path = fullfile(opts.train.expDir, ['model-vis.png']);
        system(['dot ' save_dot_path ' -Tpng -o ' save_png_path]);
        fprintf('visualization of the model saved to %s\n', save_png_path);
    end
    
    % -------------------------------------------------------------------------
    % Train
    % -------------------------------------------------------------------------
    % Launch SGD
    [net, info] = cnnTrainDag(net, imdb, ...
        getBatchWrapper(bopts, data_path_fn, data_aug_path_fn, anno_de_path_fn), ...
        getBatchWrapper(bopts_test, data_path_fn, data_aug_path_fn, anno_de_path_fn), ...
        train_batchsize, val_batchsize, opts.train, ...
        'train', train, 'val', val, 'gt_resizes', opts.gt_resizes) ;
end

% -------------------------------------------------------------------------
function fn = getBatchWrapper(opts, data_path_fn, data_aug_path_fn, anno_de_path_fn)
% -------------------------------------------------------------------------
for i = 1 : length(opts)
    fn{i} = @(imdb, batch,gt_resizes,batch_image_size_ind) getBatch(imdb, batch, ...
        gt_resizes, batch_image_size_ind, data_path_fn, data_aug_path_fn, ...
        anno_de_path_fn, opts(i)) ;
end

% -------------------------------------------------------------------------
function fn = getDataPathsWrapper(imdb)
% -------------------------------------------------------------------------
fn = @(ind) DataPaths(imdb, ind);

% -------------------------------------------------------------------------
function [rgb_path, anno_path, file_name] = DataPaths(imdb, ind)
% -------------------------------------------------------------------------
file_name = imdb.images.name{ind};
anno_path = sprintf(imdb.paths.classSegmentation, file_name) ;
rgb_path = sprintf(imdb.paths.image, file_name);

% -------------------------------------------------------------------------
function fn = getAugDataPathsWrapper(imdb)
% -------------------------------------------------------------------------
fn = @(ind, resize_img_size, img_size) AugmentDataPaths(imdb, ind, resize_img_size, img_size);

% -------------------------------------------------------------------------
function [augment_rgb_path, augment_anno_path, file_name] = AugmentDataPaths(imdb, ind, resize_image_size, image_size)
% -------------------------------------------------------------------------
dt = imdb.images.data_type(ind);
resize_name = ['sz' num2str(resize_image_size) '-cr' num2str(image_size)];
file_name = imdb.images.name{ind};
augment_anno_path = sprintf(imdb.pascal_aug_anno_path, resize_name, file_name) ;
augment_rgb_path = sprintf(imdb.pascal_aug_image_path, resize_name, file_name);

% -------------------------------------------------------------------------
function fn = getDEAnnoPathsWrapper(imdb)
% -------------------------------------------------------------------------
fn = @(ind, resize_img_size, img_size, resize_fac, strel) DEAnnoPaths(imdb, ind, resize_img_size, img_size, resize_fac, strel);

% -------------------------------------------------------------------------
function [anno_path, file_name] = DEAnnoPaths(imdb, ind, resize_image_size, image_size, resize_fac, strel)
% -------------------------------------------------------------------------
resize_name = ['sz' num2str(resize_image_size) '-cr' num2str(image_size)];
file_name = imdb.images.name{ind};
anno_path = sprintf(imdb.pascal_anno_path_rde, resize_name, file_name, num2str(resize_fac), num2str(strel));

% -------------------------------------------------------------------------
function imdb = setPascalPaths(imdb, save_augment_data_dir)
% -------------------------------------------------------------------------
resized_dir_pascal = fullfile(save_augment_data_dir, 'pascal_resize');

imdb.pascal_anno_path_rde = [sprintf('%s/anno-', resized_dir_pascal) '%s/%s-r%s-d%s.mat'];

imdb.pascal_aug_anno_path = [sprintf('%s/anno-', resized_dir_pascal) '%s/%s.png'];
imdb.pascal_aug_image_path = [sprintf('%s/images-', resized_dir_pascal) '%s/%s.jpg'];

