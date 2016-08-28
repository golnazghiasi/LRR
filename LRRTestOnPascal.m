function info = LRRTestOnPascal()

path_to_matconvnet = '../matconvnet-1.0-beta20/';
fprintf('path to matconvnet library: %s\n', path_to_matconvnet);
run(fullfile(path_to_matconvnet, 'matlab/vl_setupnn.m'));
addpath(fullfile(path_to_matconvnet, 'examples'));
addpath modelInitialization;
addpath prepareData;
addpath util;

% Experiment and data paths
opts.expDir = fullfile('models/LRR4x-VGG16-coco-pascal/');
opts.vocEdition = '11';
opts.dataDir = ['data/voc' opts.vocEdition];
opts.archiveDir = 'data/archives';
opts.modelPath = fullfile(opts.expDir , ['model.mat']);
opts.image_set = 2;
opts.imdbPath = fullfile(opts.expDir, ['imdb' '.mat']) ;
opts.vocAdditionalSegmentations = true;
opts.vocAdditionalSegmentationsMergeMode = 2 ;
opts.gpus = [2];
% Use 0 to not visualize any segmentation predictions.
opts.max_visualize = 10;
opts.resize_fractions = [0.6 0.8 1];
%resize_fractions = [1];

% -------------------------------------------------------------------------
% Setups data
% -------------------------------------------------------------------------

% Gets PASCAL VOC 11/12 segmentation dataset plus Berkeley's additional
% segmentations
if exist(opts.imdbPath, 'file')
    imdb = load(opts.imdbPath) ;
else
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
    save(opts.imdbPath, '-struct', 'imdb') ;
end

% Gets validation subset
val = find(imdb.images.set == 2 & imdb.images.segmentation) ;
fprintf('Number of validation data: %d\n', length(val));

% -------------------------------------------------------------------------
% Setups model
% -------------------------------------------------------------------------
net = load(opts.modelPath) ;
net = dagnn.DagNN.loadobj(net.net) ;

% Saves visualization of the model.
net_ = net.saveobj() ;
save_dot_path = fullfile(opts.expDir, ['model-vis.dot']);
model2dot(net_, save_dot_path, 'inputs', {'input', [224, 224, 3, 10], ...
    'label1', [224, 224, 10], 'label2', [112, 112, 10], ...
    'label4', [64, 64, 10], 'label8', [32, 32, 10]});
save_png_path = fullfile(opts.expDir, ['model-vis.png']);
system(['dot ' save_dot_path ' -Tpng -o ' save_png_path]);
fprintf('visualization of the model saved to %s\n', save_png_path);

net.mode = 'test' ;
inputVar = 'input' ;
net.meta.normalization.averageImage = ...
    reshape(net.meta.normalization.rgbMean, 1, 1, 3);

upnames ={'32x', '16x', '8x', '4x'};
% Removes objective and accuracy layers.
for i = 1 : length(upnames)
    layer_name = ['objective_' upnames{i}];
    if ~isnan(net.getLayerIndex(layer_name))
        net.removeLayer(layer_name);
    end
    layer_name = ['accuracy_' upnames{i}];
    if ~isnan(net.getLayerIndex(layer_name))
        net.removeLayer(layer_name);
    end
    layer_name = ['obj_dil_mask' upnames{i}];
    if ~isnan(net.getLayerIndex(layer_name))
        net.removeLayer(layer_name);
    end
end

prob_var_ids = zeros(1,length(upnames));
for i = 1 : length(upnames)
    prob_var_name = ['prob_' upnames{i}];
    if isnan(net.getVarIndex(prob_var_name))
        net.addLayer(prob_var_name, dagnn.SoftMax(), ...
            ['prediction_' upnames{i}], prob_var_name, {});
    end
    prob_var_ids(i) = net.getVarIndex(prob_var_name);
    net.vars(prob_var_ids(i)).precious = 1 ;
end

% -------------------------------------------------------------------------
% Testing
% -------------------------------------------------------------------------
if ~isempty(opts.gpus)
    gpuDevice(opts.gpus(1))
    net.move('gpu') ;
    pause;
end

num_classes = length(imdb.classes.name);
confusion = cell(1, length(prob_var_ids) + 1);
confusion(:) = {zeros(num_classes)};

for i = 1 : numel(val)
    fprintf('%d/%d\t', i, numel(val));
    
    % Loads an image and its gt segmentation.
    name = imdb.images.name{val(i)};
    rgb = imread(fullfile(sprintf(imdb.paths.image, name)));
    anno = imread(fullfile(sprintf(imdb.paths.classSegmentation, name))) ;
    orig_lb = single(anno) ;
    lb = mod(orig_lb + 1, 256) ; % 0 = ignore, 1 = bkg
    
    % Subtracts the mean (color).
    im = bsxfun(@minus, single(rgb), net.meta.normalization.averageImage) ;
    
    % Runs network with different scales of the input image.
    multi_scale_res = -inf;
    for ri = 1 : length(opts.resize_fractions)
        
        % Resizes input image
        % (size of the input to the network should be multipie of 32).
        [net_input, rinds, cinds] = resizeMult32(im, opts.resize_fractions(ri));
        
        if ~isempty(opts.gpus)
            net_input = gpuArray(net_input);
        end
        
        net.eval({inputVar, net_input});
        
        prob_maps = {};
        for k = 1 : length(prob_var_ids)
            prob_maps_k = gather(net.vars(prob_var_ids(k)).value) ;
            prob_maps{k} = back2ImageSize(prob_maps_k, size(net_input), ...
                size(im), rinds, cinds);
            
            % Computes multi-scale results from main output (4x output)
            % of the network.
            if k == length(prob_var_ids)
                multi_scale_res = max(multi_scale_res, prob_maps{k});
            end
        end
    end
    
    % Computes performance for intermediate and final outputs of
    % the network (when the input images of the network are resized to
    % resize_fraction(end)
    ok = lb > 0 ;
    segmentation_predictions = {};
    for pind = 1 : length(prob_var_ids)
        [~, preds_pind] = max(prob_maps{pind}, [], 3);
        confusion{pind} = confusion{pind} + ...
            accumarray([lb(ok), preds_pind(ok)], 1, [num_classes num_classes]) ;
        segmentation_predictions{pind} = preds_pind;
    end
    
    % Computes multi scale segmentation prediction for the main output of
    % the network (4x).
    [~, ms_preds] = max(multi_scale_res, [], 3);
    confusion{end} = confusion{end} + ...
        accumarray([lb(ok), ms_preds(ok)], 1, [num_classes num_classes]) ;
    
    % Saves prediction results.
    if i < opts.max_visualize
        showGroundtruth(rgb, orig_lb);
        visualizePredictions(segmentation_predictions, upnames, ms_preds);
    end
    
end

% -------------------------------------------------------------------------
% Evaluating
% -------------------------------------------------------------------------
for pind = 1 : length(confusion)
    fprintf('-----------------------------------------------------------');
    if pind == length(confusion)
        fprintf('\n4x %s\n', 'multi-scale');
    else
        fprintf('\n%s\n', upnames{pind});
    end
    clear info;
    [info.iu, info.miu, info.pacc, info.macc] = ...
        getAccuracies(confusion{pind});
    fprintf('%4.1f ', 100 * info.iu);
    fprintf('\n meanIU: %5.2f pixelAcc: %5.2f, meanAcc: %5.2f\n', ...
        100*info.miu, 100*info.pacc, 100*info.macc);
    
    if pind == length(confusion)
        figure; imagesc(normalizeConfusion(confusion{pind}));
        axis image; set(gca,'ydir','normal');
        colormap(jet);
        drawnow;
    end
end

% -------------------------------------------------------------------------
function nconfusion = normalizeConfusion(confusion)
% -------------------------------------------------------------------------
% Normalizes confusion by row (each row contains a gt label)
nconfusion = bsxfun(@rdivide, double(confusion), double(sum(confusion,2)));

% -------------------------------------------------------------------------
function [IU, meanIU, pixelAccuracy, meanAccuracy] = getAccuracies(confusion)
% -------------------------------------------------------------------------
pos = sum(confusion,2) ;
res = sum(confusion,1)' ;
tp = diag(confusion) ;
IU = tp ./ max(1, pos + res - tp) ;
meanIU = mean(IU) ;
pixelAccuracy = sum(tp) / max(1,sum(confusion(:))) ;
meanAccuracy = mean(tp ./ max(1, pos)) ;

% -------------------------------------------------------------------------
function [net_input, rinds, cinds] = resizeMult32(im, resize_frac)
% -------------------------------------------------------------------------
approximate_size = [size(im, 1) size(im, 2)] * resize_frac;
net_input_size = round(approximate_size / 32)*32;
resize_size = min(net_input_size./approximate_size - eps) * approximate_size;
im = imresize(im, resize_size, 'bicubic') ;

net_input = zeros(net_input_size(1), net_input_size(2), ...
    size(im, 3), 'single');
[rinds, cinds] = subInds(im, net_input_size);
net_input(rinds, cinds, :) = im;

% -------------------------------------------------------------------------
function score_map = back2ImageSize(score_map, net_input_size, im_size, ...
    rinds, cinds)
% -------------------------------------------------------------------------
score_map = imresize(score_map, net_input_size(1 : 2), 'bicubic') ;
score_map = score_map(rinds, cinds, :);
score_map = imresize(score_map, im_size(1 : 2), 'bicubic');

% -------------------------------------------------------------------------
function [rinds, cinds] = subInds(im, s)
% -------------------------------------------------------------------------
assert(size(im,1) <= s(1));
assert(size(im,2) <= s(2));
rb = 1 + ceil((s(1) - size(im,1))/2);
cb = 1 + ceil((s(2) - size(im,2))/2);
rinds = rb:rb-1+size(im,1);
cinds = cb:cb-1+size(im,2);

% -------------------------------------------------------------------------
function visualizePredictions(segmentation_predictions, prob_var_names, ms_pred)
% -------------------------------------------------------------------------
cmap = PascalLabelColors();
for k = 1 : length(prob_var_names)
    figure(k + 2); clf; image(segmentation_predictions{k}); colormap(cmap);
    axis 'image'; axis 'off'; title(prob_var_names{k});
end
figure(k + 3); clf; image(ms_pred); colormap(cmap);
axis 'image'; axis 'off'; title('multi-scale (4x)');
disp('Press any key to continue');
pause;

% -------------------------------------------------------------------------
function showGroundtruth(rgb, lb, save_dir)
% -------------------------------------------------------------------------
figure(1); image(rgb); axis 'image'; axis 'off';
figure(2); image(lb+1); colormap(PascalLabelColors());
title('ground-truth'); axis 'image'; axis 'off';

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

