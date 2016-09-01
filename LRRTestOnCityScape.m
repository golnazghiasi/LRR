function info = LRRTestOnCityScape()

path_to_matconvnet = '../matconvnet-1.0-beta20/';
fprintf('path to matconvnet library: %s\n', path_to_matconvnet);
run(fullfile(path_to_matconvnet, 'matlab/vl_setupnn.m'));
addpath(fullfile(path_to_matconvnet, 'examples'));
addpath modelInitialization;
addpath prepareData;
addpath util;

% Experiment and data paths
opts.expDir = fullfile('models/LRR4x-VGG16-CityScapes-coarse-and-fine/');
opts.dataDir = 'data/CityScapes/' ;
opts.includeCoarseData = 0;
opts.modelPath = fullfile(opts.expDir , ['model.mat']);
opts.image_set = 2;
opts.imdbPath = fullfile(opts.expDir, ['imdb' '.mat']) ;
opts.gpus = [];
% Use 0 to not visualize any segmentation predictions.
opts.max_visualize = 10;
opts.resize_fractions = [1];

% -------------------------------------------------------------------------
% Setups data
% -------------------------------------------------------------------------

% Gets CityScape dataset.
if exist(opts.imdbPath, 'file')
    imdb = load(opts.imdbPath) ;
else
    imdb = CityScapeSetup(opts.dataDir, opts.includeCoarseData);
    save(opts.imdbPath, '-struct', 'imdb') ;
end

% Gets validation subset.
val = find(imdb.images.set == 2) ;
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
end

% Runs model on two sub-parts of the image with overlap and
% then combines their results.
im_parts(1).r1 = 0;
im_parts(1).r2 = 1;
im_parts(1).c1 = 0;
im_parts(1).c2 = 1/2 + 1/8;

im_parts(2).r1 = 0;
im_parts(2).r2 = 1;
im_parts(2).c1 = 1/2 - 1/8;
im_parts(2).c2 = 1;
%disp(im_parts);

confusion = cell(1, length(prob_var_ids));
confusion(:) = {zeros(imdb.num_classes)};
imgs = imdb.images;

for i = 1 : numel(val)
    fprintf('%d/%d\t', i, numel(val));
    img_i = val(i) ;
    name = imdb.images.name{img_i};
    
    labelsPath = sprintf(imdb.anno_path, imgs.type{img_i}, imdb.sets.name{imgs.set(img_i)}, imgs.city{img_i}, imgs.filename{img_i});
    rgbPath = sprintf(imdb.img_path, imdb.sets.name{imgs.set(img_i)}, imgs.city{img_i}, imgs.name{img_i});
    
    % Loads an image and its gt segmentation.
    rgb = imread(rgbPath);
    anno = imread(labelsPath) ;
    
    anno_tid = double(imdb.classes.trainid(anno+1));
    anno = single(mod(anno_tid + 1, 256)); % 0: don't care
    lb = single(anno) ;
    
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
        prob_maps_ =  EvalPartsOfImage(net, net_input, im_parts, inputVar, prob_var_ids);
        %net.eval({inputVar, net_input});
        
        prob_maps = {};
        for k = 1 : length(prob_var_ids)
            prob_maps_k = prob_maps_{k}; %gather(net.vars(prob_var_ids(k)).value) ;
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
            accumarray([lb(ok), preds_pind(ok)], 1, [imdb.num_classes imdb.num_classes]) ;
        segmentation_predictions{pind} = preds_pind;
    end
    
    % Computes multi scale segmentation prediction for the main output of
    % the network (4x).
    %[~, ms_preds] = max(multi_scale_res, [], 3);
    %confusion{end} = confusion{end} + ...
    %    accumarray([lb(ok), ms_preds(ok)], 1, [imdb.num_classes imdb.num_classes]) ;
    
    % Visualizes prediction results.
    if i < opts.max_visualize
        showGroundtruth(rgb, lb);
        visualizePredictions(segmentation_predictions, upnames);
    end
    
end

% -------------------------------------------------------------------------
% Evaluating
% -------------------------------------------------------------------------
for pind = 1 : length(confusion)
    fprintf('-----------------------------------------------------------');
    fprintf('\n%s\n', upnames{pind});
    clear info;
    [info.iu, info.miu, info.pacc, info.macc] = ...
        getAccuracies(confusion{pind});
    fprintf('%4.1f ', 100 * info.iu);
    fprintf('\n meanIU: %5.2f pixelAcc: %5.2f, meanAcc: %5.2f\n', ...
        100*info.miu, 100*info.pacc, 100*info.macc);
    
    if pind == length(confusion)
        figure; imagesc(normalizeConfusion(confusion{pind}));
        axis image; set(gca, 'ydir', 'normal');
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
cmap = CityScapeLabelColors();
for k = 1 : length(prob_var_names)
    figure(k + 2); clf; image(uint8(segmentation_predictions{k})); colormap(cmap);
    axis 'image'; axis 'off'; title(prob_var_names{k});
end
if exist('ms_pred', 'var')
    figure(k + 3); clf; image(uint8(ms_pred)); colormap(cmap);
    axis 'image'; axis 'off'; title('multi-scale (4x)');
end
disp('Press any key to continue');
pause;

% -------------------------------------------------------------------------
function showGroundtruth(rgb, lb, save_dir)
% -------------------------------------------------------------------------
figure(1); image(rgb); axis 'image'; axis 'off';
figure(2); image(uint8(lb)); colormap(CityScapeLabelColors());
title('ground-truth'); axis 'image'; axis 'off';

function [prob_maps] =  EvalPartsOfImage(net, im, im_parts, inputVar, probVars);
for i = 1 : length(im_parts)
    rs = 1 + im_parts(i).r1 * size(im, 1) : im_parts(i).r2 * size(im, 1);
    cs = 1 + im_parts(i).c1 * size(im, 2) : im_parts(i).c2 * size(im, 2);
    im_ = im(rs, cs, :);
    net.eval({inputVar, im_});
    
    for k = 1 : length(probVars)
        prob_maps_ = gather(net.vars(probVars(k)).value) ;
        sz = [size(im, 1), size(im, 2)] * size(prob_maps_, 1) / size(im, 1);
        assert(size(prob_maps_, 3) == 19);
        sz(3) = size(prob_maps_, 3);
        rs = 1 + im_parts(i).r1 * sz(1) : im_parts(i).r2 * sz(1);
        cs = 1 + im_parts(i).c1 * sz(2) : im_parts(i).c2 * sz(2);
        if i == 1
            score_maps{k} = zeros(sz);
            prob_maps{k} = zeros(sz);
            score_maps_n{k} = zeros(sz);
            prob_maps_n{k} = zeros(sz);
        end
        
        b_cs = 1;
        e_cs = sz(2);
        b_cs_ = 1;
        e_cs_ = size(prob_maps_, 2);
        if cs(1) > 1 + eps
            nrem = length(cs) / 8;
            b_cs = 1 + im_parts(i).c1 * sz(2)  + nrem;
            b_cs_ = 1 + nrem;
        end
        if cs(end) < sz(2)
            nrem = length(cs) / 8;
            e_cs = im_parts(i).c2 * sz(2) - nrem;
            e_cs_ = size(prob_maps_, 2) - nrem;
        end
        cs_nborder = b_cs : e_cs;
        prob_maps{k}(rs, cs_nborder, :) =  max(prob_maps{k}(rs, cs_nborder, :), ...
            prob_maps_(:, b_cs_ : e_cs_, 1:19));
    end
end
