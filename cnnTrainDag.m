% This file is from Matconvnet library (https://github.com/vlfeat/matconvnet)
% and has minor modifications.
function [net,stats] = cnnTrainDag(net, imdb, getBatch_train, getBatch_val, batchSize_train, batchSize_val, varargin)

%CNN_TRAIN_DAG Demonstrates training a CNN using the DagNN wrapper
%    CNN_TRAIN_DAG() is similar to CNN_TRAIN(), but works with
%    the DagNN wrapper instead of the SimpleNN wrapper.

% Copyright (C) 2014-15 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

%addpath ~/export_fig/;

opts.expDir = fullfile('data','exp') ;
opts.continue = true ;
opts.batchSize = 256 ;
opts.numSubBatches = 1 ;
opts.train = [] ;
opts.val = [] ;
opts.gpus = [] ;
opts.prefetch = false ;
opts.numEpochs = 300 ;
opts.learningRate = 0.001 ;
opts.weightDecay = 0.0005 ;
opts.momentum = 0.9 ;
opts.randomSeed = 0 ;
opts.memoryMapFile = fullfile(tempdir, 'matconvnet.bin') ;
opts.profile = false;
opts.train_imageSize_multi = [0];
opts.gt_resizes = [];
opts.num_classes = 21;

opts.derOutputs = {'objective', 1} ;
opts.extractStatsFn = @extractStats ;
opts.plotStatistics = true;
opts = vl_argparse(opts, varargin) ;

if ~exist(opts.expDir, 'dir'), mkdir(opts.expDir) ; end
if isempty(opts.train), opts.train = find(imdb.images.set==1) ; end
if isempty(opts.val), opts.val = find(imdb.images.set==2) ; end
if isnan(opts.train), opts.train = [] ; end

% -------------------------------------------------------------------------
%                                                            Initialization
% -------------------------------------------------------------------------

evaluateMode = isempty(opts.train) ;
if ~evaluateMode
    if isempty(opts.derOutputs)
        error('DEROUTPUTS must be specified when training.\n') ;
    end
end
stats = [] ;

% setup GPUs
numGpus = numel(opts.gpus) ;
if numGpus > 1
    if isempty(gcp('nocreate')),
        parpool('local',numGpus) ;
        spmd, gpuDevice(opts.gpus(labindex)), end
    end
    if exist(opts.memoryMapFile)
        delete(opts.memoryMapFile) ;
    end
elseif numGpus == 1
    gpuDevice(opts.gpus)
end

% -------------------------------------------------------------------------
%                                                        Train and validate
% -------------------------------------------------------------------------

modelPath = @(ep) fullfile(opts.expDir, sprintf('net-epoch-%d.mat', ep));
modelFigPath = fullfile(opts.expDir, 'net-train.pdf') ;

start = min(opts.continue * findLastCheckpoint(opts.expDir) , opts.numEpochs);
if start >= 1
    fprintf('%s: resuming by loading epoch %d\n', mfilename, start) ;
    [net, stats] = loadState(modelPath(start)) ;
end


for epoch=start+1:opts.numEpochs
    
    rng(epoch + opts.randomSeed) ;
    
    if opts.profile
        profile 'on';
    end
    % train one epoch
    state.epoch = epoch ;
    state.learningRate = opts.learningRate(min(epoch, numel(opts.learningRate))) ;
    state.train = opts.train(randperm(numel(opts.train))) ; % shuffle
    state.val = opts.val ;
    state.imdb = imdb ;
    
    if numGpus <= 1
        state.getBatch = getBatch_train{1};
        opts.batchSize = batchSize_train;
        stats.train(epoch) = process_epoch(net, state, opts, 'train') ;
        state.getBatch = getBatch_val{1};
        opts.batchSize = batchSize_val(1);
        stats.val1(epoch) = process_epoch(net, state, opts, 'val') ;
        if length(getBatch_val) > 1
            state.getBatch = getBatch_val{2};
            opts.batchSize = batchSize_val(2);
            stats.val2(epoch) = process_epoch(net, state, opts, 'val') ;
        end
    else
        savedNet = net.saveobj() ;
        spmd
            net_ = dagnn.DagNN.loadobj(savedNet) ;
            stats_.train = process_epoch(net_, state, opts, 'train') ;
            stats_.val = process_epoch(net_, state, opts, 'val') ;
            if labindex == 1, savedNet_ = net_.saveobj() ; end
        end
        net = dagnn.DagNN.loadobj(savedNet_{1}) ;
        stats__ = accumulateStats(stats_) ;
        stats.train(epoch) = stats__.train ;
        stats.val(epoch) = stats__.val ;
        clear net_ stats_ stats__ savedNet_ ;
    end
    
    if ~evaluateMode
        saveState(modelPath(epoch), net, stats) ;
    end
    
    close all;
    for f = setdiff(fieldnames(stats.val1)', {'num', 'time'})
        h = figure;
        f = char(f);
        b_names = {'pixelAcc', 'meanAcc', 'meanIU'};
        test_names = {'train resize 384', 'val resize 384', 'val orig size'};
        line_styles = {'-', ':', '-.','-','c'};
        cols = {'g','b','r','m','c'};
        leg = {} ;
        k = 0;
        loc = 'northeast';
        for s = {'train', 'val1', 'val2'}
            k = k + 1;
            s = char(s);
            if ~isfield(stats, s) || ~isfield(stats.(s), f)
                continue;
            end
            vals = [stats.(s).(f)];
            if(size(vals, 1)> 1)
                loc = 'northwest';
            end
            for b = 1 : size(vals, 1)
                plot(1:epoch, vals(b, :), 'LineStyle', line_styles{k}, 'Color', cols{b}, 'LineWidth', 3); hold on;
                b_name = f;
                if(size(vals, 1)> 1)
                    b_name = b_names{b};
                end
                leg{end+1} = sprintf('%s (%s)', b_name, test_names{k}) ;
            end
        end
        legend(leg{:}, 'Location', loc) ; xlabel('epoch') ; ylabel('metric') ;
        grid on ; drawnow ;
        f(f=='_') = '-';
        title(f);
        save_path = [modelFigPath(1 : end - 4) '-' f '.pdf'];
        print(h, save_path, '-dpdf') ;
        %export_fig(save_path);
    end
    if opts.profile
        p = profile('info');
        save_profile_path = fullfile(opts.expDir, ['profile-epoch' num2str(epoch)] ) ;
        profsave(p, [save_profile_path 'profile_']);
    end
end

% -------------------------------------------------------------------------
function stats = process_epoch(net, state, opts, mode)
% -------------------------------------------------------------------------
disp(opts);
if 1
    if ~strcmp(mode,'train')
        % Adds bilinear upsampling up to the image size to perform
        % evaluation on predictions with the input image size.
        for i = 1 : 1 %length(opts.gt_resizes)
            
            bilinear_upsample = opts.gt_resizes(i);
            
            upsample_fac = 4 * bilinear_upsample;
            up_name = [num2str(upsample_fac) 'x'];
            prediction_var_name = ['prediction_' up_name];
            
            if bilinear_upsample > 1
                upsampled_var = [prediction_var_name '_biup'];
                net = AddBilinearUpSampling(net, prediction_var_name, upsampled_var, bilinear_upsample, opts);
                
                biup_prediction_var_name = upsampled_var;
                
                % Adds an accuracy layer
                net.addLayer(['accuracy_biup_' up_name], ...
                    SegmentationAccuracy('numClasses', opts.num_classes), ...
                    {biup_prediction_var_name, 'label'}, ['accuracy_biup_' up_name]) ;
            end
            
        end
        %net_ = net.saveobj() ;
        %save_dot_path = fullfile(opts.expDir,['model-add-accuracies.dot']);
        %model2dot(net_, save_dot_path, 'inputs',{'input',[224,224,3,10]});
        %save_png_path = fullfile(opts.expDir,['model-vis-add-accuracies.png']);
        %system(['dot ' save_dot_path ' -Tpng -o ' save_png_path]);
        %fprintf('visualization of the model saved to %s\n', save_png_path);
    end
end

if strcmp(mode,'train')
    state.momentum = num2cell(zeros(1, numel(net.params))) ;
end

numGpus = numel(opts.gpus) ;
if numGpus >= 1
    net.move('gpu') ;
    if strcmp(mode,'train')
        state.momentum = cellfun(@gpuArray,state.momentum,'UniformOutput',false) ;
    end
end
if numGpus > 1
    mmap = map_gradients(opts.memoryMapFile, net, numGpus) ;
else
    mmap = [] ;
end

stats.time = 0 ;
stats.num = 0 ;
subset = state.(mode) ;
start = tic ;
num = 0 ;

for t=1:opts.batchSize:numel(subset)
    % Computes the size_inds for scale augmentation
    if strcmp(mode,'train')
        next_size_ind = randi(length(opts.train_imageSize_multi));
    else
        % no multi resizing for validation or test
        next_size_ind = 1;
    end
    
    batchSize = min(opts.batchSize, numel(subset) - t + 1) ;
    for s=1:opts.numSubBatches
        % get this image batch and prefetch the next
        batchStart = t + (labindex-1) + (s-1) * numlabs ;
        batchEnd = min(t+opts.batchSize-1, numel(subset)) ;
        batch = subset(batchStart : opts.numSubBatches * numlabs : batchEnd) ;
        num = num + numel(batch) ;
        if numel(batch) == 0, continue ; end
        
        inputs = state.getBatch(state.imdb, batch, opts.gt_resizes, next_size_ind) ;
        
        if opts.prefetch
            if s == opts.numSubBatches
                batchStart = t + (labindex-1) + opts.batchSize ;
                batchEnd = min(t+2*opts.batchSize-1, numel(subset)) ;
            else
                batchStart = batchStart + numlabs ;
            end
            nextBatch = subset(batchStart : opts.numSubBatches * numlabs : batchEnd) ;
            if numel(nextBatch) > 0
                state.getBatch(state.imdb, nextBatch, opts.gt_resizes, next_size_ind) ;
            end
        end
        
        if strcmp(mode, 'train')
            net.mode = 'normal' ;
            net.accumulateParamDers = (s ~= 1) ;
            net.eval(inputs, opts.derOutputs) ;
        else
            net.mode = 'test' ;
            net.eval(inputs) ;
        end
    end
    
    % extract learning stats
    stats = opts.extractStatsFn(net) ;
    
    % accumulate gradient
    if strcmp(mode, 'train')
        if ~isempty(mmap)
            write_gradients(mmap, net) ;
            labBarrier() ;
        end
        state = accumulate_gradients(state, net, opts, batchSize, mmap) ;
    end
    
    % print learning statistics
    time = toc(start) ;
    stats.num = num ;
    stats.time = toc(start) ;
    
    fprintf('%s: epoch %02d: %3d/%3d: %.1f Hz', ...
        mode, ...
        state.epoch, ...
        fix(t/opts.batchSize)+1, ceil(numel(subset)/opts.batchSize), ...
        stats.num/stats.time * max(numGpus, 1)) ;
    
    for f = setdiff(fieldnames(stats)', {'num', 'time'})
        f = char(f) ;
        fprintf(' %s:', f) ;
        fprintf(' %.3f', stats.(f)) ;
    end
    fprintf('\n') ;
end

if 1
    if ~strcmp(mode,'train')
        for i = 1 : 1%length(opts.gt_resizes)
            
            bilinear_upsample = opts.gt_resizes(i);
            
            if bilinear_upsample > 1
                upsample_fac = 4 * bilinear_upsample;
                up_name = [num2str(upsample_fac) 'x'];
                prediction_var_name = ['prediction_' up_name];
                upsampled_var = [prediction_var_name '_biup'];
                deconv_name = ['dec_' upsampled_var];
                
                net.removeLayer(deconv_name);
                net.removeLayer(['accuracy_biup_' up_name]);
            end
        end
        
        %net_ = net.saveobj() ;
        %save_dot_path = fullfile(opts.expDir,['model-rem-accuracies.dot']);
        %model2dot(net_, save_dot_path, 'inputs',{'input',[224,224,3,10]});
        %save_png_path = fullfile(opts.expDir,['model-vis-rem-accuracies.png']);
        %system(['dot ' save_dot_path ' -Tpng -o ' save_png_path]);
        %fprintf('visualization of the model saved to %s\n', save_png_path);
    end
end



net.reset() ;
net.move('cpu') ;

% -------------------------------------------------------------------------
function state = accumulate_gradients(state, net, opts, batchSize, mmap)
% -------------------------------------------------------------------------
for p=1:numel(net.params)
    
    % bring in gradients from other GPUs if any
    if ~isempty(mmap)
        numGpus = numel(mmap.Data) ;
        tmp = zeros(size(mmap.Data(labindex).(net.params(p).name)), 'single') ;
        for g = setdiff(1:numGpus, labindex)
            tmp = tmp + mmap.Data(g).(net.params(p).name) ;
        end
        net.params(p).der = net.params(p).der + tmp ;
    else
        numGpus = 1 ;
    end
    
    switch net.params(p).trainMethod
        
        case 'average' % mainly for batch normalization
            thisLR = net.params(p).learningRate ;
            net.params(p).value = ...
                (1 - thisLR) * net.params(p).value + ...
                (thisLR/batchSize/net.params(p).fanout) * net.params(p).der ;
            
        case 'gradient'
            thisDecay = opts.weightDecay * net.params(p).weightDecay;
            thisLR = state.learningRate * net.params(p).learningRate;
            state.momentum{p} = opts.momentum * state.momentum{p} ...
                - thisDecay * net.params(p).value ...
                - (1 / batchSize) * net.params(p).der ;
            net.params(p).value = net.params(p).value + thisLR * state.momentum{p} ;
            
        case 'nothing'
            assert(net.params(p).learningRate == 0);
            
        case 'otherwise'
            error('Unknown training method ''%s'' for parameter ''%s''.', ...
                net.params(p).trainMethod, ...
                net.params(p).name) ;
    end
end

% -------------------------------------------------------------------------
function mmap = map_gradients(fname, net, numGpus)
% -------------------------------------------------------------------------
format = {} ;
for i=1:numel(net.params)
    format(end+1,1:3) = {'single', size(net.params(i).value), net.params(i).name} ;
end
format(end+1,1:3) = {'double', [3 1], 'errors'} ;
if ~exist(fname) && (labindex == 1)
    f = fopen(fname,'wb') ;
    for g=1:numGpus
        for i=1:size(format,1)
            fwrite(f,zeros(format{i,2},format{i,1}),format{i,1}) ;
        end
    end
    fclose(f) ;
end
labBarrier() ;
mmap = memmapfile(fname, 'Format', format, 'Repeat', numGpus, 'Writable', true) ;

% -------------------------------------------------------------------------
function write_gradients(mmap, net)
% -------------------------------------------------------------------------
for i=1:numel(net.params)
    mmap.Data(labindex).(net.params(i).name) = gather(net.params(i).der) ;
end

% -------------------------------------------------------------------------
function stats = accumulateStats(stats_)
% -------------------------------------------------------------------------

stats = struct() ;

for s = {'train', 'val'}
    s = char(s) ;
    total = 0 ;
    
    for g = 1:numel(stats_)
        stats__ = stats_{g} ;
        num__ = stats__.(s).num ;
        total = total + num__ ;
        
        for f = setdiff(fieldnames(stats__.(s))', 'num')
            f = char(f) ;
            
            if g == 1
                stats.(s).(f) = 0 ;
            end
            stats.(s).(f) = stats.(s).(f) + stats__.(s).(f) * num__ ;
            
            if g == numel(stats_)
                stats.(s).(f) = stats.(s).(f) / total ;
            end
        end
    end
    stats.(s).num = total ;
end

% -------------------------------------------------------------------------
function stats = extractStats(net)
% -------------------------------------------------------------------------
sel = find(cellfun(@(x) isa(x,'dagnn.Loss'), {net.layers.block})) ;
stats = struct() ;
for i = 1:numel(sel)
    stats.(net.layers(sel(i)).outputs{1}) = net.layers(sel(i)).block.average ;
end

% -------------------------------------------------------------------------
function saveState(fileName, net, stats)
% -------------------------------------------------------------------------
net_ = net ;
net = net_.saveobj() ;
save(fileName, 'net', 'stats') ;

% -------------------------------------------------------------------------
function [net, stats] = loadState(fileName)
% -------------------------------------------------------------------------
load(fileName, 'net', 'stats') ;
net = dagnn.DagNN.loadobj(net) ;

% -------------------------------------------------------------------------
function epoch = findLastCheckpoint(modelDir)
% -------------------------------------------------------------------------
list = dir(fullfile(modelDir, 'net-epoch-*.mat')) ;
tokens = regexp({list.name}, 'net-epoch-([\d]+).mat', 'tokens') ;
epoch = cellfun(@(x) sscanf(x{1}{1}, '%d'), tokens) ;
epoch = max([epoch 0]) ;
