function [imdb] = CityScapeSetup(dataDir, include_coarse)

%%{root}/{type}{video}/{split}/{city}/{city}_{seq:0>6}_{frame:0>6}_{type}{ext}
imdb.root_path = dataDir;
imdb.anno_path = [dataDir '/%s/%s/%s/%s'];
imdb.img_path = [dataDir '/leftImg8bit/%s/%s/%s_leftImg8bit.png'];

% Source images and classes
imdb.sets.id = uint8([1 2 3 4]) ;
imdb.sets.name = {'train', 'val', 'test', 'train_extra'} ;
imdb.classes.name = {...
    'unlabeled', 'ego vehicle', 'rectification border', 'out of roi', ...
    'static', 'dynamic', 'ground', 'road', 'sidewalk', 'parking', ...
    'rail track', 'building', 'wall', 'fence', 'guard rail', 'bridge', ...
    'tunnel', 'pole', 'polegroup', 'traffic light', 'traffic sign', ...
    'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', ...
    'bus', 'caravan', 'trailer', 'train', 'motorcycle', 'bicycle', 'license plate'};
imdb.classes.trainid = uint8([255 255 255 255 255 255 255 0 1 255 255 2 ...
    3 4 255 255 255 5 255 6 7 8 9 10 11 12 13 14 15 255 255 16 17 18 -1]);
imdb.classes.id = [0:33 -1];

assert(length(imdb.classes.name) == length(imdb.classes.id));
assert(length(imdb.classes.name) == length(imdb.classes.trainid));

imdb.images.id = [] ;
imdb.images.name = {} ;
imdb.images.set = [] ;

index = containers.Map() ;
% First adding gtFine annotations.
% If there are both fine and coarse annotation for one image, then the coarse one will not be added.
[imdb, index] = addImageSet(imdb, index, 'gtFine', 'train', 1, '_labelIds.png');
[imdb, index] = addImageSet(imdb, index, 'gtFine', 'val', 2, '_labelIds.png');
[imdb, index] = addImageSet(imdb, index, 'gtFine', 'test', 3, '_labelIds.png');
if include_coarse
    [imdb, index] = addImageSet(imdb, index, 'gtCoarse', 'train', 1, '_labelIds.png');
    [imdb, index] = addImageSet(imdb, index, 'gtCoarse', 'train_extra', 4, '_labelIds.png');
    [imdb, index] = addImageSet(imdb, index, 'gtCoarse', 'val', 2, '_labelIds.png');
    [imdb, index] = addImageSet(imdb, index, 'gtCoarse', 'test', 3, '_labelIds.png');
end
disp(length(imdb.images.id))

trainids = imdb.classes.trainid;
classes_name = [imdb.classes.name];
classes_name(trainids == 255 | trainids == -1) = [];
classes_name = classes_name(1:19);
imdb.num_classes = length(classes_name);
imdb.classes_names = classes_name;

%checkData(imdb);

% Compress data types
imdb.images.id = uint32(imdb.images.id) ;
imdb.images.set = uint8(imdb.images.set) ;

% Checks images on disk and get their size
imdb = getImageSizes(imdb) ;

% -------------------------------------------------------------------------
function [imdb, index] = addImageSet(imdb, index, type, setName, setCode, postFilename)
% -------------------------------------------------------------------------
%%{root}/{type}{video}/{split}/{city}/{city}_{seq:0>6}_{frame:0>6}_{type}{ext}

j = length(imdb.images.id) ;
pre_path = fullfile(imdb.root_path, type, setName)
cities = dir(pre_path);
for ci = 1 : length(cities)
    if(cities(ci).name(1)=='.')
        continue;
    end
    
    anno_path = fullfile(pre_path, cities(ci).name);
    disp(anno_path);
    names = dir([anno_path '/*' postFilename]);
    
    for i = 1 : length(names)
        filename = names(i).name;
        ind = findstr(filename, type);
        name = filename(1:ind-2);
        if ~index.isKey(name)
            j = j + 1 ;
            index(name) = j ;
            imdb.images.id(j) = j ;
            imdb.images.set(j) = setCode ;
            imdb.images.filename{j} = filename ;
            imdb.images.name{j} = name;
            imdb.images.city{j} = cities(ci).name;
            imdb.images.type{j} = type;
        else
            fprintf('repeated annotation for %s, previous annotation filename %s\n', filename, imdb.images.filename{index(name)});
        end
    end
end

% -------------------------------------------------------------------------
function checkData(imdb)
% -------------------------------------------------------------------------
%%{root}/{type}{video}/{split}/{city}/{city}_{seq:0>6}_{frame:0>6}_{type}{ext}
cmap = citySpaceLabelColors() ;
imgs = imdb.images;
for i = 1 : length(imgs.name)
    anno_path = sprintf(imdb.anno_path, imgs.type{i}, imdb.sets.name{imgs.set(i)}, imgs.city{i}, imgs.filename{i});
    anno = imread(anno_path);
    
    img_path = sprintf(imdb.img_path, imdb.sets.name{imgs.set(i)}, imgs.city{i}, imgs.name{i});
    I = imread(img_path);
    
    figure(1); clf; image(I);
    figure(2); clf; image(uint8(anno)) ; axis image ; colormap(cmap) ;
    pause;
end

% -------------------------------------------------------------------------
function imdb = getImageSizes(imdb)
% -------------------------------------------------------------------------
imgs = imdb.images;
for i=1:numel(imdb.images.id)
    info = imfinfo(sprintf(imdb.anno_path, imgs.type{i}, ...
        imdb.sets.name{imgs.set(i)}, imgs.city{i}, imgs.filename{i})) ;
    info = imfinfo(sprintf(imdb.img_path, imdb.sets.name{imgs.set(i)}, ...
        imgs.city{i}, imgs.name{i})) ;
    imdb.images.size(:,i) = uint16([info.Width ; info.Height]) ;
    fprintf('%s: checked image %s [%d x %d]\n', mfilename, ...
        imdb.images.name{i}, info.Height, info.Width) ;
end

