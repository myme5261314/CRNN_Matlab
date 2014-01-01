function imgFeatures = extractFeatures(img, filters, params)
%
% 1. convolve input features with kernels
%
cim = convolve(img,filters,params);

%
% 2. rectify with absolute val
%
rim = abs(cim);

%
% 3. local normalization
%
lnim = localnorm(rim);

%
% 4. average down sampling
%
imgFeatures = avdown(lnim);

clear cim rim lnim;



function out = avdown(in)
%
% Do average down sampling of input with a boxcar filter of width bw and
% downsampling step size of bs
%
% hard code downsampling window size and step
winSize = 10;
winStep = 5;

ker = zeros(winSize);
ker(:) = 1/(winSize*winSize);
oi = (size(in,2)-winSize)/winStep + 1;
oj = (size(in,3)-winSize)/winStep + 1;
out = zeros(size(in,1),oi,oj);
for i=1:size(in,1)
    from = squeeze(in(i,:,:));
    to = conv2(from,ker,'valid');
    dsi = 1:winStep:size(to,1);
    dsj = 1:winStep:size(to,2);
    out(i,:,:) = to(dsi,dsj);
end

clear ker oi oj from to;


function out = convolve(img,filters,params)
rfSize = 9;
% extract all patches from image
patches = [];
for ch = 1:size(img,3)
    patches = [patches; im2col(img(:,:,ch),[rfSize rfSize])];
end
patches = patches';

% whiten patches
patches = bsxfun(@rdivide, bsxfun(@minus, patches, mean(patches,2)), sqrt(var(patches,[],2)+10));
patches = bsxfun(@minus, patches, params.whiten.M) * params.whiten.P;

resp = filters*patches';

% conv size
respSz = size(img,1)-rfSize+1;

out = permute(reshape(resp', [respSz respSz size(filters,1)]), [ 3 1 2]);

clear patches resp;

function lnim = localnorm(in)
%
% given a set of feature maps, performs local normalization
% out = (in -mean(in))/std(in)
% mean and std are defined over local neighborhoods that span
% all feature maps and a local spatial neighborhood
%

filtSize = 9;
k = fspecial('gaussian',filtSize,1.591);

ker = zeros(size(in,1),size(k,1),size(k,2));
for i=1:size(in,1)
    ker(i,:) = k(:);
end
ker = ker / sum(ker(:));

% mu = E(in)
inmean = multiconv(in,ker);
% in-mu
inzmean = in - inmean;
% (in - mu)^2
inzmeansq = inzmean .* inzmean;
% std = sqrt ( E (in - mu)^2 )
instd = sqrt(multiconv(inzmeansq,ker));
% threshold std with the mean
mstd = mean(instd(:));
instd(instd<mstd) = mstd;
% scale the input by (in - mu) / std
lnim = inzmean ./ instd;

clear k ker inmean inzmean inzmeansq instd mstd instd;

function out = multiconv(in,ker)
%
% this is basically 3d convolution without zero padding in 3rd
% dimension
%
out = zeros(size(in));
for i=1:size(in,1)
    cin = squeeze(in(i,:,:));
    cker = squeeze(ker(i,:,:));
    out(i,:,:) = conv2(cin,cker,'same');
    clear cin cker;
end
sout = squeeze(sum(out));
for i=1:size(out,1)
    out(i,:) = sout(:);
end
clear sout;
