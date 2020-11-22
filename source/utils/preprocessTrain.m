function out = preprocessTrain(x,targetSize)
out = imread(x);
if numel(size(out))==2
    out = cat(3,out,out,out);
end
out = imresize(out,targetSize);



