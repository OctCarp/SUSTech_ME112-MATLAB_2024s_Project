%% Function

function [predictedLabels, scores] = readByPath(imageFilePath)
clc, close;

load('model/mobile_ck_final.mat', 'model');

inputSize = model.Layers(1).InputSize;


img = imread(imageFilePath);

% select face

% faceDetector = vision.CascadeObjectDetector();
% bbox = step(faceDetector, img);
% 
% if ~isempty(bbox)
%     face = bbox(1, :);
% 
%     croppedFace = imcrop(img, face);
% 
%     figure;
%     subplot(1, 2, 1);
%     imshow(img);
%     title('Original Image');
% 
%     subplot(1, 2, 2);
%     imshow(croppedFace);
%     title('Cropped Face');
% 
% else
%     disp('No face detected in the image.');
% end

gray_img = rgb2gray(img);
gray_img_rgb = repmat(gray_img, [1, 1, 3]);

resizedImg = imresize(gray_img_rgb, inputSize(1:2));
imshow(resizedImg)

[predictedLabels, scores] = classify(model, resizedImg);

figure;
bar(scores);
xticks(1:numel(model.Layers(end).Classes));
xticklabels(model.Layers(end).Classes);
xlabel('Class');
ylabel('Probability');
title('Probability of each class');

end

test = true;
if test
    disp(readByPath('image/inet_test/neutral.png'))
else
    disp(readByPath('s'))
end