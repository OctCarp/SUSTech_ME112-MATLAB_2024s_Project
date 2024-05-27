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

%% GUI
function createImageClassificationGUI()
    % 创建主 GUI 窗口
    fig = uifigure('Position', [100, 100, 900, 700], 'Name', 'Facial Expression Recognition GUI');
    fig.Color = [0.95, 0.95, 0.95]; % 设置窗口背景颜色
    
    % 在 Fig 内部上方添加文字
    annotation(fig, 'textbox', [0.1, 0.85, 0.8, 0.1], 'String', 'Facial Expression Recognition System', 'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'FontSize', 16, 'FontWeight', 'bold', 'Color', [0.2, 0.2, 0.2]);

    % 添加一个按钮用于选择图片，放在下方
    btnSelect = uibutton(fig, 'push', 'Text', 'Select Image', ...
                         'Position', [50, 50, 120, 40], ... % 调整按钮大小
                         'BackgroundColor', [0.2, 0.6, 1], ... % 设置按钮背景颜色
                         'FontColor', [1, 1, 1], ... % 设置按钮字体颜色
                         'FontSize', 12, ... % 设置按钮字体大小
                         'FontWeight', 'bold', ... % 设置按钮字体粗细
                         'ButtonPushedFcn', @(btn, event) selectImageCallback(fig));

    % 添加一个按钮用于显示灰度图像，放在下方
    btnShowGray = uibutton(fig, 'push', 'Text', 'Show Grayscale', ...
                           'Position', [200, 50, 120, 40], ...
                           'BackgroundColor', [0.2, 0.6, 1], ...
                           'FontColor', [1, 1, 1], ...
                           'FontSize', 12, ...
                           'FontWeight', 'bold', ...
                           'ButtonPushedFcn', @(btn, event) showGrayImageCallback(fig));

    % 添加一个按钮用于显示柱状图，放在下方
    btnShowBar = uibutton(fig, 'push', 'Text', 'Show Result', ...
                          'Position', [350, 50, 120, 40], ...
                          'BackgroundColor', [0.2, 0.6, 1], ...
                          'FontColor', [1, 1, 1], ...
                          'FontSize', 12, ...
                          'FontWeight', 'bold', ...
                          'ButtonPushedFcn', @(btn, event) showBarChartCallback(fig));

    % 添加一个按钮用于显示最高预测类型及其概率，放在下方
    btnShowPrediction = uibutton(fig, 'push', 'Text', 'Show Prediction', ...
                                 'Position', [500, 50, 150, 40], ...
                                 'BackgroundColor', [0.2, 0.6, 1], ...
                                 'FontColor', [1, 1, 1], ...
                                 'FontSize', 12, ...
                                 'FontWeight', 'bold', ...
                                 'ButtonPushedFcn', @(btn, event) showPredictionCallback(fig));

    % 添加一个按钮用于显示最高预测类型的概率，放在下方
    btnShowMaxProb = uibutton(fig, 'push', 'Text', 'Show Accuracy', ...
                              'Position', [680, 50, 150, 40], ...
                              'BackgroundColor', [0.2, 0.6, 1], ...
                              'FontColor', [1, 1, 1], ...
                              'FontSize', 12, ...
                              'FontWeight', 'bold', ...
                              'ButtonPushedFcn', @(btn, event) showMaxProbCallback(fig));

    % 添加一个退出按钮，放在下方
    btnExit = uibutton(fig, 'push', 'Text', 'Exit', ...
                       'Position', [850, 50, 100, 40], ...
                       'BackgroundColor', [1, 0.4, 0.4], ... % 设置退出按钮背景颜色
                       'FontColor', [1, 1, 1], ...
                       'FontSize', 12, ...
                       'FontWeight', 'bold', ...
                       'ButtonPushedFcn', @(btn, event) close(fig));

    % 添加一个轴用于显示选中的图片
    imgAxes = uiaxes(fig, 'Position', [50, 300, 300, 300]);
    imgAxes.Color = [0.9, 0.9, 0.9]; % 设置轴背景颜色
    imgAxes.XTick = [];
    imgAxes.YTick = [];
    title(imgAxes, 'Selected Image', 'Color', [0.2, 0.2, 0.2], 'FontSize', 14); % 设置标题颜色和字体

    % 添加一个轴用于显示灰度图像
    grayAxes = uiaxes(fig, 'Position', [400, 300, 200, 200]);
    grayAxes.Color = [0.9, 0.9, 0.9];
    grayAxes.XTick = [];
    grayAxes.YTick = [];
    title(grayAxes, 'Grayscale Image', 'Color', [0.2, 0.2, 0.2], 'FontSize', 12);

    % 添加一个轴用于显示分类结果
    resultAxes = uiaxes(fig, 'Position', [650, 300, 200, 200]);
    resultAxes.Color = [0.9, 0.9, 0.9];
    resultAxes.XTick = [];
    resultAxes.YTick = [];
    title(resultAxes, 'Classification Result', 'Color', [0.2, 0.2, 0.2], 'FontSize', 12);

    % 添加文本标签用于显示最高预测类型及其概率
    lblPrediction = uilabel(fig, 'Position', [400, 200, 400, 50], ...
                            'Text', 'Predicted Emotion: ', ...
                            'FontSize', 16, ...
                            'FontColor', [0.2, 0.2, 0.2], ...
                            'BackgroundColor', [0.95, 0.95, 0.95], ...
                            'HorizontalAlignment', 'left', ...
                            'VerticalAlignment', 'center');
    
    % 添加文本标签用于显示最高预测类型的概率
    lblMaxProb = uilabel(fig, 'Position', [400, 150, 400, 50], ...
                         'Text', 'Accuracy: ', ...
                         'FontSize', 16, ...
                         'FontColor', [0.2, 0.2, 0.2], ...
                         'BackgroundColor', [0.95, 0.95, 0.95], ...
                         'HorizontalAlignment', 'left', ...
                         'VerticalAlignment', 'center');

    % 存储句柄以便在回调函数中使用
    fig.UserData.imgAxes = imgAxes;
    fig.UserData.grayAxes = grayAxes;
    fig.UserData.resultAxes = resultAxes;
    fig.UserData.lblPrediction = lblPrediction;
    fig.UserData.lblMaxProb = lblMaxProb;
end

function selectImageCallback(fig)
    % 选择图片文件
    [file, path] = uigetfile({'*.jpg;*.jpeg;*.png;*.bmp', 'Image Files (*.jpg, *.jpeg, *.png, *.bmp)'});
    if isequal(file, 0)
        return; % 用户取消选择
    end

    % 读取并显示选中的图片
    imagePath = fullfile(path, file);
    img = imread(imagePath);
    imgAxes = fig.UserData.imgAxes;
    imshow(img, 'Parent', imgAxes);

    % 重置灰度图像轴
    grayAxes = fig.UserData.grayAxes;
    cla(grayAxes);
    title(grayAxes, 'Grayscale Image');

    % 重置柱状图轴
    resultAxes = fig.UserData.resultAxes;
    cla(resultAxes);
    title(resultAxes, 'Classification Result');

    % 重置预测结果标签
    lblPrediction = fig.UserData.lblPrediction;
    lblPrediction.Text = 'Predicted Emotion: ';

    % 重置最高预测类型的概率标签
    lblMaxProb = fig.UserData.lblMaxProb;
    lblMaxProb.Text = 'Accuracy: ';

    % 加载预训练模型
    load('model/mobile_ck_final.mat', 'model');

    % 处理图像以适应模型输入
    inputSize = model.Layers(1).InputSize;
    gray_img = rgb2gray(img);
    gray_img_rgb = repmat(gray_img, [1, 1, 3]);
    resizedImg = imresize(gray_img_rgb, inputSize(1:2));

    % 存储处理后的图像和分类结果在 fig.UserData 中
    fig.UserData.grayImg = gray_img;
    fig.UserData.resizedImg = resizedImg;
    [predictedLabels, scores] = classify(model, resizedImg);
    fig.UserData.scores = scores;
    fig.UserData.predictedLabels = predictedLabels;

    % 获取所有分类类别
    classes = model.Layers(end).Classes;
    fig.UserData.classes = classes;

    % 获取最高预测类型及其概率
    [maxScore, maxIndex] = max(scores);
    predictedClass = classes(maxIndex);
    fig.UserData.maxScore = maxScore;
    fig.UserData.maxIndex = maxIndex;
    fig.UserData.predictedClass = predictedClass;
end

function showGrayImageCallback(fig)
    % 在 grayAxes 上显示灰度图像
    grayAxes = fig.UserData.grayAxes;
    gray_img = fig.UserData.grayImg;
    imshow(gray_img, 'Parent', grayAxes);
end

function showBarChartCallback(fig)
    % 在 resultAxes 上显示分类结果的柱状图
    resultAxes = fig.UserData.resultAxes;
    scores = fig.UserData.scores;
    classes = fig.UserData.classes;
    b = bar(resultAxes, scores);
    xticks(resultAxes, 1:numel(classes));
    xticklabels(resultAxes, classes);
    xlabel(resultAxes, 'Emotion');
    ylabel(resultAxes, 'Probability');
    title(resultAxes, 'Probability of each Emotion');

    % 在柱状图上添加每个类别的概率值
    y = b.YData;
    for i = 1:length(y)
        text(resultAxes, i, y(i), sprintf('%.2f', y(i)), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
    end
end

function showPredictionCallback(fig)
    % 显示最高预测类型及其概率
    lblPrediction = fig.UserData.lblPrediction;
    predictedClass = fig.UserData.predictedClass;
    maxScore = fig.UserData.maxScore;
    lblPrediction.Text = sprintf('Predicted Emotion: %s', predictedClass);
end

function showMaxProbCallback(fig)
    % 显示最高预测类型的概率
    lblMaxProb = fig.UserData.lblMaxProb;
    maxScore = fig.UserData.maxScore;
    lblMaxProb.Text = sprintf('Accuracy: %.2f%%', maxScore * 100);
end

%% main
clc;
test = true;
if test
    createImageClassificationGUI()
    %disp(readByPath('image/inet_test/neutral.png'))
else
    disp(readByPath('s'))
end