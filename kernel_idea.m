% Purpose: This program illustrates the basic idea of Kernel, which is how
% adding an extra dimension to data can be helpful in classification of
% data.
%
% Record of Revisions :
%      Date           Programmer          Description of Change
%      ====           ==========          =====================
%  Dec 12th 2019    Mehrdad Kashefi           Original code 
% ...................................................................
% Define Variables:
%.............................................
clear;
clc;
close all;
%% Creating two concentric data samples 
rad = 1.4; 
marg = 0.5; % The margin between the classes
data = 3*(rand(1000,2)-0.5);
label_1 = data(:,1).^2 + data(:,2).^2 >= rad;
label_2 = data(:,1).^2 + data(:,2).^2 < rad - marg;


figure(1)
scatter(data(label_1,1),data(label_1,2));
hold on
scatter(data(label_2,1),data(label_2,2));
disp(['Num class_1: ', num2str(sum(label_1)), ' Num class_2: ', num2str(sum(label_2))])

data = [data(label_1,:); data(label_2,:)];
label = [ ones(sum(label_1),1); -1*ones(sum(label_2),1) ];

% Fit SVM on the sample data in low dimension
Mdl = fitcsvm(data,label);
pred = Mdl.predict(data);
acc= (sum(label==pred))/length(pred);
disp(['Prediction is ', num2str(acc)])

%% Apply a third dimention
data_hd = zeros(length(data),3);
for i =1: length(data)
    data_hd(i,:) = [data(i,1)^2, data(i,2)^2, sqrt(2)*data(i,1)*data(i,2)];
end

% Classification data samples in higher dimension
figure(2)
scatter3(data_hd(label==1,1),data_hd(label==1,2),data_hd(label==1,3))
hold on
scatter3(data_hd(label==-1,1),data_hd(label==-1,2),data_hd(label==-1,3))

Mdl = fitcsvm(data_hd,label);
pred = Mdl.predict(data_hd);
acc= (sum(label==pred))/length(pred);
disp(['Prediction is ', num2str(acc)])