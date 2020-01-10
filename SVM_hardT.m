% This is the code for hard-threshold SVM. In hard-theshold SVM we consider
% that all the samples points are linearly separable and the goale is to
% find the hyperplane which has the maximum distance from both classes.
%
% Record of Revisions :
%      Date           Programmer          Description of change
%      ====           ==========          =====================
%  Dec 3rd 2019    Mehrdad Kashefi           Original code 
% ...................................................................
% Define Variables:
%.............................................
clear;
clc;
close all;
%% Load a sample data
% Load simple data_set
[X,y] = simplecluster_dataset;

% Scatter plot of 4 classes (Select two linearly separable clusters)
class_1 = logical(y(1,:));
class_2 = logical(y(2,:));
class_3 = logical(y(3,:));
class_4 = logical(y(4,:));

figure(1)
hold on
scatter(X(1,class_1),X(2,class_1));
scatter(X(1,class_2),X(2,class_2));
scatter(X(1,class_3),X(2,class_3));
scatter(X(1,class_4),X(2,class_4));

X = [X(:, class_1),X(:, class_2)];
y = [ones(sum(class_1),1); -1* ones(sum(class_2),1)];
rand_index = randperm(length(y));
X = X(:,rand_index);
y = y(rand_index);

% Train-Test separation
X_train = X(:, 1:390);
y_train = y(1:390);

X_test = X(:, 390:end);
y_test = y(390:end);

X = X_train;
y = y_train;

figure(2)
hold on
scatter(X(1,y==1), X(2,y==1),'xb')
scatter(X(1,y==-1), X(2,y==-1),'xr')

%% Solve SVM problem
M = (X.*y')' * (X.*y');

H = M;
f = -ones(length(y) ,1)';
A = [];
b = [];
Aeq = y';
beq = 0;
lb = zeros(length(y),1);
ub = [];
options = optimoptions('quadprog','Display','iter');

a = quadprog(H,f,A,b,Aeq,beq,lb,ub,[],options);

sup = a>0.01;
sup_inx = find(sup==1);

figure(2)
scatter(X(1,sup), X(2,sup),'ko','LineWidth',1.5)

%% Calculate Beta and Beta0
beta = (X.*y')*a;
% for subpport vectors:
beta0 = y(sup_inx(1)) - beta'*X(:,sup_inx(1));

y_pred = beta'*X_test + beta0 ;

y_pred(y_pred>=0) = 1;
y_pred(y_pred<0) = -1;

Acc = sum(y_test==y_pred')/length(y_pred);
disp(['Accuracy in test data: ', num2str(Acc*100)])

x = linspace(-0.4,0.4,100);
f = @(x) -(x*beta(1) + beta0)/beta(2);
figure(2)
hold on
plot(x,f(x),'g--','LineWidth',2,'DisplayName','Boundary')
hold off

