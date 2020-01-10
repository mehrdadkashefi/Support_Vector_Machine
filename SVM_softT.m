% This is the code for soft margin SVM classifier. In soft-margin SVM we
% allow some sample of the data to pass the decision boundry. The parameter
% gamma controls who much off-boubry samples are tolerated.
%
% Record of Revisions :
%      Date           Programmer          Description of Change
%      ====           ==========          =====================
%  Dec 10th 2019    Mehrdad Kashefi           original code 
% ...................................................................
% Define Variables:
%.............................................
clear;
clc;
close all;

gamma = 100;  % Soft thereshold Parameter

%% Creating sample data (two multi-dimensional Gussian Discribution)
% Class_1
mu = [0,0];    % Mean
sigma = [1,0;0,1];  % Covariance
class_1 = mvnrnd(mu,sigma,100); 
% Class_1
mu = [1,3];    % Mean
sigma = [1,0;0,1];  % Covariance
class_2 = mvnrnd(mu,sigma,100); 

data = [class_1;class_2];
label = [ones(100,1);-1*ones(100,1)];
rand_idx = randperm(size(data,1));
data= data(rand_idx,:);
label = label(rand_idx,:);

x_train = data(1:80,:);
y_train = label(1:80,:);

x_test = data(80:end,:);
y_test = label(80:end,:);

X= x_train';
y = y_train;

figure(1)
scatter(data(label==1,1),data(label==1,2),'bx');
hold on
scatter(data(label==-1,1),data(label==-1,2),'rx');
disp(['Num class_1: ', num2str(sum(label==1)), ' Num class_2: ', num2str(sum(label==-1))])

%% Solving SVM
M = (X.*y')' * (X.*y');

H = M;
f = -ones(length(y) ,1)';
A = [];
b = [];
Aeq = y';
beq = 0;
lb = zeros(length(y),1);
ub = gamma*ones(length(y),1);
options = optimoptions('quadprog','Display','iter');

a = quadprog(H,f,A,b,Aeq,beq,lb,ub,[],options);

sup = a>0.01; % Find Support Vectors
sup_inx = find(sup==1);

figure(1)
hold on
scatter(X(1,sup), X(2,sup),'ko','LineWidth',1.5)

beta = (X.*y')*a;
% for subpport vectors:
beta0 = y(sup_inx(1)) - beta'*X(:,sup_inx(1));

% Test with test data 
X = x_test';
y = y_test;

y_pred = beta'*X +beta0;

y_pred(y_pred>=0) = 1;
y_pred(y_pred<0) = -1;

Acc = sum(y==y_pred')/length(y_pred);
disp(['Accuracy in test data: ', num2str(Acc*100)])
disp(['Number of Support vectors: ', num2str(length(sup_inx))])

% Plot Decision boundry for linear kernel
x = linspace(0,2,100);
f = @(x) -(x*beta(1) + beta0)/beta(2);
figure(1)
hold on
plot(x,f(x),'g--','LineWidth',2,'DisplayName','Boundary')