% This is the code for soft margin SVM + kernel classifier. In soft-margin SVM we
% allow some sample of the data to pass the decision boundry. The parameter
% gamma controls who much off-boubry samples are tolerated. The data
% sanmples are also applied to a kernel.
%
% Record of Revisions :
%      Date           Programmer          Description of Change
%      ====           ==========          =====================
%  Dec 10th 2019    Mehrdad Kashefi           original code 
% ...................................................................
% define variables:
%.............................................
clear;
clc;
close all;

% Class_1
mu = [0,0];    % Mean
sigma = [1,0;0,1];  % Covariance
class_1 = mvnrnd(mu,sigma,500); 
% Class_1
mu = [2,3];    % Mean
sigma = [1,0;0,1];  % Covariance
class_2 = mvnrnd(mu,sigma,500); 

data = [class_1;class_2];
label = [ones(500,1);-1*ones(500,1)];


figure(1)
scatter(data(label==1,1),data(label==1,2),'bx');
hold on
scatter(data(label==-1,1),data(label==-1,2),'rx');
disp(['Num class_1: ', num2str(sum(label==1)), ' Num class_2: ', num2str(sum(label==-1))])


X = data';
y = label;


gauss_kernel = @(x1,x2,sigma) exp( (-norm(x1-x2)^2)/sigma );
linear_kernel = @(x1,x2) (x1'*x2)^2;

M = (X.*y')' * (X.*y');
XX = X.*y';

M_kernel = zeros(size(M));
for i = 1:size(M_kernel,1)
    for j = 1:size(M_kernel,2)
        M_kernel(i,j) = linear_kernel(XX(:,i),XX(:,j));
    end
end

H = M_kernel;
f = -ones(length(y) ,1)';
A = [];
b = [];
Aeq = y';
beq = 0;
lb = zeros(length(y),1);
ub = 0.1*ones(length(y),1);
options = optimoptions('quadprog','Display','iter');

a = quadprog(H,f,A,b,Aeq,beq,lb,ub,[],options);

sup = a>0.01;
sup_inx = find(sup==1);

figure(1)
hold on
scatter(X(1,sup), X(2,sup),'ko','LineWidth',1.5)

% Calculate B0
temp = 0;
for i = 1:length(sup_inx)
    temp = temp +linear_kernel(X(:,sup_inx(1)), X(:,sup_inx(i)))*y(sup_inx(i))*a(sup_inx(i));
end
beta0 = y(sup_inx(1)) - temp;

% Predict each point
y_pred = zeros(length(X),1);
for i = 1: length(X)
    temp = 0;
    for j = 1:length(sup_inx)
        temp = temp + linear_kernel(X(:,i), X(:,sup_inx(j)))*y(sup_inx(j))*a(sup_inx(j)) ;
    end
    y_pred(i) = temp + beta0;
end

y_pred(y_pred>=0) = 1;
y_pred(y_pred<0) = -1;

Acc = sum(y==y_pred)/length(y_pred);
disp(['Accuracy in test data: ', num2str(Acc*100)])

% Plot Decision boundry for linear kernel
X = X.^2;
beta = (X.*y')*a;
% for subpport vectors:
beta0 = y(sup_inx(1)) - beta'*X(:,sup_inx(1));

x = linspace(-1,1,100);
f = @(x) sqrt(-(x.^2*beta(1) + beta0)/beta(2));
figure(1)
hold on
plot(x,f(x),'g--','LineWidth',2,'DisplayName','Boundary')
