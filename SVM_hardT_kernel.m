% This is the code for hard-threshold SVM. In hard-theshold + kernel we
% assume that there is a margin between the classes. The data samples sould
% net be necessarily linearly separable. 
%
% Record of Revisions :
%      Date           Programmer          Description of Change
%      ====           ==========          =====================
%  Dec 3rd 2019    Mehrdad Kashefi           original code 
% ...................................................................
% define variables:
%.............................................
clear;
clc;
close all;
%% Creating sample data (two Cocenteric data sets)
rad = 1.4;
marg = 0.2;
data = 3*(rand(1000,2)-0.5);
label_1 = data(:,1).^2 + data(:,2).^2 >= rad;
label_2 = data(:,1).^2 + data(:,2).^2 < rad - marg;


figure(1)
scatter(data(label_1,1),data(label_1,2),'bx');
hold on
scatter(data(label_2,1),data(label_2,2),'rx');
disp(['Num class_1: ', num2str(sum(label_1)), ' Num class_2: ', num2str(sum(label_2))])

data = [data(label_1,:); data(label_2,:)];
label = [ ones(sum(label_1),1); -1*ones(sum(label_2),1) ];

X = data';
y = label;

% Kernel Functions
gauss_kernel = @(x1,x2,sigma) exp( (-norm(x1-x2)^2)/sigma );
linear_kernel = @(x1,x2) (x1'*x2)^2;

M = (X.*y')' * (X.*y');
XX = X.*y';

% Apply Kernel to sample points
M_kernel = zeros(size(M));
for i = 1:size(M_kernel,1)
    for j = 1:size(M_kernel,2)
        M_kernel(i,j) = linear_kernel(XX(:,i),XX(:,j));
    end
end
%% Solving SVM
H = M_kernel;
f = -ones(length(y) ,1)';
A = [];
b = [];
Aeq = y';
beq = 0;
lb = zeros(length(y),1);
ub = [];
options = optimoptions('quadprog','Display','iter');

a = quadprog(H,f,A,b,Aeq,beq,lb,ub,[],options);

sup = a>0.001;
sup_inx = find(sup==1);

figure(1)
hold on
scatter(X(1,sup), X(2,sup),'ko','LineWidth',1.5)

%% Calculate bta0 and beta
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