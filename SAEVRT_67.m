% Enhancing Zero-Shot Scene Recognition through Semantic Autoencoders and Visual Relation Transfer
clear
clc
options = [];
options.PCARatio = 0.99;
%
load('E:\MIT-67\gnd_67');
gnd = gnd_67;
clear gnd_67 
nnClass = length(unique(gnd));  % The number of classes;
%
load('E:\MIT-67\attribute67');
F_attribute = feature'; 
%
load('E:\MIT-67\places365_resnet152_67');
[~,~,~,new_data2] = PCA(resnet152_67',options);
F_feature = normcols(new_data2');  
clear resnet152_67 new_data2

accy1 = zeros(25,1);
accy2 = zeros(25,1);
for t = 1:25
%random tran/test
randIdx = randperm(nnClass);
seenClass = 57;  %57
X_tr_cl_id = randIdx(1:seenClass);
X_te_cl_id = randIdx(seenClass+1:end);
% 
Ftr_feature = [];
Fte_feature = [];
Ftr_attributelabel = [];
Ftr_attributetabel = [];
Fte_attributetabel = [];
tr_label = [];
te_label = [];
for i = 1:length(X_tr_cl_id)
    idx = find(gnd == X_tr_cl_id(i));
    Ftr_feature = [Ftr_feature, F_feature(:,idx)];
    Ftr_attributetabel = [Ftr_attributetabel, F_attribute(:,X_tr_cl_id(i))];
    Ftr_attributelabel = [Ftr_attributelabel, repmat(F_attribute(:,X_tr_cl_id(i)), 1, length(idx))];
    tr_label = [tr_label; gnd(idx)]; 
end
for i = 1:length(X_te_cl_id)
    idx = find(gnd == X_te_cl_id(i));
    Fte_feature = [Fte_feature, F_feature(:,idx)];
    Fte_attributetabel = [Fte_attributetabel, F_attribute(:,X_te_cl_id(i))];
    te_label = [te_label; gnd(idx)]; 
end

%%  SAEVRT
% SAE initialize
alpha = 10;
bate = 100;
lamda1 = 100;
lamda2 = 1000;
lamda3 = 0.001; 
As = Ftr_attributelabel;
% At = Fte_attributetabel;
Xs = Ftr_feature;
Xt = Fte_feature;
A = As * As';
B = alpha * Xs * Xs';
C = (1+alpha) * As * Xs';
Ws = sylvester(A,B,C);
Wt = Ws;

% Visual relation transfer
dis = EuclideanDistance(Xs',Xt');
dis = ones(size(dis,1),size(dis,2)) ./ dis;
%
dis = EuclideanDistance(Xs',Xt').^2; 
sigma = 4;   %4
dis = exp(-dis/sigma);
% 
% dis1 = EuclideanDistance(Xs',Xt');
% dis1 = ones(size(dis1,1),size(dis1,2)) ./ dis1;
% dis2 = EuclideanDistance(Xt',Xt');
% dis2 = ones(size(dis2,1),size(dis2,2)) ./ dis2;
% dis2(logical(eye(size(dis2)))) = 0;
% dis2diag = diag(max(dis2,[],2));
% dis2 = dis2 + dis2diag;
% dis2 = dis2*0.01;  
% dis2 = (dis2 - min(min(dis2))) / (max(max(dis2)) - min(min(dis2)))*0.1;
% dis = dis1 * dis2;

% update
for tt = 1:50  %
    %At = (1+lamda2)*inv(Wt*Wt'+lamda2*eye(size(Wt,1)))*(Wt*Xt);
    At = inv(lamda1*(Wt*Wt'+bate*eye(size(Wt,1)))+lamda3*eye(size(Wt,1)))*((1+bate)*lamda1*(Wt*Xt)+lamda3*As*dis);
    Wt = sylvester(lamda2*eye(size(At,1))+lamda1*At*At', lamda1*bate*Xt*Xt', lamda2*Ws+lamda1*(1+bate)*At*Xt');
    Ws = sylvester(lamda2*eye(size(As,1))+As*As', alpha*Xs*Xs', lamda2*Wt+(1+alpha)*As*Xs');
end

% in visual space
Fte_pre_feature = Wt' * Fte_attributetabel;
dis = zeros(size(Fte_pre_feature,2),size(Fte_feature,2));
for i = 1 : size(Fte_pre_feature,2)
    for j = 1 : size(Fte_feature,2)
         temp = dot(Fte_pre_feature(:,i),Fte_feature(:,j)) / (norm(Fte_pre_feature(:,i)) * norm(Fte_feature(:,j))); 
         dis(i,j) = 1 - temp;
    end
end
[~,index] = min(dis',[],2);
sum = 0;
for i = 1: length(index)
    if X_te_cl_id(index(i)) == te_label(i)
        sum = sum + 1;
    end
end
accy1(t) = sum / length(index);   

% in semantic space
Fte_pre_attribute = Wt * Fte_feature;
dis = zeros(size(Fte_pre_attribute,2),size(Fte_attributetabel,2));
for i = 1 : size(Fte_pre_attribute,2)
    for j = 1 : size(Fte_attributetabel,2)
         temp = dot(Fte_pre_attribute(:,i),Fte_attributetabel(:,j)) / (norm(Fte_pre_attribute(:,i)) * norm(Fte_attributetabel(:,j))); 
         dis(i,j) = 1 - temp;
    end
end
[~,index] = min(dis,[],2);
sum = 0;
for i = 1: length(index)
    if X_te_cl_id(index(i)) == te_label(i)
        sum = sum + 1;
    end
end
accy2(t) = sum / length(index);   
end
mean(accy1)  
mean(accy2)  
std(accy1)   
std(accy2)   




