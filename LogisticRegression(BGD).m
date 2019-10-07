clear all;
clc;
load('mnist35.mat')

%%normalize  data
trainx=double(trainx)/255;
testx=double(testx)/255;

n_train=length(trainy);%total number of training samples
n_test=length(testy);%total number of test samples

m_data=size(trainx,2);%dimension of original feature vector


trainx=[trainx ones(n_train,1)];%  add dummy feature 1
testx=[testx ones(n_test,1)];%  add dummy feature 1
theta=zeros(m_data+1,1);%initialize theta, dimension is 784+1, where the last entry is b
alpha=0.5;%step size 

% Changing the label training and test data from -1 to 0 and +1 to 1
trainy(1:1000,1)= 0;
trainy(1001:2000,1)=1;

testy(1:1010,1)= 0;
testy(1011:1902,1)=1;

% To initialize l2_norm loss gradient.
h_theta=trainx*theta;
pred_label=1./(1+(exp(-h_theta)));
loss=trainx'*(trainy-pred_label);
l2_norm = norm(loss,2);

% To check the no of samples used before converging.
counter=1;

% Training the model with Batch gradient descent algorithm.
while(l2_norm > 0.01)
    h_theta=trainx*theta;
    pred_label=1./(1+(exp(-h_theta)));
    loss=trainx'*(trainy-pred_label);
    theta=theta+ alpha.*(loss);
    l2_norm = norm(loss,2);
     counter=counter+1; 
end

% Getting label comparing with threshold of 0.5
pred_label(pred_label<0.5) = 0;
pred_label(pred_label>0.5) = 1;

% Testing the trained model using test data set.
test_h_theta=testx*theta;
test_pred_label=1./(1+(exp(-test_h_theta)));
test_pred_label(test_pred_label<0.5) = 0;
test_pred_label(test_pred_label>0.5) = 1;

% Number of error in labelling the samples in train and test dataset.
train_count=sum(trainy~=pred_label);
test_count=sum(testy~=test_pred_label);

% Error percentage.
train_per=(train_count/2000)*100;
test_per=(test_count/1902)*100;

fprintf('The 0-1 error loss for BGD is %f %% for the training dataset\n',train_per);
fprintf('The 0-1 error loss for BGD is %f %% for the test dataset\n',test_per);

fprintf('Number of times dataset used before stopping is %f \n',counter);