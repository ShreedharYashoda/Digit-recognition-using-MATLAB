clear all
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

% Changing the label of training and test data from -1 to 0 and +1 to 1
trainy(1:1000,1)= 0;
trainy(1001:2000,1)=1;

testy(1:1010,1)= 0;
testy(1011:1902,1)=1;

% Shuffling the training data set for SGD algorithm.
shuffled_idx = randperm(2000);
shuffled_trainx = trainx(shuffled_idx,:);
shuffled_trainy = trainy(shuffled_idx,:);

flag = 0;

% To check the no of samples used before converging.
counter=0;

while( flag == 0)
    counter=counter+1;
  
    for i = 1:2000        
        % Updating theta value using every samples.
         i_h_theta = shuffled_trainx(i,:)*theta;
         i_pred_label = 1/(1+(exp(-i_h_theta)));
         theta = theta + alpha*(shuffled_trainx(i,:)'*(shuffled_trainy(i)-i_pred_label));
         
         % Checking stopping condition after every 200 samples.
         if (mod(i,200)==0)
              h_theta = shuffled_trainx*theta;
              pred_label = 1./(1+(exp(-h_theta)));
              loss = shuffled_trainx'*(shuffled_trainy - pred_label);
              l2_norm = norm(loss,2);
              
              if (l2_norm < 0.01)
                flag = 1;
                break;
              end
         end        
    end
end

% Getting label comparing with threshold of 0.5
train_h_theta = trainx*theta;
train_pred_label = 1./(1+(exp(-train_h_theta)));
train_pred_label(train_pred_label<0.5) = 0;
train_pred_label(train_pred_label>0.5) = 1;


% Testing the trained model using test data set.
test_h_theta = testx*theta;
test_pred_label = 1./(1+(exp(-test_h_theta)));
test_pred_label(test_pred_label<0.5) = 0;
test_pred_label(test_pred_label>0.5) = 1;

% Number of error in labelling the samples in train and test dataset.
train_count = sum(trainy~=train_pred_label);
test_count = sum(testy~=test_pred_label);

% Error percentage.
train_per = train_count/2000 * 100;
test_per = test_count/1902 * 100;

fprintf('The 0-1 error loss for SGD is %f %% for the training dataset\n',train_per);
fprintf('The 0-1 error loss for SGD is %f %% for the test dataset\n',test_per);
fprintf('Number of times dataset used before stopping is %f \n',counter);

