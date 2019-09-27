clear all;
load('mnist35.mat')

%normalize data
trainx=double(trainx)/255;
testx=double(testx)/255;

[m n] = size(trainx);

test_l2 = zeros(1902,2000);
train_l2 = zeros(2000,2000);

for i = 1:2000
    for j=1:2000
        train_l2(i,j) = norm((trainx(j,:)-trainx(i,:)));
        if i<=1902
        test_l2(i,j) = norm((trainx(j,:)-testx(i,:)));
        end
    end
end

[train_sort, train_idx] = sort(train_l2,2);
[test_sort, test_idx] = sort(test_l2,2);

test_k3(:,1:3) = test_idx(:,1:3);
test_k5(:,1:5) = test_idx(:,1:5);

train_k3(:,1:3) = train_idx(:,2:4);
train_k5(:,1:5) = train_idx(:,2:6);

for i=1:2000
    train_predk3(i,1) = sign(sum(trainy(train_k3(i,:),1)));
    train_predk5(i,1) = sign(sum(trainy(train_k5(i,:),1)));
    if i<=1902
        test_predk3(i,1) = sign(sum(trainy(test_k3(i,:),1)));
        test_predk5(i,1) = sign(sum(trainy(test_k5(i,:),1)));
    end   
end

train_k3_count = sum(trainy~=train_predk3);
train_k5_count = sum(trainy~=train_predk5);
test_k3_count = sum(testy~=test_predk3);
test_k5_count = sum(testy~=test_predk5);

train_k3_per=train_k3_count/2000 *100;
train_k5_per=train_k5_count/2000 *100;
test_k3_per=test_k3_count/1902  *100;
test_k5_per=test_k5_count/1902  *100;

train_k3_per
train_k5_per
test_k3_per
test_k5_per

d = {'3-KNN',train_k3_per,test_k3_per;'5-KNN',train_k5_per,test_k5_per};
xlswrite('cmpr_tbl', d, 1, 'A5');
