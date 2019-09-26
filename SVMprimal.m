clear all;
load('mnist35.mat')

%%normalize data
trainx=double(trainx)/255;
testx=double(testx)/255;

[m n]=size(trainx);
m
n
cvx_begin quiet 
    variables w(n) b(1)
    minimize( norm(w))
    subject to
        1<=trainy.*((trainx*w)+b)
cvx_end

train_error=sign((trainx*w)+b);
test_error = sign((testx*w)+b);


train_count = 0;
test_count = 0;
for i=1:1902
      if testy(i)~= test_error(i)
          test_count=test_count+1;
     end
  end
  
  for i=1:2000
       if trainy(i)~= train_error(i)
          train_count=train_count+1;
       end
  end


test_percent =(test_count/1902)*100;
train_percent = (train_count/2000)*100;

%w
b
test_percent
train_percent

d = {'Technique','Loss % on train','Loss % on test';'Optimal',train_percent,test_percent};
xlswrite('cmpr_tbl', d, 1, 'A1');