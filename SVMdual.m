clear all;
load('hw1_mnist35.mat')

%%normalize  data
trainx=double(trainx)/255;
testx=double(testx)/255;

[m n]=size(trainx');


one=ones(n,1);
cvx_begin quiet 
    variables a(n) 
    
    minimize( -one'*a + 1/2*(trainy.*a)'*(trainx*(trainx)')*(trainy.*a) )
    subject to
        0<=a
        trainy'*a==0
cvx_end

w = ((a.*trainy)'*trainx)';
t_w=trainx*w;
b = -( max(t_w(1:1000))  +    min((trainx*w).*(trainy==1))  ) /2
 
train_error=sign((trainx*w)+b);
ytest = sign((testx*w)+b);


min(w)
train_count = 0;
test_count = 0;
for i=1:1902
    if testy(i)~= ytest(i)
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

size(trainy)
b
test_percent
train_percent

d = {'Dual',train_percent,test_percent};
xlswrite('cmpr_tbl', d, 1, 'A3');