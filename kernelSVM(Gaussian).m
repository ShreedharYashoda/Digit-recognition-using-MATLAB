clear all;
load('mnist35.mat')

%%normalize  data
trainx=double(trainx)/255;
testx=double(testx)/255;

[m,n]=size(trainx);
one=ones(m,1);
s=1;

for i=1:2000
   for j=1:2000
      k1(i,j)=exp(-(norm(trainx(i,:)-trainx(j,:)))^2/(2*s^2));
      if (j<=1902)
      k2(i,j)=exp(-(norm(trainx(i,:)-trainx(j,:)))^2/(2*s^2));
      end
    end  
end

cvx_begin quiet
    variables alpha(m)
    minimize( -one'*alpha + 1/2*(trainy.*alpha)'*k1*(trainy.*alpha)  )
    subject to
    0<=alpha
     0==trainy'*alpha    
cvx_end

max_idx=find(trainy==-1)
min_idx=find(trainy==1)

b= -(max((alpha(max_idx).*trainy(max_idx))'*k1(max_idx,:)) + min((alpha(min_idx).*trainy(min_idx))'*k1(min_idx,:)))/2;

train_pred= sign(((alpha.*trainy)'*k1)+b);
test_pred= sign(((alpha.*trainy)'*k2)+b);

train_mismatch=sum(trainy~=train_pred');
test_mismatch=sum(testy~=test_pred');

train_per=train_mismatch/2000 * 100;
test_per=test_mismatch/1902 *100;

alpha
b

train_per
test_per

d = {'Kernel',train_per,test_per};
xlswrite('cmpr_tbl', d, 1, 'A4');






