%This manuscript is used to test the function of poison distribution
% lamda=50;
% x=1:1:200;
% [r,c]=size(x);
% y=zeros(r,c);
% for i=1:c
%     y(i)=lamda^(x(i))*exp(-50)/(factorial(x(i)));
% end
% 
% figure;
% plot(x,y);

%This is a simulation of a dynamic selection of the samples
k=10;
min=2;
max=500;
num=3000;
x=1:1:num;
y=zeros(1,num);
c=zeros(1,num);
c(1:100)=0.05;
c(101:120)=0.1;
c(121:140)=0.2 + rand(1,1)*0.1;
c(141:200)=0.4;
c(201:400)=0.3+rand(1,1)*0.1;
c(401:500)=0.1+rand(1,1)*0.1;
c(501:600)=0.34+rand(1,1)*0.1;
c(601:900)=0.1+rand(1,1)*0.1;
c(901:num)=0.01+rand(1,1)*0.01;

for i=1:num
    if i<k
        continue;
    else
        %Select current samples number y(i-1)
%         P=(1-y(i-1)/x(i))^(y(i-1));
        P=(1-y(i-1)/x(i))^(y(i-1))*(c(i));
        %Generate a randnumber between 0,1
        if(rand(1,1)<P)
            %Do not jump, add one to y (i)
            if(y(i-1)==0)
                y(i)=min;
            else
                y(i)=y(i-1)+1;
            end
        else
            y(i)=y(i-1);    %The number remains the same
            continue; %Do nothing
        end
    end
end

figure;
plot(x,y);