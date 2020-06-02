%This script is used to implement the function defined for the probability
%of selection
%x:= the total number of samples;
%y:= the number of selected samples;
%e:= the current classification error;
function [P]=PSSA(x,y,e)
    if(x==0||y==0)
        P=0.5;  %A default probability which equals to a random guess
    else
        %Use the second edition
    P=(1-y/x)^y;    %The first solution
%         P=(1-y/x)^y*e;  %The second solution
%         P=(1-y/x)^y+0.5*e;  %The third solution
%         P=e;
    end
    
end