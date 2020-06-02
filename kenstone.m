
function [model,test]=kenstone(X,k)

% -------------------------------------------------------------------------
% Function: [model,test]=kenstone(X,k)
% -------------------------------------------------------------------------
% Aim:
% Subset selection with Kennard and Stone algorithm
% -------------------------------------------------------------------------
% Input:
% X, matrix (n,p), predictor variables in columns
% k, number of objects to be selected to model set
% -------------------------------------------------------------------------
% Output:
% model, vector (k,1), list of objects selected to model set
% test, vector (n-k,1), list of objects selected to test set (optionally)
% -----------------------------------------------------------------------
% Example: 
% [model,test]=kenstone(X,10)
% [model]=kenstone(X,10)
% -----------------------------------------------------------------------
[m,n]=size(X);
if k>=m | k<=0  
    h=errordlg('Wrongly specified number of objects to be selected to model set.','Error');
    model=[];
    if nargout==2
        test=[];
    end
    waitfor(h)
    return
end

x=[[1:size(X,1)]' X];
n=size(x,2);
[i1,ind1]=min(fastdist(mean(x(:,2:n)),x(:,2:n)));
model(1)=x(ind1,1);
x(ind1,:)=[];

[i2,ind2]=max(fastdist(X(model(1),:),x(:,2:n)));
model(2)=x(ind2,1);
x(ind2,:)=[];

% h=waitbar(0,'Please wait ...'); 
% h=waitbar(0/k,h);

for d=3:k
    [ii,ww]=max(min(fastdist(x(:,2:n),X(model,:))));
	model(d)=x(ww,1);
	x(ww,:)=[];
%     h=waitbar(d/k,h);
end

if nargout==2
    test=1:size(X,1);
    test(model)=[];
end

% close(h);


function D=fastdist(x,y)

% Calculated Euclideam distances between two sets of objetcs

D=((sum(y'.^2))'*ones(1,size(x,1)))+(ones(size(y,1),1)*(sum(x'.^2)))-2*(y*x');