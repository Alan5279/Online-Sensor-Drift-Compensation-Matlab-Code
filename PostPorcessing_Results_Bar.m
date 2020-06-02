%Result processing bar
MainFolder=['Data',filesep,'SameSequence'];
Methods=cell(10,1);
Methods{1}='ELM';
Methods{2}='SVM';Methods{3}='Ensemble-ELM';Methods{4}='Ensemble-SVM';
Methods{5}='RandomForest';Methods{6}='DAELMS';Methods{7}='DAELMT';
Methods{8}='ODAELMS';Methods{9}='ODAELMT';
Flag='-Same-Time-';
Flag2='-Time-';
Index=['B'];
FType='.txt';

DA=zeros(9,4);
for i=1:9
    for j=1:9
        if j==7||j==9
            File=[MainFolder,filesep,Methods{j},num2str(2),filesep,Methods{j},Flag,Index,num2str(i),FType];
        else
            File=[MainFolder,filesep,Methods{j},filesep,Methods{j},Flag,Index,num2str(i),FType];
        end
        if j>=6
            TempData=load(File);
            if(j==7)
                File2=[MainFolder,filesep,Methods{j},filesep,Methods{j},'-Error-',Index,num2str(i),FType];
                TempData2=load(File2);
                TempData=TempData(1:size(TempData2,1),:);
            end
            if(j==9)
                File2=[MainFolder,filesep,Methods{j},filesep,Methods{j},'-Same-Error-',Index,num2str(i),FType];
                TempData2=load(File2);
                TempData=TempData(1:size(TempData2,1),:);
            end
            if(j==8)
                File2=[MainFolder,filesep,Methods{j},num2str(2),filesep,Methods{j},'-Same-Error-',Index,num2str(i),FType];
                TempData2=load(File2);
                TempData=TempData(1:size(TempData2,1),:);
            end
%             if(i==3)
%                 File2=[MainFolder,filesep,Methods{j},filesep,Methods{j},'-Same-Error-',Index,num2str(i),FType];
%                 TempData2=load(File2);
%                 size(TempData2,1)
%                 size(TempData,1)
%             end
            DA(i,j-5)=sum(TempData);
        end
    end
end

% k=2;
figure;
bar(DA(:,:));