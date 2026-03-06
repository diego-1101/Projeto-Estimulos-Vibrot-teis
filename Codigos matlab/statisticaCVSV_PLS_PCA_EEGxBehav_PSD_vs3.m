%..........................................................................
%            statistics para PROJETO VESTE - Correlação entre EEG x Desempenho
%..........................................................................

clear all
close all
clc

pc = 3;
directory = {'C:\Users\Jean Faber\OneDrive\MATLAB';...
       'C:\Users\jeanf_000\OneDrive\MATLAB';...
       'C:\Users\jeanf\OneDrive\MATLAB';...
       'C:\Users\Neuroengenharia\OneDrive\MATLAB'};        
   
addpath([directory{pc},'\MatlabFunctionsWeb']);
addpath([directory{pc},'\IINN\NewFunctions']);
addpath([directory{pc},'\functionsMatlab']);

%% Files to be read...
%% XX variables: PSDs...

fileNames = {'protA_cv_concat.mat','protA_sv_concat.mat'};
direct = 'G:\Meu Drive\UNIFESP\ALUNOS\Karina\ProjetoVeste\DATAscores\DadosProcessados\';

load([direct,'\PLS_PCA_XY\varEEGBEHAV_PSD.mat']);
load([direct,'\PLS_PCA_XY\varEEG_BEHAV.mat'],...
    'OverlapCV','OverlapSV','ComplexidadeCV','ComplexidadeSV');


%% YY variables : Desempenhos
%% Files to be read...

fileNames = {'protA_cv_concat.mat','protA_sv_concat.mat'};
direct = 'G:\Meu Drive\UNIFESP\ALUNOS\Karina\ProjetoVeste\DATAscores\DadosProcessados\';

clear cv sv
cv = load([direct, fileNames{1}]);
sv = load([direct, fileNames{2}]);

%% Seting all variables...

varNames = {'Desempenho','Acuracia','Similaridade',...
            'Taxa_de_Falsos_Positivos','Proporcao_espacial_x','Proporcao_espacial_y'};
idxNames = {'Overlap','Complexidade','Numero_da_Trajetoria'};

clc
clear XX YY Y
for ii=1:length(varNames)
    Y(:,ii) = [eval(['cv.',varNames{ii}])'; eval(['sv.',varNames{ii}])'];
end

gpGeral = [ones(1,length(eval(['cv.',varNames{1}])')) 2*ones(1,length(eval(['cv.',varNames{1}])'))];
gpOverlap = [eval(['cv.',idxNames{1}]) (eval(['cv.',idxNames{1}])+1)];
gpComplexidade = [eval(['cv.',idxNames{2}]) 10*eval(['cv.',idxNames{2}])];

%..........................................................................
% Index para serem descartados:

fileID = fopen('G:\Meu Drive\UNIFESP\ALUNOS\Karina\ProjetoVeste\CodigosDiego\lista_booleana.txt');
cc = textscan(fileID,'%f'); idx = logical(cc{1});



%% Statistics EEG... com visão (cv) x sem visão (sv)

clear gpCVSV XX YY gpOverlap gpComplexidade

gpCVSV = [ones(1,size(XCVpsdC3(2:end,2:end),1)) 2*ones(1,size(XSVpsdC3(2:end,2:end),1))]';

%..........................................................................

% XX = {XCVpsdC3(2:end,2:end), XSVpsdC4(2:end,2:end), [XCVpsdC4(2:end,2:end); XSVpsdC4(2:end,2:end)]};
XX = {[XCVpsdC3(2:end,2:end) XCVpsdCZ(2:end,2:end) XCVpsdC4(2:end,2:end)],...
      [XSVpsdC3(2:end,2:end) XSVpsdCZ(2:end,2:end) XSVpsdC4(2:end,2:end)],...
      [[XCVpsdC3(2:end,2:end); XSVpsdC3(2:end,2:end)],...
       [XCVpsdCZ(2:end,2:end); XSVpsdCZ(2:end,2:end)],...
       [XCVpsdC4(2:end,2:end); XSVpsdC4(2:end,2:end)]]};

% XX = {[XCVpsdC3(2:end,2:end) XCVpsdC4(2:end,2:end)],...
%       [XSVpsdC3(2:end,2:end) XSVpsdC4(2:end,2:end)],...
%       [[XCVpsdC3(2:end,2:end); XSVpsdC3(2:end,2:end)],...
%        [XCVpsdC4(2:end,2:end); XSVpsdC4(2:end,2:end)]]};

clear YY YYc
varY = [4:6];
YY = {Y(gpCVSV==1,varY), Y(gpCVSV==2,varY),[Y(idx,varY)]};
Ynom = [ones(1,486) 2*ones(1,567)];

%..........................................................................

gpOverlap = {OverlapCV(2:end), OverlapSV(2:end), [0.*OverlapCV(2:end); (0.*OverlapSV(2:end)+1)]};
gpComplexidade = {ComplexidadeCV(2:end),...
    ComplexidadeSV(2:end), [0.*ComplexidadeCV(2:end); (0.*ComplexidadeSV(2:end)+1)]};

clc
%..........................................................................
%% CDA and PCA:

clear dMOv pMOv stMOv dMCom pMCom stMCom ldP scP Xld Yld Xsc Ysc ldP scP
clc
for ii=1:length(XX)
    
    % CDA: PSDs
    [dMOv{ii},pMOv{ii},stMOv{ii}] = manova1(XX{ii}(:,1:400), gpOverlap{ii});
    %......................................................................
    [dMCom{ii},pMCom{ii},stMCom{ii}] = manova1(XX{ii}(:,1:400),gpComplexidade{ii});

    %______________________________________________________________________
    % CDA: Desempenho
    [dMOvDs{ii},pMOvDs{ii},stMOvDs{ii}] = manova1(YY{ii}, gpOverlap{ii});
    %......................................................................
    [dMComDs{ii},pMComDs{ii},stMComDs{ii}] = manova1(YY{ii},gpComplexidade{ii});    
   
    %______________________________________________________________________
    % PCA: PSD / Desempenho

    [ldP{ii},scP{ii}] = pca(zscore(XX{ii}));
    [ldPDs{ii},scPDs{ii}] = pca(zscore(YY{ii}));
    
    %______________________________________________________________________
    % PLS : EEG x Behavior

    [Xld{ii},Yld{ii},Xsc{ii},Ysc{ii}] = plsregress(zscore(XX{ii}(:,:)), zscore(YY{ii}), 5);
        
    %......................................................................
end

%% Ploting Desempenho:

%% Ploting X contra Y (EEG x Desempenho):

type = {'o','+','d','o','+','d'};

labelCVSV = {'CV','SV','CV + SV'};
cor = {'k','b','r','m','y','g'}; 
labelGP = {{'CV: 0', 'CV: 0.25', 'CV: 0.5'},{'SV: 0','SV: 0.25', 'SV: 0.5'},...
           {'CV','SV'}};
%CDA:

figure, 
for ii=1:length(XX)
    subplot(2,3,ii)
    
    plot3cluster([stMOvDs{ii}.canon(:,1:3)], gpOverlap{ii}, cor), alpha(0.6)
    xlabel('CAN1:DESEMPENHO'); ylabel('CAN2:DESEMPENHO'); zlabel('CAN3:DESEMPENHO');
    title(['OVERLAP: ', labelCVSV{ii}])
    legend(labelGP{ii})
   
    subplot(2,3,ii+3)
    plot3cluster([stMComDs{ii}.canon(:,1:3)], gpComplexidade{ii}, cor), alpha(0.6)
    xlabel('CAN1:DESEMPENHO'); ylabel('CAN2:DESEMPENHO'); zlabel('CAN3:DESEMPENHO');
    title(['COMPLEXIDADE: ', labelCVSV{ii}])
    legend(labelGP{ii})

end
suptitle('CDA')

%% Hypothesis Tests...

clc
for ii=1:3
    clear CDApANOv CDAtbANOv CDAstANOv CDAcANOv
    [CDApANOv, CDAtbANOv, CDAstANOv]=kruskalwallis(stMOvDs{ii}.canon(:,1), gpOverlap{ii},'off'); %figure
    CDAcANOv = multcompare(CDAstANOv, 'disp','off');
    sigCDAcANOv{ii}=significative(CDAcANOv); 


    clear CDApANCom CDAtbANCom CDAstANCom CDAcANCom
    [CDApANCom, CDAtbANCom, CDAstANCom]=kruskalwallis(stMComDs{ii}.canon(:,1), gpComplexidade{ii},'off'); %figure
    CDAcANCom = multcompare(CDAstANCom, 'disp','off');
    sigCDAcANCom{ii}=significative(CDAcANCom); 
end

%% Ploting HP of CANs and PCs...

clc
labelOVERLAP = {'CV: 0', 'CV: 0.25', 'CV: 0.5','SV: 0','SV: 0.25', 'SV: 0.5'};
labelCOMPLEX = {'CV: 4', 'CV: 6', 'CV: 8','SV: 4','SV: 6', 'SV: 8'};

figure, 
for ii=1:3
    subplot(2,3,ii)
    boxplot(stMOvDs{ii}.canon(:,1), gpOverlap{ii},'Color','k')
    set(gca,'xtick',1:6,'xticklabel',labelOVERLAP);
    sigstar(sigCDAcANOv{ii})     

    subplot(2,3,ii+3)
    boxplot(stMComDs{ii}.canon(:,1), gpComplexidade{ii},'Color','r')
    set(gca,'xtick',1:6,'xticklabel',labelCOMPLEX);
    sigstar(sigCDAcANCom{ii})     

end

%% Using the original variables...
clc
clear pANOv tbANOv stANOv cANOv pANCom tbANCom stANCom cANCom
for ii=1:3
    for jj=1%:size(YY{1},2)
        [pANOv{ii,jj}, tbANOv{ii,jj}, stANOv{ii,jj}]=kruskalwallis(YY{ii}(:,jj),gpOverlap{ii},'off'); %figure
        cANOv = multcompare(stANOv{ii,jj}, 'disp','off');  
        sigcANOv{ii}=significative(cANOv); 

        %......................................................................
        [pANCom{ii,jj}, tbANCom{ii,jj}, stANCom{ii,jj}]=kruskalwallis(YY{ii}(:,jj), gpComplexidade{ii},'off'); %figure
        cANCom = multcompare(stANCom{ii,jj}, 'disp','off'); 
        sigcANCom{ii}=significative(cANCom); 
        
        %......................................................................
%         [pAN2ComDs{ii,jj}, tbAN2ComDs{ii,jj}, stAN2ComDs{ii,jj}]=anovan(YY{ii}(:,jj),{gpOverlap{ii},gpComplexidade{ii}});%,'model',2,'disp','off'); 
%         figure, cAN2ComDs{ii,jj} = multcompare(stAN2ComDs{ii,jj}, 'Dimension',[1 2]); 
    end
end

%% Ploting statistics of original variable DESEMPENHO:

labelOVERLAP = {{'CV: 0', 'CV: 0.25', 'CV: 0.5'},{'SV: 0','SV: 0.25', 'SV: 0.5'},{'CV','SV'}};
labelCOMPLEX = {{'CV: 4', 'CV: 6', 'CV: 8'},{'SV: 4','SV: 6', 'SV: 8'},{'CV','SV'}};

figure, 
for ii=1:3
    subplot(2,3,ii)
    boxplot(YY{ii}(:,1), gpOverlap{ii},'Color','k')
    set(gca,'xtick',1:6,'xticklabel',labelOVERLAP{ii});
    sigstar(sigcANOv{ii})     
    
    %......................................................................

    subplot(2,3,ii+3)
    boxplot(YY{ii}(:,1), gpComplexidade{ii},'Color','m')
    set(gca,'xtick',1:6,'xticklabel',labelCOMPLEX{ii});
    sigstar(sigcANCom{ii})     

end

%%
%% Ploting X contra Y (EEG x Desempenho):

type = {'o','+','d','o','+','d'};

labelCVSV = {'CV','SV','CV + SV'};
cor = {'k','b','r','m','y','g'}; 

%CDA:

figure, 
for ii=1:length(XX)
    subplot(2,3,ii)
    
    plot3cluster([stMOv{ii}.canon(:,1), stMOvDs{ii}.canon(:,1:2)], gpOverlap{ii}, cor), alpha(0.6)
    xlabel('CAN1:PSD'); ylabel('CAN2:PSD'); zlabel('CAN1:DESEMPENHO');
    title(['OVERLAP: ', labelCVSV{ii}])
   
    subplot(2,3,ii+3)
    plot3cluster([stMCom{ii}.canon(:,1), stMComDs{ii}.canon(:,1:2)], gpComplexidade{ii}, cor), alpha(0.6)
    xlabel('CAN1:PSD'); ylabel('CAN2:PSD'); zlabel('CAN1:DESEMPENHO');
    title(['COMPLEXIDADE: ', labelCVSV{ii}])
end
suptitle('CDA')

%PCA:

figure, 
for ii=1:length(XX)
    subplot(2,3,ii)
    
    plot3cluster([scP{ii}(:,1), scPDs{ii}(:,1:2)], gpOverlap{ii}, cor), alpha(0.6)
    xlabel('PC1:PSD'); ylabel('PC2:PSD'); zlabel('PC1:DESEMPENHO');
    title(['OVERLAP: ', labelCVSV{ii}])
   
    subplot(2,3,ii+3)
    plot3cluster([scP{ii}(:,1), scPDs{ii}(:,1:2)], gpComplexidade{ii}, cor), alpha(0.6)
    xlabel('PC1:PSD'); ylabel('PC2:PSD'); zlabel('PC1:DESEMPENHO');
    title(['COMPLEXIDADE: ', labelCVSV{ii}])
end
suptitle('PCA')

%PLS:

figure, 
for ii=1:length(XX)
    subplot(2,3,ii)
    
    plot3cluster([Xsc{ii}(:,1:3)], gpOverlap{ii}, cor), alpha(0.6)
    xlabel('PC1:PSD'); ylabel('PC2:PSD'); zlabel('PC1:DESEMPENHO');
    title(['OVERLAP: ', labelCVSV{ii}])
   
    subplot(2,3,ii+3)
    plot3cluster([Xsc{ii}(:,1:3)], gpComplexidade{ii}, cor), alpha(0.6)
    xlabel('PC1'); ylabel('PC2'); zlabel('PC3');
    title(['COMPLEXIDADE: ', labelCVSV{ii}])
end
suptitle('PLS')

clc
%% Ploting CDA EEG in function of States (Overlap and Complexity)

labelCVSV = {'CV','SV','CV + SV'};
cor = {'k','b','r','m','y','g'}; 

figure, 
for ii=1:length(XX)
    subplot(2,3,ii)
    
    plot3cluster(stMOv{ii}.canon(:,1:3), gpOverlap{ii}, cor), alpha(0.6)
    xlabel('CAN1'); ylabel('CAN2'); zlabel('CAN3');
    title(['OVERLAP: ', labelCVSV{ii}])
   
    subplot(2,3,ii+3)
    plot3cluster(stMCom{ii}.canon(:,1:3), gpComplexidade{ii}, cor), alpha(0.6)
    xlabel('CAN1'); ylabel('CAN2'); zlabel('CAN3');
    title(['COMPLEXIDADE: ', labelCVSV{ii}])
end
suptitle('CDA')
    
figure, 
for ii=1:length(XX)
    subplot(2,3,ii)
    plot3cluster(Xsc{ii}(:,1:3), gpOverlap{ii}, cor), alpha(0.6)
    xlabel('CAN1'); ylabel('CAN2'); zlabel('CAN3');
    title(['OVERLAP: ', labelCVSV{ii}])
   
    subplot(2,3,ii+3)
    plot3cluster(Xsc{ii}(:,1:3), gpComplexidade{ii}, cor), alpha(0.6)
    xlabel('CAN1'); ylabel('CAN2'); zlabel('CAN3');
    title(['COMPLEXIDADE: ', labelCVSV{ii}])   
end
suptitle('PLS: X_{score}')

figure, 
for ii=1:length(XX)
    subplot(2,3,ii)
    plot3cluster(scP{ii}(:,1:3), gpOverlap{ii}, cor), alpha(0.6)
    xlabel('CAN1'); ylabel('CAN2'); zlabel('CAN3');
    title(['OVERLAP: ', labelCVSV{ii}])
   
    subplot(2,3,ii+3)
    plot3cluster(scP{ii}(:,1:3), gpComplexidade{ii}, cor), alpha(0.6)
    xlabel('CAN1'); ylabel('CAN2'); zlabel('CAN3');
    title(['COMPLEXIDADE: ', labelCVSV{ii}])
end
suptitle('PCA')

% figure, 
% for ii=1:length(XX)
%     subplot(2,3,ii)
%     gscatter(UCan{ii},VCan{ii}, gpOverlap{ii}), alpha(0.6)
%     xlabel('CAN1'); ylabel('CAN2'); zlabel('CAN3');
%     title(['OVERLAP: ', labelCVSV{ii}])
%    
%     subplot(2,3,ii+3)
%     gscatter(UCan{ii},VCan{ii}, gpComplexidade{ii}), alpha(0.6)
%     xlabel('CAN1'); ylabel('CAN2'); zlabel('CAN3');
%     title(['COMPLEXIDADE: ', labelCVSV{ii}])
% end
% suptitle('CCA')


%% Corrplot...










%% Ploting CDA and PCA

type = {'o','+','d','o','+','d'};
colorOver1 = {[0 0.8 0.8],[0 0.6 0.6],[0 0.4 0.4],[0.8 0 0],[0.6 0 0],[0.4 0 0]};
colorComp1 = {[0 0 0.8],[0 0 0.6],[0 0 0.4],[0.6 0 0],[0.4 0 0],[0.2 0 0]};

colorOver2 = {[0.4 0.8 0.8],[0.4 0.6 0.6],[0.4 0.4 0.4],[0.8 0 0.4],[0.6 0 0.4],[0.4 0 0.4]};
colorComp2 = {[0.4 0 0.8],[0.4 0 0.6],[0.4 0 0.4],[0.6 0 0.4],[0.4 0 0.4],[0.2 0 0.4]};

figure, 
subplot(2,2,1)
s = stMOv.canon;

uniqueGroups = unique(gpOverlap); 
colors = brewermap(length(uniqueGroups),'Set1'); 
hold on, grid minor
for k = 1:length(uniqueGroups)
      ind = gpOverlap==uniqueGroups(k); 
      plot3(s(ind,1),s(ind,2),s(ind,3),type{k},'color',colorOver1{k},'markersize',5,'LineWidth',3);
      xlim([-4 4]), ylim([-4 4]), zlim([-4 4])
end
xlabel('CAN1'), ylabel('CAN2'), zlabel('CAN3')
set(gca, 'FontSize', 10)
legend('CV: 0', 'CV: 0.25', 'CV: 0.5','SV: 0','SV: 0.25', 'SV: 0.5')
title(['CDA: OVERLAP'])  

%..........................................................................

subplot(2,2,2)
s = stMCom.canon;

uniqueGroups = unique(gpComplexidade); 
colors = brewermap(length(uniqueGroups),'Set1'); 
hold on, grid minor
for k = 1:length(uniqueGroups)
      ind = gpComplexidade==uniqueGroups(k); 
      plot3(s(ind,1),s(ind,2),s(ind,3),type{k},'color',colorComp1{k},'markersize',5,'LineWidth',3);
      xlim([-4 4]), ylim([-4 4]), zlim([-4 4])
end
xlabel('CAN1'), ylabel('CAN2'), zlabel('CAN3')
set(gca, 'FontSize', 10)
legend('CV: 4', 'CV: 6', 'CV: 8','SV: 4','SV: 6', 'SV: 8')
title(['CDA: COMPLEXITY'])  

%__________________________________________________________________________

subplot(2,2,3)
s = scP;

uniqueGroups = unique(gpOverlap); 
colors = brewermap(length(uniqueGroups),'Set1'); 
hold on, grid minor
for k = 1:length(uniqueGroups)
      ind = gpOverlap==uniqueGroups(k); 
      plot3(s(ind,1),s(ind,2),s(ind,3),type{k},'color',colorOver2{k},'markersize',5,'LineWidth',3);
      xlim([-4 4]), ylim([-4 4]), zlim([-4 4])
end
xlabel('PC1'), ylabel('PC2'), zlabel('PC3')
set(gca, 'FontSize', 10)
legend('CV: 0', 'CV: 0.25', 'CV: 0.5','SV: 0','SV: 0.25', 'SV: 0.5')
title(['PCA: OVERLAP'])  

%..........................................................................

subplot(2,2,4)
s = scP;

uniqueGroups = unique(gpComplexidade); 
colors = brewermap(length(uniqueGroups),'Set1'); 
hold on, grid minor
for k = 1:length(uniqueGroups)
      ind = gpComplexidade==uniqueGroups(k); 
      plot3(s(ind,1),s(ind,2),s(ind,3),type{k},'color',colorComp2{k},'markersize',5,'LineWidth',3);
      xlim([-4 4]), ylim([-4 4]), zlim([-4 4])
end
xlabel('PC1'), ylabel('PC2'), zlabel('PC3')
set(gca, 'FontSize', 10)
legend('CV: 4', 'CV: 6', 'CV: 8','SV: 4','SV: 6', 'SV: 8')
title(['PCA: COMPLEXITY'])  


%% Hypothesis Tests...

clear CDApANOv CDAtbANOv CDAstANOv CDAcANOv
[CDApANOv, CDAtbANOv, CDAstANOv]=anova1(stMOv.canon(:,1), gpOverlap,'off'); figure
CDAcANOv = multcompare(CDAstANOv, 'disp','on');

clear CDApANCom CDAtbANCom CDAstANCom CDAcANCom
[CDApANCom, CDAtbANCom, CDAstANCom]=anova1(stMCom.canon(:,1), gpComplexidade,'off'); figure
CDAcANCom = multcompare(CDAstANCom, 'disp','on');

clear PCApANOv PCAtbANOv PCAstANOv cANOv
[PCApANOv, PCAtbANOv, PCAstANOv]=anova1(scP(:,1), gpOverlap,'off'); figure
cANOv = multcompare(PCAstANOv, 'disp','on');

clear PCApANCom PCAtbANCom PCAstANCom cANCom
[PCApANCom, PCAtbANCom, PCAstANCom]=anova1(scP(:,1), gpComplexidade,'off'); figure
cANCom = multcompare(PCAstANCom, 'disp','on');


%% Ploting HP of CANs and PCs...

clc
labelOVERLAP = {'CV: 0', 'CV: 0.25', 'CV: 0.5','SV: 0','SV: 0.25', 'SV: 0.5'};
labelCOMPLEX = {'CV: 4', 'CV: 6', 'CV: 8','SV: 4','SV: 6', 'SV: 8'};

figure, 
subplot(2,2,1)
boxplot(stMOv.canon(:,1), gpOverlap)
set(gca,'xtick',1:6,'xticklabel',labelOVERLAP);

subplot(2,2,2)
boxplot(stMCom.canon(:,1), gpComplexidade)
set(gca,'xtick',1:6,'xticklabel',labelCOMPLEX);

subplot(2,2,3)
boxplot(scP(:,1), gpOverlap)
set(gca,'xtick',1:6,'xticklabel',labelOVERLAP);

subplot(2,2,4)
boxplot(scP(:,1), gpComplexidade)
set(gca,'xtick',1:6,'xticklabel',labelCOMPLEX);

%% Using the original variables...

clear pANOv tbANOv stANOv cANOv pANCom tbANCom stANCom cANCom
for ii=1:size(XX,2)
%     [pANOv{ii}, tbANOv{ii}, stANOv{ii}]=kruskalwallis(XX(:,ii), gpOverlap,'off'); figure
%     cANOv{ii} = multcompare(stANOv{ii}, 'disp','off');  
%     
%     %......................................................................
%     [pANCom{ii}, tbANCom{ii}, stANCom{ii}]=kruskalwallis(XX(:,ii), gpComplexidade,'off'); figure
%     cANCom{ii} = multcompare(stANCom{ii}, 'disp','off'); 
    
    %......................................................................
    [pAN2Com{ii}, tbAN2Com{ii}, stAN2Com{ii}]=anovan(XX(:,ii),{gpOverlap,gpComplexidade})%,'model',2,'disp','off'); 
    figure
     cAN2Com{ii} = multcompare(stAN2Com{ii}, 'Dimension',[1 2]); 
end

%..........................................................................
% Anova-2Way:











