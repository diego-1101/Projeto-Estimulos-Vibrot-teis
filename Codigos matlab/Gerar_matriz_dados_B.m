clear all
close all
clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Organizar Dados

id = {'01' '04' '05' '12' '18' '25' '26' '33' '38' '42'}; %CF
%id = {'02' '03' '13' '15' '28' '29' '32' '40' '43'}; %SF
load('Gab_Sequencia_Traj.mat')

for i =1:length(id)
    load(['expB_CF_' (id{i}) '.mat'])
    InfoB_SF(i,1) = str2num(id{i});
    InfoB_SF(i,2) = str2num(data.Idade);
    if data.Sexo=='f' | data.Sexo=='F'
        InfoB_SF(i,3) = 1;
    elseif data.Sexo=='m' | data.Sexo=='M'
        InfoB_SF(i,3) = 2;
    end
    InfoB_SF(i,4) = str2num(data.Peso);
    InfoB_SF(i,5) = str2num(data.Altura);
    %Info(i,6) = data.Tipo;
    %seq = cell(1,9);
    
    for j=1:length(data.res)
        
        %ProtB_SF{i}{j,1} = data.traj(j); %n?mero da trajet?ria
        %ProtB_SF{i}{j,2} = data.overlap(j); %overlap
%         ProtB_SF{i}{j,1} = data.idx(1,j); %pos
%         ProtB_SF{i}{j,2} = data.pos(j);
%         ProtB_SF{i}{j,3} = data.tempo{j,1}(1); %tempo 1
%         ProtB_SF{i}{j,4} = data.tempo{j,1}(2); %tempo 2
%         ProtB_SF{i}{j,5} = data.tempo{j,1}(3); %tempo 3
        
        for w=1:3
            ProtB_SF{i,w}{j,1} = data.idx(1,j); % trajetória
            ProtB_SF{i,w}{j,2} = data.pos(j); % sorteio
            ProtB_SF{i,w}{j,3} = data.tempo{j,w}(1); %tempo 1
            ProtB_SF{i,w}{j,4} = data.tempo{j,w}(2); %tempo 2
            ProtB_SF{i,w}{j,5} = data.tempo{j,w}(3); %tempo 3
        
            %tranSFormar desenho em blocos
            pos = data.idx(1,data.pos(find(data.pos == j)));
            if sum(max(data.res{pos,w}(:,2))-data.res{pos,w}(:,2))~=0
                data.res{pos,w}(:,2) = max(data.res{pos,w}(:,2))-data.res{pos,w}(:,2);
                
                blocox = (max(data.res{pos,w}(:,2))-min(data.res{pos,w}(:,2)))/4;
                blocoy = (max(data.res{pos,w}(:,1))-min(data.res{pos,w}(:,1)))/4;
                
                propx = (max(data.res{pos,w}(:,2))-min(data.res{pos,w}(:,2)))/520;
                propy = (max(data.res{pos,w}(:,1))-min(data.res{pos,w}(:,1)))/1043;
                
                cont=0;
                aux2{j,w} = zeros(length(data.res{pos,w}),1);
                
                for a=max(data.res{pos,w}(:,1)):-blocoy:min(data.res{pos,w}(:,1))+blocoy
                    for b=min(data.res{pos,w}(:,2)):blocox:max(data.res{pos,w}(:,2))-blocox
                        cont=cont+1;
                        aux = zeros(length(data.res{pos,w}),1);
                        aux = (find(data.res{pos,w}(:,1)<=a & data.res{pos,w}(:,1)>=a-blocoy & data.res{pos,w}(:,2)<=b+blocox & data.res{pos,w}(:,2)>=b));
                        aux2{j,w}(aux) = cont;
                        %res_blocos{j,w}(find(a>=data.res{j,w}(:,2)>a-blocoy & b>=data.res{j,w}(:,1)>b-blocox)) = cont;
                    end
                end
                
                res_blocos{j,w} = aux2{j,w}(diff([0 aux2{j,w}'])~=0);    
                
                %achar sequ?ncia de movimentos da trajet?ria
                
                for k=1:length(res_blocos{j,w})-1
                    
                    if res_blocos{j,w}(k)+1 == res_blocos{j,w}(k+1)
                        seq_completa{j,w}(k) = 1;%movimento para direita
                    elseif res_blocos{j,w}(k)-1 == res_blocos{j,w}(k+1)
                        seq_completa{j,w}(k) = 2;%movimento para esquerda
                    elseif res_blocos{j,w}(k)-4 == res_blocos{j,w}(k+1)
                        seq_completa{j,w}(k) = 3;%movimento para cima
                    elseif res_blocos{j,w}(k)+4 == res_blocos{j,w}(k+1)
                        seq_completa{j,w}(k) = 4;%movimento para baixo
                    elseif res_blocos{j,w}(k)-3 == res_blocos{j,w}(k+1)
                        seq_completa{j,w}(k) = 5;%movimento diagonal esq->dir para cima
                    elseif res_blocos{j,w}(k)+3 == res_blocos{j,w}(k+1)
                        seq_completa{j,w}(k) = 6;%movimento diagonal dir->esq para baixo
                    elseif res_blocos{j,w}(k)-5 == res_blocos{j,w}(k+1)
                        seq_completa{j,w}(k) = 7;%movimento diagonal dir->esq para cima
                    elseif res_blocos{j,w}(k)+5 == res_blocos{j,w}(k+1)
                        seq_completa{j,w}(k) = 8;%movimento diagonal esq->dir para baixo
                    else
                        seq_completa{j,w}(k) = 0;
                    end
                    
                    seq{j,w} = seq_completa{j,w}(diff([0 seq_completa{j,w}])~=0);
                    
                end
            else
                res_blocos{j,w} = 13;
                seq_completa{j,w}(1) = 9; %Ficou parado
                seq{j,w} = 9; %Ficou parado
            end
            
            %Comparar a sequ?ncia achada com o gabarito
            
            for n=1:length(seq{j,w})
                s{n} = [];
                x=1;
                for l=n:length(seq{j,w})
                    if x<=(length(gab_seq{ProtB_SF{1,1}{pos,1}}))
                        for m=x:length(gab_seq{ProtB_SF{1,1}{pos,1}})
                            if  seq{j,w}(l) == gab_seq{ProtB_SF{1,1}{pos,1}}(m)
                                s{n} = [s{n}; 1 l m];
                                x = m+1;
                                break
                            else
                                s{n} = [s{n}; 0 l m];
                            end
                        end
                    end
                end
                
                pos2 = find(s{n}(1:end-1,1)==1);
                pos3 = find(s{n}(pos2+1,1)==1);
                pontos(n) = length(pos3);
                
                clear pos2 pos3
                
            end
            
            if max(pontos)~=0
             
            pos4 = find(pontos==max(pontos),1,'first');
            
            posG = s{pos4}((find(s{pos4}(:,1)==1 & ([(s{pos4}(1:end-1,1)-s{pos4}(2:end,1)); 1]==0 | [1; (s{pos4}(2:end,1)-s{pos4}(1:end-1,1))]==0))),3);
            posR = s{pos4}((find(s{pos4}(:,1)==1 & ([(s{pos4}(1:end-1,1)-s{pos4}(2:end,1)); 1]==0 | [1; (s{pos4}(2:end,1)-s{pos4}(1:end-1,1))]==0))),2);
                
            d = [true, diff(gab_seq_completa{ProtB_SF{1,1}{pos,1}})~=0, true];
            repeticoes_G = diff(find(d));
            e = [true, diff(seq_completa{j,w})~=0, true];
            repeticoes_R = diff(find(e));
            
            score(j,w) = 0;
            for g=1:length(posG)
                if repeticoes_G(posG(g)) >= repeticoes_R(posR(g))
                    score(j,w) = score(j,w)+(repeticoes_R(posR(g))/repeticoes_G(posG(g)))/(length(gab_seq{ProtB_SF{1,1}{j,1}}));
                else
                    score(j,w) = score(j,w)+(repeticoes_G(posG(g))/repeticoes_R(posR(g)))/(length(gab_seq{ProtB_SF{1,1}{j,1}}));
                end
            end
            
            else
                score(j,w) = 0;
            end
            ProtB_SF{i,w}{j,6} = score(j,w);
            %ProtB_SF{i}{j,14+w} = score(j,w);
            score(j,w) = (propx*score(j,w) + propy*score(j,w))/2;
            
            clear l n x m pos posG posR d e pontos repeticoes_G repeticoes_R
          
%         ProtB_SF{i}{j,5+w} = score(j,w);
%         ProtB_SF{i}{j,8+w} = propx;
%         ProtB_SF{i}{j,11+w} = propy;
          ProtB_SF{i,w}{j,7} = score(j,w);
          ProtB_SF{i,w}{j,8} = propx;
          ProtB_SF{i,w}{j,9} = propy;

          stringArray1 = ['[', sprintf('%g, ', seq_completa{j,w}(1:end-1)), sprintf('%g', seq_completa{j,w}(end)), ']'];
          stringArray2 = ['[', sprintf('%g, ', seq{j,w}(1:end-1)), sprintf('%g', seq{j,w}(end)), ']'];

          ProtB_SF{i,w}{j,10} = stringArray1%trajetória completa 
          ProtB_SF{i,w}{j,11} = stringArray2 %trajetória simplificada
        end    
        
    end

end

save('ProtB_SF.mat','ProtB_SF','InfoB_SF')