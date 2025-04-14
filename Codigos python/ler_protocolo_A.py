'''
Código feito para abrir os arquivos .mat de cada Protocolo e extrair as trajetórias 
'''

#%%
from scipy.io import loadmat
import pandas as pd
import numpy as np 
import ast
import Plotar_sequencias as plotar

#%%

def transformar_protA_mat_em_df(protocolo = [], id = ['07', '10', '17','21','24','31','36','39']):
    '''
    Recebe uma strig do protocolo A que acabou de ser lido em .mat e retorna
    um DataFrame.
    '''
    from scipy.io import loadmat
    import pandas as pd
    import numpy as np 
    import ast

    # Criar uma lista para armazenar os DataFrames
    dataframes = []

    # Headers das colunas
    headers = ['Número da Trajetória', 'Overlap', 'Sorteio', 'Tempo 1',
           'Tempo 2', 'Tempo 3', 'Score total', 'Score Parcial',
           'Proporção espacial x', 'Proporção espacial y',
           'Trajetória Completa', 'Trajetória Simplificada']

    # Iterar pelas células da célula principal
    for i in range(protocolo.shape[0]):  # Itera sobre as linhas
        for j in range(protocolo.shape[1]):  # Itera sobre as colunas
            cell_data = protocolo[i, j]  # Acessa a célula individual
            # Converter a célula individual para um DataFrame
            df = pd.DataFrame(cell_data, columns=headers)  # Adiciona os headers
            # Armazenar o DataFrame com a localização
            dataframes.append({'Row': i + 1, 'Column': j + 1, 'DataFrame': df})

    # Transformando as strings de lista em lista
    '''
    Antes, cada célula estava com um array dentro de outroi array, essa parte
    é para simplificar as coisas e deixar um DataFrame mais amigável
    '''
    for i in range(len(dataframes)):
        for j in range(dataframes[i]['DataFrame']['Trajetória Completa'].shape[0]):
            if(dataframes[i]['DataFrame']['Trajetória Completa'][j] != '[, 9]' and dataframes[i]['DataFrame']['Trajetória Simplificada'][j] != '[, 9]'):
                dataframes[i]['DataFrame']['Número da Trajetória'][j] = dataframes[i]['DataFrame']['Número da Trajetória'][j][0]
                dataframes[i]['DataFrame']['Overlap'][j] = dataframes[i]['DataFrame']['Overlap'][j][0]
                dataframes[i]['DataFrame']['Sorteio'][j] = dataframes[i]['DataFrame']['Sorteio'][j][0]
                dataframes[i]['DataFrame']['Tempo 1'][j] = dataframes[i]['DataFrame']['Tempo 1'][j][0]
                dataframes[i]['DataFrame']['Tempo 2'][j] = dataframes[i]['DataFrame']['Tempo 2'][j][0]
                dataframes[i]['DataFrame']['Tempo 3'][j] = dataframes[i]['DataFrame']['Tempo 3'][j][0]
                dataframes[i]['DataFrame']['Score total'][j] = dataframes[i]['DataFrame']['Score total'][j][0]
                dataframes[i]['DataFrame']['Score Parcial'][j] = dataframes[i]['DataFrame']['Score Parcial'][j][0]
                dataframes[i]['DataFrame']['Proporção espacial x'][j] = dataframes[i]['DataFrame']['Proporção espacial x'][j][0]
                dataframes[i]['DataFrame']['Proporção espacial y'][j] = dataframes[i]['DataFrame']['Proporção espacial y'][j][0]
                dataframes[i]['DataFrame']['Trajetória Completa'][j]= ast.literal_eval(dataframes[i]['DataFrame']['Trajetória Completa'][j][0])
                dataframes[i]['DataFrame']['Trajetória Simplificada'][j]= ast.literal_eval(dataframes[i]['DataFrame']['Trajetória Simplificada'][j][0])
            else:
                dataframes[i]['DataFrame']['Número da Trajetória'][j] = dataframes[i]['DataFrame']['Número da Trajetória'][j][0]
                dataframes[i]['DataFrame']['Overlap'][j] = dataframes[i]['DataFrame']['Overlap'][j][0]
                dataframes[i]['DataFrame']['Sorteio'][j] = dataframes[i]['DataFrame']['Sorteio'][j][0]
                dataframes[i]['DataFrame']['Tempo 1'][j] = dataframes[i]['DataFrame']['Tempo 1'][j][0]
                dataframes[i]['DataFrame']['Tempo 2'][j] = dataframes[i]['DataFrame']['Tempo 2'][j][0]
                dataframes[i]['DataFrame']['Tempo 3'][j] = dataframes[i]['DataFrame']['Tempo 3'][j][0]
                dataframes[i]['DataFrame']['Score total'][j] = dataframes[i]['DataFrame']['Score total'][j][0]
                dataframes[i]['DataFrame']['Score Parcial'][j] = dataframes[i]['DataFrame']['Score Parcial'][j][0]
                dataframes[i]['DataFrame']['Proporção espacial x'][j] = dataframes[i]['DataFrame']['Proporção espacial x'][j][0]
                dataframes[i]['DataFrame']['Proporção espacial y'][j] = dataframes[i]['DataFrame']['Proporção espacial y'][j][0]
                dataframes[i]['DataFrame']['Trajetória Completa'][j] = [9]
                dataframes[i]['DataFrame']['Trajetória Simplificada'][j]= [9]
                

    #Estruturando um dataframe com todos os indivíduos
    prot_df = {}
    cabecalho = []
    for i, ID in enumerate(id):
        prot_df[f'df_ID_{ID}'] = {'Rep1': dataframes[i*3]['DataFrame'],
                                    'Rep2': dataframes[i*3+1]['DataFrame'],
                                    'Rep3': dataframes[i*3+2]['DataFrame']}
        cabecalho.append(f'df_ID_{ID}')
    prot_df = pd.DataFrame(prot_df)


    return prot_df

#%% 
#ID dos pacientes naquele protocolo
id_cv = ['07', '10', '17','21','24','31','36','39']
id_sv = ['06', '09','16','19','23', '27', '34','37']

# Carregar o arquivo .mat do protocolo 
A_CV = loadmat('Aquivos mat\ProtA_CV.mat')
A_SV = loadmat('Aquivos mat\ProtA_SV.mat')

# Acessar o conteúdo de ProtA_CV
ProtA_CV = A_CV['ProtA_CV']  # ProtA_CV é uma célula de células
ProtA_SV = A_SV['ProtA_CV'] # ProtA_CV é uma célula de células (ta so com o mesmo nome do CV, mas não é a mesma coisa)

# Convertendo para um DataFrame
protA_cv_df = transformar_protA_mat_em_df(protocolo = ProtA_CV, id = id_cv)
protA_sv_df = transformar_protA_mat_em_df(protocolo = ProtA_SV, id = id_sv)


#%% plotando algumas trajetórias
"""for i, seq in enumerate(protA_cv_df['df_ID_10']['Rep2']['Trajetória Completa']):
    plotar.plotar_trajetoria(seq = seq,individuo= 'ID10 na primeira repetição')
"""

# %% Carregando os gabaritos
gabarito = pd.read_csv('Gabaritos\gab_seq_completa_converted.csv')
gabarito_simplificado = pd.read_csv('Gabaritos\gab_seq_converted.csv')
for i in gabarito.columns:
    gabarito[i][0] = ast.literal_eval(gabarito[i][0])
    gabarito_simplificado[i][0] = ast.literal_eval(gabarito_simplificado[i][0])

# %% Ideia 1 - Comparação simples
pontuacao = []

for i, num in enumerate(teste['Número da Trajetória']):
    tamanho = 0
    coincidencia = 0
    reincidencia = 0 
    #pegando a sequencia a ser comparada
    seq = teste['Trajetória Completa'][i]
    #pegando a sequência do gabarito correspondente à sequência que a pessoa fez
    certo = gabarito[f'{num}'][0]
    
    if(len(certo) == len(seq)):
        tamanho += 1
    elif len(seq) > len(certo):
        certo = np.pad(certo,(0,len(seq)-len(certo)), mode ='constant', constant_values = 0)
    else:
        certo = np.pad(certo,(0,len(certo)-len(seq)), mode ='constant', constant_values = 0)

    for j in range(len(seq)):
        #verifico se a direção atual é igual a direção do gabarito
        if(seq[j] == certo[j]):
            coincidencia += 1
            #se for, vejo se a direção seguinte também é igual
            if(j < len(seq)-1):
                if(seq[j+1] == certo[j+1]):
                    reincidencia += 1
      
    ponto = tamanho + coincidencia + reincidencia
    
    pontuacao.append(ponto) 
          

print(pontuacao)

#%% Ideia 2 - Comparação por matchs dinâmicos
def avaliar_match_dinamico(v1 = [],v2 =[]):
    '''
    Compara dinamicamente os vetores v1 e v2
    No caso do código, v1 será a sequencia do individuo e v2 a sequencia do gabarito
    '''
    resultado = []
    resultado_uns = [] 
    # Loop sobre posições iniciais dinamicamente
    for start in range(len(v1)):  # Iterar por todas as posições iniciais
        temp_list = []  # Lista temporaria para essa iteração
        bin_temp_list = []
        # Loop para verificar correspondências crescentes a partir da posição inicial atual
        for i in range(start, len(seq)):  
            if np.array_equal(v1[start:i+1], v2[start:i+1]):  # Compara os subarrays
                temp_list.append(i - start + 1)  #  Append no comprimento do match
                bin_temp_list.append(1)
        resultado.append(temp_list)  # Armazena o resultado pelo seus índices de início 
        resultado_uns.append(bin_temp_list)
    
    return resultado, resultado_uns
# %% Ideia 3- Comparação de padrão de imagem 
def rotate(array):
    """Recebe uma matriz e rotaciona ela no sentido anti-horário

    Args:
        array (list): array que você quer rotacionar

    Returns:
        rotArr (list): array rotacionado no sentido anti-horário
    """
    '''
    
    '''
    import numpy as np

    # tamanho de linhas e colunas da matriz rotacionada
    L,C = len(array), len(array[0]) 

    # criando a matriz que irá conter o resultado
    rotArr = [[None]*L for _ in range(C)]

    # rotacionando o array
    for c in range(C):
        for l in range(L-1,-1,-1):
            rotArr[C-c-1][l] = array[l][c]
    
    return np.array(rotArr)

def traj_to_point(traj =[]):
    """Decodifica a trajetória de números para pontos em um sistema de coordenadas (x,y)

    Args:
        traj (list, optional): trajetória que você quer decodificar. Defaults to [].

    Returns:
        x: vetor de posições em x
        y: vetor de posições em y
        sequencia: sequência em forma de flechas
    """

    
    # Vetores contendo todas as posições x e y 
    x = [0]
    y = [0]

    #sequencia em setas para um print amigável
    sequencia = []

    for num in traj:
        match num:
            case 1:
                # Movimento para Direita
                y.append(y[-1])
                x.append(x[-1]+1)
                sequencia.append('⮕')
                #print('⮕')
                pass
            case 2:
                #Movimento para Esquerda
                y.append(y[-1])
                x.append(x[-1]-1)
                sequencia.append('⬅')
                #print('⬅')
                pass
            case 3:
                # Movimento para Cima 
                y.append(y[-1]+1)
                x.append(x[-1])
                sequencia.append('⬆')
                #print('⬆')
                pass
            case 4:
                #Movimento para Baixo
                y.append(y[-1]-1)
                x.append(x[-1])
                sequencia.append('⬇')
                #print('⬇')
                pass
            case 5:
                # Movimento Diagonal esq->dir para cima
                y.append(y[-1]+1)
                x.append(x[-1]+1)
                sequencia.append('⬈')
                #print('⬈')
                pass
            case 6:
                #Movimento Diagonal dir->esq para baixo
                y.append(y[-1]-1)
                x.append(x[-1]-1)
                sequencia.append('⬋')
                #print('⬋')
                pass
            case 7:
                # Movimento Diagonal dir->esq para cima
                y.append(y[-1]+1)
                x.append(x[-1]-1)
                sequencia.append('⬉')
                #print('⬉')
                pass
            case 8:
                #Movimento Diagonal esq->dir para baixo
                y.append(y[-1]-1)
                x.append(x[-1]+1)
                sequencia.append('⬊')
                #print('⬊')
                pass
            case 9:
                #Parado
                y.append(y[-1])
                x.append(x[-1])
                sequencia.append('Parado')
                #print('Parado')
                pass
    
    return x,y,sequencia

def comparar_imagem(seq1=[], seq2=[], plotar_imagens = False):
    """Compara espacialmente duas sequencias

    Args:
        seq1 (list): sequencia executada pelo individuo. Defaults to [].
        seq2 (list): gabarito. Defaults to [].
        plotar_imagens (bool): se quer ou não plotar as imagens das matrizes binarizadas. Defaults to False
    """
    import numpy as np 
    import matplotlib.pyplot as plt
    
    #Pegando os vetores x e y de cada sequência
    x1,y1,_= traj_to_point(seq1)
    x2,y2,_= traj_to_point(seq2)

    # Verificando qual o tamanho máximo explorado na sequência para fazer posteriormente a matriz de zeros
    tamanho = np.max(np.concatenate((x1,x2,y1,y2)))
    #print(tamanho)

    #criando as sequências que vão ser a imagem binarizada
    v1_bin = np.zeros((tamanho+1,tamanho+1)) 
    v2_bin = np.zeros((tamanho+1,tamanho+1))

    #colocando os 1's de acordo com a trajetória
    for i in zip(x1,y1):
        #se os valores de x e y forem iguais a zero eu apenas desconsidero na conversão
        if((i[0]>=0)&(i[1]>=0)):
            v1_bin[i[0]][i[1]] = 1

    #se os valores de x e y forem iguais a zero eu apenas desconsidero na conversão    
    for i in zip(x2,y2):
        if((i[0]>=0)&(i[1]>=0)):
            v2_bin[i[0]][i[1]] = 1
    
    #rotacionando as matrizes para ficar de um jeito amigável para printar
    v1_bin = rotate(v1_bin)
    v2_bin = rotate(v2_bin)
    #print("Trajetória 1 Binarizada")
    #print(v1_bin)
    #print('--'*100)
    #print('Trajetória 2 Binarizada')
    #print(v2_bin)

    #Multiplicando ponto a ponto das matrizes
    multiplicacao = v1_bin*v2_bin
    #print('--'*100)
    #print("Matriz binarizada da sobreposição das trajetórias:")
    #print(multiplicacao)
    
    # Calcula o que está exclusivamente ou em v1_bin ou em v2_bin
    xor = v1_bin.astype(int) ^ v2_bin.astype(int) 
    
    '''
    xor_sum = np.sum(xor)
    print('--'*100)
    print('Matriz XOR binarizada:')
    print(xor)
    # Score xor é o score de quanto a pessoa errou dado o quanto ela poderia ter errado
    # quanto mais próximo de 1 mais ela errou
    #score_xor = xor_sum/(np.sum(1-v2_bin) if np(1-v2_bin) != 0 else 1)"""
    '''

    # ----- Calculando as métricas de comparação 
    TP = np.sum((v1_bin.astype(int) == 1) & (v2_bin.astype(int) == 1)) # verdadeiros positivos
    FP = np.sum((v1_bin.astype(int) == 1) & (v2_bin.astype(int) == 0)) # falsos positivos
    TN = np.sum((v1_bin.astype(int) == 0) & (v2_bin.astype(int) == 0)) # verdadeiros negativos
    FN = np.sum((v1_bin.astype(int) == 0) & (v2_bin.astype(int) == 1)) # falsos negativos

    # Mértricas
    acuracia = (TP+TN)/(TP+TN+FP+FN)
    precisao = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    taxa_FP = FP / (FP + TN) if (FP + TN) > 0 else 0

    # Exibir resultados
    #print(f"Acurácia: {acuracia:.4f}")
    #print(f"Precisão: {precisao:.4f}")
    #print(f"Recall (Sensibilidade): {recall:.4f}")
    #print(f"Taxa de Falsos Positivos (FPR): {taxa_FP:.4f}")

    # ----- Visualizando os resultados
    if(plotar_imagens):
        # Criar a figura e subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))  # Criar uma linha com 4 colunas

        # Lista das imagens e títulos correspondentes
        imagens = [v1_bin, v2_bin, multiplicacao,xor]
        titulos = ['Trajetória 1 Binarizada (realizada)', 'Trajetória 2 Binarizada (gabarito)', 
           'Sobreposição das trajetórias Binarizada', 'Matriz XOR binarizada']
        
        #Loop ara exibir cada imagem no subplot correspondente
        for ax, img, titulo in zip(axes.ravel(), imagens, titulos):
            ax.imshow(img, cmap='gray', interpolation='nearest')

            # Percorrer todos os pixels da matriz e adicionar um "1" onde houver um pixel ativado
            for i in range(img.shape[0]):  # Linhas
                for j in range(img.shape[1]):  # Colunas
                    if img[i, j] == 1:
                        ax.text(j, i, '1', ha='center', va='center', color='red', fontsize=12, fontweight='bold')

            ax.set_title(titulo,fontsize = 16)
            ax.axis('off')  # Remover os eixos para melhor visualização

        # Ajustar layout
        plt.tight_layout()
        plt.show()
    

    return [acuracia,precisao,recall,taxa_FP]

# %% Ideia 4- Comparação por correlação máxima normalizada
def calcular_similaridade(seq1,seq2):
    """
    Calcula a métrica de similaridade entre seq1 e seq2 usando correlação cruzada normalizada.
    Retorna um valor entre 0 e 1 indicando a semelhança.
    """
    from  scipy.signal import correlate
    
    #como quero testar quanto a sequencia executada é semelhante ao gabarito, seq1 (executada) é fixada e seq2 (gabarito) será deslocado
    corr = correlate(seq1,seq2,mode = 'full') 

    #pegando o melhor alinhamento possível da correlação
    max_corr = np.max(corr)

    # Normalizando pelo produto das energias das sequências
    fator_normalizacao = np.sqrt(np.sum(seq1**2) * np.sum(seq2**2))

    # Calculando a similaridade evitando divisão por 0
    similaridade = (max_corr/fator_normalizacao) if fator_normalizacao != 0 else print('Não é possível calcular a similaridade, pois a energia de uma das sequencias é 0')

    return(similaridade)


#%% Calculando os resultados para uma repetição de um dado individuo

teste = protA_cv_df['df_ID_21']['Rep2']
#plotar.plotar_trajetoria(seq = teste['Trajetória Completa'][8])

resultados = []
for i, num in enumerate(teste['Número da Trajetória']):
    # Armazenando as sequencias da vez em cada variavel
    seq1 = np.array(teste['Trajetória Completa'][i]) # sequencia realizada
    seq2 = np.array(gabarito[f'{num}'][0]) # sequencia gabarito
    
    #---- Avaliando o match perfeito (IDEIA 2)
    tamanho = 0
    coincidencia = 0
    reincidencia = 0 
    #pegando a sequencia a ser comparada
    seq = seq1
    #pegando a sequência do gabarito correspondente à sequência que a pessoa fez
    certo = seq2
    
    if(len(certo) == len(seq)):
        tamanho += 1
    elif len(seq) > len(certo):
        certo = np.pad(certo,(0,len(seq)-len(certo)), mode ='constant', constant_values = 0)
    else:
        certo = np.pad(certo,(0,len(certo)-len(seq)), mode ='constant', constant_values = 0)
    
    # 1) Avaliar o match perfeito para poder normalizar depois
    soma_perfeita = 0
    soma_perfeita_uns = 0 
    resultado_perfeito, resultado_perfeito_uns  = avaliar_match_dinamico(certo, certo)
    
    for j in resultado_perfeito:
        soma_perfeita += np.sum(j)
    
    for j in resultado_perfeito_uns:
        soma_perfeita_uns += np.sum(j)

    # 2) Avaliar o match real
    soma_real = 0
    soma_real_uns = 0

    resultado_real, resultado_real_uns = avaliar_match_dinamico(seq, certo)
    
    for j in resultado_real:
        soma_real += np.sum(j)
    
    for j in resultado_real_uns:
        soma_real_uns += np.sum(j)

    # 3) Dividir o match perfeito pelo real para obter o score parcial
    score_parcial = soma_real/soma_perfeita
    score_parcial_uns = soma_real_uns/soma_perfeita_uns

    # 4) score total sendo a média do score parcial ponderado pela proporção explorada em x e em y
    score_total = ((score_parcial*teste['Proporção espacial x'][i]) + 
                    (score_parcial*teste['Proporção espacial y'][i]) 
                    )/2
    
    score_total_uns = ((score_parcial_uns*teste['Proporção espacial x'][i]) + 
                    (score_parcial_uns*teste['Proporção espacial y'][i]) 
                    )/2
    
    """print('--'*100)
    print(f'Trajetória {num}')
    print(f'Score parcial = {score_parcial} \nScore Total = {score_total}')
    print(f'Trajetória {num}')
    print(f'Score parcial uns = {score_parcial_uns} \nScore Total = {score_total_uns}')"""

    #---- Avaliando por comparação de imagem (IDEIA 3)
    resultado_ideia3 = comparar_imagem(seq1=seq1,seq2=seq2,plotar_imagens = False)

    #---- Avaliando por similaridade com correlação cruzada normalizada (IDEIA 4)
    resultado_ideia4 = calcular_similaridade(seq1,seq2)
    
    resultados.append([score_parcial_uns,score_total_uns,resultado_ideia3[0],resultado_ideia3[1], resultado_ideia3[2], resultado_ideia3[3],resultado_ideia4])

    # Plotando o comparativo das trajetórias
    plotar.plot_comparacao(gabarito=seq2, seq= seq1)

    # Printando os resultados
    print('--------- ' + f'Trajetória {num}' + ' ---------')
    print(f'-> Resultado da ideia 2 (comparação por matchs perfeitos):\nScore parcial: {score_parcial_uns} | Score total: {score_total_uns}')
    print(f'-> Resultado da ideia 3 (comparação de imagens):\nAcurácia: {resultado_ideia3[0]:.4f} | Precisão: {resultado_ideia3[1]:.4f}\nRecall: {resultado_ideia3[2]:.4f} | FPR: {resultado_ideia3[3]:.4f}')
    print(f'-> Resultado da ideia 4 (comparação por similaridade):\nSimilaridade entre as trajetórias: {resultado_ideia4}')
    print('--'*100)

# %% Vendo a distribuição dos resultados obtidos acima 
resultados_df = pd.DataFrame(resultados)
resultados_df.columns = ['Score Parcial','Score Total', 'Acurácia','Precisão','Recall','Taxa de Falsos Positivos','Similaridade']
import matplotlib.pyplot as plt 
import seaborn as sns

for parametro in resultados_df.columns:
    data = resultados_df[parametro]

    # Criando a figura com 2 subplots (1 linha, 2 colunas)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [3, 1]})
    # Histograma
    ax[0].hist(data, bins=20, color='blue', alpha=0.7, edgecolor='black',label ='Histograma')
    # Adicionar a linha de tendência (KDE)
    sns.kdeplot(data, color='red', linewidth=2, label="Curva KDE",ax=ax[0])
    # Calcular e destacar a média no gráfico
    media = np.mean(data)
    ax[0].axvline(media, color='black', linestyle='dashed', linewidth=2, label=f"Média: {media:.2f}")
    ax[0].set_title(f"Histograma de {parametro}")
    ax[0].set_xlabel("Valores")
    ax[0].set_ylabel("Frequência")
    ax[0].legend()
    ax[0].grid(True)
    #Boxplot
    ax[1].boxplot(data,vert=False, patch_artist=True, boxprops=dict(facecolor='lightblue'),label='boxplot')
    ax[1].axvline(max(data), color='gray', linestyle='dotted', linewidth=2, label=f"Valor máximo: {max(data):.2f}",alpha = 0.7)
    ax[1].axvline(media, color='black', linestyle='dashed', linewidth=2, label=f"Média: {media:.2f}",alpha = 0.7)
    ax[1].set_title(f"Box Plot de {parametro}")
    ax[1].set_xlabel("Valores")
    ax[1].legend()
    
    fig.suptitle(f'Visualização de {parametro}')
    plt.tight_layout()
    plt.show()

# %%#%% Calculando os resultados para todas as repetições de todos os individuos

sparcial =[]
stotal = []
acur = []
prec = []
rcll = []
fpr =[]
sim = []

for individuo in protA_cv_df.columns:
    for rep in protA_cv_df.index:
        teste = protA_cv_df[individuo][rep]
    
        for i, num in enumerate(teste['Número da Trajetória']):
            # Armazenando as sequencias da vez em cada variavel
            seq1 = np.array(teste['Trajetória Completa'][i]) # sequencia realizada
            seq2 = np.array(gabarito[f'{num}'][0]) # sequencia gabarito
            
            #---- Avaliando o match perfeito (IDEIA 2)
            tamanho = 0
            coincidencia = 0
            reincidencia = 0 
            #pegando a sequencia a ser comparada
            seq = seq1
            #pegando a sequência do gabarito correspondente à sequência que a pessoa fez
            certo = seq2
            
            if(len(certo) == len(seq)):
                tamanho += 1
            elif len(seq) > len(certo):
                certo = np.pad(certo,(0,len(seq)-len(certo)), mode ='constant', constant_values = 0)
            else:
                certo = np.pad(certo,(0,len(certo)-len(seq)), mode ='constant', constant_values = 0)
            
            # 1) Avaliar o match perfeito para poder normalizar depois
            soma_perfeita = 0
            soma_perfeita_uns = 0 
            resultado_perfeito, resultado_perfeito_uns  = avaliar_match_dinamico(certo, certo)
            
            for j in resultado_perfeito:
                soma_perfeita += np.sum(j)
            
            for j in resultado_perfeito_uns:
                soma_perfeita_uns += np.sum(j)

            # 2) Avaliar o match real
            soma_real = 0
            soma_real_uns = 0

            resultado_real, resultado_real_uns = avaliar_match_dinamico(seq, certo)
            
            for j in resultado_real:
                soma_real += np.sum(j)
            
            for j in resultado_real_uns:
                soma_real_uns += np.sum(j)

            # 3) Dividir o match perfeito pelo real para obter o score parcial
            score_parcial = soma_real/soma_perfeita
            score_parcial_uns = soma_real_uns/soma_perfeita_uns

            # 4) score total sendo a média do score parcial ponderado pela proporção explorada em x e em y
            score_total = ((score_parcial*teste['Proporção espacial x'][i]) + 
                            (score_parcial*teste['Proporção espacial y'][i]) 
                            )/2
            
            score_total_uns = ((score_parcial_uns*teste['Proporção espacial x'][i]) + 
                            (score_parcial_uns*teste['Proporção espacial y'][i]) 
                            )/2
            
            """print('--'*100)
            print(f'Trajetória {num}')
            print(f'Score parcial = {score_parcial} \nScore Total = {score_total}')
            print(f'Trajetória {num}')
            print(f'Score parcial uns = {score_parcial_uns} \nScore Total = {score_total_uns}')"""

            #---- Avaliando por comparação de imagem (IDEIA 3)
            resultado_ideia3 = comparar_imagem(seq1=seq1,seq2=seq2,plotar_imagens = False)

            #---- Avaliando por similaridade com correlação cruzada normalizada (IDEIA 4)
            resultado_ideia4 = calcular_similaridade(seq1,seq2)
            
            #resultados.append([score_parcial_uns,score_total_uns,resultado_ideia3[0],resultado_ideia3[1], resultado_ideia3[2], resultado_ideia3[3],resultado_ideia4])
            sparcial.append(score_parcial_uns)
            stotal.append(score_total_uns)
            acur.append(resultado_ideia3[0])
            prec.append(resultado_ideia3[1])
            rcll.append(resultado_ideia3[2])
            fpr.append(resultado_ideia3[3])
            sim.append(resultado_ideia4)

        """ # Plotando o comparativo das trajetórias
            plotar.plot_comparacao(gabarito=seq2, seq= seq1)

            # Printando os resultados
            print('--------- ' + f'Trajetória {num}' + ' ---------')
            print(f'-> Resultado da ideia 2 (comparação por matchs perfeitos):\nScore parcial: {score_parcial_uns} | Score total: {score_total_uns}')
            print(f'-> Resultado da ideia 3 (comparação de imagens):\nAcurácia: {resultado_ideia3[0]:.4f} | Precisão: {resultado_ideia3[1]:.4f}\nRecall: {resultado_ideia3[2]:.4f} | FPR: {resultado_ideia3[3]:.4f}')
            print(f'-> Resultado da ideia 4 (comparação por similaridade):\nSimilaridade entre as trajetórias: {resultado_ideia4}')
            print('--'*100)"""
        


# %% Vendo a distribuição dos resultados obtidos acima 
resultados = {
    'Score Parcial':sparcial,
    'Score Total':stotal,
    'Acurácia':acur,
    'Precisão':prec,
    'Recall':rcll,
    'Taxa de Falsos Positivos':fpr,
    'Similaridade':sim
}

resultados_df = pd.DataFrame(resultados)
#%%
import matplotlib.pyplot as plt 
import seaborn as sns

for parametro in resultados_df.columns:
    data = resultados_df[parametro]

    # Criando a figura com 2 subplots (1 linha, 2 colunas)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [3, 1]})
    # Histograma
    ax[0].hist(data, bins=20, color='blue', alpha=0.7, edgecolor='black',label ='Histograma',density=True)
    # Adicionar a linha de tendência (KDE)
    sns.kdeplot(data, color='red', linewidth=2, label="Curva KDE",ax=ax[0], bw_adjust=0.5)
    # Calcular e destacar a média no gráfico
    media = np.mean(data)
    ax[0].axvline(media, color='black', linestyle='dashed', linewidth=2, label=f"Média: {media:.2f}")
    ax[0].set_title(f"Histograma de {parametro}")
    ax[0].set_xlabel("Valores")
    ax[0].set_ylabel("Frequência")
    ax[0].legend()
    ax[0].grid(True)
    #Boxplot
    ax[1].boxplot(data,vert=False, patch_artist=True, boxprops=dict(facecolor='lightblue'),label='boxplot')
    ax[1].axvline(max(data), color='gray', linestyle='dotted', linewidth=2, label=f"Valor máximo: {max(data):.2f}",alpha = 0.7)
    ax[1].axvline(media, color='black', linestyle='dashed', linewidth=2, label=f"Média: {media:.2f}",alpha = 0.7)
    ax[1].set_title(f"Box Plot de {parametro}")
    ax[1].set_xlabel("Valores")
    ax[1].legend()
    
    fig.suptitle(f'Visualização de {parametro}')
    plt.tight_layout()
    plt.show()

# %%
