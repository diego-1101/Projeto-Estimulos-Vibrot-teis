'''
Functions developed by Diego de Sá Dias to evaluate and give scores into 
differents trajectories
'''

#---------------------------- Funções de avaliação de trajetória
#%% Ideia 2 - Comparação por matchs dinâmicos
def avaliar_match_dinamico(v1 = [],v2 =[]):
    """Compara de acordo com um match dinâmico os vetores 1 e 2

    Args:
        v1 (list, optional): _description_. Defaults to [].
        v2 (list, optional): _description_. Defaults to [].

    Returns:
        _type_: _description_
    """
    import numpy as np

    resultado = []
    resultado_uns = [] 
    # Loop sobre posições iniciais dinamicamente
    for start in range(len(v1)):  # Iterar por todas as posições iniciais
        temp_list = []  # Lista temporaria para essa iteração
        bin_temp_list = []
        # Loop para verificar correspondências crescentes a partir da posição inicial atual
        for i in range(start, len(v2)):  
            if np.array_equal(v1[start:i+1], v2[start:i+1]):  # Compara os subarrays
                temp_list.append(i - start + 1)  #  Append no comprimento do match
                bin_temp_list.append(1)
        resultado.append(temp_list)  # Armazena o resultado pelo seus índices de início 
        resultado_uns.append(bin_temp_list)
    
    return resultado, resultado_uns

#%% Ideia 3- Comparação de padrão de imagem 
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

#%% Ideia 4- Comparação por correlação máxima normalizada
def calcular_similaridade(seq1,seq2):
    """
    Calcula a métrica de similaridade entre seq1 e seq2 usando correlação cruzada normalizada.
    Retorna um valor entre 0 e 1 indicando a semelhança.
    """
    import numpy as np
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
#%%
def plotar_distribuicoes_resultados(resultados_df = None, titulo = None):
    """Plota um histograma e um boxplot dos resultados do dataframe inserido 

    Args:
        resultados_df (pandas Data Frame, required): os resultados das métricas de coomparação de trajetória. Defaults to None.
        titulo (String, optional): O protocolo que você queira que esteja no título. Defaults to None.
    """
    import matplotlib.pyplot as plt 
    import seaborn as sns
    import pandas as pd
    import numpy as np
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
        
        fig.suptitle(f'Visualização de {parametro} {titulo}')
        plt.tight_layout()
        plt.show()
