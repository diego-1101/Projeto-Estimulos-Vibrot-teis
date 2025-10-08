'''
Functions developed by Diego de Sá Dias 

!!!!! Write here when finished !!!! 
'''

#%%#---------------------------- Funções de avaliação de trajetória
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

#%% Função para calcular o desempenho
def calcular_desempenho(acur, fpr, sim, propx, propy):
    """Calcula o desempenho geral com base nas métricas de acurácia, especificidade (1-taxa de falsos positivos), similaridade e proporções.
    O desempenho é calculado como a média das métricas, o desempenho normalizado é calculado por essa média e normalizada pelas proporções fornecidas.

    Args:
        acur (float): acurácia da comparação de trajetórias.
        fpr (float): taxa de falsos positivos da comparação de trajetórias.
        sim (float): similaridade da comparação de trajetórias.
        propx (float): proporção em x para normalização do desempenho.
        propy (float): proporção em y para normalização do desempenho.

    Returns:
        desempenho (float): desempenho geral calculado como a média das métricas.
        desempenho_norm (float): desempenho normalizado calculado como a média das métricas e normalizado pelas proporções.
    """

    import numpy as np 
    
    desempenho = np.mean([acur, (1-fpr), sim])
    desempenho_norm = np.mean([propx, propy]) * desempenho 

    return desempenho, desempenho_norm

#%% Funções para teste de normalidade em cima dos dados agrupados
def teste_normalidade_shapiro(df, titulo = '', alpha=0.05):
    """Realiza o teste de normalidade de Shapiro-Wilk para as colunas de desempenho médio e ponderado.

    Args:
        df (Pandas DataFrame, required): DataFrame contendo as colunas 'Media_Desempenho' e 'Media_Desempenho_Ponderado'.
        titulo (str, optional): Título para o teste de normalidade, usado na impressão dos resultados. Defaults to ''.
        alpha (float, optional): Nível de significância para o teste de normalidade. Defaults to 0.05. 

    Returns:
        dict: Dicionário com os resultados do teste de normalidade para cada coluna.
            1. 'Media_Desempenho': Lista com o valor W, o p-valor e se é ou não normal de acordo com alfa (0-> Não Normal, 1-> Normal) do teste de normalidade.
            2. 'Media_Desempenho_Ponderado': Lista com o valor W, o p-valor e se é ou não normal de acordo com alfa (0-> Não Normal, 1-> Normal) do teste de normalidade.
    """


    import pandas as pd
    from scipy.stats import shapiro
    
    # Desempenho médio
    w1,p1 = shapiro(df['Media_Desempenho']) 
    # Desempenho médio ponderado 
    w2,p2 = shapiro(df['Media_Desempenho_Ponderado']) 

    print(f'--- Teste de normalidade por Shapiro-Wilk ({titulo}) ---')
    print(f"Desempenho médio: W={w1:.4f}, p-valor={p1:.4f} -> ",
           f"✅Normal (alpha={alpha})" if p1 > alpha else f"❌Não normal (alpha={alpha})")
    print(f"Desempenho médio ponderado: W={w2:.4f}, p-valor={p2:.4f} -> ", 
          f"✅Normal (alpha={alpha})" if p2 > alpha else f"❌Não normal (alpha={alpha})")
    
    if p1> alpha:
        return {'Media_Desempenho':[w1, p1,1], 'Media_Desempenho_Ponderado': [w2, p2,1]}
    else:
        return {'Media_Desempenho':[w1, p1,0], 'Media_Desempenho_Ponderado': [w2, p2,0]}
    

def teste_normalidade_kstest(df, titulo = '', alpha=0.05):
    """Realiza o teste de normalidade de Kolmogorov-Smirnov para as colunas de desempenho médio e ponderado.

    Args:
        df (Pandas DataFrame, required): DataFrame contendo as colunas 'Media_Desempenho' e 'Media_Desempenho_Ponderado'.
        titulo (str, optional): Título para o teste de normalidade, usado na impressão dos resultados. Defaults to ''.
        alpha (float, optional): Nível de significância para o teste de normalidade. Defaults to 0.05. 

    Returns:
        dict: Dicionário com os resultados do teste de normalidade para cada coluna.
            1. 'Media_Desempenho': Lista com o valor D, o p-valor e se é ou não normal de acordo com alfa (0-> Não Normal, 1-> Normal) do teste de normalidade.
            2. 'Media_Desempenho_Ponderado': Lista com o valor D, o p-valor e se é ou não normal de acordo com alfa (0-> Não Normal, 1-> Normal) do teste de normalidade.
    """

    import pandas as pd
    from scipy.stats import kstest
    
    media1 = df['Media_Desempenho'].mean()
    media2 = df['Media_Desempenho_Ponderado'].mean()
    desvio1  = df['Media_Desempenho'].std()
    desvio2  = df['Media_Desempenho_Ponderado'].std()
    # Desempenho médio
    d1,p1 = kstest(df['Media_Desempenho'], 'norm', args=(media1, desvio1)) 
    # Desempenho médio ponderado 
    d2,p2 = kstest(df['Media_Desempenho_Ponderado'], 'norm', args=(media2, desvio2)) 

    print(f'--- Teste de normalidade por Kolmogorov-Smirnov ({titulo}) ---')
    print(f"Desempenho médio: D={d1:.4f}, p-valor={p1:.4f} -> ",
           f"✅Normal (alpha={alpha})" if p1 > alpha else f"❌Não normal (alpha={alpha})")
    print(f"Desempenho médio ponderado: D={d2:.4f}, p-valor={p2:.4f} -> ", 
          f"✅Normal (alpha={alpha})" if p2 > alpha else f"❌Não normal (alpha={alpha})")
    
    if d1> alpha:
        return {'Media_Desempenho':[d1, p1,1], 'Media_Desempenho_Ponderado': [d2, p2,1]}
    else:
        return {'Media_Desempenho':[d1, p1,0], 'Media_Desempenho_Ponderado': [d2, p2,0]}

#%% Funções que testam normalidade com shapiro e kstest com o desempenho inteiro

def teste_normalidade_completo(desempenho, alpha=0.05):
    """Realiza os testes de normalidade de Shapiro-Wilk e Kolmogorov-Smirnov para a coluna de desempenho.

    Args:
        desempenho (Pandas Column DataFrame, required): Coluna de Desempenho a ser avaliada.
        titulo (str, optional): Título para o teste de normalidade, usado na impressão dos resultados. Defaults to ''.
        alpha (float, optional): Nível de significância para o teste de normalidade. Defaults to 0.05.
    Returns:
        dict: Dicionário com os resultados do teste de normalidade para a coluna de desempenho.
            1. 'Shapiro-Wilk': Lista com o valor W, o p-valor e se é ou não normal de acordo com alfa (0-> Não Normal, 1-> Normal) do teste de normalidade.
            2. 'Kolmogorov-Smirnov': Lista com o valor D, o p-valor e se é ou não normal de acordo com alfa (0-> Não Normal, 1-> Normal) do teste de normalidade.
    """

    import pandas as pd
    from scipy.stats import shapiro, kstest
    w,p = shapiro(desempenho)
    print(f'Estatística W: {w}, p-valor: {p}')
    print(f"✅Normal (alpha={alpha}, p-value= {p})" if p > alpha else f"❌Não normal (alpha={alpha}, p-value= {p})")
    d,p = kstest(desempenho, 'norm', args=(desempenho.mean(), desempenho.std()))
    print(f'Estatística D: {d}, p-valor: {p}')
    print(f"✅Normal (alpha={alpha}, p-value= {p})" if p > alpha else f"❌Não normal (alpha={alpha}, p-value= {p})")

    return {'Shapiro-Wilk': [w,p, 1 if p>alpha else 0], 'Kolmogorov-Smirnov': [d,p, 1 if p>alpha else 0]}

#%%#---------------------------- Funções auxiliares gerais ----------------------------

def map_complexidade(num_traj):
    """
    Mapeia o número da trajetória para um nível de complexidade.

    Args:
        num_traj (int): qual o número da trajetória realizada

    Returns:
        (int): valor da complexidade daquela trajetória
    """
    if num_traj in [1,2,3]:
        return 4
    if num_traj in [4,5,6]:
        return 6
    if num_traj in [7,8,9]:
        return 8

def map_niveis(num_complexidade):
    """
    Mapeia o número da complexidade para um nível de complexidade.

    Args:
        num_complexidade (int): qual o número da complexidade

    Returns:
        (str): valor do nível daquela complexidade
    """
    if num_complexidade in [4]:
        return 'Fácil'
    if num_complexidade in [6]:
        return 'Médio'
    if num_complexidade in [8]:
        return 'Difícil'    

def map_overlap(num_overlap):
    """
    Mapeia o número do overlap para um nível de overlap.

    Args:
        num_overlap (float): qual o número do overlap

    Returns:
        (str): valor do nível daquele overlap
    """
    if num_overlap in [0.0]:
        return 'Lento'
    if num_overlap in [0.25]:
        return 'Médio'
    if num_overlap in [0.5]:
        return 'Rápido'
    
def calcular_desempenhos_medios(df_concat,ids=[],complexidades=[], overlaps= None, trajetorias = []):

    import pandas as pd
    
    # Caso esteja no protocolo A, terá o overlap para avaliar, se não, teremos que avaliar sem overlap
    if overlaps != None:
        res_1 = []
        res_2 = []
        res_3 = []
        res_4 = []
        for id in ids:
            # Desempenho médio e médio ponderado por Overlap
            for overlap in overlaps:
                dados_filtrados=df_concat[(df_concat['ID'] == id) & 
                                                (df_concat['Overlap'] == overlap)
                                                ]
                media = dados_filtrados['Desempenho'].mean()
                media_nomalizada = dados_filtrados['Desempenho ponderado com proporção'].mean()
                res_1.append({'ID': id, 'Overlap': overlap, 'Media_Desempenho': media, 'Media_Desempenho_Ponderado': media_nomalizada})
            # Desempenho médio e médio ponderado por Complexidade
            for complexidade in complexidades:
                dados_filtrados=df_concat[(df_concat['ID'] == id) & 
                                                (df_concat['Complexidade'] == complexidade)
                                                ]
                media = dados_filtrados['Desempenho'].mean()
                media_nomalizada = dados_filtrados['Desempenho ponderado com proporção'].mean()
                res_2.append({'ID': id, 'Complexidade': complexidade, 'Media_Desempenho': media, 'Media_Desempenho_Ponderado': media_nomalizada})
            # Desempenho médio e médio ponderado por Complexidade e Overlap
            for overlap in overlaps:
                for complexidade in complexidades:
                    dados_filtrados=df_concat[(df_concat['ID'] == id) & 
                                                    (df_concat['Complexidade'] == complexidade) & 
                                                    (df_concat['Overlap'] == overlap)
                                                    ]
                    media = dados_filtrados['Desempenho'].mean()
                    media_nomalizada = dados_filtrados['Desempenho ponderado com proporção'].mean()
                    res_3.append({'ID': id, 'Complexidade': complexidade, 'Overlap': overlap, 'Media_Desempenho': media,'Media_Desempenho_Ponderado': media_nomalizada})
            # Desempenho médio e médio ponderado por Trajetória 
            for overlap in overlaps:
                for traj in trajetorias:
                    dados_filtrados=df_concat[(df_concat['ID'] == id) & 
                                                    (df_concat['Overlap'] == overlap) & 
                                                    (df_concat['Número da Trajetória'] == traj)
                                                    ]
                    media = dados_filtrados['Desempenho'].mean()
                    media_nomalizada = dados_filtrados['Desempenho ponderado com proporção'].mean()
                    res_4.append({'ID': id, 'Número da trajetoria': traj, 'Overlap': overlap, 'Media_Desempenho': media,'Media_Desempenho_Ponderado': media_nomalizada})

        desempenho = {
            'por_overlap': pd.DataFrame(res_1),
            'por_complexidade': pd.DataFrame(res_2),
            'por_complexidade_por_overlap': pd.DataFrame(res_3),
            'por_trajetoria_por_overlap': pd.DataFrame(res_4)
        }
    else:
        # Caso não tenha o overlap, vamos calcular as médias sem ele
        res_1 = []
        res_2 = []
        for id in ids:            
            # Desempenho médio e médio ponderado por Complexidade
            for complexidade in complexidades:
                dados_filtrados=df_concat[(df_concat['ID'] == id) & 
                                                (df_concat['Complexidade'] == complexidade)
                                                ]
                media = dados_filtrados['Desempenho'].mean()
                media_nomalizada = dados_filtrados['Desempenho ponderado com proporção'].mean()
                res_1.append({'ID': id, 'Complexidade': complexidade, 'Media_Desempenho': media, 'Media_Desempenho_Ponderado': media_nomalizada})
            # Desempenho médio e médio ponderado por Trajetória 
            for traj in trajetorias:
                dados_filtrados=df_concat[(df_concat['ID'] == id)  & 
                                                (df_concat['Número da Trajetória'] == traj)
                                                ]
                media = dados_filtrados['Desempenho'].mean()
                media_nomalizada = dados_filtrados['Desempenho ponderado com proporção'].mean()
                res_2.append({'ID': id, 'Número da trajetoria': traj, 'Media_Desempenho': media,'Media_Desempenho_Ponderado': media_nomalizada})

        desempenho = {
            'por_complexidade': pd.DataFrame(res_1),
            'por_trajetoria': pd.DataFrame(res_2)
        }

    return desempenho

def selecao_vetorial1(x1 = [], y1 = [], nomes_carac = [], k = 3, plotar = False, interativo = False):
    
    if k not in [2,3]:
        raise ValueError('Só são permitidos valores de k em [2,3]')

    import numpy as np 
    from itertools import combinations
    import pandas as pd
    
    classes = np.unique(y1)
    mediaTotal = np.mean(x1,axis=0) # média total do vetor de características

    #criando as matrizes de espalhamento
    Sw = np.zeros((x1.shape[1], x1.shape[1])) # espalhamento intra-classes -> Inicia uma matriz (15, 15) com zeros 
    Sb = np.zeros((x1.shape[1], x1.shape[1])) # espalhamento entre-classes

    #Calculando Sw e Sb
    for c in classes:
        xc = x1[y1==c] #pegando os dados da classe c
        media_classe = np.mean(xc,axis=0) #média da classe c

        Sw += (xc-media_classe).T  @ (xc-media_classe) #espalhamento intra-classes -> dimensões= (15,amostras_classe) @ (amostras_classe,15) = (15,15)
        n = xc.shape[0] #tamanho da classe c
        diferenca = (media_classe - mediaTotal).reshape(-1,1) #diferença entre a média da classe e a média total -> eu fiz o reshape para que ela fique com dimensão (15,1)
        Sb += n * (diferenca @ diferenca.T) #espalhamento entre-classes -> dimensões = (15,1) @ (1,15) = (15,15)

    print('Sw shape:', Sw.shape)
    print('Sb shape:', Sb.shape)
    # Utilizando o critério de Fisher para selecionar as melhores características

    # 3) Critério de seleção vetorial (Fisher ratio)
    def fisher_ratio(Sw, Sb, features):
        # Submatrizes para o subconjunto de features
        Sw_f = Sw[np.ix_(features, features)]
        Sb_f = Sb[np.ix_(features, features)]
        # Razão de Fisher = trace(Sb) / trace(Sw)
        return np.trace(Sb_f) / np.trace(Sw_f)
    
    p = x1.shape[1]   # número total de features
    scores = []

    for comb in combinations(range(p), k): # combinação das p características tomadas k a k 
        J = fisher_ratio(Sw, Sb, comb)
        scores.append((comb, J))

    # Ordena pelo maior Fisher ratio
    scores_sorted = sorted(scores, key=lambda t: t[1], reverse=True)

    # Top 5 combinações
    print("Top 5 combinações de características (índices) e seus Fisher ratios:")
    for comb, J in scores_sorted[:5]:
        print("Features:", comb, " -> J =", J)
    
    if plotar:
        if k == 3:
            #Indice melhores caracteristicas 
            for i in range(0,5):
                c1,c2,c3 = scores_sorted[i][0]
                dados_melhores = x1[:,[c1,c2,c3]] #pego todas as linhas das colunas das melhores características
                print(dados_melhores.shape)

                #Plotando as 3 melhores características para distinguir entre os estados de sono
                from mpl_toolkits.mplot3d import Axes3D
                import matplotlib.pyplot as plt

                fig = plt.figure(figsize=(18,15))
                ax = fig.add_subplot(111, projection='3d')
                for classe in np.unique(y1):
                    ax.scatter(dados_melhores[y1==classe,0], dados_melhores[y1==classe,1], dados_melhores[y1== classe,2], label =classe) #ploto todos do estágio 1
                
                ax.set_xlabel(f'{nomes_carac[c1]}' if nomes_carac else f'Feature {c1+1}')
                ax.set_ylabel(f'{nomes_carac[c2]}' if nomes_carac else f'Feature {c2+1}')
                ax.set_zlabel(f'{nomes_carac[c3]}' if nomes_carac else f'Feature {c3+1}')
                ax.set_title('Melhores características para distinguir as classes')
                ax.legend()
        elif k == 2:
            for i in range(0,5):
                c1,c2 = scores_sorted[i][0]
                dados_melhores = x1[:,[c1,c2]] #pego todas as linhas das colunas das melhores características
                print(dados_melhores.shape)

                #Plotando as 3 melhores características para distinguir entre os estados de sono

                import matplotlib.pyplot as plt

                fig = plt.figure(figsize=(18,15))
                for classe in np.unique(y1):
                    plt.scatter(dados_melhores[y1==classe,0], dados_melhores[y1==classe,1], label =classe) #ploto todos do estágio 1
                
                plt._xlabel(f'{nomes_carac[c1]}' if nomes_carac else f'Feature {c1+1}')
                plt.ylabel(f'{nomes_carac[c2]}' if nomes_carac else f'Feature {c2+1}')
                plt.title('Melhores características para distinguir as classes')
                plt.legend()
            
    return Sw, Sb, scores_sorted

def selecao_vetorial(
    x1 = [],
    y1 = [],
    nomes_carac = [],
    k = 3,
    plotar = False,
    interativo = False,
    salvar_interativo = False
):
    """
    Seleção vetorial de características pelo critério de Fisher (traço) e plot das
    melhores combinações (k=2 ou k=3).

    A função calcula as matrizes de espalhamento intra-classes (Sw) e entre-classes (Sb),
    avalia todas as combinações de features p tomadas k a k usando a razão de Fisher:
        J(features) = trace(Sb_features) / trace(Sw_features)
    e retorna a lista ordenada (decrescente) dessas combinações. Opcionalmente plota
    as top-5 combinações em 2D (k=2) ou 3D (k=3), com Matplotlib (padrão) ou Plotly
    (interativo=True). Se `salvar_interativo=True`, o gráfico interativo é salvo como
    arquivo HTML com o nome igual ao título do gráfico.

    Parâmetros
    ----------
    x1 : array-like (n_amostras, n_features)
        Matriz de dados (características). Deve ser conversível para NumPy 2D.
    y1 : array-like (n_amostras,)
        Vetor de rótulos/classe para cada amostra (qualquer tipo hashable).
    nomes_carac : list[str], opcional
        Lista com os nomes das features (len == n_features). Se ausente, usa "Feature i".
    k : int, {2, 3}
        Dimensionalidade do subespaço para avaliação/plot (pares ou trincas de features).
    plotar : bool
        Se True, plota as top-5 combinações de acordo com k.
    interativo : bool
        Se True, usa Plotly para gráficos interativos; caso contrário usa Matplotlib.
    salvar_interativo : bool
        Se True, salva os gráficos interativos em arquivos .html (o nome é o título do gráfico).

    Retorna
    -------
    Sw : np.ndarray (p, p)
        Matriz de espalhamento intra-classes.
    Sb : np.ndarray (p, p)
        Matriz de espalhamento entre-classes.
    scores_sorted : list[tuple[tuple[int,...], float]]
        Lista ordenada das combinações e seus escores de Fisher.

    Notas
    -----
    - O plot (se habilitado) mostra até as 5 melhores combinações.
    - Para gráficos interativos é necessário ter `plotly` instalado no mesmo ambiente.
    - Se `salvar_interativo=True`, os arquivos são salvos com o título do gráfico como nome.
    """
    import numpy as np
    from itertools import combinations

    if k not in [2, 3]:
        raise ValueError('Só são permitidos valores de k em [2,3]')

    x1 = np.asarray(x1)
    y1 = np.asarray(y1)
    classes = np.unique(y1)
    mediaTotal = np.mean(x1, axis=0)

    Sw = np.zeros((x1.shape[1], x1.shape[1]))
    Sb = np.zeros((x1.shape[1], x1.shape[1]))

    for c in classes:
        xc = x1[y1 == c]
        media_classe = np.mean(xc, axis=0)
        Sw += (xc - media_classe).T @ (xc - media_classe)
        n = xc.shape[0]
        diferenca = (media_classe - mediaTotal).reshape(-1, 1)
        Sb += n * (diferenca @ diferenca.T)

    def fisher_ratio(Sw, Sb, features):
        Sw_f = Sw[np.ix_(features, features)]
        Sb_f = Sb[np.ix_(features, features)]
        return np.trace(Sb_f) / np.trace(Sw_f)

    p = x1.shape[1]
    scores = [(comb, fisher_ratio(Sw, Sb, comb)) for comb in combinations(range(p), k)]
    scores_sorted = sorted(scores, key=lambda t: t[1], reverse=True)

    print("Top 5 combinações de características (índices) e seus Fisher ratios:")
    for comb, J in scores_sorted[:5]:
        print("Features:", comb, " -> J =", J)

    if plotar:
        import matplotlib.pyplot as plt
        if interativo:
            import plotly.express as px
            import pandas as pd

        def nome(i):
            return (nomes_carac[i] if nomes_carac else f"Feature {i+1}")

        topo = min(5, len(scores_sorted))
        for i in range(topo):
            comb = scores_sorted[i][0]
            dados_melhores = x1[:, list(comb)]
            title = f"Top {i+1} — ({', '.join([nome(c) for c in comb])})"

            if interativo:
                df = pd.DataFrame({
                    nome(comb[0]): dados_melhores[:, 0],
                    nome(comb[1]): dados_melhores[:, 1],
                    "classe": y1.astype(str)
                })
                if k == 3:
                    df[nome(comb[2])] = dados_melhores[:, 2]
                    fig = px.scatter_3d(df, x=nome(comb[0]), y=nome(comb[1]), z=nome(comb[2]),
                                        color="classe", title=title)
                else:
                    fig = px.scatter(df, x=nome(comb[0]), y=nome(comb[1]),
                                     color="classe", title=title)

                fig.update_traces(marker=dict(size=6))
                fig.update_layout(scene_aspectmode="data")
                fig.show()

                if salvar_interativo:
                    file_name = f"{title.replace('—', '-').replace(' ', '_').replace('(', '').replace(')', '').replace(',', '_')}.html"
                    fig.write_html(file_name)
                    print(f"Gráfico interativo salvo em: {file_name}")

            else:
                if k == 3:
                    from mpl_toolkits.mplot3d import Axes3D
                    fig = plt.figure(figsize=(10, 8))
                    ax = fig.add_subplot(111, projection='3d')
                    for classe in np.unique(y1):
                        sel = (y1 == classe)
                        ax.scatter(dados_melhores[sel, 0], dados_melhores[sel, 1],
                                   dados_melhores[sel, 2], label=str(classe))
                    ax.set_xlabel(nome(comb[0])); ax.set_ylabel(nome(comb[1])); ax.set_zlabel(nome(comb[2]))
                    ax.set_title(title)
                    ax.legend()
                    plt.show()
                else:
                    fig = plt.figure(figsize=(10, 8))
                    for classe in np.unique(y1):
                        sel = (y1 == classe)
                        plt.scatter(dados_melhores[sel, 0], dados_melhores[sel, 1],
                                    label=str(classe))
                    plt.xlabel(nome(comb[0]))
                    plt.ylabel(nome(comb[1]))
                    plt.title(title)
                    plt.legend()
                    plt.show()

    return Sw, Sb, scores_sorted

def manova1_py(
    X,
    groups,
    k_plot=3,
    plotar=True,
    interativo=False,
    salvar_interativo=False,
    title_prefix="MANOVA1 / CDA"
):
    """
    Executa MANOVA de 1 fator (one-way) com variáveis canônicas (CDA) no estilo do MATLAB manova1.

    Parâmetros
    ----------
    X : array-like (n_amostras, n_variaveis)
        Matriz de dados (variáveis dependentes).
    groups : array-like (n_amostras,)
        Vetor com o fator (grupos; 1 fator). Pode ser string, int etc.
    k_plot : int, {2,3}, opcional (default=3)
        Dimensionalidade para o plot (2D ou 3D) usando as primeiras variáveis canônicas.
    plotar : bool, opcional (default=True)
        Se True, gera o gráfico das variáveis canônicas.
    interativo : bool, opcional (default=False)
        Se True, usa Plotly para gráfico interativo; senão, Matplotlib.
    salvar_interativo : bool, opcional (default=False)
        Se True e interativo=True, salva o gráfico .html com o título como nome do arquivo.
    title_prefix : str, opcional
        Prefixo do título do gráfico (também base do nome do arquivo .html).

    Retorna
    -------
    D : np.ndarray (g, g)
        Matriz de distâncias de Mahalanobis entre médias de grupos (simétrica, diagonal zero).
    P : np.ndarray (m,)
        p-values sequenciais (teste de Wilks para raízes 1..m com aproximação qui-quadrado).
        m = min(p, g-1).
    stats : dict
        Dicionário com campos típicos do manova1:
        - 'W', 'B', 'T' : matrizes de espalhamento within/between/total (p x p)
        - 'eigvals'     : autovalores de inv(W)@B (roots canônicas), decrescentes (m,)
        - 'eigvecs'     : autovetores/coeficientes canônicos (p x m)
        - 'scores'      : escores canônicos Z = (X - mean) @ eigvecs (n x m)
        - 'overall_mean': média global (p,)
        - 'group_means' : DataFrame com médias por grupo (g x p)
        - 'group_sizes' : Series com n_i por grupo
        - 'labels'      : array de labels (ordem usada internamente)
        - 'wilks_lambda_seq' : lambdas sequenciais para raízes 1..m
        - 'chi2'        : estatística qui-quadrado de Bartlett por raiz (1..m)
        - 'df'          : graus de liberdade do teste de Bartlett por raiz (1..m)

    Observações
    -----------
    - A MANOVA aqui é one-way (um único fator em 'groups').
    - Teste global baseado em Wilks' lambda com aproximação de Bartlett.
    - D (Mahalanobis) usa a covariância pooled (W/(N-g)) entre médias de grupos.
    - Para gráficos interativos é necessário ter plotly instalado no MESMO ambiente.
    """
    import numpy as np
    import pandas as pd
    from numpy.linalg import inv, eig
    from scipy.stats import chi2

    # --- checagens e preparo ---
    X = np.asarray(X, dtype=float)
    gvec = np.asarray(groups)
    if X.ndim != 2:
        raise ValueError("X deve ser 2D (n_amostras, n_variaveis).")
    n, p = X.shape
    labels = pd.unique(gvec)
    g = len(labels)
    if g < 2:
        raise ValueError("É necessário pelo menos 2 grupos.")

    # index por grupo
    idx_by = {lab: np.where(gvec == lab)[0] for lab in labels}
    n_i = {lab: len(idx_by[lab]) for lab in labels}
    if any(v == 0 for v in n_i.values()):
        raise ValueError("Algum grupo está vazio.")

    # médias
    overall_mean = X.mean(axis=0)
    means = {lab: X[idx_by[lab]].mean(axis=0) for lab in labels}

    # --- matrizes de espalhamento ---
    W = np.zeros((p, p))
    for lab in labels:
        Xi = X[idx_by[lab]]
        dif = Xi - means[lab]
        W += dif.T @ dif
    B = np.zeros((p, p))
    for lab in labels:
        d = (means[lab] - overall_mean).reshape(-1, 1)
        B += n_i[lab] * (d @ d.T)
    T = W + B

    # --- autovalores/vetores de inv(W)B ---
    # (se W for singular, uma regularização leve pode ser necessária no mundo real)
    evals, evecs = eig(inv(W) @ B)
    order = np.argsort(-evals.real)
    evals = evals[order].real
    evecs = evecs[:, order].real
    m = min(p, g - 1)
    eigvals = evals[:m]
    eigvecs = evecs[:, :m]

    # --- escores canônicos ---
    Z = (X - overall_mean) @ eigvecs  # n x m

    # --- Wilks' lambda sequencial + Bartlett chi2 approx e p-values ---
    # Lambda_total = Π 1/(1+λ_j). Para teste sequencial da raiz r..m:
    # lambda_r = Π_{j=r..m} 1/(1+λ_j)
    # Bartlett: X2 = -[(N - 1) - (p + g)/2] * ln(lambda_r), df = (p - r + 1)*(g - r)
    lambdas_seq = []
    chi2_seq = []
    df_seq = []
    pvalues = []
    # N_effective ~ N_total - (g), mas a fórmula clássica abaixo usa (N - 1) - (p + g)/2
    N_eff = n
    for r in range(1, m + 1):
        lam_r = np.prod(1.0 / (1.0 + eigvals[r - 1:]))
        lambdas_seq.append(lam_r)
        t = (N_eff - 1) - (p + g) / 2.0
        t = max(t, 1e-9)  # proteção
        chi2_stat = -t * np.log(lam_r)
        df_r = (p - r + 1) * (g - r)
        df_r = int(max(df_r, 1))
        chi2_seq.append(chi2_stat)
        df_seq.append(df_r)
        pval = 1.0 - chi2.cdf(chi2_stat, df_r)
        pvalues.append(pval)
    P = np.asarray(pvalues)

    # --- distâncias de Mahalanobis entre médias dos grupos (pooled covariance) ---
    # Sp = W / (N - g)
    Sp = W / max(n - g, 1)
    Sp_inv = inv(Sp)
    D = np.zeros((g, g))
    means_mat = np.vstack([means[lab] for lab in labels])
    for i in range(g):
        for j in range(i + 1, g):
            diff = means_mat[i] - means_mat[j]
            dij2 = float(diff.T @ Sp_inv @ diff)
            D[i, j] = D[j, i] = np.sqrt(max(dij2, 0.0))

    # --- montar stats ---
    stats = {
        "W": W,
        "B": B,
        "T": T,
        "eigvals": eigvals,
        "eigvecs": eigvecs,       # coeficientes canônicos (colunas = variáveis canônicas)
        "scores": Z,              # escores canônicos (linhas = amostras)
        "overall_mean": overall_mean,
        "group_means": pd.DataFrame(means_mat, index=labels, columns=[f"Var{i+1}" for i in range(p)]),
        "group_sizes": pd.Series(n_i)[labels],
        "labels": labels,
        "wilks_lambda_seq": np.array(lambdas_seq),
        "chi2": np.array(chi2_seq),
        "df": np.array(df_seq),
    }

    # --- plot opcional ---
    if plotar:
        k_plot = 3 if k_plot not in (2, 3) else k_plot
        titulo = f"{title_prefix} — Canônicas ({'3D' if k_plot==3 else '2D'})"
        if interativo:
            try:
                import plotly.express as px
            except Exception as e:
                raise ImportError("Plot interativo requer 'plotly'. Instale com: pip install plotly") from e

            df_plot = pd.DataFrame({"grupo": gvec.astype(str)})
            df_plot["Can1"] = Z[:, 0]
            if m >= 2:
                df_plot["Can2"] = Z[:, 1]
            else:
                df_plot["Can2"] = 0.0
            if k_plot == 3:
                if m >= 3:
                    df_plot["Can3"] = Z[:, 2]
                else:
                    df_plot["Can3"] = 0.0

            if k_plot == 3:
                fig = px.scatter_3d(df_plot, x="Can1", y="Can2", z="Can3", color="grupo", title=titulo)
            else:
                fig = px.scatter(df_plot, x="Can1", y="Can2", color="grupo", title=titulo)

            fig.update_traces(marker=dict(size=6))
            fig.update_layout(legend_title_text="Grupo")
            fig.show()

            if salvar_interativo:
                safe = (titulo.replace("—", "-")
                             .replace(" ", "_")
                             .replace("/", "_")
                             .replace("(", "")
                             .replace(")", ""))
                fname = f"{safe}.html"
                fig.write_html(fname)
                print(f"💾 Gráfico interativo salvo em: {fname}")

        else:
            import matplotlib.pyplot as plt
            if k_plot == 3:
                from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
                # coordenadas
                x = Z[:, 0]
                y = Z[:, 1] if m >= 2 else np.zeros_like(x)
                z = Z[:, 2] if m >= 3 else np.zeros_like(x)
                for lab in labels:
                    sel = (gvec == lab)
                    ax.scatter(x[sel], y[sel], z[sel], label=str(lab), s=16)
                ax.set_xlabel("Can1"); ax.set_ylabel("Can2"); ax.set_zlabel("Can3")
                ax.set_title(titulo)
                ax.legend()
                plt.show()
            else:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, 8))
                x = Z[:, 0]
                y = Z[:, 1] if m >= 2 else np.zeros_like(x)
                for lab in labels:
                    sel = (gvec == lab)
                    plt.scatter(x[sel], y[sel], label=str(lab), s=16)
                plt.xlabel("Can1"); plt.ylabel("Can2")
                plt.title(titulo); plt.legend()
                plt.show()

    return D, P, stats




#%%#---------------------------- Funções de Plot dos resultados ----------------------------

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

def plotar_desempenhos(data,titulo,parametro):
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    fig, ax = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [3, 1]})
    # Histograma
    ax[0].hist(data, bins=20, color='blue', alpha=0.7, edgecolor='black', label='Histograma', density=True)
    sns.kdeplot(data, color='red', linewidth=2, label="Curva KDE", ax=ax[0], bw_adjust=0.5)
    media = np.mean(data)
    ax[0].axvline(media, color='black', linestyle='dashed', linewidth=2, label=f"Média: {media:.2f}")
    ax[0].set_title(f"Histograma de {parametro}")
    ax[0].set_xlabel("Valores")
    ax[0].set_ylabel("Frequência")
    ax[0].legend()
    ax[0].grid(True)

    # Boxplot
    ax[1].boxplot(data, vert=False, patch_artist=True, boxprops=dict(facecolor='lightblue'), labels=[''])
    ax[1].scatter(data, np.ones_like(data), color='red', alpha=0.6, s=20, label='Pontos')  
    ax[1].axvline(max(data), color='gray', linestyle='dotted', linewidth=2, label=f"Valor máximo: {max(data):.2f}", alpha=0.7)
    ax[1].axvline(media, color='black', linestyle='dashed', linewidth=2, label=f"Média: {media:.2f}", alpha=0.7)
    ax[1].set_title(f"Box Plot de {parametro}")
    ax[1].set_xlabel("Valores")
    ax[1].legend()

    fig.suptitle(f'Visualização de {parametro} {titulo}')
    plt.tight_layout()
    plt.show()

def dot_ic_sig1(
    df, x, y='Desempenho',
    order=None,
    alpha=0.05,
    show_p_text=False,
    star_thresh=((0.001,'***'), (0.01,'**'), (0.05,'*')),
    figsize=(20,10),
    jitter=True, dot_alpha=0.4, dot_color='gray',
    annotate_means=True, text_offset=0.01,
    y_pad=0.02, step=0.03, cap_width=0.08, line_w=1.5,
    ylim=(0,1.1), title=None, grid=True, savepath=None,
    seed=None):
    """
    Dotplot + média ± IC95% + chaves automáticas de significância (Tukey HSD).

    Parâmetros principais:
      df: DataFrame com ao menos as colunas [x, y]
      x:  coluna categórica que define os grupos no eixo x
      y:  coluna numérica do desfecho (default 'Desempenho')
      order: ordem dos níveis em x (se None, usa ordem categórica ou sorted)
      alpha: nível de significância para Tukey
      show_p_text: True -> escreve p; False -> usa estrelas
      star_thresh: tuplas (limiar, '***', '**', '*') para mapear p em estrelas
      ylim: tupla para limites de y (None mantém automático)
      title: título opcional
      savepath: caminho para salvar a figura (png/svg/pdf), se desejado

    Retorna:
      fig, ax, tukey_df, sig_pairs, group_stats
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import t
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    import pandas.api.types as ptypes

    if seed is not None:
        np.random.seed(seed)

    data = df[[x, y]].dropna().copy()

    # Determinar ordem dos grupos
    if order is None:
        if ptypes.is_categorical_dtype(data[x]):
            order = list(data[x].cat.categories)
        else:
            order = sorted(data[x].unique().tolist())

    # Funções auxiliares internas --------------------------
    def _p_to_text(p):
        if show_p_text:
            return f"p={p:.3g}"
        for thr, star in star_thresh:
            if p < thr:
                return star
        return 'ns'

    def _draw_sig_bracket(ax, x1, x2, y0, text):
        ax.plot([x1, x1, x2, x2], [y0, y0+step*0.25, y0+step*0.25, y0], color='k', lw=line_w)
        ax.plot([x1, x1-cap_width], [y0, y0], color='k', lw=line_w)
        ax.plot([x2, x2+cap_width], [y0, y0], color='k', lw=line_w)
        ax.text((x1+x2)/2, y0 + step*0.28, text, ha='center', va='bottom', fontsize=12)

    # ------------------------------------------------------

    fig, ax = plt.subplots(figsize=figsize)

    # Dotplot (dispersão por grupo)
    sns.stripplot(
        data=data, x=x, y=y, order=order,
        jitter=jitter, color=dot_color, alpha=dot_alpha, ax=ax
    )

    # Estatísticas por grupo: média ± IC95%
    g = (data.groupby(x)[y]
         .agg(mean='mean', std='std', count='count')
         .reindex(order)
         .reset_index())
    # CI 95% com t de Student
    def _ci95(std, n):
        if n and n > 1 and pd.notnull(std):
            sem = std / np.sqrt(n)
            return t.ppf(0.975, df=n-1) * sem
        return np.nan
    g['ci95'] = [_ci95(s, n) for s, n in zip(g['std'], g['count'])]

    # Plotar média ± IC e rótulos
    for i, row in g.iterrows():
        m, ci = row['mean'], row['ci95']
        ax.errorbar(i, m, yerr=ci, fmt='o', color='blue', capsize=5, markersize=8,
                    label='Média ± IC95%' if i == 0 else "")
        if annotate_means:
            off = text_offset if pd.notnull(ci) else text_offset*2
            txt_ci = f"±{ci:.2f}" if pd.notnull(ci) else "n/a"
            ax.text(i, (m + (ci if pd.notnull(ci) else 0)) + off,
                    f"Média: {m:.2f}\nIC95: {txt_ci}",
                    ha='center', va='bottom', fontsize=9, color='black')

    # Topo de cada coluna para posicionar chaves
    tops = (g['mean'] + g['ci95'].fillna(0)).values
    x_pos = {lvl: i for i, lvl in enumerate(order)}

    # Tukey HSD automático
    tukey = pairwise_tukeyhsd(endog=data[y].values, groups=data[x].values, alpha=alpha)
    res = tukey.summary()
    tukey_df = pd.DataFrame(res.data[1:], columns=res.data[0])
    tukey_df['p_adj']  = pd.to_numeric(tukey_df['p-adj'], errors='coerce')
    tukey_df['reject'] = tukey_df['reject'].astype(str).str.lower().map({'true': True, 'false': False})

    sig_pairs = tukey_df[tukey_df['reject']].copy()
    if not sig_pairs.empty:
        sig_pairs['x1'] = sig_pairs['group1'].map(x_pos)
        sig_pairs['x2'] = sig_pairs['group2'].map(x_pos)
        sig_pairs[['xa','xb']] = np.sort(sig_pairs[['x1','x2']].values, axis=1)
        sig_pairs = sig_pairs.sort_values(by=['xb','xa'])

        # Empilhar chaves sem sobreposição
        y_base = tops.max() + y_pad
        levels = []

        def _get_free_level(a, b):
            for lvl, intervals in enumerate(levels):
                # conflito se (a,b) SOBREPOE qualquer (ia,ib)
                if any(not (b <= ia or a >= ib) for ia, ib in intervals):
                    continue
                intervals.append((a, b))
                return lvl
            levels.append([(a, b)])
            return len(levels)-1

        for _, r in sig_pairs.iterrows():
            xa, xb = int(r['xa']), int(r['xb'])
            local_top = max(tops[xa], tops[xb]) + y_pad
            lvl = _get_free_level(xa, xb)
            y0 = max(y_base + lvl*step, local_top + lvl*step*0.6)
            _draw_sig_bracket(ax, xa, xb, y0, _p_to_text(r['p_adj']))

    # Estética final
    ttl = title if title else f"Dotplot + IC95% + Tukey (α={alpha})"
    ax.set_title(ttl)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if grid:
        ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches='tight')

    # Retornos úteis para relatório
    group_stats = g.rename(columns={x: 'group'})
    return fig, ax, tukey_df, sig_pairs, group_stats

def dot_ic_sig(
    df, x, y='Desempenho',
    order=None,
    alpha=0.05,
    show_sig_bars=False,       # << novo: só desenha as barras se True
    show_p_text=False,         # se False, usa estrelas
    star_thresh=((0.001,'***'), (0.01,'**'), (0.05,'*')),
    test='auto',               # 'auto' -> t-test se 2 grupos, Tukey se >=3
    equal_var=False,           # Welch (False) por padrão no t-test
    figsize=(12,6),
    jitter=True, dot_alpha=0.5, dot_color='gray',
    annotate_means=True, text_offset=0.01,
    y_pad=0.02, step=0.04, cap_width=0.08, line_w=1.6,
    ylim=None, title=None, grid=True, savepath=None,
    seed=None
):
    """
    Dot plot por grupo com média ± IC95% e (opcionalmente) barras de significância no topo.

    O que faz:
      - Plota a dispersão dos dados (um ponto por amostra) para cada nível de `x`;
      - Desenha, para cada grupo, a média e o IC95% (Student t);
      - Se `show_sig_bars=True`, testa diferenças entre grupos e desenha chaves no topo:
          * Se houver exatamente 2 grupos: t-teste (Welch por padrão: equal_var=False).
            Também calcula e retorna o tamanho de efeito (Cohen's d).
          * Se houver ≥3 grupos: teste post-hoc de Tukey HSD.

    Parâmetros
    ----------
    df : pandas.DataFrame
        DataFrame contendo ao menos as colunas `x` (categórica) e `y` (numérica).
    x : str
        Nome da coluna categórica (grupos).
    y : str, default 'Desempenho'
        Nome da coluna numérica do desfecho.
    order : list[str] | None
        Ordem dos níveis de `x`. Se None, usa a ordem categórica ou sorted(unique).
    alpha : float, default 0.05
        Nível de significância.
    show_sig_bars : bool, default False
        Se True, desenha as barras de significância no topo.
    show_p_text : bool, default False
        Se True, escreve "p=..." nas barras; senão usa estrelas conforme `star_thresh`.
    star_thresh : tuple
        Mapeamento (limiar, símbolo) para converter p-values em estrelas.
    test : {'auto','ttest','tukey'}
        Estratégia do teste. 'auto' escolhe t-teste (2 grupos) ou Tukey (≥3).
    equal_var : bool
        Suposição de variâncias iguais no t-teste. Por padrão False (Welch).
    figsize : tuple, default (12,6)
        Tamanho da figura matplotlib.
    jitter : bool, default True
        Liga/desliga jitter nos pontos (seaborn.stripplot).
    dot_alpha : float, default 0.5
        Transparência dos pontos.
    dot_color : str, default 'gray'
        Cor dos pontos.
    annotate_means : bool, default True
        Escreve média e IC acima do marcador da média.
    text_offset : float
        Deslocamento vertical do texto da média/IC, em fração da altura do eixo y.
    y_pad, step, cap_width, line_w : floats
        Parâmetros geométricos das barras de significância (altura base, passo, largura da “aba”, espessura).
    ylim : tuple | None
        Limites do eixo y. None mantém automático.
    title : str | None
        Título.
    grid : bool
        Grade no fundo do gráfico.
    savepath : str | None
        Caminho para salvar a figura (png/svg/pdf).
    seed : int | None
        Semente para reprodutibilidade do jitter.

    Retorna
    -------
    fig, ax : matplotlib Figure, Axes
    stats_table : pandas.DataFrame
        Tabela com média, desvio e IC95% por grupo.
    sig_results : pandas.DataFrame | None
        * Se 2 grupos e show_sig_bars=True: DataFrame com t, df, p, Cohen's d.
        * Se ≥3 grupos e show_sig_bars=True: DataFrame do Tukey HSD.
        * Caso contrário, None.

    Observações
    -----------
    - Para Tukey HSD requer statsmodels instalado.
    - Para 2 grupos usa scipy.stats.ttest_ind.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas.api.types as ptypes
    from scipy.stats import t as t_dist
    from scipy.stats import ttest_ind

    if seed is not None:
        np.random.seed(seed)

    # --- preparar dados ---
    data = df[[x, y]].dropna().copy()

    # ordem dos grupos
    if order is None:
        if ptypes.is_categorical_dtype(data[x]):
            order = list(data[x].cat.categories)
        else:
            order = sorted(data[x].unique().tolist())

    # --- figura base ---
    fig, ax = plt.subplots(figsize=figsize)

    # Dispersão (dotplot)
    sns.stripplot(
        data=data, x=x, y=y, order=order,
        jitter=jitter, color=dot_color, alpha=dot_alpha, ax=ax
    )

    # --- estatísticas por grupo ---
    g = (data.groupby(x)[y]
         .agg(mean='mean', std='std', count='count')
         .reindex(order)
         .reset_index())

    # IC95% (Student t)
    def _ci95(std, n):
        if n and n > 1 and pd.notnull(std):
            sem = std / np.sqrt(n)
            return t_dist.ppf(0.975, df=n-1) * sem
        return np.nan
    g['ci95'] = [_ci95(s, n) for s, n in zip(g['std'], g['count'])]

    # Média ± IC
    # para posicionar textos com offset relativo ao range do eixo
    y_vals = data[y].values
    y_range = (np.nanmax(y_vals) - np.nanmin(y_vals)) if len(y_vals) else 1.0
    for i, row in g.iterrows():
        m, ci = row['mean'], row['ci95']
        ax.errorbar(i, m, yerr=ci, fmt='o', color='blue', capsize=5, markersize=8,
                    label='Média ± IC95%' if i == 0 else "")
        if annotate_means:
            off = text_offset * y_range
            txt_ci = f"±{ci:.3g}" if pd.notnull(ci) else "n/a"
            ax.text(i, (m + (ci if pd.notnull(ci) else 0)) + off,
                    f"Média: {m:.3g}\nIC95: {txt_ci}",
                    ha='center', va='bottom', fontsize=9, color='black')

    # Estética base
    ttl = title if title else f"Dotplot + IC95%"
    ax.set_title(tl := (ttl if not show_sig_bars else f"{ttl} (α={alpha})"))
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if grid:
        ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    # -----------------------------------------
    #  SIGNIFICÂNCIA (barras no topo) opcional
    # -----------------------------------------
    sig_results = None
    if show_sig_bars:
        # funções auxiliares
        def _p_to_text(p):
            if show_p_text:
                return f"p={p:.3g}"
            for thr, star in star_thresh:
                if p < thr:
                    return star
            return 'ns'

        def _draw_sig_bracket(ax, x1, x2, y0, text):
            ax.plot([x1, x1, x2, x2], [y0, y0+step*0.25, y0+step*0.25, y0], color='k', lw=line_w)
            ax.plot([x1, x1-cap_width], [y0, y0], color='k', lw=line_w)
            ax.plot([x2, x2+cap_width], [y0, y0], color='k', lw=line_w)
            ax.text((x1+x2)/2, y0 + step*0.28, text, ha='center', va='bottom', fontsize=12)

        # topo para empilhar barras
        tops = (g['mean'] + g['ci95'].fillna(0)).values
        x_pos = {lvl: i for i, lvl in enumerate(order)}

        levels = []  # para empilhar sem colisão
        def _get_free_level(a, b):
            for lvl, intervals in enumerate(levels):
                # conflito se (a,b) sobrepõe qualquer (ia,ib)
                if any(not (b <= ia or a >= ib) for ia, ib in intervals):
                    continue
                intervals.append((a, b))
                return lvl
            levels.append([(a, b)])
            return len(levels)-1

        unique_groups = order
        k_groups = len(unique_groups)

        # Caso 1: exatamente 2 grupos -> t-test
        if (test == 'ttest') or (test == 'auto' and k_groups == 2):
            g1, g2 = unique_groups[0], unique_groups[1]
            d1 = data.loc[data[x] == g1, y].values
            d2 = data.loc[data[x] == g2, y].values
            tt = ttest_ind(d1, d2, equal_var=equal_var, nan_policy='omit')
            # Cohen's d (pooled) – versão robusta
            n1, n2 = len(d1), len(d2)
            s1, s2 = np.nanstd(d1, ddof=1), np.nanstd(d2, ddof=1)
            sp = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / max(n1+n2-2, 1))
            d_cohen = (np.nanmean(d1) - np.nanmean(d2)) / sp if sp > 0 else np.nan
            df_t = n1 + n2 - 2 if equal_var else np.nan  # Welch tem df efetivo; aqui omitimos

            sig_results = pd.DataFrame({
                'group1':[g1], 'group2':[g2],
                'statistic':[tt.statistic], 'pvalue':[tt.pvalue],
                'df':[df_t], 'cohens_d':[d_cohen],
                'test':['t-test (Welch)' if not equal_var else 't-test (equal var)']
            })

            # desenhar 1 barra
            xa, xb = x_pos[g1], x_pos[g2]
            y_base = tops.max() + y_pad*(np.nanmax(y_vals)-np.nanmin(y_vals) if ylim is None else (ylim[1]-ylim[0]))
            local_top = max(tops[xa], tops[xb]) + y_pad
            lvl = _get_free_level(min(xa, xb), max(xa, xb))
            y0 = max(y_base + lvl*step, local_top + lvl*step*0.6)
            _draw_sig_bracket(ax, xa, xb, y0, _p_to_text(tt.pvalue))

        # Caso 2: ≥ 3 grupos -> Tukey HSD
        elif (test == 'tukey') or (test == 'auto' and k_groups >= 3):
            from statsmodels.stats.multicomp import pairwise_tukeyhsd
            tukey = pairwise_tukeyhsd(endog=data[y].values, groups=data[x].values, alpha=alpha)
            res = tukey.summary()
            tk = pd.DataFrame(res.data[1:], columns=res.data[0])
            # normalizar colunas
            tk['p_adj']  = pd.to_numeric(tk['p-adj'], errors='coerce')
            tk['reject'] = tk['reject'].astype(str).str.lower().map({'true': True, 'false': False})
            sig_results = tk.copy()

            sig_pairs = tk[tk['reject']].copy()
            if not sig_pairs.empty:
                sig_pairs['x1'] = sig_pairs['group1'].map(x_pos)
                sig_pairs['x2'] = sig_pairs['group2'].map(x_pos)
                sig_pairs[['xa','xb']] = np.sort(sig_pairs[['x1','x2']].values, axis=1)
                sig_pairs = sig_pairs.sort_values(by=['xb','xa'])

                # base de altura
                auto_span = (np.nanmax(y_vals) - np.nanmin(y_vals)) if ylim is None else (ylim[1]-ylim[0])
                y_base = tops.max() + y_pad*auto_span

                for _, r in sig_pairs.iterrows():
                    xa, xb = int(r['xa']), int(r['xb'])
                    local_top = max(tops[xa], tops[xb]) + y_pad
                    lvl = _get_free_level(xa, xb)
                    y0 = max(y_base + lvl*step, local_top + lvl*step*0.6)
                    txt = (f"p={r['p_adj']:.3g}" if show_p_text else _p_to_text(r['p_adj']))
                    _draw_sig_bracket(ax, xa, xb, y0, txt)

        else:
            raise ValueError("Parâmetro 'test' inválido. Use 'auto', 'ttest' ou 'tukey'.")

        plt.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches='tight')

    stats_table = g.rename(columns={x: 'group'})
    return fig, ax, stats_table, sig_results

def bar_ic95(
    df, x, y='Desempenho', hue=None,
    order=None, hue_order=None,
    palette=None,                 # dict {'CV':'#..', 'SV':'#..'} ou lista de cores
    figsize=(10, 6),
    bar_total_width=0.8,          # largura total ocupada pelo grupo no eixo x
    edgecolor='black', alpha=1.0,
    capsize=5, linewidth=1,
    annotate=True, text_fmt="Média: {mean:.2f}\nIC95: ±{ci:.2f}", text_pad=0.01,
    ylim=(0, 1.1), rotate_xticks=45,
    title=None, xlabel=None, ylabel=None,
    grid=True):
    """
    Barplot modular com IC95% (t-Student). Se 'hue' for informado, plota barras agrupadas
    com uma cor por grupo.

    Parâmetros
    ----------
    df : DataFrame
    x  : str, coluna categórica para o eixo x
    y  : str, coluna numérica (default 'Desempenho')
    hue: str|None, segunda categoria para agrupar e colorir (ex.: 'grupo')
    order, hue_order : listas com a ordem desejada das categorias
    palette : dict|list|None, paleta para os níveis de 'hue'
    bar_total_width : float, largura total ocupada por cada posição de x (<=1.0)
    annotate : bool, escreve média e IC acima das barras
    text_fmt : str, formato do texto; usa {mean} e {ci}
    text_pad : float, espaço vertical adicional acima do topo da barra
    ylim : tuple|None, limites do eixo y
    title, xlabel, ylabel : str|None
    grid : bool, liga grade pontilhada

    Retorna
    -------
    fig, ax, stats : (matplotlib.figure.Figure, matplotlib.axes.Axes, DataFrame de estatísticas)
    """

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import t

    data = df[[c for c in [x, y, hue] if c is not None]].dropna().copy()

    # Definir ordem dos níveis
    def _sorted_or_levels(s, given_order):
        if given_order is not None:
            return list(given_order)
        if pd.api.types.is_categorical_dtype(s):
            return list(s.cat.categories)
        return sorted(s.unique().tolist())

    x_levels = _sorted_or_levels(data[x], order)
    if hue is not None:
        h_levels = _sorted_or_levels(data[hue], hue_order)
    else:
        h_levels = [None]

    # Agregar estatísticas por (x[, hue])
    group_cols = [x] + ([hue] if hue is not None else [])
    stats = (data
             .groupby(group_cols, dropna=False)[y]
             .agg(mean='mean', std='std', count='count')
             .reset_index())

    # Garantir presença de todas as combinações (para manter espaçamentos)
    if hue is not None:
        stats = (stats
                 .set_index(group_cols)
                 .reindex(pd.MultiIndex.from_product([x_levels, h_levels], names=group_cols))
                 .reset_index())

    # CI95% com t de Student (n>1)
    def _ci95(std, n):
        if (pd.notnull(std)) and (pd.notnull(n)) and (n > 1):
            sem = std / np.sqrt(n)
            return t.ppf(0.975, df=int(n)-1) * sem
        return np.nan

    stats['ci95'] = stats.apply(lambda r: _ci95(r['std'], r['count']), axis=1)

    # Preparar cores
    if hue is not None:
        if palette is None:
            palette = sns.color_palette(None, n_colors=len(h_levels))
        if isinstance(palette, dict):
            color_map = {lvl: palette.get(lvl, '#999999') for lvl in h_levels}
        else:
            # lista -> mapear por posição
            color_map = {lvl: palette[i % len(palette)] for i, lvl in enumerate(h_levels)}
    else:
        # sem hue: uma única cor
        color_map = {None: sns.color_palette(None, 1)[0]}

    # Figura
    fig, ax = plt.subplots(figsize=figsize)

    # Geometria das barras
    nx = len(x_levels)
    nh = len(h_levels)
    base_x = np.arange(nx)

    if hue is None:
        bw = bar_total_width
        offsets = np.zeros(1)
    else:
        bw = bar_total_width / nh
        # offsets centrados em torno do zero para ficar simétrico
        start = -bar_total_width/2 + bw/2
        offsets = np.array([start + i*bw for i in range(nh)])

    # Plotar barras (com erro)
    for i, xlvl in enumerate(x_levels):
        for j, hlvl in enumerate(h_levels):
            if hue is None:
                row = stats.loc[(stats[x] == xlvl)]
            else:
                row = stats.loc[(stats[x] == xlvl) & (stats[hue] == hlvl)]

            # Se a combinação não existe, pula (evita "Média: nan / IC95: ±nan")
            if row.empty or pd.isna(row['mean'].iloc[0]):
                continue

            mean = float(row['mean'].iloc[0])
            ci   = float(row['ci95'].iloc[0]) if pd.notnull(row['ci95'].iloc[0]) else np.nan

            xpos = base_x[i] + (offsets[j] if hue is not None else 0.0)
            ax.bar(xpos, mean, width=bw,
                   color=color_map[hlvl], edgecolor=edgecolor,
                   alpha=alpha, linewidth=linewidth)

            if pd.notnull(ci):
                ax.errorbar(xpos, mean, yerr=ci, fmt='none',
                            ecolor='black', elinewidth=linewidth, capsize=capsize)

            if annotate:
                y_text = mean + (ci if pd.notnull(ci) else 0.0) + text_pad
                txt = text_fmt.format(mean=mean, ci=(ci if pd.notnull(ci) else float('nan')))
                ax.text(xpos, y_text, txt, ha='center', va='bottom', fontsize=8, color='black')

    # Eixo x
    ax.set_xticks(base_x)
    ax.set_xticklabels(x_levels, rotation=rotate_xticks)
    ax.set_xlim(-0.5, nx - 0.5)

    # Legenda
    if hue is not None:
        handles = [plt.Line2D([0],[0], marker='s', color=color_map[hl], markersize=10,
                               linewidth=0, label=str(hl), markerfacecolor=color_map[hl],
                               markeredgecolor=edgecolor)
                   for hl in h_levels]
        ax.legend(handles=handles, title=hue)

    # Estética
    ax.set_title(title if title else f'Barplot com IC95% por {x}' + (f' (hue={hue})' if hue else ''))
    ax.set_xlabel(xlabel if xlabel else x)
    ax.set_ylabel(ylabel if ylabel else y)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if grid:
        ax.grid(True, linestyle='--', alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig, ax, stats

def interaction_plot(
    df,
    x,                  # coluna categórica no eixo X (ex.: "Complexidade")
    line,               # coluna que define as linhas (ex.: "velocidade")
    y='Desempenho',     # coluna do desfecho
    facet=None,         # coluna para facetar (ex.: "grupo") ou None
    fixed=None,         # dict para filtrar dados, e.g. {"velocidade": "Médio"} ou {"velocidade":["Médio","Rápido"]}
    # Ordem e rótulos
    x_order=None, line_order=None, facet_order=None,
    x_map=None, line_map=None, facet_map=None,
    # Visual / estatística
    ci=0.95,            # nível de confiança (ex.: 0.95)
    interativo=False,   # Plotly (True) ou Matplotlib (False)
    salvar_interativo=False,  # se interativo=True, salva .html com o título como nome do arquivo
    title=None,
    figsize=(12, 5),
    grid=True,
    markers=None,       # dict opcional: {nível_line: 'o'|'s'|...} (usado no matplotlib)
    colors=None,        # dict opcional: {nível_line: '#hex'...}
    percent_auto=True,   # se True e ~>60% dos valores de y estiverem em [0,1], plota em %
    ylim=(0.0, 1.1), 
):
    """
    Faz um interaction plot (média ± IC) entre `x` e `line`, com opção de facet por `facet`.

    Exemplos de uso:
    ----------------
    # 1) “Como o desempenho varia com Complexidade (linhas = Velocidade), facetado por Grupo”
    interaction_plot(df_protA, x='Complexidade', line='velocidade', y='Desempenho',
                     facet='grupo', x_order=[4,6,8],
                     line_order=['Lento','Médio','Rápido'],
                     facet_order=['CV','SV'],
                     x_map={4:'Fácil',6:'Intermediário',8:'Difícil'},
                     title='Complexidade × Velocidade | facetado por Grupo')

    # 2) “Como o desempenho varia com Velocidade (linhas = Complexidade), dado Complexidade=6 (filtro)”
    interaction_plot(df_protA, x='velocidade', line='grupo', y='Desempenho',
                     fixed={'Complexidade': 6},
                     x_order=['Lento','Médio','Rápido'],
                     line_order=['CV','SV'],
                     title='Velocidade × Grupo (Complexidade = 6)')

    Parâmetros
    ----------
    df : DataFrame
        Tabela com os dados.
    x : str
        Coluna categórica usada no eixo X.
    line : str
        Coluna categórica que define as diferentes linhas (cores).
    y : str, default 'Desempenho'
        Coluna numérica do desfecho.
    facet : str | None
        Coluna categórica para facetar (um subplot/face por nível).
    fixed : dict | None
        Filtros a aplicar antes de agregar, ex.: {'velocidade':'Médio'} ou {'velocidade':['Lento','Médio']}.
    x_order, line_order, facet_order : list | None
        Ordens desejadas para os níveis de cada fator.
    x_map, line_map, facet_map : dict | None
        Mapeamentos de rótulos para eixos/legendas (ex.: {4:'Fácil', 6:'Médio', 8:'Difícil'}).
    ci : float
        Nível de confiança para barras de erro (Student t).
    interativo : bool
        Se True, usa Plotly; senão, Matplotlib.
    salvar_interativo : bool
        Se True e interativo=True, salva o gráfico .html com o título como nome do arquivo.
    title : str | None
        Título do gráfico.
    figsize : tuple
        Tamanho da figura (no Matplotlib).
    grid : bool
        Mostra grid (no Matplotlib).
    markers : dict | None
        Marcadores por nível de `line` (Matplotlib). Ex.: {'Lento':'o','Médio':'s','Rápido':'D'}
    colors : dict | None
        Cores por nível de `line`. Ex.: {'Lento':'#F59E0B', 'Médio':'#3B82F6', 'Rápido':'#10B981'}
    percent_auto : bool
        Se True, e se ~>60% dos valores de y ∈ [0,1], converte y em porcentagem.

    Retorna
    -------
    plot_obj : (fig, ax_or_axes) no Matplotlib, ou `fig` do Plotly
    stats_df : DataFrame de agregação com mean, std, count, ci para cada combinação.
    """
    import numpy as np
    import pandas as pd
    from scipy.stats import t as t_dist

    data = df[[c for c in [x, line, y, facet] if c is not None]].dropna().copy()

    # Aplicar filtros (fixed)
    if fixed:
        for col, val in fixed.items():
            if isinstance(val, (list, tuple, set, np.ndarray, pd.Series)):
                data = data[data[col].isin(list(val))]
            else:
                data = data[data[col] == val]

    # percent_auto: converter y em % se fizer sentido
    y_vals = data[y].dropna().values
    use_pct = False
    if percent_auto and len(y_vals):
        frac_01 = np.nanmean((y_vals >= 0) & (y_vals <= 1))
        use_pct = frac_01 > 0.6
    y_plot_col = f"{y}_plot"
    data[y_plot_col] = data[y] * (100.0 if use_pct else 1.0)
    y_label = f"{y} (%)" if use_pct else y

    scale = 100.0 if use_pct else 1.0
    ylim_plot = (ylim[0]*scale, ylim[1]*scale) if ylim is not None else None
    
    # Aplicar ordens se fornecidas
    if x_order is not None:
        data[x] = pd.Categorical(data[x], categories=x_order, ordered=True)
    if line_order is not None:
        data[line] = pd.Categorical(data[line], categories=line_order, ordered=True)
    if facet and (facet_order is not None):
        data[facet] = pd.Categorical(data[facet], categories=facet_order, ordered=True)

    # Agregar: média, desvio, n e IC
    group_cols = [c for c in [facet, x, line] if c is not None]
    stats = (data.groupby(group_cols, dropna=False)[y_plot_col]
             .agg(mean='mean', std='std', count='count')
             .reset_index())

    def _ci_level(std, n, ci=0.95):
        if n and n > 1 and pd.notnull(std):
            sem = std / np.sqrt(n)
            # bilateral
            q = 0.5 + ci/2.0
            return t_dist.ppf(q, df=n-1) * sem
        return np.nan

    stats['ci'] = [_ci_level(s, n, ci=ci) for s, n in zip(stats['std'], stats['count'])]

    # Funções helpers para rótulos
    def _label_map(val, mapping):
        return mapping.get(val, val) if mapping else val

    # ---------- PLOTLY (interativo) ----------
    if interativo:
        try:
            import plotly.express as px
            import plotly.graph_objects as go
        except Exception as e:
            raise ImportError("interativo=True requer o pacote 'plotly' instalado.") from e

        title_eff = title or f"Interaction plot: {x} × {line}" + (f" | facet: {facet}" if facet else "")

        # Rótulos amigáveis nos eixos/legenda
        stats['_xlab'] = stats[x].map(lambda v: _label_map(v, x_map))
        stats['_llab'] = stats[line].map(lambda v: _label_map(v, line_map))
        if facet:
            stats['_flab'] = stats[facet].map(lambda v: _label_map(v, facet_map))

        # Para linhas com barras de erro: usar go.Figure e adicionar traces por (facet,line)
        if facet:
            fig = go.Figure()
            # garantir a ordem
            f_levels = stats['_flab'].dropna().unique().tolist()
            if facet_order is not None:
                f_levels = [_label_map(v, facet_map) for v in facet_order if v in stats[facet].unique()]
            for fval in f_levels:
                sub_f = stats[stats['_flab'] == fval]
                # ordem de lines:
                l_levels = sub_f['_llab'].dropna().unique().tolist()
                if line_order is not None:
                    l_levels = [_label_map(v, line_map)
                                for v in line_order
                                if _label_map(v, line_map) in l_levels]
                for lval in l_levels:
                    sub = sub_f[sub_f['_llab'] == lval].copy()
                    # ordenar por x
                    if x_order is not None:
                        sub['_xlab'] = pd.Categorical(sub['_xlab'],
                                                      categories=[_label_map(v, x_map) for v in x_order], ordered=True)
                        sub = sub.sort_values('_xlab')
                    fig.add_trace(go.Scatter(
                        x=sub['_xlab'], y=sub['mean'],
                        error_y=dict(type='data', array=sub['ci'], visible=True),
                        mode='lines+markers',
                        name=f"{lval} | {fval}",
                    ))
            fig.update_layout(
                title=title_eff,
                xaxis_title=_label_map(x, None) if not x_map else x,
                yaxis_title=y_label,
                legend_title_text=line if not line_map else line,
                yaxis_range=ylim_plot  # <<< aplicar limites
            )
        else:
            # sem facet: um gráfico só, várias linhas
            fig = go.Figure()
            l_levels = stats[line].dropna().unique().tolist()
            if line_order is not None:
                l_levels = [lvl for lvl in line_order if lvl in stats[line].unique()]
            for lval in l_levels:
                sub = stats[stats[line] == lval].copy()
                # ordenar por x
                if x_order is not None:
                    sub[x] = pd.Categorical(sub[x], categories=x_order, ordered=True)
                    sub = sub.sort_values(x)
                fig.add_trace(go.Scatter(
                    x=sub[x].map(lambda v: _label_map(v, x_map)),
                    y=sub['mean'],
                    error_y=dict(type='data', array=sub['ci'], visible=True),
                    mode='lines+markers',
                    name=_label_map(lval, line_map),
                    line=dict(color=(colors.get(lval) if colors else None)),
                    marker=dict(symbol=None)  # Plotly escolhe símbolo padrão; pode customizar
                ))
            fig.update_layout(
                title=title_eff,
                xaxis_title=x if not x_map else x,
                yaxis_title=y_label,
                legend_title_text=line if not line_map else line,
            )

        fig.show()

        if salvar_interativo:
            safe = (title_eff or "interaction_plot").replace("—","-").replace(" ", "_").replace("/","_")
            fig.write_html(f"{safe}.html")
            print(f"💾 Gráfico interativo salvo em: {safe}.html")

        return fig, stats

    # ---------- MATPLOTLIB ----------
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    # ordens “efetivas” para plotar
    x_levels = stats[x].dropna().unique().tolist()
    l_levels = stats[line].dropna().unique().tolist()

    if x_order is not None:
        x_levels = [lvl for lvl in x_order if lvl in stats[x].unique()]
    if line_order is not None:
        l_levels = [lvl for lvl in line_order if lvl in stats[line].unique()]

    # markers default
    default_markers = ['o', 's', 'D', '^', 'v', '<', '>', 'P', 'X']
    if markers is None:
        markers = {lvl: default_markers[i % len(default_markers)] for i, lvl in enumerate(l_levels)}

    # cores default
    default_colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['#1f77b4','#ff7f0e','#2ca02c'])
    if colors is None:
        colors = {lvl: default_colors[i % len(default_colors)] for i, lvl in enumerate(l_levels)}

    # Preparar figure/axes
    if facet:
        f_levels = stats[facet].dropna().unique().tolist()
        if facet_order is not None:
            f_levels = [lvl for lvl in facet_order if lvl in stats[facet].unique()]
        nF = len(f_levels)# tornar os subplots mais largos
        fig, axes = plt.subplots(1, nF, figsize=(14,4), sharey=True)
        if nF == 1:
            axes = [axes]
    else:
        fig, ax = plt.subplots(figsize=figsize)
        axes = [ax]
        f_levels = [None]

    for ax, fval in zip(axes, f_levels):
        sub = stats if fval is None else stats[stats[facet] == fval]

        # para cada linha (nível de `line`)
        for lval in l_levels:
            subl = sub[sub[line] == lval].copy()
            # ordenar por x
            if x_order is not None:
                subl[x] = pd.Categorical(subl[x], categories=x_order, ordered=True)
                subl = subl.sort_values(x)

            xx = subl[x].map(lambda v: _label_map(v, x_map)).values
            yy = subl['mean'].values.astype(float)
            ee = subl['ci'].values.astype(float)

            ax.errorbar(
                np.arange(len(xx)), yy, yerr=ee,
                fmt=markers[lval], ms=7, lw=2, capsize=4,
                color=colors[lval], label=_label_map(lval, line_map)
            )
            ax.plot(np.arange(len(xx)), yy, '-', color=colors[lval], lw=2, alpha=0.9)

        # eixos / título
        ax.set_xticks(np.arange(len(x_levels)))
        ax.set_xticklabels([_label_map(v, x_map) for v in x_levels])
        ax.set_xlabel(x)
        if ylim_plot is not None:      # <<< aplicar limites
            ax.set_ylim(*ylim_plot)
        if grid:
            ax.grid(True, ls='--', alpha=0.3)

        if fval is not None:
            ax.set_title(f"{facet}: {_label_map(fval, facet_map)}")

    # y label, título e legenda global
    axes[0].set_ylabel(y_label)
    ttl = title or f"Interaction plot: {x} × {line}" + (f" | facet: {facet}" if facet else "")
    fig.suptitle(ttl, y=1.02, fontsize=12)

    handles = [Line2D([0],[0], marker=markers[lvl], linestyle='-',
                      color=colors[lvl], lw=2, markersize=7, label=_label_map(lvl, line_map))
               for lvl in l_levels]
    fig.legend(handles, [h.get_label() for h in handles], title=line,
               loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=False)

    fig.tight_layout(rect=[0, 0, 0.86, 1])
    fig.tight_layout()
    return (fig, axes if facet else axes[0]), stats
