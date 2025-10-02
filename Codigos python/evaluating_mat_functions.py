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

def dot_ic_sig(
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