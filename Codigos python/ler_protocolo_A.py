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
#%% 
teste = protA_cv_df['df_ID_10']['Rep1']
plotar.plotar_trajetoria(seq = teste['Trajetória Completa'][8])

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
    
    print('--'*100)
    print(f'Trajetória {num}')
    print(f'Score parcial = {score_parcial} \nScore Total = {score_total}')
    print(f'Trajetória {num}')
    print(f'Score parcial uns = {score_parcial_uns} \nScore Total = {score_total_uns}')
    #print('--'*100)

    # Plotando os gráficos comparativos
    #plotar.plotar_trajetoria(seq = gabarito[f'{num}'][0], individuo= f"Gabarito trajetória {num}")
    #plotar.plotar_trajetoria(seq = seq, individuo= f"'df_ID_10' repetição 1")
    plotar.plot_comparacao(gabarito=gabarito[f'{num}'][0], seq= seq,
                           score_parcial=score_parcial_uns, 
                           score_total=score_total_uns)


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
                x.append(x[-1]-11)
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

def comparar_imagem(seq1=[], seq2=[]):
    """Compara espacialmente duas sequencias

    Args:
        seq1 (list): sequencia executada pelo individuo. Defaults to [].
        seq2 (list): gabarito. Defaults to [].
    """
    import numpy as np 
    import matplotlib.pyplot as plt
    
    #Pegando os vetores x e y de cada sequência
    x1,y1,_= traj_to_point(seq1)
    x2,y2,_= traj_to_point(seq2)

    # Verificando qual o tamanho máximo explorado na sequência para fazer posteriormente a matriz de zeros
    _,contx1 = np.unique(x1, return_counts=True)
    _,conty1= np.unique(y1, return_counts=True)
    _,contx2 = np.unique(x2, return_counts=True)
    _,conty2= np.unique(y2, return_counts=True)

    tamanho = np.max(np.concatenate((contx1,conty1,contx2,conty2)))
    
    #criando as sequências que vão ser a imagem binarizada
    v1_bin = np.zeros((tamanho+1,tamanho+1)) 
    v2_bin = np.zeros((tamanho+1,tamanho+1))

    #colocando os 1's de acordo com a trajetória
    for i in zip(x1,y1):
        v1_bin[i[0]][i[1]] = 1

    for i in zip(x2,y2):
        v2_bin[i[0]][i[1]] = 1
    
    #rotacionando as matrizes para ficar de um jeito amigável para printar
    v1_bin = rotate(v1_bin)
    v2_bin = rotate(v2_bin)
    print("Trajetória Binarizada Sequência1 x Sequência2:")
    #print(v1_bin)
    plt.figure()
    plt.imshow(v1_bin, cmap='gray_r', interpolation='nearest')
    plt.axis('off')
    #print(v2_bin)
    plt.figure()
    plt.imshow(v2_bin, cmap='gray_r', interpolation='nearest')
    plt.axis('off')

    #Multiplicando ponto a ponto das matrizes
    multiplicacao = v1_bin*v2_bin
    print("Matriz binarizada da sobreposição das trajetórias:")
    #print(multiplicacao)
    plt.figure()
    plt.imshow(multiplicacao, cmap='gray_r', interpolation='nearest')
    plt.axis('off')
    #Somando todos os 1's da Matriz binarizada da sobreposição das trajetórias
    soma = np.sum(multiplicacao)

    #Calculando o valor ideal (comparando gabarito x gabarito)
    soma_ideal = np.sum(v2_bin) 

    #Score calculado pela comparação entre as duas imagens binarizadas e o ideal
    score = soma/soma_ideal

    return score



# %%
resultado = comparar_imagem(teste['Trajetória Completa'][8],gabarito[f'{teste['Número da Trajetória'][8]}'][0])

print(resultado)


# %%
plotar.plot_comparacao(gabarito[f'{teste['Número da Trajetória'][8]}'][0],teste['Trajetória Completa'][8],resultado)
# %%
