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


# %% 

# %%
