'''
Código feito para abrir os arquivos .mat de cada Protocolo e extrair as trajetórias 
'''

#%%
from scipy.io import loadmat
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
import ast
import Plotar_sequencias as plotar
import evaluating_mat_functions as ev


def transformar_protA_mat_em_df(protocolo = [], id = []):
    """
    Transforma a estrutura MATLAB ProtA (carregada com scipy.io.loadmat) em um DataFrame de DataFrames
    organizado por participante e subdividido em três repetições experimentais.

    A função é voltada para o Protocolo A do experimento, que envolve:
    - Apresentação de trajetórias com diferentes níveis de sobreposição (overlap)
    - Reprodução manual das trajetórias com ou sem feedback vibro-tátil
    Cada participante realiza três repetições do protocolo.

    Os dados de cada tentativa são organizados em DataFrames por repetição, com limpeza e conversão de tipos,
    como listas codificadas em strings e arrays aninhados, para formatos Python nativos.

    Parâmetros:
    ----------
    protocolo : np.ndarray
        Estrutura carregada do arquivo .mat referente ao ProtA, normalmente de tamanho (n_individuos, 3)
    id : list of str
        Lista com os identificadores dos participantes, na ordem das linhas de `protocolo`

    Retorna:
    -------
    prot_df : pd.DataFrame
        DataFrame com as colunas sendo os participantes (`df_ID_XX`) e três linhas por coluna:
        - 'Rep1', 'Rep2', 'Rep3': DataFrames com as tentativas de cada repetição do experimento
          As colunas de cada DataFrame são:
            ['Número da Trajetória', 'Overlap', 'Sorteio', 'Tempo 1', 'Tempo 2', 'Tempo 3',
             'Score total', 'Score Parcial', 'Proporção espacial x', 'Proporção espacial y',
             'Trajetória Completa', 'Trajetória Simplificada']

    Observações:
    -----------
    - A coluna "Overlap" representa a porcentagem de sobreposição entre estímulos vibro-táteis.
    - Trajetórias ausentes ou inválidas são preenchidas com o código [9] (indicando que o participante ficou parado).
    - As listas codificadas como strings no MATLAB são convertidas para listas reais com `ast.literal_eval`.
    - As repetições são assumidas como organizadas sequencialmente na matriz de entrada (Rep1, Rep2, Rep3 por ID).
    """

    import pandas as pd
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
'''Lendo os arquivos .mat do Protocolo A CV e SV'''

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


# plotando algumas trajetórias
"""for i, seq in enumerate(protA_cv_df['df_ID_10']['Rep2']['Trajetória Completa']):
    plotar.plotar_trajetoria(seq = seq,individuo= 'ID10 na primeira repetição')
"""

# Carregando os gabaritos
gabarito = pd.read_csv('Gabaritos\gab_seq_completa_converted.csv')
gabarito_simplificado = pd.read_csv('Gabaritos\gab_seq_converted.csv')
for i in gabarito.columns:
    gabarito[i][0] = ast.literal_eval(gabarito[i][0])
    gabarito_simplificado[i][0] = ast.literal_eval(gabarito_simplificado[i][0])


#%% ---------------------- Calculando as métricas em cima de cada protocolo -------------------

#---- Protocolo A CV ----
'''Calculando os resultados das métricas de comparação de trajetória para todas as repetições
  de todos os individuos do protocolo A CV
'''

#prec = []
#rcll = []
propx = []
propy = []
acur = []
fpr =[]
sim = []
desempenho = []
desempenho_norm = []

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
            
            #---- Avaliando por comparação de imagem (IDEIA 3)
            resultado_ideia3 = ev.comparar_imagem(seq1=seq1,seq2=seq2,plotar_imagens = False)

            #---- Avaliando por similaridade com correlação cruzada normalizada (IDEIA 4)
            resultado_ideia4 = ev.calcular_similaridade(seq1,seq2)
            
            #sparcial.append(score_parcial_uns)
            #stotal.append(score_total_uns)
            propx.append(float(teste['Proporção espacial x'][i]))
            propy.append(float(teste['Proporção espacial y'][i]))
            #prec.append(resultado_ideia3[1])
            #rcll.append(resultado_ideia3[2])
            acur.append(resultado_ideia3[0])
            fpr.append(resultado_ideia3[3])
            sim.append(resultado_ideia4)
            # Desempenho calculado através da ideia de uma combinação das métricas escolhidas
            r1, r2 = ev.calcular_desempenho( acur=resultado_ideia3[0], 
                                                                 fpr=resultado_ideia3[3], 
                                                                 sim = resultado_ideia4, 
                                                                 propx=float(teste['Proporção espacial x'][i]), 
                                                                 propy=float(teste['Proporção espacial y'][i]))
            desempenho.append(r1)
            desempenho_norm.append(r2)
        #protA_cv_df[individuo][rep]['Precisão'] = prec 
        #protA_cv_df[individuo][rep]['Media Prec'] = np.mean(prec)
        #protA_cv_df[individuo][rep]['Recall'] = rcll
        #protA_cv_df[individuo][rep]['Media Recall'] = np.mean(rcll)
        protA_cv_df[individuo][rep]['Acuracia'] = acur
        protA_cv_df[individuo][rep]['Media Acuracia'] = np.mean(acur)
        protA_cv_df[individuo][rep]['Taxa de Falsos Positivos'] = fpr
        protA_cv_df[individuo][rep]['Media FPR'] = np.mean(fpr)
        protA_cv_df[individuo][rep]['Similaridade'] = sim
        protA_cv_df[individuo][rep]['Media Similaridade'] = np.mean(sim)
        protA_cv_df[individuo][rep]['Desempenho'] = desempenho
        protA_cv_df[individuo][rep]['Desempenho ponderado com proporção'] = desempenho_norm
        
        #Reiniciando as listas para a próxima iteração
        #prec = []
        #rcll = []
        acur = []
        fpr =[]
        sim = []
        desempenho = []
        desempenho_norm = []

#---- Protocolo A SV ----
'''Calculando os resultados das métricas de comparação de trajetória para todas as repetições
  de todos os individuos do protocolo A SV
'''
#sparcial =[]
#stotal = []
propx = []
propy = []
acur = []
prec = []
rcll = []
fpr =[]
sim = []

for individuo in protA_sv_df.columns:
    for rep in protA_sv_df.index:
        teste = protA_sv_df[individuo][rep]

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
            
            #---- Avaliando por comparação de imagem (IDEIA 3)
            resultado_ideia3 = ev.comparar_imagem(seq1=seq1,seq2=seq2,plotar_imagens = False)

            #---- Avaliando por similaridade com correlação cruzada normalizada (IDEIA 4)
            resultado_ideia4 = ev.calcular_similaridade(seq1,seq2)
            
            #sparcial.append(score_parcial_uns)
            #stotal.append(score_total_uns)
            propx.append(float(teste['Proporção espacial x'][i]))
            propy.append(float(teste['Proporção espacial y'][i]))
            #prec.append(resultado_ideia3[1])
            #rcll.append(resultado_ideia3[2])
            acur.append(resultado_ideia3[0])
            fpr.append(resultado_ideia3[3])
            sim.append(resultado_ideia4)
            # Desempenho calculado através da ideia de uma combinação das métricas escolhidas
            r1, r2 = ev.calcular_desempenho( acur=resultado_ideia3[0], 
                                                                 fpr=resultado_ideia3[3], 
                                                                 sim = resultado_ideia4, 
                                                                 propx=float(teste['Proporção espacial x'][i]), 
                                                                 propy=float(teste['Proporção espacial y'][i]))
            desempenho.append(r1)
            desempenho_norm.append(r2)
        #protA_sv_df[individuo][rep]['Precisão'] = prec 
        #protA_sv_df[individuo][rep]['Media Prec'] = np.mean(prec)
        #protA_sv_df[individuo][rep]['Recall'] = rcll
        #protA_sv_df[individuo][rep]['Media Recall'] = np.mean(rcll)
        protA_sv_df[individuo][rep]['Acuracia'] = acur
        protA_sv_df[individuo][rep]['Media Acuracia'] = np.mean(acur)
        protA_sv_df[individuo][rep]['Taxa de Falsos Positivos'] = fpr
        protA_sv_df[individuo][rep]['Media FPR'] = np.mean(fpr)
        protA_sv_df[individuo][rep]['Similaridade'] = sim
        protA_sv_df[individuo][rep]['Media Similaridade'] = np.mean(sim)
        protA_sv_df[individuo][rep]['Desempenho'] = desempenho
        protA_sv_df[individuo][rep]['Desempenho ponderado com proporção'] = desempenho_norm
        
        #Reiniciando as listas para a próxima iteração
        #prec = []
        #rcll = []
        acur = []
        fpr =[]
        sim = []
        desempenho = []
        desempenho_norm = []


#%%  # Vendo a distribuição dos resultados obtidos acima
""" 
resultados_A_CV = {
    #'Score Parcial':sparcial,
    #'Score Total':stotal,
    'Propx':propx,
    'Propy':propy,
    'Acurácia':acur,
    'Precisão':prec,
    'Recall':rcll,
    'Taxa de Falsos Positivos':fpr,
    'Similaridade':sim
}

resultados_A_CV_df = pd.DataFrame(resultados_A_CV)

#Plotando as distribuições 
ev.plotar_distribuicoes_resultados(resultados_A_CV_df, titulo = '(Protocolo A CV)')"""

"""# Vendo a distribuição dos resultados obtidos acima 
resultados_A_SV = {
    #'Score Parcial':sparcial,
    #'Score Total':stotal,
    'Propx':propx,
    'Propy':propy,
    'Acurácia':acur,
    'Precisão':prec,
    'Recall':rcll,
    'Taxa de Falsos Positivos':fpr,
    'Similaridade':sim
}

resultados_A_SV_df = pd.DataFrame(resultados_A_SV)

#Plotando as distribuições 
ev.plotar_distribuicoes_resultados(resultados_A_SV_df, titulo = '(Protocolo A SV)')
#  Plotando tudo junto para poder ver a distribuição conjunta
resultado_A = pd.concat([resultados_A_CV_df,resultados_A_SV_df], axis = 0,ignore_index=True)
ev.plotar_distribuicoes_resultados(resultado_A, titulo = '(Protocolo A completo (CV e SV juntos))')"""
#%% ------- Criando o DataFrame para os desempenhos medios em cada protocolo ---------


# 1) Juntando todas as repetições de cada indivíduo 
# 1.1) Criando o dataframe com tudo junto

lista_concatenada = []

for individuo in protA_cv_df.columns:
    for rep_label in protA_cv_df[individuo].index:  # 'Rep1', 'Rep2', 'Rep3'
        df_tmp = protA_cv_df[individuo][rep_label].copy()
        df_tmp['ID'] = individuo
        df_tmp['Repeticao'] = rep_label
        lista_concatenada.append(df_tmp)

#concatena todos os indivíduos em um só dataframe para facilitar a análise
df_concat_protA_cv = pd.concat(lista_concatenada, ignore_index=True)

lista_concatenada = []

for individuo in protA_sv_df.columns:
    for rep_label in protA_sv_df[individuo].index:  # 'Rep1', 'Rep2', 'Rep3'
        df_tmp = protA_sv_df[individuo][rep_label].copy()
        df_tmp['ID'] = individuo
        df_tmp['Repeticao'] = rep_label
        lista_concatenada.append(df_tmp)

#concatena todos os indivíduos em um só dataframe para facilitar a análise
df_concat_protA_sv = pd.concat(lista_concatenada, ignore_index=True)

# 1.2) Adicionando a complexidade ao DataFrame de cada trajetória para futuro filtro
df_concat_protA_cv['Complexidade'] = df_concat_protA_cv['Número da Trajetória'].apply(ev.map_complexidade)
df_concat_protA_sv['Complexidade'] = df_concat_protA_sv['Número da Trajetória'].apply(ev.map_complexidade)

# 1.3) Organizando as colunas necessárrias para futuros filtros
df_concat_protA_cv['Overlap'] = df_concat_protA_cv['Overlap'].astype(float)
df_concat_protA_cv['ID'] = df_concat_protA_cv['ID'].astype(str)
df_concat_protA_cv['Número da Trajetória'] = df_concat_protA_cv['Número da Trajetória'].astype(int)

df_concat_protA_sv['Overlap'] = df_concat_protA_sv['Overlap'].astype(float)
df_concat_protA_sv['ID'] = df_concat_protA_sv['ID'].astype(str)
df_concat_protA_sv['Número da Trajetória'] = df_concat_protA_sv['Número da Trajetória'].astype(int)

# 2) Calculando os desempenhos médio e médio ponderado por ID, Complexidade, Overlap e Trajetória

#Listas únicas de cada ID, complexidade, overlap e trajetória Protocol A CV
ids = df_concat_protA_cv['ID'].unique()
complexidades = sorted(df_concat_protA_cv['Complexidade'].unique())
overlaps = sorted(df_concat_protA_cv['Overlap'].unique())
trajetorias = sorted(df_concat_protA_cv['Número da Trajetória'].unique())

desempenho_A_cv = ev.calcular_desempenhos_medios(df_concat_protA_cv,ids,
                                                 complexidades,overlaps,
                                                 trajetorias)

#Listas únicas de cada ID, complexidade, overlap e trajetória Protocol A SV
ids = df_concat_protA_sv['ID'].unique()
complexidades = sorted(df_concat_protA_sv['Complexidade'].unique())
overlaps = sorted(df_concat_protA_sv['Overlap'].unique())
trajetorias = sorted(df_concat_protA_sv['Número da Trajetória'].unique())

desempenho_A_sv = ev.calcular_desempenhos_medios(df_concat_protA_sv,ids,
                                                 complexidades,overlaps,
                                                 trajetorias)

#%%# Plotando os desempenhos médios

print('---'*100)
print('Desempenhos médios do Protocolo A CV')
print('---'*100)    
for tipo_desempenho in desempenho_A_cv.keys():
    parametro = f'Desempenho médio ponderado {tipo_desempenho}'
    titulo = 'do Protocolo A CV'
    #data = desempenho_A_cv[tipo_desempenho]['Media_Desempenho']
    data = desempenho_A_cv[tipo_desempenho]['Media_Desempenho']
    
    #Plotando com a função de plotar desempenhos
    ev.plotar_desempenhos(data,titulo,parametro)

print('---'*100)
print('Desempenhos médios do Protocolo A SV')
print('---'*100)    

for tipo_desempenho in desempenho_A_sv.keys():
    parametro = f'Desempenho médio {tipo_desempenho}'
    titulo = 'do Protocolo A SV'
    data = desempenho_A_sv[tipo_desempenho]['Media_Desempenho']
    
    #Plotando com a função de plotar desempenhos
    ev.plotar_desempenhos(data,titulo,parametro)
#%% #Desempenho de todo o protocolo A CV e SV

print('---'*100)
print('Desempenhos de todo Protocolo A CV')
print('---'*100)    

parametro = f'Desempenho'
titulo = 'Desempenho de todo Protocolo A CV'
#data = df_concat_protA_cv['Desempenho']
data = df_concat_protA_cv['Desempenho']

ev.plotar_desempenhos(data,titulo,parametro)

print('---'*100)
print('Desempenhos de todo Protocolo A SV')
print('---'*100)  
parametro = f'Desempenho'
titulo = 'Desempenho de todo Protocolo A SV'
data = df_concat_protA_sv['Desempenho']

ev.plotar_desempenhos(data,titulo,parametro)


"""# %%
for i in gabarito.columns:
plotar.plotar_trajetoria(gabarito[i][0],individuo= f'Gabarito {i}')

"""
# %% --------------------------------------- Teste de Normalidade ---------------------------------------

#%%
"""for tipo_desempenho in desempenho_A_cv.keys():
    '''
    resultado = ev.teste_normalidade_shapiro(desempenho_A_sv[tipo_desempenho],
                                          titulo=tipo_desempenho,alpha=0.025)
    print(resultado)
    '''
    resultado = ev.teste_normalidade_kstest(desempenho_A_sv[tipo_desempenho],
                                          titulo=tipo_desempenho,alpha=0.025)
    print(resultado)
"""

#%% --------------------------------------- Teste de Hipótese  ---------------------------------------
from scipy.stats import ttest_rel

#Teste: Há diferença no desempenho de acordo o overlap?
overlap = sorted(df_concat_protA_cv['Overlap'].unique())
# Inicializa DataFrame vazio
matriz_overlap = pd.DataFrame(index=overlaps, columns=overlaps)
A_cv_overlap = desempenho_A_cv['por_overlap'].pivot(index='ID', columns='Overlap', values='Media_Desempenho')

for i in overlaps:
    for j in overlaps:
        if i == j:
            matriz_overlap.loc[i, j] = (np.nan, np.nan, np.nan)
        else:
            stat, p = ttest_rel(A_cv_overlap[i], A_cv_overlap[j], nan_policy='omit')
            resultado = 'A diferença é estatisticamente significativa✅' if p < 0.05 else 'A diferença não é estatisticamente significativa❌'
            matriz_overlap.loc[i, j] = (stat, p, resultado)

#Teste: Há diferença no desempenho de acordo a complexidade?
complexidade = sorted(df_concat_protA_cv['Complexidade'].unique())
# Inicializa DataFrame vazio
matriz_complexidade = pd.DataFrame(index=complexidade, columns=complexidade)
A_cv_complexidade = desempenho_A_cv['por_complexidade'].pivot(index='ID', columns='Complexidade', values='Media_Desempenho')

for i in complexidade:
    for j in complexidade:
        if i == j:
            matriz_complexidade.loc[i, j] = (np.nan, np.nan,np.nan)
        else:
            stat, p = ttest_rel(A_cv_complexidade[i], A_cv_complexidade[j], nan_policy='omit')
            resultado = 'A diferença é estatisticamente significativa✅' if p < 0.05 else 'A diferença não é estatisticamente significativa❌'
            matriz_complexidade.loc[i, j] = (stat, p, resultado)

#Teste: Há diferença no desempenho de acordo a complexidade e overlap?
df= desempenho_A_cv['por_complexidade_por_overlap']
# Criação de chave única para linha e coluna: '1-Overlap 0.0'
df['Condicao'] = 'Complexidade ' + df['Complexidade'].astype(str) + '-Overlap ' + df['Overlap'].astype(str)

df_pivot = df.pivot(index='ID', columns= 'Condicao', values='Media_Desempenho')

condicoes = df_pivot.columns

# Inicializa DataFrame vazio
matriz_complexidade_overlap = pd.DataFrame(index=condicoes, columns=condicoes)
for i in condicoes:
    for j in condicoes:
        if i == j:
            matriz_complexidade_overlap.loc[i, j] = (np.nan, np.nan, np.nan)
        else:
            stat, p = ttest_rel(df_pivot[i], df_pivot[j], nan_policy='omit')
            resultado = 'A diferença é estatisticamente significativa✅' if p < 0.05 else 'A diferença não é estatisticamente significativa❌'
            matriz_complexidade_overlap.loc[i, j] = (stat, p, resultado)

#Teste: Há diferença no desempenho de acordo a trajetória e overlap?

df = desempenho_A_cv['por_trajetoria_por_overlap']
# Criação de chave única para linha e coluna: '1-Overlap 0.0'
df['Condicao'] = df['Número da trajetoria'].astype(str) + '-Overlap ' + df['Overlap'].astype(str)

# Pivotando para ter cada coluna como uma condição (por ID)
df_pivot = df.pivot(index='ID', columns='Condicao', values='Media_Desempenho')

condicoes = df_pivot.columns

matriz_trajetoria_overlap = pd.DataFrame(index=condicoes, columns=condicoes)

for i in condicoes:
    for j in condicoes:
        if i == j:
            matriz_trajetoria_overlap.loc[i, j] = (np.nan, np.nan, np.nan)
        else:
            stat, p = ttest_rel(df_pivot[i], df_pivot[j], nan_policy='omit')
            resultado = 'A diferença é estatisticamente significativa✅' if p < 0.05 else 'A diferença não é estatisticamente significativa❌'
            matriz_trajetoria_overlap.loc[i, j] = (stat, p, resultado)


#%% Testando se há diferença entre protocolos CV e SV

# Há diferença entre o desempenho do protocolo A CV e A SV agrupando-os por overlap?
A_cv_overlap = desempenho_A_cv['por_overlap'].pivot(index='ID', columns='Overlap', values='Media_Desempenho')
A_sv_overlap = desempenho_A_sv['por_overlap'].pivot(index='ID', columns='Overlap', values='Media_Desempenho')

overlaps = sorted(A_cv_overlap.columns)
overlaps_sv = list(map(lambda x: f'{x}-SV',overlaps))
overlaps_cv = list(map(lambda x: f'{x}-CV',overlaps))

# Inicializa DataFrame vazio
matriz_cv_sv_overlap = pd.DataFrame(index=overlaps_cv, columns=overlaps_sv)

for i in overlaps:
    for j in overlaps:
        stat, p = ttest_rel(A_cv_overlap[i], A_sv_overlap[j], nan_policy='omit')
        resultado = 'A diferença é estatisticamente significativa✅' if p < 0.05 else 'A diferença não é estatisticamente significativa❌'
        matriz_cv_sv_overlap.loc[f'{i}-CV', f'{j}-SV'] = (stat, p, resultado)

# Há diferença entre o desempenho do protocolo A CV e A SV agrupando-os por complexidade?
A_cv_complexidade = desempenho_A_cv['por_complexidade'].pivot(index='ID', columns='Complexidade', values='Media_Desempenho')
A_sv_complexidade = desempenho_A_sv['por_complexidade'].pivot(index='ID', columns='Complexidade', values='Media_Desempenho')
complexidades = sorted(A_cv_complexidade.columns)
complexidades_sv = list(map(lambda x: f'{x}-SV',complexidades))
complexidades_cv = list(map(lambda x: f'{x}-CV',complexidades))

# Inicializa DataFrame vazio
matriz_cv_sv_complexidade = pd.DataFrame(index=complexidades_cv, columns=complexidades_sv)
for i in complexidades:
    for j in complexidades:
        stat, p = ttest_rel(A_cv_complexidade[i], A_sv_complexidade[j], nan_policy='omit')
        resultado = 'A diferença é estatisticamente significativa✅' if p < 0.05 else 'A diferença não é estatisticamente significativa❌'
        matriz_cv_sv_complexidade.loc[f'{i}-CV', f'{j}-SV'] = (stat, p, resultado)

# Há diferença entre o desempenho do protocolo A CV e A SV agrupando-os por complexidade e overlap?
df_cv = desempenho_A_cv['por_complexidade_por_overlap']
df_cv['Condicao'] = 'Complexidade ' + df_cv['Complexidade'].astype(str) + '-Overlap ' + df_cv['Overlap'].astype(str)
df_cv_pivot = df_cv.pivot(index='ID', columns='Condicao', values='Media_Desempenho')
df_sv = desempenho_A_sv['por_complexidade_por_overlap']
df_sv['Condicao'] = 'Complexidade ' + df_sv['Complexidade'].astype(str) + '-Overlap ' + df_sv['Overlap'].astype(str)
df_sv_pivot = df_sv.pivot(index='ID', columns='Condicao', values='Media_Desempenho')
condicoes_cv = df_cv_pivot.columns
condicoes_sv = df_sv_pivot.columns
# Inicializa DataFrame vazio
matriz_cv_sv_complexidade_overlap = pd.DataFrame(index=condicoes_cv, columns=condicoes_sv)
for i in condicoes_cv:
    for j in condicoes_sv:
        stat, p = ttest_rel(df_cv_pivot[i], df_sv_pivot[j], nan_policy='omit')
        resultado = 'A diferença é estatisticamente significativa✅' if p < 0.05 else 'A diferença não é estatisticamente significativa❌'
        matriz_cv_sv_complexidade_overlap.loc[i, j] = (stat, p, resultado)

# Há diferença entre o desempenho do protocolo A CV e A SV agrupando-os por trajetória e overlap?
df_cv = desempenho_A_cv['por_trajetoria_por_overlap']
df_cv['Condicao'] = df_cv['Número da trajetoria'].astype(str) + '-Overlap ' + df_cv['Overlap'].astype(str)
df_cv_pivot = df_cv.pivot(index='ID', columns='Condicao', values='Media_Desempenho')
df_sv = desempenho_A_sv['por_trajetoria_por_overlap']
df_sv['Condicao'] = df_sv['Número da trajetoria'].astype(str) + '-Overlap ' + df_sv['Overlap'].astype(str)
df_sv_pivot = df_sv.pivot(index='ID', columns='Condicao', values='Media_Desempenho')
condicoes_cv = df_cv_pivot.columns
condicoes_sv = df_sv_pivot.columns
# Inicializa DataFrame vazio
matriz_cv_sv_trajetoria_overlap = pd.DataFrame(index=condicoes_cv, columns=condicoes_sv)
for i in condicoes_cv:
    for j in condicoes_sv:
        stat, p = ttest_rel(df_cv_pivot[i], df_sv_pivot[j], nan_policy='omit')
        resultado = 'A diferença é estatisticamente significativa✅' if p < 0.05 else 'A diferença não é estatisticamente significativa❌'
        matriz_cv_sv_trajetoria_overlap.loc[i, j] = (stat, p, resultado)

#%% 
df_cv = desempenho_A_cv['por_complexidade_por_overlap']
df_cv['Condicao'] = 'Complexidade ' + df_cv['Complexidade'].astype(str) + '-Overlap ' + df_cv['Overlap'].astype(str)
df_cv_pivot = df_cv.pivot(index='ID', columns='Condicao', values='Media_Desempenho')
df_sv = desempenho_A_sv['por_complexidade_por_overlap']
df_sv['Condicao'] = 'Complexidade ' + df_sv['Complexidade'].astype(str) + '-Overlap ' + df_sv['Overlap'].astype(str)
df_sv_pivot = df_sv.pivot(index='ID', columns='Condicao', values='Media_Desempenho')
condicoes_cv = df_cv_pivot.columns
condicoes_sv = df_sv_pivot.columns
for i in condicoes_cv:
    for j in condicoes_sv:
        if (i == j):
            if(matriz_cv_sv_complexidade_overlap[i][j][2] == 'A diferença é estatisticamente significativa✅'):
                print(f'{i} - {j} = {matriz_cv_sv_complexidade_overlap[i][j]}')
            #print(f'{i} - {j} = {matriz_cv_sv_trajetoria_overlap[i][j]}')

#%%
stat, p = ttest_rel(df_concat_protA_cv['Desempenho'], df_concat_protA_sv['Desempenho'], nan_policy='omit')
resultado = 'A diferença é estatisticamente significativa✅' if p < 0.05 else 'A diferença não é estatisticamente significativa❌'
print(f'Diferença entre Protocolo A CV e SV: {resultado} (stat={stat}, p={p})')	



#%% --------------------------------------- Teste de hipótese por ANOVA ---------------------------------------

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import MultiComparison
from statsmodels.stats.libqsturng import psturng  # usado internamente

df_cv = desempenho_A_cv['por_complexidade_por_overlap']
df_cv['grupo'] = 'CV'
df_sv = desempenho_A_sv['por_complexidade_por_overlap']
df_sv['grupo'] = 'SV'
df = pd.concat([df_cv, df_sv], axis=0, ignore_index=True)
df['grupo'] = df['grupo'].astype('category')
df['Complexidade'] = df['Complexidade'].astype('category')
df['Overlap'] = df['Overlap'].astype('category')
df['Media_Desempenho'] = df['Media_Desempenho'].astype(float)


# Assumindo que 'desempenho' é contínua, e grupo, complexidade e overlap são fatores:
modelo = ols('Media_Desempenho ~ grupo * Complexidade * Overlap', data=df).fit()
anova = sm.stats.anova_lm(modelo, typ=3)  
print("Resultados da ANOVA:")
print(anova)
print("-"*100)

# Suponha que você quer comparar as médias de 'desempenho' entre os grupos de complexidade:
mc = MultiComparison(df['Media_Desempenho'], df['Complexidade'])
resultado = mc.tukeyhsd()
#print(resultado)

# Print do resumo das comparações
print(resultado.summary())
#%%
# Boxplot com notches por Complexidade
plt.figure(figsize=(8, 6))
df.boxplot(column='Media_Desempenho', by='Complexidade', notch=True, grid=False)
plt.title('Boxplot com Notch por Complexidade')
plt.suptitle('')
plt.xlabel('Complexidade')
plt.ylabel('Desempenho')
plt.show()
#%%
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.boxplot(x='Complexidade', y='Media_Desempenho', hue='grupo', data=df, notch=True)
plt.title('Boxplot com Notch por Complexidade e Grupo')
plt.xlabel('Complexidade')
plt.ylabel('Desempenho')
plt.legend(title='Grupo')
plt.show()
#%%
#--------------------------------------- Boxplot por Grupo + Complexidade ---------------------------------------
import matplotlib.pyplot as plt

# Agrupamento por Complexidade e Grupo (ex: visao_4, visao_6 etc.)
df['grupo_complexidade'] = df['grupo'].astype(str) + '_C' + df['Complexidade'].astype(str)

# Lista ordenada de grupos para manter a ordem
grupos_ordenados = sorted(df['grupo_complexidade'].unique())

# Dados organizados por grupo
dados_por_grupo = [df[df['grupo_complexidade'] == grupo]['Media_Desempenho'] for grupo in grupos_ordenados]

# Plot
plt.figure(figsize=(12, 6))
plt.boxplot(dados_por_grupo, notch=True, labels=grupos_ordenados)
plt.title('Boxplot com Notch por Grupo + Complexidade')
plt.xlabel('Grupo e Complexidade')
plt.ylabel('Desempenho')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# %% Salvando em .mat para mandar para o Jean   

from scipy.io import savemat

protA_cv = protA_cv_df.copy()
protA_sv = protA_sv_df.copy()

for indiv in protA_cv_df.columns:
    for rep in protA_cv_df[indiv].index:
        protA_cv[indiv][rep] = protA_cv_df[indiv][rep].drop(columns=['Score total', 
                                                                     'Score Parcial',
                                                                     'Trajetória Simplificada',
                                                                     'Média Acurácia',
                                                                     'Precisão',
                                                                     'Média Prec',
                                                                     'Recall',
                                                                     'Média Recall',
                                                                     'Média FPR',
                                                                     'Média Similaridade',
                                                                     'Desempenho ponderado com proporção']).to_dict("list")

for indiv in protA_sv_df.columns:
    for rep in protA_sv_df[indiv].index:
        protA_sv[indiv][rep] = protA_sv_df[indiv][rep].drop(columns=['Score total', 
                                                                     'Score Parcial',
                                                                     'Trajetória Simplificada',
                                                                     'Média Acurácia',
                                                                     'Precisão',
                                                                     'Média Prec',
                                                                     'Recall',
                                                                     'Média Recall',
                                                                     'Média FPR',
                                                                     'Média Similaridade',
                                                                     'Desempenho ponderado com proporção']).to_dict("list")
protA_cv = protA_cv.to_dict("list")
protA_sv = protA_sv.to_dict("list")

protA_cv_concat = df_concat_protA_cv.copy()
protA_sv_concat = df_concat_protA_sv.copy()

protA_cv_concat = protA_cv_concat.drop(columns=['Score total', 
                                                'Score Parcial',
                                                'Trajetória Simplificada',
                                                'Média Acurácia',
                                                'Precisão',
                                                'Média Prec',
                                                'Recall',
                                                'Média Recall',
                                                'Média FPR',
                                                'Média Similaridade',
                                                'Desempenho ponderado com proporção']).to_dict("list")

protA_sv_concat = protA_sv_concat.drop(columns=['Score total', 
                                                'Score Parcial',
                                                'Trajetória Simplificada',
                                                'Média Acurácia',
                                                'Precisão',
                                                'Média Prec',
                                                'Recall',
                                                'Média Recall',
                                                'Média FPR',
                                                'Média Similaridade',
                                                'Desempenho ponderado com proporção']).to_dict("list")


desempenho_protA_cv = desempenho_A_cv.copy()
desempenho_protA_sv = desempenho_A_sv.copy()
for desempenhos in desempenho_A_cv.keys():
    desempenho_protA_cv[desempenhos] = desempenho_A_cv[desempenhos].drop(columns=['Media_Desempenho_Ponderado']).to_dict("list")
    desempenho_protA_sv[desempenhos] = desempenho_A_sv[desempenhos].drop(columns=['Media_Desempenho_Ponderado']).to_dict("list")

desempenho_concat_comp_overlap = df.copy()
desempenho_concat_comp_overlap = desempenho_concat_comp_overlap.drop(columns=['Media_Desempenho_Ponderado']).to_dict("list")

import unicodedata

def remover_acentos(texto):
    # Remove acentos e normaliza para ASCII
    texto_ascii = unicodedata.normalize('NFKD', str(texto)).encode('ASCII', 'ignore').decode('ASCII')
    # Substitui espaços por underline
    texto_formatado = texto_ascii.replace(' ', '_')
    return texto_formatado

def normalizar_dicionario(d):
    if isinstance(d, dict):
        return {remover_acentos(str(k)): normalizar_dicionario(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [normalizar_dicionario(i) for i in d]
    else:
        return d

protA_cv = normalizar_dicionario(protA_cv)
protA_sv = normalizar_dicionario(protA_sv)
protA_cv_concat = normalizar_dicionario(protA_cv_concat)
protA_sv_concat = normalizar_dicionario(protA_sv_concat)
desempenho_protA_cv = normalizar_dicionario(desempenho_protA_cv)
desempenho_protA_sv = normalizar_dicionario(desempenho_protA_sv)
desempenho_concat_comp_overlap = normalizar_dicionario(desempenho_concat_comp_overlap)  

#Salvando os dados em um arquivo .mat
"""savemat('protA_cv.mat', protA_cv)
savemat('protA_sv.mat', protA_sv)
savemat('protA_cv_concat.mat', protA_cv_concat)
savemat('protA_sv_concat.mat', protA_sv_concat)
savemat('desempenho_protA_cv.mat', desempenho_protA_cv)
savemat('desempenho_protA_sv.mat', desempenho_protA_sv)
savemat('desempenho_concat_comp_overlap.mat', desempenho_concat_comp_overlap)"""


# %%Teste de se há diferença entre as variâncias 
# teste de Levene 
from scipy.stats import levene
# Teste de Levene para verificar a homogeneidade das variâncias
from itertools import product

comps = df['Complexidade'].unique()
overs = df['Overlap'].unique()

for c, o in product(comps, overs):
    subset = df[(df['Complexidade'] == c) & (df['Overlap'] == o)]
    sv = subset[subset['grupo'] == 'SV']['Media_Desempenho']
    cv = subset[subset['grupo'] == 'CV']['Media_Desempenho']
    
    stat, p = levene(sv, cv)
    print(f"Levene (Comp={c}, Overlap={o}): stat={stat:.3f}, p={p:.3f}")
    print (f"Resultado: {'Variâncias homogêneas' if p > 0.05 else 'Variâncias heterogêneas'}\n")
    print('-'*100)

#%% 