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

#prec = []
#rcll = []
propx = []
propy = []
acur = []
fpr =[]
sim = []
desempenho = []
desempenho_norm = []

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
    parametro = f'Desempenho médio {tipo_desempenho}'
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
#%% Plotando os desempenhos de todo o protocolo A CV e SV

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
alpha = 0.05

print('---'*100)
print('Teste de Normalidade dos desempenhos do Protocolo A CV')
print('---'*100)

resultado_normalidade_shapiro_cv = []
resultado_normalidade_shapiro_sv = []
resultado_normalidade_kstest_cv = []
resultado_normalidade_kstest_sv = []

for tipo_desempenho in desempenho_A_cv.keys():
    
    resultado = ev.teste_normalidade_shapiro(desempenho_A_cv[tipo_desempenho],
                                          titulo=tipo_desempenho,alpha=alpha)
    print(resultado)
    resultado_normalidade_shapiro_cv.append(resultado)
    
    resultado = ev.teste_normalidade_kstest(desempenho_A_cv[tipo_desempenho],
                                          titulo=tipo_desempenho,alpha=alpha)
    print(resultado)
    resultado_normalidade_kstest_cv.append(resultado)

print('\n')
print('---'*100)
print('Teste de Normalidade dos desempenhos do Protocolo A SV')
print('---'*100)

for tipo_desempenho in desempenho_A_sv.keys():
    resultado = ev.teste_normalidade_shapiro(desempenho_A_sv[tipo_desempenho],
                                          titulo=tipo_desempenho,alpha=alpha)
    print(resultado)
    resultado_normalidade_shapiro_sv.append(resultado)
    
    resultado = ev.teste_normalidade_kstest(desempenho_A_sv[tipo_desempenho],
                                          titulo=tipo_desempenho,alpha=alpha)
    print(resultado)
    resultado_normalidade_kstest_sv.append(resultado)

#Normalidade de todo o protocolo A CV e SV
from scipy.stats import shapiro, kstest

print('\n')
print('---'*100)
print('Teste de Normalidade dos desempenhos de todo Protocolo A CV')
print('---'*100)
w_cv, p_shapiro_cv = shapiro(df_concat_protA_cv['Desempenho'])
print(f'Estatística W: {w_cv}, p-valor: {p_shapiro_cv}')
print(f"✅Normal (alpha={alpha})" if p_shapiro_cv > alpha else f"❌Não normal (alpha={alpha})")
d_cv, p_kstest_cv = kstest(df_concat_protA_cv['Desempenho'], 'norm', args=(df_concat_protA_cv['Desempenho'].mean(), df_concat_protA_cv['Desempenho'].std()))
print(f'Estatística D: {d_cv}, p-valor: {p_kstest_cv}')
print(f"✅Normal (alpha={alpha})" if p_kstest_cv > alpha else f"❌Não normal (alpha={alpha})")   

print('\n')
print('---'*100)
print('Teste de Normalidade dos desempenhos de todo Protocolo A SV')
print('---'*100)
w_sv, p_shapiro_sv = shapiro(df_concat_protA_sv['Desempenho'])
print(f'Estatística W: {w_sv}, p-valor: {p_shapiro_sv}')
print(f"✅Normal (alpha={alpha})" if p_shapiro_sv > alpha else f"❌Não normal (alpha={alpha})")
d_sv, p_kstest_sv = kstest(df_concat_protA_sv['Desempenho'], 'norm', args=(df_concat_protA_sv['Desempenho'].mean(), df_concat_protA_sv['Desempenho'].std()))
print(f'Estatística D: {d_sv}, p-valor: {p_kstest_sv}')
print(f"✅Normal (alpha={alpha})" if p_kstest_sv > alpha else f"❌Não normal (alpha={alpha})")

#%% ------------- Seleção dos dados que iremos fazer o teste de homogeneidade e ANOVA ---------------

'''
Para testar homogeneidade de variância e futuramente ANOVA, vamos utilizar um Data Frame que 
engloba os dois procolos (A CV e A SV). 
Esse Data Frame será criado a partir dos desempenhos médios agrupados
por complexidade e por overlap.
'''

df_cv = desempenho_A_cv['por_complexidade_por_overlap']
df_cv['grupo'] = 'CV'
df_sv = desempenho_A_sv['por_complexidade_por_overlap']
df_sv['grupo'] = 'SV'
df_desempenho_protA = pd.concat([df_cv, df_sv], axis=0, ignore_index=True)
df_desempenho_protA['grupo'] = df_desempenho_protA['grupo'].astype('category')
df_desempenho_protA['Complexidade'] = df_desempenho_protA['Complexidade'].astype('category')
df_desempenho_protA['Overlap'] = df_desempenho_protA['Overlap'].astype('category')
df_desempenho_protA['Media_Desempenho'] = df_desempenho_protA['Media_Desempenho'].astype(float)

df = df_desempenho_protA.copy()

#%% Plotando os boxplots
# Boxplot com notches por Complexidade
plt.figure(figsize=(8, 6))
df.boxplot(column='Media_Desempenho', by='Complexidade', notch=True, grid=False)
plt.title('Boxplot com Notch por Complexidade')
plt.suptitle('')
plt.xlabel('Complexidade')
plt.ylabel('Desempenho')
plt.show()

import seaborn as sns
plt.figure(figsize=(10, 6))
sns.boxplot(x='Complexidade', y='Media_Desempenho', hue='grupo', data=df, notch=True)
plt.title('Boxplot com Notch por Complexidade e Grupo')
plt.xlabel('Complexidade')
plt.ylabel('Desempenho')
plt.legend(title='Grupo')
plt.show()

# Boxplot por Grupo + Complexidade
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

#%% --------------------------------------- Teste de Homogeneidade de Variância ---------------------------------------
# teste de Levene 
from scipy.stats import levene
from itertools import product

comps = df['Complexidade'].unique()
overs = df['Overlap'].unique()

for c, o in product(comps, overs):
    subset = df[(df['Complexidade'] == c) & (df['Overlap'] == o)]
    sv = subset[subset['grupo'] == 'SV']['Media_Desempenho']
    cv = subset[subset['grupo'] == 'CV']['Media_Desempenho']
    
    stat, p = levene(sv, cv)
    print(f"Levene (Comp={c}, Overlap={o}): stat={stat:.3f}, p={p:.3f}")
    print (f"Resultado: {'Variâncias homogêneas ✅' if p > 0.05 else 'Variâncias heterogêneas ❌'}\n")
    print('-'*100)

#%% --------------------------------------- Teste de hipótese por ANOVA ---------------------------------------

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import MultiComparison
from statsmodels.stats.libqsturng import psturng  # usado internamente

# Assumindo que 'desempenho' é contínua, e grupo, complexidade e overlap são fatores:
modelo = ols('Media_Desempenho ~ grupo * Complexidade * Overlap', data=df).fit()
anova = sm.stats.anova_lm(modelo, typ=3)  
print("Resultados da ANOVA:")
print(anova)
print("-"*100)

'''
Como na ANOVA, vemmos que o único fator que é significativo é a junção entre grupo e complexidade,
ou seja, a interação entre o grupo (CV ou SV) e a complexidade da tarefa, então
faremos um post hoc para explorar essa interação mais a fundo.
'''
df['grupo_complexidade'] = df['grupo'].astype(str) + '_C' + df['Complexidade'].astype(str)

# Comparar as médias de desempenho entre as complexidade:
mc = MultiComparison(df['Media_Desempenho'], df['grupo_complexidade'])
resultado = mc.tukeyhsd()
#print(resultado)

# Print do resumo das comparações
print(resultado.summary())

#%% -------------------- ANOVA sem agrupar (com todo o desempenho) --------------------
df_concat_protA_cv['grupo'] = 'CV'
df_concat_protA_sv['grupo'] = 'SV'
df_protA = pd.concat([df_concat_protA_cv, df_concat_protA_sv], axis=0, ignore_index=True)
df_protA['grupo'] = df_protA['grupo'].astype('category')

df_protA['Especificidade'] = 1 - df_protA['Taxa de Falsos Positivos']

modelo = ols('Desempenho ~ C(grupo) * C(Complexidade) * C(Overlap)', data=df_protA).fit()
anova = sm.stats.anova_lm(modelo, typ=2)
print("Resultados da ANOVA sem agrupar:")
print(anova)

mc = MultiComparison(df_protA['Desempenho'], df_protA['Complexidade'])
resultado = mc.tukeyhsd()
print(resultado.summary())

#%% #Boxplots
# Boxplots do desempenho sem agrupar
sns.boxplot(x='Complexidade', y='Desempenho', hue='grupo', data=df_protA, notch=True)
plt.title('Boxplot com Notch por Complexidade e Grupo (Desempenho)')
plt.xlabel('Complexidade')
plt.ylabel('Desempenho')
plt.yticks(np.arange(0, 1.5, 0.1))
plt.legend(title='Grupo')

# Boxplot do desempenho por Overlap e Grupo
plt.figure(figsize=(10, 6))
sns.boxplot(x='Overlap', y='Desempenho', hue='grupo', data=df_protA, notch=True)
plt.title('Boxplot com Notch por Overlap e Grupo (Desempenho)')
plt.xlabel('Overlap')
plt.ylabel('Desempenho')
plt.yticks(np.arange(0, 1.5, 0.1))
plt.legend(title='Grupo')

# Boxplot do desempenho por grupo de acordo com a complexidade
plt.figure(figsize=(10, 6))
sns.boxplot(x='grupo', y='Desempenho', hue='Complexidade', data=df_protA, notch=True, palette='Set2')
plt.title('Boxplot com Notch do Desempenho do Grupo de acordo com a Complexidade')
plt.xlabel('Grupo')
plt.ylabel('Desempenho')
plt.yticks(np.arange(0, 1.5, 0.1))
plt.legend(title='Complexidade')
#%% -------------------- ANOVA depois de fazer PCA --------------------
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

X = df_protA[['Acuracia','Especificidade','Similaridade','Proporção espacial x','Proporção espacial y']]  

# Padronizar os dados
X_scaled = StandardScaler().fit_transform(X)

# Aplicar PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)
#X_pca = pca.fit_transform(X)

# Variância explicada
exp_var = pca.explained_variance_ratio_
cum_var = np.cumsum(exp_var)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(exp_var) + 1), exp_var, marker='o', label='Variância explicada (individual)')
plt.plot(range(1, len(cum_var) + 1), cum_var, marker='s', linestyle='--', color='orange', label='Variância acumulada')

# Marcar o ponto onde a variância acumulada atinge 90%
for i, v in enumerate(cum_var):
    if v >= 0.90:
        plt.axvline(x=i + 1, color='red', linestyle='--', label='90% da variância')
        break

plt.title('Scree Plot e Variância Acumulada')
plt.xlabel('Número de Componentes Principais')
plt.ylabel('Proporção da Variância Explicada')
plt.xticks(range(1, len(exp_var) + 1))
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Mostrar em formato de tabela
for i, (v, c) in enumerate(zip(exp_var, cum_var), 1):
    print(f'PC{i}: {v:.4f} ({c:.2%} acumulado)')

print('---'*100)
print('Quanto cada variável pesa em cada componente principal:')
print('| Acurácia | Especificidade | Similaridade | Prop x | Prop y |')
print(pca.components_)
print('---'*100)

#    Fazendo ANOVA com PCA
df_protA['PC1'] = X_pca[:, 0]  # Projeção no primeiro componente principal

modelo = ols('PC1 ~ grupo * Complexidade * Overlap', data=df_protA).fit()
anova = sm.stats.anova_lm(modelo, typ=3)
print(anova)

"""mc = MultiComparison(df_protA['PC1'], df_protA['Overlap'])
resultado = mc.tukeyhsd()
#print(resultado)

# Print do resumo das comparações
print(resultado.summary())"""

"""
#Boxplot com Notch para comparar
plt.figure(figsize=(10, 6))
sns.boxplot(x='Complexidade', y='PC1', hue='grupo', data=df_protA, notch=True)
plt.title('Boxplot com Notch por Complexidade e Grupo')
plt.xlabel('Complexidade')
plt.ylabel('PC1')
plt.legend(title='Grupo')
plt.show()"""

#%% -------------------- ANOVA depois de fazer LDA --------------------
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Selecionar variáveis contínuas
X = df_protA[['Acuracia', 'Especificidade', 'Similaridade', 'Proporção espacial x', 'Proporção espacial y']]
y = df_protA['grupo']

# Padronizar os dados
X_scaled = StandardScaler().fit_transform(X)

# Aplicar CDA / LDA
lda = LinearDiscriminantAnalysis()
X_lda = lda.fit_transform(X_scaled, y)

# Variância explicada por cada discriminante
exp_var_lda = lda.explained_variance_ratio_
cum_var_lda = np.cumsum(exp_var_lda)

df_protA['LD1'] = X_lda[:, 0]  # Projeção no primeiro discriminante


# Plotar a projeção dos grupos ao longo do LD1
plt.figure(figsize=(10, 5))
for group in df_protA['grupo'].cat.categories:
    plt.hist(df_protA[df_protA['grupo'] == group]['LD1'], bins=20, alpha=0.6, label=group)
plt.title('Projeção dos grupos na LD1 (Discriminante Canônico)')
plt.xlabel('LD1')
plt.ylabel('Contagem')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Coeficientes de cada variável na LD1
coef_ld1 = pd.DataFrame({'Variável': X.columns, 'Coeficiente_LD1': lda.coef_[0]})
print("Coeficientes da LD1:")
print(coef_ld1)

#    Fazendo ANOVA com LDA

modelo = ols('LD1 ~ grupo * Complexidade * Overlap', data=df_protA).fit()
anova = sm.stats.anova_lm(modelo, typ=3)
print(anova)


#Boxplot com Notch para comparar
plt.figure(figsize=(10, 6))
sns.boxplot(x='Complexidade', y='LD1', hue='grupo', data=df_protA, notch=True)
plt.title('Boxplot com Notch por Complexidade e Grupo')
plt.xlabel('Complexidade')
plt.ylabel('LD1')
plt.legend(title='Grupo')
plt.show()

#%% ------------------- Teste não paramétrico de Kruskal-Wallis -------------------
from scipy.stats import kruskal
import scikit_posthocs as sp

# Entre grupos

print('---'*100)
print('Teste de Kruskal-Wallis para Grupos') 
stat, p_value = kruskal(*[group['Desempenho'].values for name, group in df_protA.groupby('grupo')])
print(f'Estatística de Kruskal-Wallis: {stat}, p-valor: {p_value}')
if p_value < 0.05:
    print("Há diferenças significativas entre os grupos (grupo) no desempenho.") 
else:
    print("Não há diferenças significativas entre os grupos (grupo) no desempenho.")

posthoc = sp.posthoc_dunn(df_protA, val_col='Desempenho', group_col='grupo', p_adjust='bonferroni')
print(posthoc)

df_protA.boxplot(column='Desempenho', by='grupo', grid=False, notch=True)

# Entre Complexidades
print('---'*100)
print('Teste de Kruskal-Wallis para Complexidade')
stat, p_value = kruskal(*[group['Desempenho'].values for name, group in df_protA.groupby('Complexidade')])
print(f'Estatística de Kruskal-Wallis: {stat}, p-valor: {p_value}')
if p_value < 0.05:
    print("Há diferenças significativas entre os grupos (Complexidade) no desempenho.") 
else:
    print("Não há diferenças significativas entre os grupos (Complexidade) no desempenho.")

posthoc = sp.posthoc_dunn(df_protA, val_col='Desempenho', group_col='Complexidade', p_adjust='bonferroni')
print(posthoc)

df_protA.boxplot(column='Desempenho', by='Complexidade', grid=False, notch=True)

#para Overlap
print('---'*100)
print('Teste de Kruskal-Wallis para Overlap')
stat, p_value = kruskal(*[group['Desempenho'].values for name, group in df_protA.groupby('Overlap')])
print(f'Estatística de Kruskal-Wallis: {stat}, p-valor: {p_value}')
if p_value < 0.05:
    print("Há diferenças significativas entre os grupos (Overlap) no desempenho.")
else:
    print("Não há diferenças significativas entre os grupos (Overlap) no desempenho.")
posthoc = sp.posthoc_dunn(df_protA, val_col='Desempenho', group_col='Overlap', p_adjust='bonferroni')
print(posthoc)

#%% Teste de hipótese não paramétrico de Friedman
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp
# Agrupar os dados por ID e calcular a média do desempenho para cada grupo
#df_friedman = df_protA.groupby(['Overlap', 'Complexidade','grupo'])['Desempenho'].mean().unstack().unstack()
#df_friedman = df_protA.groupby(['Overlap', 'Complexidade','grupo'])['Desempenho'].mean().unstack('grupo').unstack('Overlap')
df_friedman = df_protA.groupby(['ID','Overlap', 'Complexidade'])['Desempenho'].mean().unstack('Complexidade').unstack('Overlap')
"""print(df_friedman.head())
df_friedman = df_friedman.dropna()  # Remover linhas com valores NaN
print(df_friedman.head())"""
# Realizar o teste de Friedman
stat, p_value = friedmanchisquare(*[df_friedman[col] for col in df_friedman.columns])
print(f'Estatística de Friedman: {stat}, p-valor: {p_value}')
if p_value < 0.05:
    print("Há diferenças significativas no desempenho.")

    #Post-hoc de Nemenyi

    #passando os dados para o formato correto
    df_friedman .columns = [f'C{cx}|O{ov}' for cx, ov in df_friedman.columns]
    df_friedman.reset_index('ID').melt(id_vars='ID', var_name='Condicao', value_name='Desempenho')
    #fazendo o post-hoc
    posthoc = sp.posthoc_nemenyi_friedman(df_friedman)
    #print(posthoc)

    sns.heatmap(posthoc, annot=True, fmt=".3f", 
    cmap="Reds", cbar_kws={"label": "p-valor"}, center=0.05)
    plt.title("Post-hoc de Nemenyi (p-valores)")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
else:
    print("Não há diferenças significativas no desempenho.")

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



#%% 
# %%
