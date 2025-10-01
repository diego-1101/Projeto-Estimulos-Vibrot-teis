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

#%% ------------- Criação dos Data Frames Finais ---------------

'''
Criando o Data Frame que engloba os dois procolos (A CV e A SV).
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

 # Criando o Data Frame do protocolo A sem sem agrupar
df_concat_protA_cv['grupo'] = 'CV'
df_concat_protA_sv['grupo'] = 'SV'
df_protA = pd.concat([df_concat_protA_cv, df_concat_protA_sv], axis=0, ignore_index=True)
df_protA['grupo'] = df_protA['grupo'].astype('category')

df_protA['Especificidade'] = 1 - df_protA['Taxa de Falsos Positivos']

df_protA['grupo_complexidade'] = df_protA['grupo'].astype(str) + '_C' + df_protA['Complexidade'].astype(str)
df_protA['grupo_overlap'] = df_protA['grupo'].astype(str) + '_O' + df_protA['Overlap'].astype(str)
df_protA['grupo_complexidade_overlap'] = df_protA['grupo'].astype(str) + '_C' + df_protA['Complexidade'].astype(str) + '_O' + df_protA['Overlap'].astype(str)


df_protA.to_csv('df_protA.csv', index=False)

'''#%%# Plotando os desempenhos médios caso queira

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
    
# Plotando os desempenhos de todo o protocolo A CV e SV

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

"""'''