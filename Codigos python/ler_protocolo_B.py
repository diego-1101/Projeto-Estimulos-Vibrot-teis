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


def transformar_protB_mat_em_df(protocolo = [], id = []):
    """
    Transforma a estrutura MATLAB ProtB (carregada com scipy.io.loadmat) em um DataFrame de DataFrames
    organizado por participante e subdividido em três repetições experimentais.

    A função é voltada para o Protocolo B do experimento, que envolve:
    - Visualização de uma trajetória em vídeo
    - Reprodução manual da trajetória com e sem feedback vibro-tátil
    Cada participante realiza três repetições do protocolo.

    A função organiza os dados de cada tentativa em DataFrames, convertendo os campos aninhados do MATLAB
    (como listas em string e arrays) em formatos Python nativos (listas e floats), para facilitar a análise.

    Parâmetros:
    ----------
    protocolo : np.ndarray
        Estrutura carregada do arquivo .mat referente ao ProtB, normalmente de tamanho (n_individuos, 3)
    id : list of str
        Lista com os identificadores dos participantes, na ordem das linhas de `protocolo`

    Retorna:
    -------
    prot_df : pd.DataFrame
        DataFrame com as colunas sendo os participantes (`df_ID_XX`) e três linhas por coluna:
        - 'Rep1', 'Rep2', 'Rep3': DataFrames com as tentativas de cada repetição do experimento
          As colunas de cada DataFrame são:
            ['Número da Trajetória', 'Sorteio', 'Tempo 1', 'Tempo 2', 'Tempo 3',
             'Score total', 'Score Parcial', 'Proporção espacial x', 'Proporção espacial y',
             'Trajetória Completa', 'Trajetória Simplificada']

    Observações:
    -----------
    - Trajetórias vazias ou inválidas são preenchidas com o código [9] (indicando que o participante ficou parado).
    - As listas codificadas como strings no MATLAB são convertidas para listas reais de inteiros com `ast.literal_eval`.
    - As repetições são assumidas como organizadas sequencialmente na matriz de entrada (Rep1, Rep2, Rep3 por ID).
    """

    import pandas as pd
    import ast

    # Criar uma lista para armazenar os DataFrames
    dataframes = []

    # Headers das colunas
    headers = ['Número da Trajetória', 'Sorteio', 'Tempo 1',
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
    Antes, cada célula estava com um array dentro de outro array, essa parte
    é para simplificar as coisas e deixar um DataFrame mais amigável
    '''
    for i in range(len(dataframes)):
        for j in range(dataframes[i]['DataFrame']['Trajetória Completa'].shape[0]):
            if(dataframes[i]['DataFrame']['Trajetória Completa'][j] != '[, 9]' and dataframes[i]['DataFrame']['Trajetória Simplificada'][j] != '[, 9]'):
                dataframes[i]['DataFrame']['Número da Trajetória'][j] = dataframes[i]['DataFrame']['Número da Trajetória'][j][0]
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


#ID dos pacientes naquele protocolo
id_cf = ['01', '04', '05', '12', '18', '25', '26', '33', '38', '42']
id_sf = ['02', '03', '13', '15', '28', '29', '32', '40', '43']

# Carregar o arquivo .mat do protocolo 
B_CF = loadmat('Aquivos mat\ProtB_CF.mat')
B_SF = loadmat('Aquivos mat\ProtB_SF.mat')

# Acessar o conteúdo de ProtB_SF
ProtB_CF = B_CF['ProtB_SF']  # ProtB_SF é uma célula de células
ProtB_SF = B_SF['ProtB_SF'] # ProtB_SF é uma célula de células (ta so com o mesmo nome do CV, mas não é a mesma coisa)

# Convertendo para um DataFrame
protB_cf_df = transformar_protB_mat_em_df(protocolo = ProtB_CF, id = id_cf)
protB_sf_df = transformar_protB_mat_em_df(protocolo = ProtB_SF, id = id_sf)


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

#%% Remover indivíduo 1 do protocolo B CF caso necessário
'''
    Verificando nos códigos do matlab, vi que o indivíduo 1 foi removido do protocolo C e do B.
    Provavelmente porque não houve algum erro durante a execução do protocolo c que acabou 
não gravando as posições que esse indivíduo estava executando.

    Por isso, fiz essa parte para remover (caso necessário) o indivíduo 1 antes de fazer as análises das métricas.
'''
'''novo_B_cf_df = protB_cf_df.drop(columns='df_ID_01')
#protB_cf_df = protB_cf_df.drop(columns='df_ID_01')'''

#%% ---------------------- Calculando as métricas em cima de cada protocolo -------------------

'''Calculando os resultados das métricas de comparação de trajetória para todas as repetições
  de todos os individuos do protocolo B CF
'''
#sparcial =[]
#stotal = []
#prec = []
#rcll = []

propx = []
propy = []
acur = []
fpr =[]
sim = []
desempenho = []
desempenho_norm = []

for individuo in protB_cf_df.columns:
    for rep in protB_cf_df.index:
        teste = protB_cf_df[individuo][rep]
    
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
            
            '''
            # 1) Avaliar o match perfeito para poder normalizar depois
            soma_perfeita = 0
            soma_perfeita_uns = 0 
            resultado_perfeito, resultado_perfeito_uns  = ev.avaliar_match_dinamico(certo, certo)
            
            for j in resultado_perfeito:
                soma_perfeita += np.sum(j)
            
            for j in resultado_perfeito_uns:
                soma_perfeita_uns += np.sum(j)

            # 2) Avaliar o match real
            soma_real = 0
            soma_real_uns = 0

            resultado_real, resultado_real_uns = ev.avaliar_match_dinamico(seq, certo)
            
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
            '''    
            #---- Avaliando por comparação de imagem (IDEIA 3)
            resultado_ideia3 = ev.comparar_imagem(seq1=seq1,seq2=seq2,plotar_imagens = False)

            #---- Avaliando por similaridade com correlação cruzada normalizada (IDEIA 4)
            resultado_ideia4 = ev.calcular_similaridade(seq1,seq2)
            
            #sparcial.append(score_parcial_uns)
            #stotal.append(score_total_uns)
            propx.append(float(teste['Proporção espacial x'][i]))
            propy.append(float(teste['Proporção espacial y'][i]))
            acur.append(resultado_ideia3[0])
            #prec.append(resultado_ideia3[1])
            #rcll.append(resultado_ideia3[2])
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
        #protB_cf_df[individuo][rep]['Precisão'] = prec 
        #protB_cf_df[individuo][rep]['Media Prec'] = np.mean(prec)
        #protB_cf_df[individuo][rep]['Recall'] = rcll
        #protB_cf_df[individuo][rep]['Media Recall'] = np.mean(rcll)
        protB_cf_df[individuo][rep]['Acuracia'] = acur
        protB_cf_df[individuo][rep]['Media Acuracia'] = np.mean(acur)
        protB_cf_df[individuo][rep]['Taxa de Falsos Positivos'] = fpr
        protB_cf_df[individuo][rep]['Media FPR'] = np.mean(fpr)
        protB_cf_df[individuo][rep]['Similaridade'] = sim
        protB_cf_df[individuo][rep]['Media Similaridade'] = np.mean(sim)
        protB_cf_df[individuo][rep]['Desempenho'] = desempenho
        protB_cf_df[individuo][rep]['Desempenho ponderado com proporção'] = desempenho_norm
        
        #Reiniciando as listas para a próxima iteração
        #prec = []
        #rcll = []
        acur = []
        fpr =[]
        sim = []
        desempenho = []
        desempenho_norm = []

#---- Protocolo B SF ----
'''Calculando os resultados das métricas de comparação de trajetória para todas as repetições
  de todos os individuos do protocolo A SF
'''
#sparcial =[]
#stotal = []
#prec = []
#rcll = []
propx = []
propy = []
acur = []
fpr =[]
sim = []
desempenho = []
desempenho_norm = []

for individuo in protB_sf_df.columns:
    for rep in protB_sf_df.index:
        teste = protB_sf_df[individuo][rep]

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
        #protB_sf_df[individuo][rep]['Precisão'] = prec 
        #protB_sf_df[individuo][rep]['Media Prec'] = np.mean(prec)
        #protB_sf_df[individuo][rep]['Recall'] = rcll
        #protB_sf_df[individuo][rep]['Media Recall'] = np.mean(rcll)
        protB_sf_df[individuo][rep]['Acuracia'] = acur
        protB_sf_df[individuo][rep]['Media Acuracia'] = np.mean(acur)
        protB_sf_df[individuo][rep]['Taxa de Falsos Positivos'] = fpr
        protB_sf_df[individuo][rep]['Media FPR'] = np.mean(fpr)
        protB_sf_df[individuo][rep]['Similaridade'] = sim
        protB_sf_df[individuo][rep]['Media Similaridade'] = np.mean(sim)
        protB_sf_df[individuo][rep]['Desempenho'] = desempenho
        protB_sf_df[individuo][rep]['Desempenho ponderado com proporção'] = desempenho_norm
        
        #Reiniciando as listas para a próxima iteração
        #prec = []
        #rcll = []
        acur = []
        fpr =[]
        sim = []
        desempenho = []
        desempenho_norm = []


#%% # Vendo a distribuição dos resultados obtidos acima 
"""resultados_B_CF = {
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

resultados_B_CF_df = pd.DataFrame(resultados_B_CF)

#Plotando as distribuições 
ev.plotar_distribuicoes_resultados(resultados_B_CF_df, titulo = '(Protocolo B CF)')"""

# Vendo a distribuição dos resultados obtidos acima 
"""resultados_B_SF = {
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

resultados_B_SF_df = pd.DataFrame(resultados_B_SF)

#Plotando as distribuições 
ev.plotar_distribuicoes_resultados(resultados_B_SF_df, titulo = '(Protocolo B SF)')"""
#%% ------- Criando o DataFrame para os desempenhos medios em cada protocolo ---------

lista_concatenada = []

for individuo in protB_cf_df.columns:
    for rep_label in protB_cf_df[individuo].index:  # 'Rep1', 'Rep2', 'Rep3'
        df_tmp = protB_cf_df[individuo][rep_label].copy()
        df_tmp['ID'] = individuo
        df_tmp['Repeticao'] = rep_label
        lista_concatenada.append(df_tmp)

df_concat_protB_cf = pd.concat(lista_concatenada, ignore_index=True)

lista_concatenada = []

for individuo in protB_sf_df.columns:
    for rep_label in protB_sf_df[individuo].index:  # 'Rep1', 'Rep2', 'Rep3'
        df_tmp = protB_sf_df[individuo][rep_label].copy()
        df_tmp['ID'] = individuo
        df_tmp['Repeticao'] = rep_label
        lista_concatenada.append(df_tmp)

df_concat_protB_sf = pd.concat(lista_concatenada, ignore_index=True)

# Adicionando a complexidade
df_concat_protB_cf['Complexidade'] = df_concat_protB_cf['Número da Trajetória'].apply(ev.map_complexidade)
df_concat_protB_sf['Complexidade'] = df_concat_protB_sf['Número da Trajetória'].apply(ev.map_complexidade)

# Tipagem correta
df_concat_protB_cf['ID'] = df_concat_protB_cf['ID'].astype(str)
df_concat_protB_sf['ID'] = df_concat_protB_sf['ID'].astype(str)
df_concat_protB_cf['Número da Trajetória'] = df_concat_protB_cf['Número da Trajetória'].astype(int)
df_concat_protB_sf['Número da Trajetória'] = df_concat_protB_sf['Número da Trajetória'].astype(int)

ids_cf = df_concat_protB_cf['ID'].unique()
ids_sf = df_concat_protB_sf['ID'].unique()
complexidades = sorted(df_concat_protB_cf['Complexidade'].unique())
trajetorias = sorted(df_concat_protB_cf['Número da Trajetória'].unique())

desempenho_B_cf = ev.calcular_desempenhos_medios(df_concat_protB_cf, ids_cf, complexidades, overlaps =None, trajetorias = trajetorias)
desempenho_B_sf = ev.calcular_desempenhos_medios(df_concat_protB_sf, ids_sf, complexidades, overlaps = None, trajetorias= trajetorias)


#%% Plotando os desempenhos médios do Protocolo B CF e SF
print('---'*100)
print('Desempenhos médios do Protocolo B CF')
print('---'*100)
for tipo_desempenho in desempenho_B_cf.keys():
    parametro = f'Desempenho médio {tipo_desempenho}'
    titulo = 'do Protocolo B CF'
    data = desempenho_B_cf[tipo_desempenho]['Media_Desempenho']
    ev.plotar_desempenhos(data, titulo, parametro)

print('---'*100)
print('Desempenhos médios do Protocolo B SF')
print('---'*100)
for tipo_desempenho in desempenho_B_sf.keys():
    parametro = f'Desempenho médio {tipo_desempenho}'
    titulo = 'do Protocolo B SF'
    data = desempenho_B_sf[tipo_desempenho]['Media_Desempenho']
    ev.plotar_desempenhos(data, titulo, parametro)

#%% -------- Plotando o desempenho global por grupo --------

print('---'*100)
print('Desempenhos de todo Protocolo B CF')
print('---'*100)
ev.plotar_desempenhos(df_concat_protB_cf['Desempenho'], 'Desempenho de todo Protocolo B CF', 'Desempenho')

print('---'*100)
print('Desempenhos de todo Protocolo B SF')
print('---'*100)
ev.plotar_desempenhos(df_concat_protB_sf['Desempenho'], 'Desempenho de todo Protocolo B SF', 'Desempenho')

# %% Teste de normalidade
alpha = 0.05

print('---'*100)
print('Teste de Normalidade dos desempenhos do Protocolo B CF')
print('---'*100)

resultado_normalidade_shapiro_cf = []
resultado_normalidade_shapiro_sf = []
resultado_normalidade_kstest_cf = []
resultado_normalidade_kstest_sf = []

for tipo_desempenho in desempenho_B_cf.keys():
    resultado = ev.teste_normalidade_shapiro(desempenho_B_cf[tipo_desempenho],
                                             titulo=tipo_desempenho, alpha=alpha)
    print(resultado)
    resultado_normalidade_shapiro_cf.append(resultado)

    resultado = ev.teste_normalidade_kstest(desempenho_B_cf[tipo_desempenho],
                                             titulo=tipo_desempenho, alpha=alpha)
    print(resultado)
    resultado_normalidade_kstest_cf.append(resultado)

print('\n')
print('---'*100)
print('Teste de Normalidade dos desempenhos do Protocolo B SF')
print('---'*100)

for tipo_desempenho in desempenho_B_sf.keys():
    resultado = ev.teste_normalidade_shapiro(desempenho_B_sf[tipo_desempenho],
                                             titulo=tipo_desempenho, alpha=alpha)
    print(resultado)
    resultado_normalidade_shapiro_sf.append(resultado)

    resultado = ev.teste_normalidade_kstest(desempenho_B_sf[tipo_desempenho],
                                             titulo=tipo_desempenho, alpha=alpha)
    print(resultado)
    resultado_normalidade_kstest_sf.append(resultado)

#Normalidade d etodo o protocolo B CF e SV
from scipy.stats import shapiro, kstest

print('\n')
print('---'*100)
print('Teste de Normalidade dos desempenhos de todo Protocolo B CF')
print('---'*100)
w_cv, p_shapiro_cv = shapiro(df_concat_protB_cf['Desempenho'])
print(f'Estatística W: {w_cv}, p-valor: {p_shapiro_cv}')
print(f"✅Normal (alpha={alpha})" if p_shapiro_cv > alpha else f"❌Não normal (alpha={alpha})")
d_cv, p_kstest_cv = kstest(df_concat_protB_cf['Desempenho'], 'norm', args=(df_concat_protB_cf['Desempenho'].mean(), df_concat_protB_cf['Desempenho'].std()))
print(f'Estatística D: {d_cv}, p-valor: {p_kstest_cv}')
print(f"✅Normal (alpha={alpha})" if p_kstest_cv > alpha else f"❌Não normal (alpha={alpha})")   

print('\n')
print('---'*100)
print('Teste de Normalidade dos desempenhos de todo Protocolo B SF')
print('---'*100)
w_sv, p_shapiro_sv = shapiro(df_concat_protB_sf['Desempenho'])
print(f'Estatística W: {w_sv}, p-valor: {p_shapiro_sv}')
print(f"✅Normal (alpha={alpha})" if p_shapiro_sv > alpha else f"❌Não normal (alpha={alpha})")
d_sv, p_kstest_sv = kstest(df_concat_protB_sf['Desempenho'], 'norm', args=(df_concat_protB_sf['Desempenho'].mean(), df_concat_protB_sf['Desempenho'].std()))
print(f'Estatística D: {d_sv}, p-valor: {p_kstest_sv}')
print(f"✅Normal (alpha={alpha})" if p_kstest_sv > alpha else f"❌Não normal (alpha={alpha})")

#%% ------------- Seleção dos dados que iremos fazer o teste de homogeneidade e ANOVA ---------------

'''
Para testar homogeneidade de variância e futuramente ANOVA, vamos utilizar um Data Frame que 
engloba os dois procolos (b CF e A Sf). 
Esse Data Frame será criado a partir dos desempenhos médios agrupados
por complexidade.
'''

df_cf = desempenho_B_cf['por_complexidade']
df_cf['grupo'] = 'CF'
df_sf = desempenho_B_sf['por_complexidade']
df_sf['grupo'] = 'SF'
df_desempenho_protB = pd.concat([df_cf, df_sf], axis=0, ignore_index=True)
df_desempenho_protB['grupo'] = df_desempenho_protB['grupo'].astype('category')
df_desempenho_protB['Complexidade'] = df_desempenho_protB['Complexidade'].astype('category')
df_desempenho_protB['Media_Desempenho'] = df_desempenho_protB['Media_Desempenho'].astype(float)

df = df_desempenho_protB.copy()

#%% ------------- Teste de Homogeneidade de Variância ---------------
# teste de Levene 
from scipy.stats import levene

comps = df['Complexidade'].unique()

for c in comps:
    subset = df[(df['Complexidade'] == c)]
    sf = subset[subset['grupo'] == 'SF']['Media_Desempenho']
    cf = subset[subset['grupo'] == 'CF']['Media_Desempenho']
    
    stat, p = levene(sf, cf)
    print(f"Levene (Comp={c}): stat={stat:.3f}, p={p:.3f}")
    print (f"Resultado: {'Variâncias homogêneas ✅' if p > 0.05 else 'Variâncias heterogêneas ❌'}\n")
    print('-'*100)

#%% Plotando os Boxplots

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
plt.yticks(np.arange(0, 1.1, 0.1))
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


#%% --------------------------------------- Teste de hipótese por ANOVA ---------------------------------------

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import MultiComparison
from statsmodels.stats.libqsturng import psturng  # usado internamente

# Assumindo que 'desempenho' é contínua e grupo e complexidade  são fatores:
modelo = ols('Media_Desempenho ~ grupo * Complexidade', data=df).fit()
anova = sm.stats.anova_lm(modelo, typ=3)  
print("Resultados da ANOVA:")
print(anova)
print("-"*100)

# Comparar as médias de desempenho entre as complexidade:
mc = MultiComparison(df['Media_Desempenho'], df['Complexidade'])
resultado = mc.tukeyhsd()
#print(resultado)

# Print do resumo das comparações
print(resultado.summary())

#%% Juntando tudo em um data frame só para poder fazer outros testes

df_concat_protB_cf['grupo'] = 'CF'
df_concat_protB_sf['grupo'] = 'SF'
df_protB = pd.concat([df_concat_protB_cf, df_concat_protB_sf], ignore_index=True)
df_protB['grupo'] = df_protB['grupo'].astype('category')
df_protB['Especificidade'] = 1 - df_protB['Taxa de Falsos Positivos']
df_protB = df_protB[df_protB['ID']!= 'df_ID_01']

#%% -------- ANOVA sem agrupar por complexidade --------
from statsmodels.formula.api import ols
import statsmodels.api as sm

modelo = ols('Desempenho ~ grupo * Complexidade', data=df_protB).fit()
anova = sm.stats.anova_lm(modelo, typ=3) 
print('---'*100)
print('ANOVA sem agrupar por complexidade - Protocolo B')
print(anova)
print('---'*100)

"""# Se tiver diferença significativa, fazer o teste de Tukey
mc = MultiComparison(df_protB['Desempenho'], df_protB['grupo'])
resultado = mc.tukeyhsd()
print(resultado.summary())"""

#%% Boxplots
plt.figure(figsize=(10, 6))
sns.boxplot(x='Complexidade', y='Desempenho', hue='grupo', data=df_protB, notch=True)
plt.title('Boxplot com Notch por Complexidade e Grupo')
plt.yticks(np.arange(0, 1.5, 0.1))

plt.figure(figsize=(10, 6))
sns.boxplot(x='grupo', y='Desempenho', hue='Complexidade', data=df_protB, notch=True, palette='Set2')
plt.title('Boxplot com Notch do Desempenho por Grupo e Complexidade')
plt.yticks(np.arange(0, 1.5, 0.1))

#%% -------- PCA + ANOVA --------
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

X = df_protB[['Acuracia','Especificidade','Similaridade','Proporção espacial x','Proporção espacial y']]
X_scaled = StandardScaler().fit_transform(X)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

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

df_protB['PC1'] = X_pca[:, 0]

modelo_pca = ols('PC1 ~ grupo * Complexidade', data=df_protB).fit()
anova_pca = sm.stats.anova_lm(modelo_pca, typ=3)
print('---'*100)
print('ANOVA com PCA (PC1) - Protocolo B')
print('---'*100)
print(anova_pca)

#%%
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
X = df_protB[['Acuracia', 'Especificidade', 'Similaridade', 
              'Proporção espacial x', 'Proporção espacial y']]
y = df_protB['grupo']
X_scaled = StandardScaler().fit_transform(X)

lda = LinearDiscriminantAnalysis()
X_lda = lda.fit_transform(X_scaled, y)

df_protB['LD1'] = X_lda[:, 0]

# Plotar a projeção dos grupos ao longo do LD1
plt.figure(figsize=(10, 5))
for group in df_protB['grupo'].cat.categories:
    plt.hist(df_protB[df_protB['grupo'] == group]['LD1'], bins=20, alpha=0.6, label=group)
plt.title('Projeção dos grupos na LD1 (Discriminante Canônico)')
plt.xlabel('LD1')
plt.ylabel('Contagem')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
for group in df_protB['grupo'].cat.categories:
    plt.scatter(df_protB[df_protB['grupo'] == group]['LD1'], df_protB[df_protB['grupo'] == group]['LD1'],label=group, alpha=0.6)
plt.title('Projeção dos grupos na LD1 (Discriminante Canônico)')
plt.xlabel('LD1')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Coeficientes de cada variável na LD1
coef_ld1 = pd.DataFrame({'Variável': X.columns, 'Coeficiente_LD1': lda.coef_[0]})
print("Coeficientes da LD1:")
print(coef_ld1)

modelo_lda = ols('LD1 ~ grupo * Complexidade', data=df_protB).fit()
anova_lda = sm.stats.anova_lm(modelo_lda, typ=3)
print('---'*100)
print('ANOVA com LDA (LD1) - Protocolo B')
print('---'*100)
print(anova_lda)
#%%

plt.figure(figsize=(12, 2.5))
sns.stripplot(data=df_protB, x='LD1', hue='grupo', dodge=True, size=6, alpha=0.7, palette='Set1', orient='h', jitter=False)

plt.title('Distribuição 1D da projeção LD1 por grupo')
plt.xlabel('LD1')
plt.yticks([])
plt.legend(title='Grupo', bbox_to_anchor=(1.01, 1), borderaxespad=0)
plt.grid(True, axis='x')
plt.tight_layout()

#%%
plt.figure(figsize=(12, 4))

# Gráfico de densidade por grupo
sns.violinplot(data=df_protB, x='LD1', y='grupo', inner=None, palette='Set1', linewidth=1.2)

# Gráfico de pontos por cima (stripplot)
sns.stripplot(data=df_protB, x='LD1', y='grupo', color='k', alpha=0.5, size=5, jitter=False)

plt.title('Distribuição da projeção LD1 por grupo (Violin + Stripplot)')
plt.xlabel('LD1')
plt.ylabel('')
plt.grid(True, axis='x')
plt.tight_layout()
plt.show()

#%% Boxplots para comparar

plt.figure(figsize=(10, 6))
sns.boxplot(x='Complexidade', y='LD1', hue='grupo', data=df_protB, notch=True)
plt.title('Boxplot com Notch por Complexidade e Grupo')
plt.xlabel('Complexidade')
plt.ylabel('LD1')
plt.legend(title='Grupo')
plt.show()

#%% ---------- Teste de hipótese não paramétrico (Kruskal-Wallis) ---------------
# Entre grupos
from scipy.stats import kruskal
# Teste de Kruskal-Wallis
stat, p_value = kruskal(*[group['Desempenho'].values for name, group in df_protB.groupby('grupo')])
print(f'Estatística de Kruskal-Wallis: {stat}, p-valor: {p_value}')
if p_value < 0.05:
    print("Há diferenças significativas entre os grupos (grupo) no desempenho.") 
else:
    print("Não há diferenças significativas entre os grupos (grupo) no desempenho.")

import scikit_posthocs as sp

posthoc = sp.posthoc_dunn(df_protB, val_col='Desempenho', group_col='grupo', p_adjust='bonferroni')
print(posthoc)

df_protB.boxplot(column='Desempenho', by='grupo', grid=False, notch=True)

# Entre Complexidades

# Teste de Kruskal-Wallis
stat, p_value = kruskal(*[group['Desempenho'].values for name, group in df_protB.groupby('Complexidade')])
print(f'Estatística de Kruskal-Wallis: {stat}, p-valor: {p_value}')
if p_value < 0.05:
    print("Há diferenças significativas entre os grupos (Complexidade) no desempenho.") 
else:
    print("Não há diferenças significativas entre os grupos (Complexidade) no desempenho.")

posthoc = sp.posthoc_dunn(df_protB, val_col='Desempenho', group_col='Complexidade', p_adjust='bonferroni')
print(posthoc)

df_protB.boxplot(column='Desempenho', by='Complexidade', grid=False, notch=True)

#%% Teste de Friedman 
#%% Teste de hipótese não paramétrico de Friedman
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp
# Agrupar os dados por ID e calcular a média do desempenho para cada grupo
#df_friedman = df_protB.groupby(['Overlap', 'Complexidade','grupo'])['Desempenho'].mean().unstack('grupo').unstack('Overlap')
df_friedman = df_protB.groupby(['ID','Complexidade'])['Desempenho'].mean().unstack('Complexidade')
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
    df_friedman.columns = [f'C{cx}' for cx in df_friedman.columns]
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
