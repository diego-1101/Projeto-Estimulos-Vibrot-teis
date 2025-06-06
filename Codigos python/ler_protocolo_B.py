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

#%% 
#ID dos pacientes naquele protocolo
id_cf = ['01', '04', '05', '12', '18', '25', '26', '33', '38', '42']
id_sf = ['02', '03', '13', '15', '28', '29', '32', '40', '43']

# Carregar o arquivo .mat do protocolo 
B_CF = loadmat('Aquivos mat\ProtB_CF.mat')
B_SF = loadmat('Aquivos mat\ProtB_SF.mat')
#%%
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

#%%
'''
    Verificando nos códigos do matlab, vi que o indivíduo 1 foi removido do protocolo C e do B.
    Provavelmente porque não houve algum erro durante a execução do protocolo c que acabou 
não gravando as posições que esse indivíduo estava executando.

    Por isso, fiz essa parte para remover (caso necessário) o indivíduo 1 antes de fazer as análises das métricas.
'''
'''novo_B_cf_df = protB_cf_df.drop(columns='df_ID_01')
#protB_cf_df = protB_cf_df.drop(columns='df_ID_01')'''

#%% 
'''Calculando os resultados das métricas de comparação de trajetória para todas as repetições
  de todos os individuos do protocolo B CF
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
        
# Vendo a distribuição dos resultados obtidos acima 
resultados_B_CF = {
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
ev.plotar_distribuicoes_resultados(resultados_B_CF_df, titulo = '(Protocolo B CF)')
#%% 
'''Calculando os resultados das métricas de comparação de trajetória para todas as repetições
  de todos os individuos do protocolo B SF
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
        
# Vendo a distribuição dos resultados obtidos acima 
resultados_B_SF = {
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
ev.plotar_distribuicoes_resultados(resultados_B_SF_df, titulo = '(Protocolo B SF)')

# %% Salvando os resultados em .csv
'''
# Arquivo B CF
resultados_B_CF_df.to_csv('Resultados Metricas ProtB CF.csv')
# Arquivo A SV
resultados_B_SF_df.to_csv('Resultados Metricas ProtB SF.csv')'''

# %% Plotando tudo junto para poder ver a distribuição conjunta

resultado_B = pd.concat([resultados_B_CF_df,resultados_B_SF_df], axis = 0,ignore_index=True)
ev.plotar_distribuicoes_resultados(resultado_B, titulo = '(Protocolo B completo (CF e SF juntos))')

# %%
