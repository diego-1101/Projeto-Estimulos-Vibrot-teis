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


def transformar_protC_mat_em_df(protocolo, id=['08', '11', '14', '20', '22', '30', '35', '41', '44']):
    """
    Transforma a estrutura MATLAB ProtC (carregada com scipy.io.loadmat) em um DataFrame de DataFrames
    organizado por participante e dividido nas fases do experimento.

    A função é voltada para o Protocolo C do experimento, que envolve duas fases:
    - Fase de Exploração: onde o participante apenas observa/recebe os estímulos
    - Fase de Execução: onde o participante tenta reproduzir o que percebeu

    Parâmetros:
    ----------
    protocolo : np.ndarray
        Estrutura carregada do arquivo .mat referente ao ProtC, de tamanho (9, 2)
    id : list of str
        Lista com os identificadores dos participantes, na ordem das linhas de `protocolo`

    Retorna:
    -------
    df_final : pd.DataFrame
        DataFrame com as colunas sendo os participantes (`df_ID_XX`) e duas linhas por coluna:
        - 'Fase de Exploração': DataFrame com colunas ['Número da Trajetória', 'Sorteio', 'Tempo 1', 'Tempo 2']
        - 'Fase de Execução': DataFrame com colunas:
            ['Número da Trajetória', 'Sorteio', 'Tempo 1', 'Tempo 2',
             'Score Bruto', 'Score Ponderado', 'Proporção X', 'Proporção Y',
             'Trajetória Completa', 'Trajetória Simplificada']
    """

    headers_exploracao = ['Número da Trajetória', 'Sorteio', 'Tempo 1', 'Tempo 2']
    headers_execucao = ['Número da Trajetória', 'Sorteio', 'Tempo 1',
           'Tempo 2', 'Score total', 'Score Parcial',
           'Proporção espacial x', 'Proporção espacial y',
           'Trajetória Completa', 'Trajetória Simplificada']

    prot_df = {}

    for i, ID in enumerate(id):
        # Fase de exploração (coluna 0)
        exploracao_raw = protocolo[i, 0]
        exploracao_df = pd.DataFrame(exploracao_raw, columns=headers_exploracao)
        for col in headers_exploracao:
            exploracao_df[col] = exploracao_df[col].apply(
                lambda x: x.item() if isinstance(x, np.ndarray) and x.size == 1 else x
            )

        # Fase de execução (coluna 1)
        execucao_raw = protocolo[i, 1]
        execucao_df = pd.DataFrame(execucao_raw, columns=headers_execucao)
        for col in headers_execucao[:-2]:  # colunas numéricas
            execucao_df[col] = execucao_df[col].apply(
                lambda x: x.item() if isinstance(x, np.ndarray) and x.size == 1 else x
            )

        # Transformar trajetórias (strings) em listas de inteiros
        execucao_df['Trajetória Completa'] = execucao_df['Trajetória Completa'].apply(
            lambda x: ast.literal_eval(x[0]) if isinstance(x, np.ndarray) and isinstance(x[0], str) else [9]
        )
        execucao_df['Trajetória Simplificada'] = execucao_df['Trajetória Simplificada'].apply(
            lambda x: ast.literal_eval(x[0]) if isinstance(x, np.ndarray) and isinstance(x[0], str) else [9]
        )

        # Armazenar os DataFrames dessa pessoa
        prot_df[f'df_ID_{ID}'] = {
            'Fase de Exploração': exploracao_df,
            'Fase de Execução': execucao_df
        }

    # Estrutura final: DataFrame com os indivíduos como colunas
    df_final = pd.DataFrame(prot_df)

    return df_final


#%% 
#ID dos pacientes naquele protocolo
id = ['08', '11', '14', '20', '22', '30', '35', '41', '44']

# Carregar o arquivo .mat do protocolo 
C = loadmat('Aquivos mat\ProtC.mat')
#%%
# Acessar o conteúdo de ProtB_SF
ProtC = C['ProtC']  # ProtB_SF é uma célula de células !!!!!!!!!!!!!


# Convertendo para um DataFrame
protC_df = transformar_protC_mat_em_df(protocolo = ProtC, id = id)

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
'''Calculando os resultados das métricas de comparação de trajetória para todas as repetições
  de todos os individuos do protocolo C
'''
sparcial =[]
stotal = []
acur = []
prec = []
rcll = []
fpr =[]
sim = []

for individuo in protC_df.columns:
    #como para esse protocolo só na fase de execução tem avaliação do desempenho, só irei usar a parte de 'Fase de Execução' do data frame
    teste = protC_df[individuo]['Fase de Execução']
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

        #---- Avaliando por comparação de imagem (IDEIA 3)
        resultado_ideia3 = ev.comparar_imagem(seq1=seq1,seq2=seq2,plotar_imagens = False)

        #---- Avaliando por similaridade com correlação cruzada normalizada (IDEIA 4)
        resultado_ideia4 = ev.calcular_similaridade(seq1,seq2)
        
        #resultados.append([score_parcial_uns,score_total_uns,resultado_ideia3[0],resultado_ideia3[1], resultado_ideia3[2], resultado_ideia3[3],resultado_ideia4])
        sparcial.append(score_parcial_uns)
        stotal.append(score_total_uns)
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
resultadosC = {
    'Score Parcial':sparcial,
    'Score Total':stotal,
    'Acurácia':acur,
    'Precisão':prec,
    'Recall':rcll,
    'Taxa de Falsos Positivos':fpr,
    'Similaridade':sim
}

resultados_C_df = pd.DataFrame(resultadosC)

#Plotando as distribuições 
ev.plotar_distribuicoes_resultados(resultados_C_df, titulo = '(Protocolo C)')

#%% Salvando os resultados das métricas

"""# Arquivo C
resultados_C_df.to_csv('Resultados Metricas ProtC.csv')"""
# %%
