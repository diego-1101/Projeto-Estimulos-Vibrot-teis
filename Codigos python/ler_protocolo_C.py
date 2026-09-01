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
            'Fase Exploracao': exploracao_df,
            'Fase Execucao': execucao_df
        }

    # Estrutura final: DataFrame com os indivíduos como colunas
    df_final = pd.DataFrame(prot_df)

    return df_final


#ID dos pacientes naquele protocolo
id = ['08', '11', '14', '20', '22', '30', '35', '41', '44']

# Carregar o arquivo .mat do protocolo 
prot_C = loadmat('Aquivos mat\ProtC.mat')

# Acessar o conteúdo de ProtC
ProtC = prot_C['ProtC']  

# Convertendo para um DataFrame
protC_df = transformar_protC_mat_em_df(protocolo = ProtC, id = id)

# plotando algumas trajetórias
"""for i, seq in enumerate(protA_cv_df['df_ID_10']['Rep2']['Trajetória Completa']):
    plotar.plotar_trajetoria(seq = seq,individuo= 'ID10 na primeira repetição')
"""

# Carregando os gabaritos
gabarito = pd.read_csv('Gabaritos\gabarito_protocolo_C.csv')
for i in gabarito.columns:
    gabarito[i][0] = ast.literal_eval(gabarito[i][0])


#%% 
'''Calculando os resultados das métricas de comparação de trajetória para todas as repetições
  de todos os individuos do protocolo C
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

for individuo in protC_df.columns:
    #como para esse protocolo só na fase de execução tem avaliação do desempenho, só irei usar a parte de 'Fase de Execução' do data frame
    teste = protC_df[individuo]['Fase Execucao']

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
        #Desempenho calculado através da ideia de combinação das métricas escolhidas
        r1, r2 = ev.calcular_desempenho( acur=resultado_ideia3[0], 
                                                                 fpr=resultado_ideia3[3], 
                                                                 sim = resultado_ideia4, 
                                                                 propx=float(teste['Proporção espacial x'][i]), 
                                                                 propy=float(teste['Proporção espacial y'][i]))
        desempenho.append(r1)
        desempenho_norm.append(r2)
    protC_df[individuo]['Fase Execucao']['Acuracia'] = acur
    protC_df[individuo]['Fase Execucao']['Media Acuracia'] = np.mean(acur)
    protC_df[individuo]['Fase Execucao']['Taxa de Falsos Positivos'] = fpr
    protC_df[individuo]['Fase Execucao']['Media FPR'] = np.mean(fpr)
    protC_df[individuo]['Fase Execucao']['Similaridade'] = sim
    protC_df[individuo]['Fase Execucao']['Media Similaridade'] = np.mean(sim)
    protC_df[individuo]['Fase Execucao']['Desempenho'] = desempenho
    protC_df[individuo]['Fase Execucao']['Desempenho ponderado com proporção'] = desempenho_norm
    
    #Reiniciando as listas para a próxima iteração
    #prec = []
    #rcll = []
    acur = []
    fpr =[]
    sim = []
    desempenho = []
    desempenho_norm = []

#%% Vendo a distribuição dos resultados obtidos acima 
"""resultadosC = {
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

resultados_C_df = pd.DataFrame(resultadosC)

#Plotando as distribuições 
ev.plotar_distribuicoes_resultados(resultados_C_df, titulo = '(Protocolo C)')"""


#%% ------- Criando o DataFrame para os desempenhos medios do protocolo ---------
lista_concatenada = []

for individuo in protC_df.columns:
    for fase in protC_df[individuo].index:
        df_tmp = protC_df[individuo][fase].copy()
        df_tmp['ID'] = individuo
        df_tmp['Fase'] = fase
        lista_concatenada.append(df_tmp)

df_concat_protC = pd.concat(lista_concatenada, ignore_index=True)

df_concat_protC['Complexidade'] = df_concat_protC['Número da Trajetória'].apply(ev.map_complexidade)

# Tipagem correta
df_concat_protC['ID'] = df_concat_protC['ID'].astype(str)
df_concat_protC['Fase'] = df_concat_protC['Fase'].astype(str)
df_concat_protC['Número da Trajetória'] = df_concat_protC['Número da Trajetória'].astype(int)

#Adicionando a interpretação de Nível de acordo com a Complexidade
df_concat_protC['nivel'] = df_concat_protC['Complexidade'].apply(ev.map_niveis)

df = df_concat_protC[df_concat_protC['Fase']== 'Fase Execucao'].copy()
df['Complexidade'] = df['Complexidade'].astype('category')

#Salvando todo o protocolo C em um Data Frame
df_concat_protC.to_csv('df_protC.csv', index = False)

# %% Distribuição do desempenho da fase de execução do Protocolo C 

print('---'*100)
print('Desempenhos da fase de execução do Protocolo C')
print('---'*100)
ev.plotar_desempenhos(df['Desempenho'], 'Desempenhos da fase de execução do Protocolo C',
                       'Desempenho')


# %% Salvando em .mat para o Jean analisar

import unicodedata
from scipy.io import savemat

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
    
protC = df_concat_protC.to_dict('list')

protC = normalizar_dicionario(protC)

# Salvando o dicionário em um arquivo .mat
savemat('ProtC_normalizado.mat', protC)
