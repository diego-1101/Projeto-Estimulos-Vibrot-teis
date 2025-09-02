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
C = loadmat('Aquivos mat\ProtC.mat')

# Acessar o conteúdo de ProtC
ProtC = C['ProtC']  

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

df = df_concat_protC[df_concat_protC['Fase']== 'Fase Execucao'].copy()

# %% Distribuição do desempenho da fase de execução do Protocolo C 

print('---'*100)
print('Desempenhos da fase de execução do Protocolo C')
print('---'*100)
ev.plotar_desempenhos(df['Desempenho'], 'Desempenhos da fase de execução do Protocolo C',
                       'Desempenho')

#%% Normalidade 
from scipy.stats import shapiro, kstest
print('\n')
print('---'*100)
print('Teste de Normalidade dos desempenhos da fase de execução do Protocolo C')
print('---'*100)
alpha = 0.05
w_c, p_shapiro_c = shapiro(df['Desempenho'])
print(f'Estatística W: {w_c}, p-valor: {p_shapiro_c}')
print(f"✅Normal (alpha={alpha})" if p_shapiro_c > alpha else f"❌Não normal (alpha={alpha})")
d_c, p_kstest_c = kstest(df['Desempenho'], 'norm', args=(df['Desempenho'].mean(), df['Desempenho'].std()))
print(f'Estatística D: {d_c}, p-valor: {p_kstest_c}')
print(f"✅Normal (alpha={alpha})" if p_kstest_c > alpha else f"❌Não normal (alpha={alpha})")  
print('---'*100)

#%% Teste de Homogeneidade de Variâncias
from scipy.stats import levene
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#%% Boxplot

# Boxplot com notches por Complexidade
plt.figure(figsize=(8, 6))
sns.boxplot(x='Complexidade', y='Desempenho', data=df, notch=True)
plt.title('Boxplot com Notch por Complexidade')
plt.suptitle('')
plt.yticks(np.arange(0, 1.5, 0.1), fontsize=10)
plt.xlabel('Complexidade')
plt.ylabel('Desempenho')
plt.show()

#%% --------------------------------------- Teste de hipótese por ANOVA ---------------------------------------

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import MultiComparison
from statsmodels.stats.libqsturng import psturng  # usado internamente

# Assumindo que 'desempenho' é contínua, e grupo, complexidade e overlap são fatores:
modelo = ols('Desempenho ~ Complexidade', data=df).fit()
anova = sm.stats.anova_lm(modelo, typ=3)  
print("Resultados da ANOVA:")
print(anova)
print("-"*100)

# Comparar as médias de desempenho entre as complexidade:
mc = MultiComparison(df['Desempenho'], df['Complexidade'])
resultado = mc.tukeyhsd()
#print(resultado)

# Print do resumo das comparações
print(resultado.summary())

#%% -------------------- PCA + ANOVA --------------------
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
df_concat_protC['Especificidade'] = 1 - df_concat_protC['Taxa de Falsos Positivos']
df = df_concat_protC[df_concat_protC['Fase']== 'Fase Execucao'].copy()

X = df[['Acuracia', 'Especificidade', 'Proporção espacial x', 'Proporção espacial y', 'Similaridade']]

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

#    Fazendo ANOVA com PCA
df['PC1'] = X_pca[:, 0]  # Usando o primeiro componente principal
modelo = ols('PC1 ~ Complexidade', data=df).fit()
anova = sm.stats.anova_lm(modelo, typ=3)
print(anova)

# Comparar as médias de desempenho entre as complexidade:
mc = MultiComparison(df['PC1'], df['Complexidade'])
resultado = mc.tukeyhsd()
#print(resultado)

# Print do resumo das comparações
print(resultado.summary())

#Box plot do PCA
# Boxplot com notches por Complexidade
plt.figure(figsize=(8, 6))
df.boxplot(column='PC1', by='Complexidade', notch=True, grid=False)
plt.title('Boxplot com Notch por Complexidade')
plt.suptitle('')
plt.xlabel('Complexidade')
plt.ylabel('PC1')
plt.show()


#%% -------------------- LDA + ANOVA --------------------
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# Selecionar variáveis contínuas
df['Complexidade'] = df['Complexidade'].astype('category')  # Garantir que Complexidade é categórica
X = df[['Acuracia', 'Especificidade', 'Similaridade', 'Proporção espacial x', 'Proporção espacial y']]
y = df['Complexidade'] 

# Padronizar os dados
X_scaled = StandardScaler().fit_transform(X)

# Aplicar CDA / LDA
lda = LinearDiscriminantAnalysis()
X_lda = lda.fit_transform(X_scaled, y)

# Variância explicada por cada discriminante
exp_var_lda = lda.explained_variance_ratio_
cum_var_lda = np.cumsum(exp_var_lda)

df['LD1'] = X_lda[:, 0]  # Projeção no primeiro discriminante

# Plotar a projeção dos grupos ao longo do LD1
plt.figure(figsize=(10, 5))
for comp in df['Complexidade'].cat.categories:
    plt.hist(df[df['Complexidade'] == comp]['LD1'], bins=20, alpha=0.6, label=comp)
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
modelo = ols('LD1 ~ Complexidade', data=df).fit()
anova = sm.stats.anova_lm(modelo, typ=3)
print(anova)
# Comparar as médias de desempenho entre as complexidade:
mc = MultiComparison(df['LD1'], df['Complexidade'])
resultado = mc.tukeyhsd()
print(resultado)


#Boxplot com Notch para comparar
plt.figure(figsize=(10, 6))
df.boxplot(column='LD1', by='Complexidade', notch=True, grid=False, patch_artist=True)
plt.title('Boxplot com Notch por Complexidade e Grupo')
plt.xlabel('Complexidade')
plt.ylabel('LD1')
plt.legend(title='Grupo')
plt.show()

#%% qq plot dos resíduos
import statsmodels.api as sm

sm.qqplot(df, line='s')
plt.title('QQ Plot dos Resíduos do Modelo LDA')
plt.grid()

#%% ------------- Teste não paramétrico de Kruskal-Wallis -------------
from scipy.stats import kruskal
# Teste de Kruskal-Wallis
stat, p_value = kruskal(*[group['Desempenho'].values for name, group in df.groupby('Complexidade')])
print(f'Estatística de Kruskal-Wallis: {stat}, p-valor: {p_value}')
if p_value < 0.05:
    print("Há diferenças significativas entre os grupos (Complexidade) no desempenho.") 
else:
    print("Não há diferenças significativas entre os grupos (Complexidade) no desempenho.")

import scikit_posthocs as sp

posthoc = sp.posthoc_dunn(df, val_col='Desempenho', group_col='Complexidade', p_adjust='bonferroni')
print(posthoc)

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
#%%
protC = df_concat_protC.to_dict('list')

protC = normalizar_dicionario(protC)

# Salvando o dicionário em um arquivo .mat
savemat('ProtC_normalizado.mat', protC)
