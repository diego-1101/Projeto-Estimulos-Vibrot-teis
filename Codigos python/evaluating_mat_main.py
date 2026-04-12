#%% Modules
from scipy.io import loadmat
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
import ast
import Plotar_sequencias as plotar
import evaluating_mat_functions as ev

#%% Loadding all protocols files 
df_protA = pd.read_csv('df_protA.csv', index_col=0)
df_protB = pd.read_csv('df_protB.csv', index_col=0)
df_protC = pd.read_csv('df_protC.csv', index_col=0)
df_protC = df_protC[df_protC['Fase'] == 'Fase Execucao'].copy() #Comment if want entire protC

# Renaming the weighted performance
df_protA.rename(columns={'Desempenho ponderado com proporção': 'Desempenho_ponderado'},inplace=True)
df_protB.rename(columns={'Desempenho ponderado com proporção': 'Desempenho_ponderado'},inplace=True)
df_protC.rename(columns={'Desempenho ponderado com proporção': 'Desempenho_ponderado'},inplace=True)

#%% A Protocol
# --------------------------------------- Teste de Normalidade ---------------------------------------
alpha = 0.05
print('---'*100)
print('Teste de Normalidade dos desempenhos do Protocolo A CV')
print('---'*100)
normalidade_A_cv = ev.teste_normalidade_completo(df_protA[df_protA['grupo']=='CV']['Desempenho'])   

print('\n')
print('---'*100)
print('Teste de Normalidade dos desempenhos de todo Protocolo A SV')
print('---'*100)
normalidade_A_sv = ev.teste_normalidade_completo(df_protA[df_protA['grupo']=='SV']['Desempenho'])

print('\n')
print('---'*100)
print('Teste de Normalidade dos desempenhos de todo Protocolo A')
print('---'*100)
normalidade_A = ev.teste_normalidade_completo(df_protA['Desempenho'])

# --------------------------------------- ANOVA ---------------------------------------
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import MultiComparison

modelo = ols('Desempenho_ponderado ~ C(grupo) * C(Complexidade) * C(Overlap)', data=df_protA).fit()
anova = sm.stats.anova_lm(modelo, typ=3)
print("Resultados da ANOVA do protocolo A:")
print(anova)
print('---'*100)

# Post-hoc test (Tukey's HSD)
mc = MultiComparison(df_protA['Desempenho_ponderado'], df_protA['grupo'])
resultado = mc.tukeyhsd()
print(resultado.summary())
print('---'*100)

#%% --------------------------------------- Plots ---------------------------------------
#%% ---------- Desempenho
# 1. Dotplot
classes = ['grupo','grupo_complexidade']
for classe in classes:
    ev.dot_ic_sig(
        df=df_protA,
        x= classe,
        y='Desempenho',
        order=df_protA[classe].unique(),  
        show_sig_bars=True,       # << novo: só desenha as barras se True
        show_p_text=False, 
        alpha=0.05,         # True para escrever p-values
        ylim=(0, 1.1),
        title=f'Prot A — {classe}'
    )
ev.dot_ic_sig(
        df=df_protA,
        x= 'grupo_complexidade_overlap',
        y='Desempenho',
        order=df_protA['grupo_complexidade_overlap'].unique(),  
        show_sig_bars=True,       # << novo: só desenha as barras se True
        show_p_text=False, 
        alpha=0.05,         # True para escrever p-values
        ylim=(0, 1.1),
        figsize=((18,10)),
        step=0.025,
        title=f'Prot A — {'grupo_complexidade_overlap'}'
    )

# 2. Barplot
ev.bar_ic95(
    df=df_protA,
    x='grupo_complexidade',
    y='Desempenho', # coluna do grupo
    hue='grupo',
    hue_order=['CV','SV'],                  # opcional
    palette={'CV':'#4C78A8','SV':'#F58518'},  # opcional (ou passe lista de cores)
    ylim=(0, 1.1),
    rotate_xticks=45,
    title='Prot A — grupo_complexidade'
)

# 3. Interaction Plot

# 3.1. Interaction plot Desempenho x Complexidades (linhas= Velocidade/Overlap)
#3.1.1. Facetado 
(_, _), stats = ev.interaction_plot(
    df=df_protA,
    x='Complexidade',
    line='velocidade',
    y='Desempenho',
    facet='grupo',
    x_order=[4,6,8],                       # ordem do eixo X
    line_order=['Lento','Médio','Rápido'], # ordem das linhas
    facet_order=['CV','SV'],               # ordem dos facets
    x_map={4:'Fácil', 6:'Médio',8:'Difícil'}, # rótulos bonitos
    title='Desempenho × Complexidade (linhas= Overlaps) | Facet: Grupo',
    figsize=(9,4)
)
#3.1.2. Não Facetado 
(_, _), stats = ev.interaction_plot(
    df=df_protA,
    x='Complexidade',
    line='velocidade',
    y='Desempenho',
    x_order=[4,6,8],                       # ordem do eixo X
    line_order=['Lento','Médio','Rápido'], # ordem das linhas
    x_map={4:'Fácil', 6:'Médio',8:'Difícil'}, # rótulos bonitos
    title='Desempenho × Complexidade (linhas= Overlaps)',
    figsize=(9,4)
)

# 3.2. Interaction plot Desempenho x Overlaps (linhas= Niveis/Complexidade)
#3.2.1. Facetado 
(_, _), stats = ev.interaction_plot(
    df=df_protA,
    x='Overlap',
    line='nivel',
    y='Desempenho',
    facet='grupo',
    x_order=[0.0,0.25,0.5],                       # ordem do eixo X
    line_order=['Fácil','Médio','Difícil'], # ordem das linhas
    facet_order=['CV','SV'],               # ordem dos facets
    x_map={0.0:'Lento', 0.25:'Médio',0.5:'Rápido'}, # rótulos bonitos
    title='Desempenho × Overlap (linhas= Complexidades) | Facet: Grupo',
    figsize=(9,4)
)
#3.2.2. Não Facetado 
(_, _), stats = ev.interaction_plot(
    df=df_protA,
    x='Overlap',
    line='nivel',
    y='Desempenho',
    x_order=[0.0,0.25,0.5],                       # ordem do eixo X
    line_order=['Fácil','Médio','Difícil'], # ordem das linhas
    x_map={0.0:'Lento', 0.25:'Médio',0.5:'Rápido'}, # rótulos bonitos
    title='Desempenho × Overlap (linhas= Complexidades)',
    figsize=(9,4)
)

# 3.3. Desempenho × Grupo (linhas = Complexidades)
# 3.3.1. Facetado 
(fig, axes), stats = ev.interaction_plot(
    df=df_protA,
    x='grupo',
    line='nivel',                      # ← linhas = complexidades
    y='Desempenho',
    facet='Overlap',                   # ← um subplot por Overlap
    x_order=['CV','SV'],
    line_order=['Fácil','Médio','Difícil'],
    facet_order=[0.0, 0.25, 0.5],
    facet_map={0.0:'Lento', 0.25:'Médio', 0.5:'Rápido'},
    title='Desempenho × Grupo (linhas = Complexidades) | Facet: Overlap',
    figsize=(12,4)
)

# 3.3.2. Não Facetado 
(fig, axes), stats = ev.interaction_plot(
    df=df_protA,
    x='grupo',
    line='nivel',                      # ← linhas = complexidades
    y='Desempenho',                  
    x_order=['CV','SV'],
    line_order=['Fácil','Médio','Difícil'],
    facet_order=[0.0, 0.25, 0.5],
    facet_map={0.0:'Lento', 0.25:'Médio', 0.5:'Rápido'},
    title='Desempenho × Grupo (linhas = Complexidades)',
    figsize=(12,4)
)

# 3.4. Desempenho × Grupo (linhas = Overlap)
# 3.4.1. Facetado 
(fig, axes), stats = ev.interaction_plot(
    df=df_protA,
    x='grupo',
    line='Overlap',                    # ← linhas = overlaps
    y='Desempenho',
    facet='nivel',                     # ← um subplot por complexidade
    x_order=['CV','SV'],
    line_order=[0.0, 0.25, 0.5],
    line_map={0.0:'Lento', 0.25:'Médio', 0.5:'Rápido'},
    facet_order=['Fácil','Médio','Difícil'],
    title='Desempenho × Grupo (linhas = Overlaps) | Facet: Complexidade',
    figsize=(12,4)
)


# 3.4.2. Não Facetado 
(fig, axes), stats = ev.interaction_plot(
    df=df_protA,
    x='grupo',
    line='Overlap',                    # ← linhas = overlaps
    y='Desempenho',
    x_order=['CV','SV'],
    line_order=[0.0, 0.25, 0.5],
    line_map={0.0:'Lento', 0.25:'Médio', 0.5:'Rápido'},
    facet_order=['Fácil','Médio','Difícil'],
    title='Desempenho × Grupo (linhas = Overlaps)',
    figsize=(12,4)
)

#%% ---------- Desempenho_ponderado
# 1. Dotplot
classes = ['grupo','grupo_complexidade']
for classe in classes:
    ev.dot_ic_sig(
        df=df_protA,
        x= classe,
        y='Desempenho_ponderado',
        order=df_protA[classe].unique(),  
        show_sig_bars=True,       # << novo: só desenha as barras se True
        show_p_text=False, 
        alpha=0.05,         # True para escrever p-values
        ylim=(0, 1.1),
        title=f'Prot A — {classe}'
    )

ev.dot_ic_sig(
        df=df_protA,
        x= 'grupo_complexidade_overlap',
        y='Desempenho_ponderado',
        order=df_protA['grupo_complexidade_overlap'].unique(),  
        show_sig_bars=False,       # << novo: só desenha as barras se True
        show_p_text=False, 
        alpha=0.05,         # True para escrever p-values
        ylim=(0, 1.1),
        figsize=((18,10)),
        step=0.025,
        title=f'Prot A — {'grupo_complexidade_overlap'}'
    )

# 2. Barplot
ev.bar_ic95(
    df=df_protA,
    x='grupo_complexidade_overlap',
    y='Desempenho_ponderado', # coluna do grupo
    hue='grupo',
    hue_order=['CV','SV'],                  # opcional
    palette={'CV':'#4C78A8','SV':'#F58518'},  # opcional (ou passe lista de cores)
    ylim=(0, 1.1),
    rotate_xticks=45,
    title='Prot A — grupo_complexidade_overlap'
)

# 3. Interaction Plot

# 3.1. Interaction plot Desempenho_ponderado x Complexidades (linhas= Velocidade/Overlap)
#3.1.1. Facetado 
(_, _), stats = ev.interaction_plot(
    df=df_protA,
    x='Complexidade',
    line='velocidade',
    y='Desempenho_ponderado',
    facet='grupo',
    x_order=[4,6,8],                       # ordem do eixo X
    line_order=['Lento','Médio','Rápido'], # ordem das linhas
    facet_order=['CV','SV'],               # ordem dos facets
    x_map={4:'Fácil', 6:'Médio',8:'Difícil'}, # rótulos bonitos
    title='Desempenho_ponderado × Complexidade (linhas= Overlaps) | Facet: Grupo',
    figsize=(9,4)
)
#3.1.2. Não Facetado 
(_, _), stats = ev.interaction_plot(
    df=df_protA,
    x='Complexidade',
    line='velocidade',
    y='Desempenho_ponderado',
    x_order=[4,6,8],                       # ordem do eixo X
    line_order=['Lento','Médio','Rápido'], # ordem das linhas
    x_map={4:'Fácil', 6:'Médio',8:'Difícil'}, # rótulos bonitos
    title='Desempenho_ponderado × Complexidade (linhas= Overlaps)',
    figsize=(9,4)
)

# 3.2. Interaction plot Desempenho_ponderado x Overlaps (linhas= Niveis/Complexidade)
#3.2.1. Facetado 
(_, _), stats = ev.interaction_plot(
    df=df_protA,
    x='Overlap',
    line='nivel',
    y='Desempenho_ponderado',
    facet='grupo',
    x_order=[0.0,0.25,0.5],                       # ordem do eixo X
    line_order=['Fácil','Médio','Difícil'], # ordem das linhas
    facet_order=['CV','SV'],               # ordem dos facets
    x_map={0.0:'Lento', 0.25:'Médio',0.5:'Rápido'}, # rótulos bonitos
    title='Desempenho_ponderado × Overlap (linhas= Complexidades) | Facet: Grupo',
    figsize=(9,4)
)
#3.2.2. Não Facetado 
(_, _), stats = ev.interaction_plot(
    df=df_protA,
    x='Overlap',
    line='nivel',
    y='Desempenho_ponderado',
    x_order=[0.0,0.25,0.5],                       # ordem do eixo X
    line_order=['Fácil','Médio','Difícil'], # ordem das linhas
    x_map={0.0:'Lento', 0.25:'Médio',0.5:'Rápido'}, # rótulos bonitos
    title='Desempenho_ponderado × Overlap (linhas= Complexidades)',
    figsize=(9,4)
)

# 3.3. Desempenho_ponderado × Grupo (linhas = Complexidades)
# 3.3.1. Facetado 
(fig, axes), stats = ev.interaction_plot(
    df=df_protA,
    x='grupo',
    line='nivel',                      # ← linhas = complexidades
    y='Desempenho_ponderado',
    facet='Overlap',                   # ← um subplot por Overlap
    x_order=['CV','SV'],
    line_order=['Fácil','Médio','Difícil'],
    facet_order=[0.0, 0.25, 0.5],
    facet_map={0.0:'Lento', 0.25:'Médio', 0.5:'Rápido'},
    title='Desempenho_ponderado × Grupo (linhas = Complexidades) | Facet: Overlap',
    figsize=(12,4)
)

# 3.3.2. Não Facetado 
(fig, axes), stats = ev.interaction_plot(
    df=df_protA,
    x='grupo',
    line='nivel',                      # ← linhas = complexidades
    y='Desempenho_ponderado',                  
    x_order=['CV','SV'],
    line_order=['Fácil','Médio','Difícil'],
    facet_order=[0.0, 0.25, 0.5],
    facet_map={0.0:'Lento', 0.25:'Médio', 0.5:'Rápido'},
    title='Desempenho_ponderado × Grupo (linhas = Complexidades)',
    figsize=(12,4)
)

# 3.4. Desempenho_ponderado × Grupo (linhas = Overlap)
# 3.4.1. Facetado 
(fig, axes), stats = ev.interaction_plot(
    df=df_protA,
    x='grupo',
    line='Overlap',                    # ← linhas = overlaps
    y='Desempenho_ponderado',
    facet='nivel',                     # ← um subplot por complexidade
    x_order=['CV','SV'],
    line_order=[0.0, 0.25, 0.5],
    line_map={0.0:'Lento', 0.25:'Médio', 0.5:'Rápido'},
    facet_order=['Fácil','Médio','Difícil'],
    title='Desempenho_ponderado × Grupo (linhas = Overlaps) | Facet: Complexidade',
    figsize=(12,4)
)


# 3.4.2. Não Facetado 
(fig, axes), stats = ev.interaction_plot(
    df=df_protA,
    x='grupo',
    line='Overlap',                    # ← linhas = overlaps
    y='Desempenho_ponderado',
    x_order=['CV','SV'],
    line_order=[0.0, 0.25, 0.5],
    line_map={0.0:'Lento', 0.25:'Médio', 0.5:'Rápido'},
    facet_order=['Fácil','Médio','Difícil'],
    title='Desempenho_ponderado × Grupo (linhas = Overlaps)',
    figsize=(12,4)
)


#%% B Protocol
# --------------------------------------- Teste de Normalidade ---------------------------------------
alpha = 0.05
print('---'*100)
print('Teste de Normalidade dos desempenhos do Protocolo B CF')
print('---'*100)
normalidade_B_cv = ev.teste_normalidade_completo(df_protB[df_protB['grupo']=='CF']['Desempenho'])   

print('\n')
print('---'*100)
print('Teste de Normalidade dos desempenhos de todo Protocolo B SF')
print('---'*100)
normalidade_B_sv = ev.teste_normalidade_completo(df_protB[df_protB['grupo']=='SF']['Desempenho'])

print('\n')
print('---'*100)
print('Teste de Normalidade dos desempenhos de todo Protocolo B')
print('---'*100)
normalidade_B = ev.teste_normalidade_completo(df_protB['Desempenho'])

# --------------------------------------- ANOVA ---------------------------------------
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import MultiComparison

modelo = ols('Desempenho ~ C(grupo) * C(Complexidade)', data=df_protB).fit()
anova = sm.stats.anova_lm(modelo, typ=3)
print("Resultados da ANOVA do protocolo B:")
print(anova)
print('---'*100)

# Post-hoc test (Tukey's HSD)
mc = MultiComparison(df_protB['Desempenho'], df_protB['Complexidade'])
resultado = mc.tukeyhsd()
print(resultado.summary())
print('---'*100)

#%% --------------------------------------- Plots ---------------------------------------

x = 'grupo'
y= 'Proporção espacial y'

# 1. Dotplot
ev.dot_ic_sig(
    df=df_protB,
    x=x,
    y=y,
    order=df_protB[x].unique(),  
    alpha=0.05,
    show_sig_bars=True, 
    show_p_text=False,          # True para escrever p-values
    ylim=(0, 1.1),
    title=f'Prot B — {x}'
)

#%% 2. Barplot
ev.bar_ic95(
    df=df_protB,
    x=x,
    y='Desempenho',
    hue='grupo',                            # coluna do grupo
    hue_order=['CF','SF'],                  # opcional
    palette={'CF':'#4C78B8','SF':'#F58518'},  # opcional (ou passe lista de cores)
    ylim=(0, 1.1),
    rotate_xticks=45,
    title=f'Prot B — {x}'
)

#%% 3. Interaction Plot
# 3.1. Desempenho × Grupo (linhas = Complexidade)
# 3.1.1. Não Facetado 
(fig, axes), stats = ev.interaction_plot(
    df=df_protB,
    x='grupo',
    line='nivel',                      # ← linhas = complexidades
    y='Desempenho_ponderado',                  
    x_order=['CF','SF'],
    line_order=['Fácil','Médio','Difícil'],
    title='Desempenho_ponderado × Grupo (linhas = Complexidades)',
    figsize=(12,4),
    ylim=(0.2,1)
)
# 3.2. Desempenho_ponderado × Complexidade (linhas = Grupo)
# 3.2.1. Não Facetado 
(fig, axes), stats = ev.interaction_plot(
    df=df_protB,
    x='nivel',
    line='grupo',                      # ← linhas = complexidades
    y='Desempenho_ponderado',
    x_order = ['Fácil','Médio','Difícil'],
    line_order=['CF','SF'],
    title='Desempenho_ponderado × Complexidade (linhas = Grupos)',
    figsize=(12,4),
    ylim=(0.2,1)
)

#%% C Protocol

# 1. Dotplot
ev.dot_ic_sig(
    df=df_protC,
    x='nivel',
    y='Desempenho',
    order=df_protC['nivel'].unique(),  
    alpha=0.05,
    show_p_text=True,          # True para escrever p-values
    ylim=(0, 1.1),
    title='Prot C — Nível'
)
# 2. Barplot

# 3. Interaction Plot

# %% Fazendo uma seleção vetorial para melhor visualizar os clusters de cada classe
#%% Protocolo A
# Normalizando os dados, para isso vou usar o método Standardscaler do sklearn

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
'''
pca = PCA()
x1 = pca.fit_transform(x1)'''

caracteristicas = ['Proporção espacial x', 'Proporção espacial y','Acuracia',
               'Similaridade', 'Especificidade']
#caracteristicas = ['Acuracia',
#               'Similaridade', 'Especificidade']
x1 = df_protA[caracteristicas].to_numpy() #vetor de características
for classe in ['grupo', x,'grupo_complexidade_overlap']:
    print('--'*100)
    print(f'Classe: {classe}')
    y1 = df_protA[classe].to_numpy() #vetor de classes
    y1 = y1.reshape(-1) # fazendo o flatten do vetor de classes (removendo o (x,1))
    x1 = StandardScaler().fit_transform(x1) #normalização
    ev.selecao_vetorial(x1 = x1, y1 = y1, nomes_carac = caracteristicas, k = 3, plotar = True, 
                        interativo = False, salvar_interativo=False)
    print('--'*100)

#%% Protocolo B
# Normalizando os dados, para isso vou usar o método Standardscaler do sklearn

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
'''
pca = PCA()
x1 = pca.fit_transform(x1)'''

caracteristicas = ['Proporção espacial x', 'Proporção espacial y','Acuracia',
               'Similaridade', 'Especificidade']
#caracteristicas = ['Acuracia',
#               'Similaridade', 'Especificidade']
x1 = df_protB[caracteristicas].to_numpy() #vetor de características
for classe in ['grupo', x]:
    print('--'*100)
    print(f'Classe: {classe}')
    y1 = df_protB[classe].to_numpy() #vetor de classes
    y1 = y1.reshape(-1) # fazendo o flatten do vetor de classes (removendo o (x,1))
    x1 = StandardScaler().fit_transform(x1) #normalização
    ev.selecao_vetorial(x1 = x1, y1 = y1, nomes_carac = caracteristicas, k = 3, plotar = True, 
                        interativo = False, salvar_interativo=False)
    print('--'*100)

# %% Fazendo os plots de CDA
#%% Protocolo A
x1 = df_protA[caracteristicas].to_numpy() #vetor de características
for classe in ['grupo', x,'grupo_complexidade_overlap','Overlap', 'Complexidade']:
    print('--'*100)
    print(f'Classe: {classe}')
    y1 = df_protA[classe].to_numpy() #vetor de classes
    y1 = y1.reshape(-1) # fazendo o flatten do vetor de classes (removendo o (x,1))
    x1 = StandardScaler().fit_transform(x1) #normalização
    ev.manova1_py(
    X= x1,
    groups = y1,
    k_plot=2,
    plotar=True,
    interativo=True,
    salvar_interativo=False,
    title_prefix="MANOVA1 / CDA"
)
    print('--'*100)

#%% Protocolo B
x1 = df_protB[caracteristicas].to_numpy() #vetor de características
for classe in ['grupo', 'grupo_complexidade', 'Complexidade']:
    print('--'*100)
    print(f'Classe: {classe}')
    y1 = df_protB[classe].to_numpy() #vetor de classes
    y1 = y1.reshape(-1) # fazendo o flatten do vetor de classes (removendo o (x,1))
    x1 = StandardScaler().fit_transform(x1) #normalização
    ev.manova1_py(
    X= x1,
    groups = y1,
    k_plot=2,
    plotar=True,
    interativo=True,
    salvar_interativo=False,
    title_prefix="MANOVA1 / CDA"
)
    print('--'*100)

#%% Teste de hipótese em cima das proporções
from scipy import stats
g1_propx = df_protA[(df_protA['grupo']=='CV') & (df_protA['Proporção espacial x']<=1.0)]['Proporção espacial x']
g2_propx = df_protA[df_protA['grupo']=='SV']['Proporção espacial x']

# Run the independent t-test (Welch’s version by default)
t_stat, p_value = stats.ttest_ind(g1_propx, g2_propx, equal_var=False)

print('--'*10)
print('Teste-t para propx entre CV x SV (sem outlier)')
print(f"T-statistic = {t_stat:.3f}")
print(f"P-value = {p_value:.4f}")
print('--'*20)

g1_propy = df_protA[df_protA['grupo']=='CV']['Proporção espacial y']
g2_propy = df_protA[df_protA['grupo']=='SV']['Proporção espacial y']

# Run the independent t-test (Welch’s version by default)
t_stat, p_value = stats.ttest_ind(g1_propy, g2_propy, equal_var=False)

print('--'*10)
print('Teste-t para propy entre CV x SV (sem outlier)')
print(f"T-statistic = {t_stat:.3f}")
print(f"P-value = {p_value:.4f}")
print('--'*20)

#%% 
g1_propx = df_protB[(df_protB['grupo']=='CF')]['Proporção espacial x']
g2_propx = df_protB[df_protB['grupo']=='SF']['Proporção espacial x']

# Run the independent t-test (Welch’s version by default)
t_stat, p_value = stats.ttest_ind(g1_propx, g2_propx, equal_var=False)

print('--'*10)
print('Teste-t para propx entre CF x SF (com outlier)')
print(f"T-statistic = {t_stat:.3f}")
print(f"P-value = {p_value:.4f}")
print('--'*20)

g1_propy = df_protB[df_protB['grupo']=='CF']['Proporção espacial y']
g2_propy = df_protB[df_protB['grupo']=='SF']['Proporção espacial y']

# Run the independent t-test (Welch’s version by default)
t_stat, p_value = stats.ttest_ind(g1_propy, g2_propy, equal_var=False)

print('--'*10)
print('Teste-t para propy entre CF x SF (com outlier)')
print(f"T-statistic = {t_stat:.3f}")
print(f"P-value = {p_value:.4f}")
print('--'*20)
# %%
