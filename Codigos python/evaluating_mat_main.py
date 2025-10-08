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
df_protC = df_protC[df_protC['Fase'] == 'Fase Execucao'].copy()

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

#%% --------------------------------------- ANOVA ---------------------------------------
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import MultiComparison

modelo = ols('desempenho_ponderado ~ C(grupo) * C(Complexidade) * C(Overlap)', data=df_protA).fit()
anova = sm.stats.anova_lm(modelo, typ=3)
print("Resultados da ANOVA do protocolo A:")
print(anova)
print('---'*100)

# Post-hoc test (Tukey's HSD)
mc = MultiComparison(df_protA['desempenho_ponderado'], df_protA['grupo'])
resultado = mc.tukeyhsd()
print(resultado.summary())
print('---'*100)

#%% --------------------------------------- Plots ---------------------------------------

# 1. Dotplot
ev.dot_ic_sig(
    df=df_protA,
    x='grupo',
    y='desempenho_ponderado',
    order=df_protA['grupo'].unique(),  
    alpha=0.05,
    show_p_text=True,          # True para escrever p-values
    ylim=(0, 1.1),
    title='Prot A — grupo'
)

# 2. Barplot
ev.bar_ic95(
    df=df_protA,
    x='grupo_complexidade',
    y='desempenho_ponderado',
    hue='grupo',                            # coluna do grupo
    hue_order=['CV','SV'],                  # opcional
    palette={'CV':'#4C78A8','SV':'#F58518'},  # opcional (ou passe lista de cores)
    ylim=(0, 1.1),
    rotate_xticks=0,
    title='Prot A — Overlap por Grupo'
)

# 3. Interaction Plot

from scipy.stats import t

#Mapas de rótulos e ordens dos fatores
comp_order   = [4, 6, 8]
vel_order    = ['Lento', 'Médio', 'Rápido']
group_order  = ['CV', 'SV']

map_comp   = {4: 'Fácil', 6: 'Intermediário', 8: 'Difícil'}
map_group  = {'CV': 'Com visão', 'SV': 'Sem visão'}

colors  = {'Lento':'#F59E0B', 'Médio':'#3B82F6', 'Rápido':'#10B981'}
markers = {'Lento':'o',       'Médio':'s',        'Rápido':'D'}

#Preparar dados (ordem categórica e desempenho em %)
df = df_protA.copy()
df['Complexidade'] = pd.Categorical(df['Complexidade'], categories=comp_order, ordered=True)
df['velocidade']   = pd.Categorical(df['velocidade'],   categories=vel_order,   ordered=True)
df['grupo']        = pd.Categorical(df['grupo'],        categories=group_order, ordered=True)
df['Desempenho_%'] = 100 * df['Desempenho']  # para ficar na escala do seu exemplo

#Agregar: média, desvio, n e IC95% por (grupo, complexidade, velocidade)
stats = (df.groupby(['grupo','Complexidade','velocidade'])['Desempenho_%']
           .agg(mean='mean', std='std', count='count').reset_index())

def ci95(row):
    n = int(row['count'])
    if n > 1 and pd.notnull(row['std']):
        sem = row['std'] / np.sqrt(n)
        return t.ppf(0.975, df=n-1) * sem
    return np.nan

stats['ci95'] = stats.apply(ci95, axis=1)

#Interaction plot facetado por Grupo (média ± IC95%)
xpos = np.arange(len(comp_order))
xticklabels = [map_comp[c] for c in comp_order]

fig, axes = plt.subplots(1, len(group_order), figsize=(14, 4), sharey=True)
fig.suptitle('Gráfico de Interação - ANOVA 3-way (Complexidade × Velocidade × Grupo)', y=1.10, fontsize=12)

for ax, gcode in zip(axes, group_order):
    sub = stats[stats['grupo'] == gcode].set_index(['Complexidade','velocidade']).sort_index()

    for v in vel_order:
        # sequência de médias/IC seguindo a ordem da complexidade
        ymean = [sub.loc[(c, v), 'mean']  if (c, v) in sub.index else np.nan for c in comp_order]
        yerr  = [sub.loc[(c, v), 'ci95']  if (c, v) in sub.index else np.nan for c in comp_order]

        ax.errorbar(xpos, ymean, yerr=yerr, fmt=markers[v], ms=7, lw=2,
                    capsize=4, color=colors[v], label=v)
        ax.plot(xpos, ymean, '-', color=colors[v], lw=2, alpha=0.9)

    ax.set_title(f"Grupo: {map_group[gcode]}")
    ax.set_xticks(xpos); ax.set_xticklabels(xticklabels)
    ax.set_xlabel('Complexidade')
    ax.grid(True, ls='--', alpha=0.3)

axes[0].set_ylabel('Desempenho médio (%)')
# legenda ao lado
from matplotlib.lines import Line2D

handles = [
    Line2D([0],[0], marker=markers[v], linestyle='-',
           color=colors[v], lw=2, markersize=7, label=v)
    for v in vel_order
]
fig.legend(handles, [h.get_label() for h in handles], title='Velocidade',
           loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=False)
fig.tight_layout(rect=[0, 0, 0.86, 1])  # reserva margem direita
fig.tight_layout()
plt.show()


# Interaction Plot sem facetar por Grupo 


# Ordens e rótulos
comp_order = [4,6,8]
vel_order  = ['Lento','Médio','Rápido']
map_comp   = {4:'Fácil', 6:'Intermediário', 8:'Difícil'}

colors  = {'Lento':'#F59E0B','Médio':'#3B82F6','Rápido':'#10B981'}
markers = {'Lento':'o',     'Médio':'s',      'Rápido':'D'}

# Preparar dados
df = df_protA.copy()
df['Complexidade'] = pd.Categorical(df['Complexidade'], categories=comp_order, ordered=True)
df['velocidade']   = pd.Categorical(df['velocidade'],   categories=vel_order,  ordered=True)
df['Desempenho_%'] = 100*df['Desempenho']

# Agregar CV+SV juntos: média, desvio, n e IC95% por (Complexidade, Velocidade)
stats = (df.groupby(['Complexidade','velocidade'])['Desempenho']
           .agg(mean='mean', std='std', count='count').reset_index())

def ci95(std, n):
    if (n is not None) and (n>1) and pd.notnull(std):
        return t.ppf(0.975, df=int(n)-1) * (std/np.sqrt(n))
    return np.nan

stats['ci95'] = stats.apply(lambda r: ci95(r['std'], r['count']), axis=1)

# Plot único
xpos = np.arange(len(comp_order))
xticklabels = [map_comp[c] for c in comp_order]

fig, ax = plt.subplots(figsize=(9,4))

for v in vel_order:
    rows = (stats[stats['velocidade']==v]
            .set_index('Complexidade')
            .reindex(comp_order))
    ymean = rows['mean'].values.astype(float)
    yerr  = rows['ci95'].values.astype(float)

    ax.errorbar(xpos, ymean, yerr=yerr, fmt=markers[v], ms=7, lw=2,
                capsize=4, color=colors[v], label=v)
    ax.plot(xpos, ymean, '-', color=colors[v], lw=2)

ax.set_xticks(xpos); ax.set_xticklabels(xticklabels)
ax.set_xlabel('Complexidade')
ax.set_ylabel('Desempenho médio (%)')
ax.grid(True, ls='--', alpha=.3)
plt.title('Gráfico de Interação - ANOVA 3-way (Complexidade × Velocidade)')
# Legenda ao lado
from matplotlib.lines import Line2D
handles = [Line2D([0],[0], marker=markers[v], linestyle='-', color=colors[v],
                   lw=2, markersize=7, label=v) for v in vel_order]
ax.legend(handles=handles, title='Velocidade',
          loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False)

fig.tight_layout()
fig.subplots_adjust(right=0.82)
plt.show()

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

# --------------------------------------- BNOVB ---------------------------------------
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

# --------------------------------------- Plots ---------------------------------------

# 1. Dotplot
ev.dot_ic_sig(
    df=df_protB,
    x='grupo_complexidade',
    y='Desempenho',
    order=df_protB['grupo_complexidade'].unique(),  
    alpha=0.05,
    show_p_text=True,          # True para escrever p-values
    ylim=(0, 1.1),
    title='Prot B — grupo_complexidade'
)

# 2. Barplot
ev.bar_ic95(
    df=df_protB,
    x='grupo_complexidade',
    y='Desempenho',
    hue='grupo',                            # coluna do grupo
    hue_order=['CF','SF'],                  # opcional
    palette={'CF':'#4C78B8','SF':'#F58518'},  # opcional (ou passe lista de cores)
    ylim=(0, 1.1),
    rotate_xticks=0,
    title='Prot B — Complexidade por Grupo'
)

# 3. Interaction Plot
from scipy.stats import t
import re


# Garantir tipos e escala do desfecho 
df['Desempenho'] = pd.to_numeric(df['Desempenho'], errors='coerce')
use_pct = df['Desempenho'].dropna().between(0, 1).mean() > 0.6  # auto: 0–1 vira %
df['Desempenho_plot'] = df['Desempenho'] * (100 if use_pct else 1.0)
y_label = 'Desempenho médio (%)' if use_pct else 'Desempenho médio'

# Normalizar níveis de Complexidade e Grupo 
comp_raw = df['Complexidade'].astype(str).str.strip()
comp_num = comp_raw.str.extract(r'(\d+)')[0].astype(float)
df['_COMP_LABEL_'] = comp_raw
df['_COMP_NUM_'] = comp_num

# ordem preferida 4–6–8 (ou a ordem numérica presente)
comp_order_num = [c for c in [4, 6, 8] if c in comp_num.dropna().unique()]
if not comp_order_num:
    comp_order_num = sorted(comp_num.dropna().unique().tolist())

# mapeia de volta para os rótulos originais (C4 ou 4) na ordem desejada
order_comp_labels = []
for c in comp_order_num:
    lbls = df.loc[df['_COMP_NUM_'] == c, '_COMP_LABEL_'].dropna().unique()
    order_comp_labels.append(lbls[0] if len(lbls) else str(int(c)))

# rótulos “bonitos” do eixo x (6 -> 'Média')
ticklabels = []
for lab in order_comp_labels:
    m = re.search(r'(\d+)', str(lab))
    if m:
        val = int(m.group(1))
        ticklabels.append({4:'Fácil', 6:'Média', 8:'Difícil'}.get(val, str(lab)))
    else:
        ticklabels.append(str(lab))

# Grupos agora são CF (Com feedback) e SF (Sem feedback)
df['_GRUPO_'] = df['grupo'].astype(str).str.strip().str.upper()
group_order = [g for g in ['CF', 'SF'] if g in df['_GRUPO_'].unique()]
if not group_order:
    group_order = sorted(df['_GRUPO_'].unique().tolist())
title_map = {'CF':'Com feedback', 'SF':'Sem feedback'}

# Agregar: média, desvio, n e IC95% por (Grupo, Complexidade)
stats = (df[['Desempenho_plot', '_GRUPO_', '_COMP_LABEL_']]
         .dropna()
         .groupby(['_GRUPO_', '_COMP_LABEL_'])['Desempenho_plot']
         .agg(mean='mean', std='std', count='count')
         .reset_index())

def ci95(std, n):
    if pd.notnull(std) and n and n > 1:
        sem = std / np.sqrt(n)
        return t.ppf(0.975, df=int(n)-1) * sem
    return np.nan

stats['ci95'] = stats.apply(lambda r: ci95(r['std'], r['count']), axis=1)

# Garante grade completa na ordem desejada
idx = pd.MultiIndex.from_product([group_order, order_comp_labels],
                                 names=['_GRUPO_', '_COMP_LABEL_'])
stats = stats.set_index(['_GRUPO_', '_COMP_LABEL_']).reindex(idx).reset_index()

# Plot: dois painéis lado a lado, média ± IC95% 
xpos = np.arange(len(order_comp_labels))
fig, axes = plt.subplots(1, len(group_order), figsize=(12, 4), sharey=True)
if len(group_order) == 1:
    axes = [axes]

for ax, g in zip(axes, group_order):
    sub = stats[stats['_GRUPO_'] == g].set_index('_COMP_LABEL_').reindex(order_comp_labels)
    ymean = sub['mean'].values.astype(float)
    yerr  = sub['ci95'].values.astype(float)

    ax.errorbar(xpos, ymean, yerr=yerr, fmt='o', ms=7, lw=2, capsize=4)
    ax.plot(xpos, ymean, '-', lw=2, alpha=0.9)

    ax.set_title(f"Grupo: {title_map.get(g, g)}")
    ax.set_xticks(xpos); ax.set_xticklabels(ticklabels)
    ax.set_xlabel('Complexidade')
    ax.grid(True, ls='--', alpha=.3)

axes[0].set_ylabel(y_label)
fig.suptitle('Prot B — Interação Complexidade × Grupo (média ± IC95%)', y=1.05, fontsize=12)
fig.tight_layout()
plt.show()


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

# 2 Barplot
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import t
import numpy as np

# Calcular estatísticas de grupo
group_stats = df_protC.groupby('nivel')['Desempenho'].agg(['mean', 'std', 'count']).reset_index()
group_stats['sem'] = group_stats['std'] / np.sqrt(group_stats['count'])          # Erro padrão
group_stats['t_crit'] = group_stats['count'].apply(lambda n: t.ppf(0.975, df=n-1))  # t crítico IC95%
group_stats['ci95'] = group_stats['t_crit'] * group_stats['sem']                 # Intervalo de confiança
group_stats['nivel'] = pd.Categorical(group_stats['nivel'],categories = ['Fácil', 'Médio', 'Difícil'],ordered=True)
group_stats = group_stats.sort_values('nivel').reset_index(drop=True)
# Renomear colunas para facilitar o uso no barplot
group_stats.rename(columns={'mean': 'Desempenho'}, inplace=True)

# Plot
plt.figure(figsize=(8, 6))
ax = sns.barplot(
    data=group_stats,
    x='nivel',
    y='Desempenho',
    ci=None,                      # não deixa o seaborn desenhar o erro
    color='skyblue',
    edgecolor='black'
)

# HASTE + CAP (limites) do IC95% — agora com a barra do meio!
for i, row in group_stats.iterrows():
    media = row['Desempenho']
    ci = row['ci95']
    ax.errorbar(
        i, media, yerr=ci,
        fmt='none',
        ecolor='black',
        elinewidth=1.5,           # <— HASTE VERTICAL visível
        capsize=6, capthick=1.5   # <— “orelhas” nos limites
    )
    ax.text(i, media + ci + 0.01, f"Média: {media:.2f}\nIC95: ±{ci:.2f}",
            ha='center', va='bottom', fontsize=8, color='black')

# Estética
plt.title('Barplot com Média e Intervalo de Confiança 95% por Complexidade')
plt.ylabel('Desempenho')
plt.xlabel('Complexidade')
plt.xticks(rotation=0)
plt.ylim(0, 1.1)
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()


# 3. Interaction Plot
# --- Interaction plot só por Complexidade (Protocolo C) ---
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t
import re
# (opcional) se quiser ler do CSV:
# import pandas as pd
# df = pd.read_csv('df_protC_execucao.csv')

# 1) Garantir tipos e escala do desfecho
df['Desempenho'] = pd.to_numeric(df['Desempenho'], errors='coerce')
use_pct = df['Desempenho'].dropna().between(0, 1).mean() > 0.6  # auto: 0–1 vira %
df['Desempenho_plot'] = df['Desempenho'] * (100 if use_pct else 1.0)
y_label = 'Desempenho médio (%)' if use_pct else 'Desempenho médio'

# 2) Normalizar níveis de Complexidade
comp_raw = df['Complexidade'].astype(str).str.strip()
comp_num = comp_raw.str.extract(r'(\d+)')[0].astype(float)
df['_COMP_LABEL_'] = comp_raw
df['_COMP_NUM_']   = comp_num

# ordem preferida 4–6–8 (ou a ordem numérica presente)
comp_order_num = [c for c in [4, 6, 8] if c in comp_num.dropna().unique()]
if not comp_order_num:
    comp_order_num = sorted(comp_num.dropna().unique().tolist())

# mapeia para os rótulos originais (ex.: 'C4' ou '4') na ordem desejada
order_comp_labels = []
for c in comp_order_num:
    lbls = df.loc[df['_COMP_NUM_'] == c, '_COMP_LABEL_'].dropna().unique()
    order_comp_labels.append(lbls[0] if len(lbls) else str(int(c)))

# rótulos “bonitos” do eixo x (6 -> 'Média')
ticklabels = []
for lab in order_comp_labels:
    m = re.search(r'(\d+)', str(lab))
    if m:
        val = int(m.group(1))
        ticklabels.append({4:'Fácil', 6:'Média', 8:'Difícil'}.get(val, str(lab)))
    else:
        ticklabels.append(str(lab))

# 3) Agregar: média, desvio, n e IC95% por Complexidade
stats = (df[['Desempenho_plot', '_COMP_LABEL_']]
         .dropna()
         .groupby('_COMP_LABEL_')['Desempenho_plot']
         .agg(mean='mean', std='std', count='count')
         .reindex(order_comp_labels)   # garante a ordem desejada no eixo x
         .reset_index())

def ci95(std, n):
    if np.isfinite(std) and n and n > 1:
        sem = std / np.sqrt(n)
        return t.ppf(0.975, df=int(n)-1) * sem
    return np.nan

stats['ci95'] = [ci95(s, n) for s, n in zip(stats['std'], stats['count'])]

# 4) Plot: média ± IC95% ao longo das complexidades
xpos = np.arange(len(order_comp_labels))
ymean = stats['mean'].values.astype(float)
yerr  = stats['ci95'].values.astype(float)

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.errorbar(xpos, ymean, yerr=yerr, fmt='o', ms=7, lw=2, capsize=4)
ax.plot(xpos, ymean, '-', lw=2, alpha=0.9)

ax.set_xticks(xpos); ax.set_xticklabels(ticklabels)
ax.set_xlabel('Complexidade')
ax.set_ylabel(y_label)
ax.set_title('Prot C — Desempenho por Complexidade (média ± IC95%)')
ax.grid(True, ls='--', alpha=.3)
fig.tight_layout()
plt.show()

#%% Testes para ver como está sendo calculado o intervalo de confiança 
from scipy import stats
print('Resultados do IC para grupo_complexidade') 
for i in df_protA['grupo_complexidade'].unique():
    dados = df_protA[df_protA['grupo_complexidade'] == f'{i}']['Desempenho']
    n = len(dados)
    desvio = np.std(dados,ddof = 1) # Amostral
    # Nível de confiança (ex: 95%)
    conf = 0.95
    # Valor crítico t (bilateral)
    t_crit = stats.t.ppf((1 + conf) / 2, df=n-1)
    erro_padrao = (desvio / np.sqrt(n))
    ic = t_crit * (desvio / np.sqrt(n))
    print('--'*5, f'grupo_complexidade: {i}', '--'*5)
    print(f'Dp = {desvio}, n = {n}, erro padrão = {erro_padrao} ic = {ic}')
#%%
print('Resultados do IC para grupo')
for i in df_protA['grupo'].unique():
    dados = df_protA[df_protA['grupo'] == f'{i}']['Desempenho']
    n = len(dados)
    desvio = np.std(dados,ddof = 1) # Amostral
    # Nível de confiança (ex: 95%)
    conf = 0.95
    # Valor crítico t (bilateral)
    t_crit = stats.t.ppf((1 + conf) / 2, df=n-1)
    erro_padrao = (desvio / np.sqrt(n))
    ic = t_crit * (desvio / np.sqrt(n))
    print('--'*5, f'Grupo: {i}', '--'*5)
    print(f'Dp = {desvio}, n = {n}, erro padrão = {erro_padrao} ic = {ic}')
#%%
print('Resultados do IC para grupo_complexidade_overlap')
for i in df_protA['grupo_complexidade_overlap'].unique():
    dados = df_protA[df_protA['grupo_complexidade_overlap'] == f'{i}']['Desempenho']
    n = len(dados)
    desvio = np.std(dados,ddof = 1) # Amostral
    # Nível de confiança (ex: 95%)
    conf = 0.95
    # Valor crítico t (bilateral)
    t_crit = stats.t.ppf((1 + conf) / 2, df=n-1)
    erro_padrao = (desvio / np.sqrt(n))
    ic = t_crit * (desvio / np.sqrt(n))
    print('--'*5, f'grupo_complexidade_overlap: {i}', '--'*5)
    print(f'Dp = {desvio}, n = {n}, erro padrão = {erro_padrao} ic = {ic}')
# %% Fazendo uma seleção vetorial para melhor visualizar os clusters de cada classe

# 1) Normalizando os dados, para isso vou usar o método Standardscaler do sklearn

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Protocolo A


'''
pca = PCA()
x1 = pca.fit_transform(x1)'''

#caracteristicas = ['Proporção espacial x', 'Proporção espacial y','Acuracia',
#               'Similaridade', 'Especificidade']
caracteristicas = ['Acuracia',
               'Similaridade', 'Especificidade']
x1 = df_protA[caracteristicas].to_numpy() #vetor de características
for classe in ['grupo', 'grupo_complexidade','grupo_complexidade_overlap']:
    print('--'*100)
    print(f'Classe: {classe}')
    y1 = df_protA[classe].to_numpy() #vetor de classes
    y1 = y1.reshape(-1) # fazendo o flatten do vetor de classes (removendo o (x,1))
    x1 = StandardScaler().fit_transform(x1) #normalização
    ev.selecao_vetorial(x1 = x1, y1 = y1, nomes_carac = caracteristicas, k = 2, plotar = True, 
                        interativo = False, salvar_interativo=False)
    print('--'*100)

# %%
