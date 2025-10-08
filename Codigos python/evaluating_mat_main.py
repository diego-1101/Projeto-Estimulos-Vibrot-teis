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

#%% --------------------------------------- ANOVA ---------------------------------------
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

# %% Fazendo uma seleção vetorial para melhor visualizar os clusters de cada classe

# Protocolo A
# 1) Normalizando os dados, para isso vou usar o método Standardscaler do sklearn

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
for classe in ['grupo', 'grupo_complexidade','grupo_complexidade_overlap']:
    print('--'*100)
    print(f'Classe: {classe}')
    y1 = df_protA[classe].to_numpy() #vetor de classes
    y1 = y1.reshape(-1) # fazendo o flatten do vetor de classes (removendo o (x,1))
    x1 = StandardScaler().fit_transform(x1) #normalização
    ev.selecao_vetorial(x1 = x1, y1 = y1, nomes_carac = caracteristicas, k = 3, plotar = True, 
                        interativo = False, salvar_interativo=False)
    print('--'*100)

# %% Fazendo os plots de CDA

x1 = df_protA[caracteristicas].to_numpy() #vetor de características
for classe in ['grupo', 'grupo_complexidade','grupo_complexidade_overlap','Overlap', 'Complexidade']:
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
