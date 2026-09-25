#%% PLS Regression
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_predict, KFold, LeaveOneOut



# ==============================================================================
# FUNÇÃO PLSR COM CÁLCULO DE VIP E PLOTS DE RESULTADOS
# ==============================================================================

def calculate_vip(model: PLSRegression) -> np.array:
    """
    Calcula o VIP Score (Variable Importance in Projection) para um modelo PLS.
    VIP > 1 é o limiar de significância.
    """
    t = model.x_scores_
    w = model.x_weights_
    q = model.y_loadings_
    p, h = w.shape
    vips = np.zeros((p,))
    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    total_s = np.sum(s)
    
    # VIP é calculado como a soma ponderada das variâncias de Y explicadas por cada componente.
    for i in range(p):
        weight = np.array([ (w[i,j] / np.linalg.norm(w[:,j]))**2 for j in range(h) ])
        vips[i] = np.sqrt(p * (s.T @ weight) / total_s)
    return vips


def plsr_and_plot(X: np.array, Y: np.array, feature_names: list = [], 
                  n_components: int = 5,
                  metrics_names: list = [], 
                  main_title: str = "Análise PLS Regression") -> dict:
    """
    Executa a PLS Regression, calcula métricas (incluindo VIP) e gera plots 
    dos scores, VIP Scores e Coeficientes de Regressão.
    
    Args:
        X (np.array): Matriz de preditores (EEG), shape (n_amostras, n_features).
        Y (np.array): Matriz de respostas (Desempenho), shape (n_amostras, n_targets).
        feature_names (list): Nomes das features de X para rotulagem dos gráficos.
        metrics_names (list): Nomes das métricas de Y (e.g., ['Acurácia', 'Similaridade']).
        n_components (int): Número de componentes latentes a extrair.
        main_title (str): Título principal para a figura (suptitle).
        
    Returns:
        dict: Dicionário com as principais métricas e resultados do modelo.
    """
    
    # Força Y a ser 2D, se for um vetor (para compatibilidade com sklearn PLSRegression)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    # Captura as dimensões atuais
    n_features = X.shape[1] 
    n_targets = Y.shape[1]
        
    # 1. Configuração e Treino do Modelo PLS
    # scale=True: Normaliza (Z-score) X e Y automaticamente.
    pls = PLSRegression(n_components=n_components, scale=True)
    pls.fit(X, Y)

    # 2. Extração de Resultados (O que você pediu para retornar)
    T = pls.x_scores_ 	   # T: Scores de X (coordenadas dos sujeitos)
    U = pls.y_scores_ 	   # U: Scores de Y
    beta = pls.coef_ 	   # Coeficientes de Regressão (Beta)
    X_loadings = pls.x_loadings_
    Y_loadings = pls.y_loadings_
    vip_scores = calculate_vip(pls)

    # --- CORREÇÃO DE DIMENSÃO DO BETA E CRIAÇÃO DO DF_BETA ---

    # 3. Preparação dos Nomes das Features
    if not feature_names:
        feature_names = [f'feature_{i}' for i in range(n_features)]
        
    df_vip = pd.DataFrame({'Feature': feature_names, 'VIP': vip_scores})
    
    # --- CORREÇÃO UNIVERSAL PARA O SHAPE DO BETA ---
    # beta sempre deve ficar (n_features, n_targets)
    beta = np.atleast_2d(beta)  # garante 2D

    # Se o número de linhas de beta não bate com o número de features,
    # é porque veio como (n_targets, n_features) e precisa transpor.
    if beta.shape[0] != n_features:
        beta = beta.T

    # Validação e correção dos nomes das métricas
    if not metrics_names:
        metrics_names_final = [f'metrica_{i+1}' for i in range(n_targets)]
    elif len(metrics_names) != n_targets:
        raise ValueError(
            f'Tamanho dos nomes das métricas inválido!\n'
            f'Esperado: {n_targets} (colunas em Y), Recebido {len(metrics_names)}'
        )
    else:
        metrics_names_final = metrics_names
    
    # Criação do DataFrame de Coeficientes Beta (n_features rows, n_targets columns)
    df_beta = pd.DataFrame(beta, columns=metrics_names_final)
    df_beta['Feature'] = feature_names
    df_beta = df_beta.set_index('Feature')


    # 4. Cálculo de Métricas de Qualidade (RESS e R-squared)
    Y_predicted = pls.predict(X)
    
    # Usa 'uniform_average' para lidar corretamente com múltiplos alvos (retorna a média)
    r2 = r2_score(Y, Y_predicted, multioutput='uniform_average')
    rmse = np.sqrt(mean_squared_error(Y, Y_predicted, multioutput='uniform_average'))

    # --- 5. Visualização (Figura 1: Scores e VIP) ---
    
    plt.figure(figsize=(13, 6))
    # APLICANDO O TÍTULO PRINCIPAL (SUPER-TÍTULO)
    plt.suptitle(main_title, fontsize=16, fontweight='bold')
    
    # A) plot dos Coeficientes Beta do modelo de regressão
    plt.subplot(1,2,1)
    
    # Plotar a métrica atual
    df_beta_plot = df_beta[['Desempenho']].sort_values(by='Desempenho', ascending=True)
    
    # Usar cores para indicar a direção da associação (+ ou -)
    colors_beta = ['green' if x > 0 else 'red' for x in df_beta_plot['Desempenho']]
    
    plt.barh(df_beta_plot.index, df_beta_plot['Desempenho'], color=colors_beta)
    plt.axvline(x=0, color='gray', linestyle='-')
    
    # Título referente à métrica
    plt.title(f'Coeficientes Beta: X -> {'Desempenho'}', fontsize=14)
    plt.xlabel('Coeficiente Beta')
    
    plt.grid(axis='x', alpha=0.3)
    
    '''
    # A) Plot dos Scores (Mapa dos Sujeitos)
    plt.subplot(1, 2, 1)
    
    # Colore os sujeitos pelo valor da primeira métrica (índice 0)
    color_data = Y[:, 0].flatten() # Garante que seja 1D para o scatter
    color_label = metrics_names_final[0]
    
    scatter = plt.scatter(T[:, 0], T[:, 1], c=color_data, cmap='viridis', edgecolor='k', alpha=0.7)
    plt.xlabel(f'Componente Latente 1 (T1)')
    plt.ylabel(f'Componente Latente 2 (T2)')
    plt.title(f'Espaço Latente de Sujeitos (Colorido por {color_label})')
    plt.colorbar(scatter, label=color_label)
    plt.grid(True, alpha=0.3)
    '''
    
    # B) Plot da Importância (VIP Scores)
    plt.subplot(1, 2, 2)
    df_vip_plot = df_vip.sort_values(by='VIP', ascending=True) # Ordenar para plot mais limpo
    colors = ['red' if x > 1 else 'gray' for x in df_vip_plot['VIP']]
    
    plt.barh(df_vip_plot['Feature'], df_vip_plot['VIP'], color=colors)
    plt.axvline(x=1, color='blue', linestyle='--', label='Limiar VIP > 1')
    plt.xlabel('VIP Score (Importância da Variável)')
    plt.title('Importância das Features (VIP Score)')
    plt.legend()
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.9])
    plt.show()


    # --- 6. Visualization of Beta Coefficients (Separate Figure for EACH Metric) ---
    
    """# Loop para criar uma figura separada para cada métrica
    for target_col_name in metrics_names_final:
        plt.figure(figsize=(8, 6)) 
        
        # Plotar a métrica atual
        df_beta_plot = df_beta[[target_col_name]].sort_values(by=target_col_name, ascending=True)
        
        # Usar cores para indicar a direção da associação (+ ou -)
        colors_beta = ['green' if x > 0 else 'red' for x in df_beta_plot[target_col_name]]
        
        plt.barh(df_beta_plot.index, df_beta_plot[target_col_name], color=colors_beta)
        plt.axvline(x=0, color='gray', linestyle='-')
        
        # Título referente à métrica
        plt.title(f'Coeficientes Beta: X -> {target_col_name}', fontsize=14)
        plt.xlabel('Coeficiente Beta')
        
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.show()"""


    # 7. Dicionário de Resultados para Return
    results = {
        'n_components': n_components,
        'R2_score': r2,
        'RMSE': rmse,
        'X_scores': T,
        'Y_scores': U,
        'X_loadings': X_loadings,
        'Y_loadings': Y_loadings,
        'Regr_Coefficients_Beta': beta,
        'VIP_Scores': vip_scores,
        'VIP_df': df_vip.sort_values(by='VIP', ascending=False)
    }
    
    return results

def plsr_permutation_bootstrap_validation(
    X: np.array,
    Y: np.array,
    feature_names: list,
    metrics_names: list,
    n_components: int = 5,
    n_permutations: int = 500,
    n_bootstrap: int = 500,
    main_title: str = "Validação PLSR"
) -> dict:
    """
    Faz teste de permutação + bootstrap para um modelo PLSR.

    - Permutação: avalia se o R² do modelo é maior que o esperado ao acaso.
    - Bootstrap: avalia a estabilidade dos coeficientes beta e dos VIP scores.
    """

    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    n_samples, n_features = X.shape
    n_targets = Y.shape[1]

    # ==========================
    # 1. MODELO ORIGINAL
    # ==========================
    pls = PLSRegression(n_components=n_components, scale=True)
    pls.fit(X, Y)

    Y_pred = pls.predict(X)
    r2_real = r2_score(Y, Y_pred, multioutput='uniform_average')

    # Corrige shape dos betas
    beta = np.atleast_2d(pls.coef_)
    if beta.shape[0] != n_features:
        beta = beta.T  # (n_features, n_targets)

    vip = calculate_vip(pls)

    # ==========================
    # 2. TESTE DE PERMUTAÇÃO
    # ==========================
    r2_perm = np.zeros(n_permutations)

    for i in range(n_permutations):
        Y_perm = np.random.permutation(Y)
        pls_perm = PLSRegression(n_components=n_components, scale=True)
        pls_perm.fit(X, Y_perm)
        Y_pred_perm = pls_perm.predict(X)
        r2_perm[i] = r2_score(Y_perm, Y_pred_perm, multioutput='uniform_average')

    # p-valor empírico (proporção de permutações com R² >= R² real)
    p_value = np.mean(r2_perm >= r2_real)

    # ---- Plot Permutation ----
    plt.figure(figsize=(18, 5))
    plt.hist(r2_perm, bins=30, alpha=0.7, color='gray')
    plt.axvline(r2_real, color='red', linewidth=2,
                label=f"R² real = {r2_real:.3f}")
    plt.title(f"{main_title}\nTeste de Permutação (R²) – p = {p_value:.4f}", fontsize=20)
    plt.xlabel("R² com Y permutado (distribuição nula)")
    plt.ylabel("Frequência")
    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.show()

    # ==========================
    # 3. BOOTSTRAP DOS BETAS E VIP
    # ==========================
    betas_boot = np.zeros((n_bootstrap, n_features, n_targets))
    vips_boot = np.zeros((n_bootstrap, n_features))

    for b in range(n_bootstrap):
        idx = np.random.randint(0, n_samples, size=n_samples)
        Xb = X[idx]
        Yb = Y[idx]

        pls_b = PLSRegression(n_components=n_components, scale=True)
        pls_b.fit(Xb, Yb)

        beta_b = np.atleast_2d(pls_b.coef_)
        if beta_b.shape[0] != n_features:
            beta_b = beta_b.T

        betas_boot[b] = beta_b
        vips_boot[b] = calculate_vip(pls_b)

    # Médias e desvios
    beta_mean = betas_boot.mean(axis=0)          # (n_features, n_targets)
    beta_std = betas_boot.std(axis=0, ddof=1)    # (n_features, n_targets)

    vip_mean = vips_boot.mean(axis=0)            # (n_features,)
    vip_std = vips_boot.std(axis=0, ddof=1)      # (n_features,)

    # Evita divisão por zero
    beta_std[beta_std == 0] = np.nan
    vip_std[vip_std == 0] = np.nan

    beta_br = beta_mean / beta_std              # bootstrap ratio ~ z-score
    vip_br = vip_mean / vip_std                 # (n_features,)

    # ==========================
    # 4. PLOTS – BOOTSTRAP RATIOS
    # ==========================

    # ---- VIP BR ----
    order_vip = np.argsort(np.abs(vip_br))  # ordena por importância absoluta
    plt.figure(figsize=(8, 6))
    plt.barh(
        np.array(feature_names)[order_vip],
        vip_br[order_vip],
        color=['green' if x > 0 else 'red' for x in vip_br[order_vip]]
    )
    plt.axvline(0, color='black', linewidth=1)
    plt.axvline(2, color='blue', linestyle='--', label='|BR| = 2')
    plt.axvline(-2, color='blue', linestyle='--')
    plt.title(f"{main_title}\nBootstrap ratio dos VIP (estabilidade das features)")
    plt.xlabel("Bootstrap ratio (VIP)")
    plt.ylabel("Feature")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ---- Betas BR por métrica ----
    for t in range(n_targets):
        metric_name = metrics_names[t] if metrics_names else f"Métrica_{t+1}"
        br_t = beta_br[:, t]
        order_beta = np.argsort(np.abs(br_t))

        plt.figure(figsize=(8, 6))
        plt.barh(
            np.array(feature_names)[order_beta],
            br_t[order_beta],
            color=['green' if x > 0 else 'red' for x in br_t[order_beta]]
        )
        plt.axvline(0, color='black', linewidth=1)
        plt.axvline(2, color='blue', linestyle='--', label='|BR| = 2')
        plt.axvline(-2, color='blue', linestyle='--')
        plt.title(f"{main_title}\nBootstrap ratio dos coeficientes Beta – {metric_name}")
        plt.xlabel("Bootstrap ratio (Beta)")
        plt.ylabel("Feature")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Resultado para você guardar se quiser
    return {
        "r2_real": r2_real,
        "r2_perm_distribution": r2_perm,
        "p_value_perm": p_value,
        "beta_mean": beta_mean,
        "beta_br": beta_br,
        "vip_mean": vip_mean,
        "vip_br": vip_br
    }


#psd_canais = ['psd_delta_CZ', 'psd_theta_CZ', 'psd_alfa_CZ', 'psd_beta_CZ', 'psd_gamma_CZ',
#       'psd_delta_C3', 'psd_theta_C3', 'psd_alfa_C3', 'psd_beta_C3', 'psd_gamma_C3',
#       'psd_delta_C4', 'psd_theta_C4', 'psd_alfa_C4', 'psd_beta_C4', 'psd_gamma_C4']
psd_canais = ['psd_norm_delta_CZ', 'psd_norm_theta_CZ', 'psd_norm_alfa_CZ',
       'psd_norm_beta_CZ', 'psd_norm_gamma_CZ', 'psd_norm_delta_C3',
       'psd_norm_theta_C3', 'psd_norm_alfa_C3', 'psd_norm_beta_C3',
       'psd_norm_gamma_C3', 'psd_norm_delta_C4', 'psd_norm_theta_C4',
       'psd_norm_alfa_C4', 'psd_norm_beta_C4', 'psd_norm_gamma_C4']


# --------- Protocolo A --------- 

Y_desempenho = df_A_final['Desempenho'].to_numpy()
Y_desempenho = Y_desempenho[:,np.newaxis]
Y_metricas = df_A_final[['Acuracia','Similaridade','Especificidade']].to_numpy()

dict_X_A_bandas = {
    'CV': df_A_final[df_A_final['grupo']=='CV'][psd_canais].to_numpy(),
    'SV': df_A_final[df_A_final['grupo']=='SV'][psd_canais].to_numpy(),
    'juntos': df_A_final[psd_canais].to_numpy(),
}

''' Parte com a PSD completa (sem segmentar por bandas)
# Cria a nova coluna 'CZ'
df_A_final['CZ'] = df_A_final['psd_trecho'].apply(lambda x: x.get('CZ')[1])

# Cria a nova coluna 'C3'
df_A_final['C3'] = df_A_final['psd_trecho'].apply(lambda x: x.get('C3')[1])

# Cria a nova coluna 'C4'
df_A_final['C4'] = df_A_final['psd_trecho'].apply(lambda x: x.get('C4')[1])

df_CZ_psd = pd.DataFrame(df_A_final['CZ'].tolist())
df_CZ_psd['grupo'] = df_A_final['grupo'].to_numpy()

df_C3_psd = pd.DataFrame(df_A_final['C3'].tolist())
df_C3_psd['grupo'] = df_A_final['grupo'].to_numpy()

df_C4_psd = pd.DataFrame(df_A_final['C4'].tolist())
df_C4_psd['grupo'] = df_A_final['grupo'].to_numpy()

dict_canais = {
    'CZ': df_CZ_psd,
    'C3': df_C3_psd,
    'C4': df_C4_psd
}

grupos = ['CV', 'SV']

df_canais_psd = pd.concat([df_CZ_psd[df_CZ_psd.columns[:-1]],
                           df_C3_psd[df_C3_psd.columns[:-1]],
                           df_C4_psd], 
                           axis=1)

dict_X_A = {
    'CV': df_canais_psd[df_canais_psd['grupo'] == 'CV'][df_canais_psd.columns[:-1]].to_numpy(),
    'SV': df_canais_psd[df_canais_psd['grupo'] == 'SV'][df_canais_psd.columns[:-1]].to_numpy(),
    'juntos': df_canais_psd[df_canais_psd.columns[:-1]].to_numpy(),
}


#----- Fazendo a PLSR para X = psd dos canais e Y = Desempenho -----
results_psd_completo_1 = []
for grupo, X in dict_X_A.items():
    if grupo != 'juntos':
        Y = Y_desempenho[df_A_final['grupo'] == grupo]
    else: Y = Y_desempenho
    results_psd_completo_1.append(plsr_and_plot(X = X,Y= Y,feature_names = []))

#----- Fazendo a PLSR para X = psd dos canais e Y = métrica -----
results_psd_completo_2 = []
for grupo, X in dict_X_A.items():
    if grupo != 'juntos':
        Y = Y_metricas[df_A_final['grupo'] == grupo]
    else: Y = Y_metricas
    plsr_and_plot(X = X,Y= Y,feature_names = [])
    results_psd_completo_2.append(plsr_and_plot(X = X,Y= Y,feature_names = []))
'''


channels = ['Cz', 'C3', 'C4']
bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
feature_names = [f"{ch}_{b}" for ch in channels for b in bands]

#----- Fazendo a PLSR para X = potencia bandas e Y = Desempenho -----
results_plsr_A1 = []
for grupo, X in dict_X_A_bandas.items():
    if grupo != 'juntos':
        Y = Y_desempenho[df_A_final['grupo'] == grupo]
        titulo = f'Análise PLS Regression do protocolo A {grupo}'
    else: 
        Y = Y_desempenho
        titulo = f'Análise PLS Regression do protocolo A completo'
    results_plsr_A1.append(plsr_and_plot(X = X,Y= Y,feature_names = feature_names,
                                         main_title=titulo,metrics_names=['Desempenho']))

    
# ===== Validação – Protocolo A – Y = Desempenho =====
valid_plsr_A1 = {}  # dicionário pra guardar, se quiser

for grupo, X in dict_X_A_bandas.items():
    if grupo != 'juntos':
        Y = Y_desempenho[df_A_final['grupo'] == grupo]
        titulo_val = f"PLSR – Protocolo A – {grupo} – Desempenho"
    else:
        Y = Y_desempenho
        titulo_val = f"PLSR – Protocolo A – juntos – Desempenho"

    print(f"\n\n### Validação PLSR – Protocolo A – {grupo} – Desempenho ###")
    valid_plsr_A1[grupo] = plsr_permutation_bootstrap_validation(
        X=X,
        Y=Y,
        feature_names=feature_names,
        metrics_names=['Desempenho'],
        n_components=5,          # mesmo valor usado na plsr_and_plot
        n_permutations=500,      # ajuste se estiver pesado
        n_bootstrap=500,
        main_title=titulo_val
    )

#%%----- Fazendo a PLSR para X = potencia bandas e Y = métricas -----

results_plsr_A2 = []
metricas = ['Acuracia','Similaridade','Especificidade']
for grupo, X in dict_X_A_bandas.items():
    if grupo != 'juntos':
        Y = Y_metricas[df_A_final['grupo'] == grupo]
        titulo = f'Análise PlS Regression do protocolo A {grupo}'
    else: 
        Y = Y_metricas
        titulo = f'Análise PlS Regression do protocolo A completo'
    results_plsr_A2.append(plsr_and_plot(X = X,Y= Y,feature_names = feature_names,
                                         main_title=titulo,metrics_names=metricas))

# ===== Validação – Protocolo A – Y = Acurácia, Similaridade, Especificidade =====
valid_plsr_A2 = {}

metricas = ['Acuracia','Similaridade','Especificidade']

for grupo, X in dict_X_A_bandas.items():
    if grupo != 'juntos':
        Y = Y_metricas[df_A_final['grupo'] == grupo]
        titulo_val = f"PLSR – Protocolo A – {grupo} – Métricas"
    else:
        Y = Y_metricas
        titulo_val = f"PLSR – Protocolo A – juntos – Métricas"

    print(f"\n\n### Validação PLSR – Protocolo A – {grupo} – Métricas (Acurácia, Similaridade, Especificidade) ###")
    valid_plsr_A2[grupo] = plsr_permutation_bootstrap_validation(
        X=X,
        Y=Y,
        feature_names=feature_names,
        metrics_names=metricas,
        n_components=5,
        n_permutations=500,
        n_bootstrap=500,
        main_title=titulo_val
    )


# --------- Protocolo B --------- 

Y_desempenho = df_B_final['Desempenho'].to_numpy()
Y_desempenho = Y_desempenho[:,np.newaxis]
Y_metricas = df_B_final[['Acuracia','Similaridade','Especificidade']].to_numpy()

dict_X_B_bandas = {
    'CF': df_B_final[df_B_final['grupo']=='CF'][psd_canais].to_numpy(),
    'SF': df_B_final[df_B_final['grupo']=='SF'][psd_canais].to_numpy(),
    'juntos': df_B_final[psd_canais].to_numpy(),
}

#----- Fazendo a PLSR para X = potencia bandas e Y = Desempenho -----
results_plsr_B1 = []
for grupo, X in dict_X_B_bandas.items():
    if grupo != 'juntos':
        Y = Y_desempenho[df_B_final['grupo'] == grupo]
        titulo = f'Análise PLS Regression do protocolo B {grupo}'
    else: 
        Y = Y_desempenho
        titulo = f'Análise PLS Regression do protocolo B completo'
    results_plsr_B1.append(plsr_and_plot(X = X,Y= Y,feature_names = feature_names,
                                         main_title=titulo,metrics_names=['Desempenho']))

# ===== Validação – Protocolo B – Y = Desempenho =====
valid_plsr_B1 = {}

for grupo, X in dict_X_B_bandas.items():
    if grupo != 'juntos':
        Y = Y_desempenho[df_B_final['grupo'] == grupo]
        titulo_val = f"PLSR – Protocolo B – {grupo} – Desempenho"
    else:
        Y = Y_desempenho
        titulo_val = f"PLSR – Protocolo B – juntos – Desempenho"

    print(f"\n\n### Validação PLSR – Protocolo B – {grupo} – Desempenho ###")
    valid_plsr_B1[grupo] = plsr_permutation_bootstrap_validation(
        X=X,
        Y=Y,
        feature_names=feature_names,
        metrics_names=['Desempenho'],
        n_components=5,
        n_permutations=500,
        n_bootstrap=500,
        main_title=titulo_val
    )


#%%----- Fazendo a PLSR para X = potencia bandas e Y = métricas -----

results_plsr_B2 = []
metricas = ['Acuracia','Similaridade','Especificidade']
for grupo, X in dict_X_B_bandas.items():
    if grupo != 'juntos':
        Y = Y_metricas[df_B_final['grupo'] == grupo]
        titulo = f'Análise PlS Regression do protocolo B {grupo}'
    else: 
        Y = Y_metricas
        titulo = f'Análise PlS Regression do protocolo B completo'
    results_plsr_B2.append(plsr_and_plot(X = X,Y= Y,feature_names = feature_names,
                                         main_title=titulo,metrics_names=metricas))

# ===== Validação – Protocolo B – Y = Acurácia, Similaridade, Especificidade =====
valid_plsr_B2 = {}

metricas = ['Acuracia','Similaridade','Especificidade']

for grupo, X in dict_X_B_bandas.items():
    if grupo != 'juntos':
        Y = Y_metricas[df_B_final['grupo'] == grupo]
        titulo_val = f"PLSR – Protocolo B – {grupo} – Métricas"
    else:
        Y = Y_metricas
        titulo_val = f"PLSR – Protocolo B – juntos – Métricas"

    print(f"\n\n### Validação PLSR – Protocolo B – {grupo} – Métricas ###")
    valid_plsr_B2[grupo] = plsr_permutation_bootstrap_validation(
        X=X,
        Y=Y,
        feature_names=feature_names,
        metrics_names=metricas,
        n_components=5,
        n_permutations=500,
        n_bootstrap=500,
        main_title=titulo_val
    )


#%% Plotando o espaço latente dos protocolos A e B em 2D e em 3D
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # necessário para 3D

def plot_pls_latent_space(results_plsr: dict,
                          group_labels,
                          n_dims: int = 2,
                          title: str = "Espaço latente PLSR (Protocolo A - CV + SV)"):
    """
    Plota os dados projetados no espaço latente da PLSR (scores de X),
    usando as duas ou três primeiras componentes (PLs) e colorindo
    CV e SV com cores diferentes.

    Args
    ----
    results_plsr : dict
        Dicionário retornado pela função plsr_and_plot (precisa conter 'X_scores').
    group_labels : array-like, shape (n_samples,)
        Vetor com o grupo de cada amostra (ex.: 'CV' ou 'SV').
        A ordem deve ser a mesma das linhas usadas na PLSR.
    n_dims : int, {2, 3}
        Número de dimensões a plotar (2D ou 3D).
    title : str
        Título do gráfico.
    """

    # Scores de X (T) – já vêm do modelo PLS treinado
    T = results_plsr['X_scores']          # shape (n_samples, n_components)
    n_samples, n_components = T.shape

    group_labels = np.array(group_labels)

    if group_labels.shape[0] != n_samples:
        raise ValueError(
            f"Número de labels ({group_labels.shape[0]}) "
            f"não bate com número de amostras em T ({n_samples})."
        )

    if n_dims not in [2, 3]:
        raise ValueError("n_dims deve ser 2 ou 3.")
    if n_dims > n_components:
        raise ValueError(
            f"n_dims={n_dims}, mas o modelo só tem {n_components} componentes."
        )

    grupos_unicos = np.unique(group_labels)
    cores = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

    if n_dims == 2:
        plt.figure(figsize=(7, 6))
        for i, g in enumerate(grupos_unicos):
            mask = (group_labels == g)
            plt.scatter(T[mask, 0],
                        T[mask, 1],
                        label=str(g),
                        color=cores[i % len(cores)],
                        alpha=0.8,
                        edgecolor='k')
        plt.axhline(0, color='gray', linewidth=0.5)
        plt.axvline(0, color='gray', linewidth=0.5)
        plt.xlabel("PL1 (T1)")
        plt.ylabel("PL2 (T2)")
        plt.title(title + " - 2D")
        plt.legend(title="Grupo")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    else:  # n_dims == 3
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection='3d')

        for i, g in enumerate(grupos_unicos):
            mask = (group_labels == g)
            ax.scatter(T[mask, 0],
                       T[mask, 1],
                       T[mask, 2],
                       label=str(g),
                       color=cores[i % len(cores)],
                       alpha=0.8,
                       edgecolor='k')

        ax.set_xlabel("PL1 (T1)")
        ax.set_ylabel("PL2 (T2)")
        ax.set_zlabel("PL3 (T3)")
        ax.set_title(title + " - 3D")
        ax.legend(title="Grupo")
        plt.tight_layout()
        plt.show()

# --------- Protocolo A --------- 

# Métrica: Desempenho
# Pegando o dicionário de resultados do caso 'juntos'
results_plsr_A1_juntos = None
for (grupo, _), res in zip(dict_X_A_bandas.items(), results_plsr_A1):
    if grupo == 'juntos':
        results_plsr_A1_juntos = res
        break

# Labels de grupo (CV / SV) para TODAS as amostras usadas em 'juntos'
labels_juntos = df_A_final['grupo'].values   # mesma ordem de X_juntos

# Plot 2D (PL1 x PL2)
plot_pls_latent_space(results_plsr_A1_juntos,
                      group_labels=labels_juntos,
                      n_dims=2,
                      title="Protocolo A - CV + SV- Métrica: Desempenho")

# Se quiser em 3D (PL1 x PL2 x PL3):
plot_pls_latent_space(results_plsr_A1_juntos,
                      group_labels=labels_juntos,
                      n_dims=3,
                      title="Protocolo A - CV + SV- Métrica: Desempenho")

# -- Métrica: ['Acuracia','Similaridade','Especificidade']

# Pegando o dicionário de resultados do caso 'juntos'
results_plsr_A2_juntos = None
for (grupo, _), res in zip(dict_X_A_bandas.items(), results_plsr_A2):
    if grupo == 'juntos':
        results_plsr_A2_juntos = res
        break

# Labels de grupo (CV / SV) para TODAS as amostras usadas em 'juntos'
labels_juntos = df_A_final['grupo'].values   # mesma ordem de X_juntos

# Plot 2D (PL1 x PL2)
plot_pls_latent_space(results_plsr_A2_juntos,
                      group_labels=labels_juntos,
                      n_dims=2,
                      title="Protocolo A - CV + SV- Métrica: Acurácia, Similaridade, Especificidade")

# Se quiser em 3D (PL1 x PL2 x PL3):
plot_pls_latent_space(results_plsr_A2_juntos,
                      group_labels=labels_juntos,
                      n_dims=3,
                      title="Protocolo A - CV + SV- Métrica:  Acurácia, Similaridade, Especificidade")

# --------- Protocolo B --------- 

# -- Métrica: Desempenho
# Pegando o dicionário de resultados do caso 'juntos'
results_plsr_B1_juntos = None
for (grupo, _), res in zip(dict_X_B_bandas.items(), results_plsr_B1):
    if grupo == 'juntos':
        results_plsr_B1_juntos = res
        break

# Labels de grupo (CF / SF) para TODAS as amostras usadas em 'juntos'
labels_juntos = df_B_final['grupo'].values   # mesma ordem de X_juntos

# Plot 2D (PL1 x PL2)
plot_pls_latent_space(results_plsr_B1_juntos,
                      group_labels=labels_juntos,
                      n_dims=2,
                      title="Protocolo B - CF + SF- Métrica: Desempenho")

# Se quiser em 3D (PL1 x PL2 x PL3):
plot_pls_latent_space(results_plsr_B1_juntos,
                      group_labels=labels_juntos,
                      n_dims=3,
                      title="Protocolo B - CF + SF- Métrica: Desempenho")

# -- Métrica: ['Acuracia','Similaridade','Especificidade']
# Pegando o dicionário de resultados do caso 'juntos'
results_plsr_B2_juntos = None
for (grupo, _), res in zip(dict_X_B_bandas.items(), results_plsr_B2):
    if grupo == 'juntos':
        results_plsr_B2_juntos = res
        break

# Labels de grupo (CF / SF) para TODAS as amostras usadas em 'juntos'
labels_juntos = df_B_final['grupo'].values   # mesma ordem de X_juntos

# Plot 2D (PL1 x PL2)
plot_pls_latent_space(results_plsr_B2_juntos,
                      group_labels=labels_juntos,
                      n_dims=2,
                      title="Protocolo B - CF + SF- Métrica:  Acurácia, Similaridade, Especificidade")

# Se quiser em 3D (PL1 x PL2 x PL3):
plot_pls_latent_space(results_plsr_B2_juntos,
                      group_labels=labels_juntos,
                      n_dims=3,
                      title="Protocolo B - CF + SF- Métrica:  Acurácia, Similaridade, Especificidade")




#%% PLOTANDO AS PSD'S médias e cada canal
import numpy as np
import matplotlib.pyplot as plt

# ======================================================
# 1) Função auxiliar: calcula PSD média ± SD em dB
# ======================================================
def compute_psd_stats(df, canal, grupo=None):
    """
    Calcula frequências, média e desvio-padrão da PSD em dB
    para um determinado canal e (opcionalmente) um grupo.

    df: dataframe (df_A_final ou df_B_final)
    canal: 'CZ', 'C3' ou 'C4'
    grupo: valor em df['grupo'] ou None para TODOS os trials
    """
    mask = df['psd_trecho'].notna()
    if grupo is not None:
        mask &= (df['grupo'] == grupo)

    psd_series = df.loc[mask, 'psd_trecho'].apply(lambda d: d.get(canal) if d is not None else None)

    freqs = None
    psd_stack = []

    for item in psd_series:
        if item is None:
            continue
        f, p = item[0], item[1]   # [0] = frequências, [1] = PSD (linear)
        if freqs is None:
            freqs = np.asarray(f)
        psd_stack.append(np.asarray(p))

    # Se não tiver dado suficiente
    if len(psd_stack) == 0:
        return None, None, None

    psd_stack = np.vstack(psd_stack)  # (n_trials, n_freqs)

    # --- Converter cada trial para dB (forma correta) ---
    psd_db = 10 * np.log10(psd_stack + 1e-12)  # evita log(0)

    mean_psd_db = psd_db.mean(axis=0)
    sd_psd_db   = psd_db.std(axis=0)

    return freqs, mean_psd_db, sd_psd_db


# ======================================================
# 2) Função principal: Figura 1 (PSD média ± SD em dB)
# ======================================================
def plot_psd_figure1(df, protocolo_label, group_order, palette_group,
                     canais=('CZ', 'C3', 'C4'), freq_max=None):
    """
    Gera Figura 1: PSD média ± 1 SD em dB por canal,
    comparando grupos + curva GERAL (todos os trials).

    df: df_A_final ou df_B_final
    protocolo_label: string para título geral (ex.: 'Protocolo A')
    group_order: lista com nomes dos grupos (ex.: ['CV', 'SV'])
    palette_group: dict {grupo: cor}
    canais: tupla com canais a plotar
    freq_max: frequência máxima a ser mostrada (Hz), ex.: 40
    """
    stats_dict = {}
    y_min, y_max = np.inf, -np.inf

    # -----------------------------------------------
    # Pré-cálculo para todos os canais e grupos
    # -----------------------------------------------
    for canal in canais:
        stats_dict[canal] = {}

        # Grupos
        for g in group_order:
            freqs, mean_psd, sd_psd = compute_psd_stats(df, canal, grupo=g)
            if freqs is None:
                continue

            stats_dict[canal][g] = (freqs, mean_psd, sd_psd)

            y_min = min(y_min, np.min(mean_psd - sd_psd))
            y_max = max(y_max, np.max(mean_psd + sd_psd))

        # Curva GERAL (todos os trials)
        freqs_all, mean_all, sd_all = compute_psd_stats(df, canal, grupo=None)
        if freqs_all is None:
            continue

        stats_dict[canal]['GERAL'] = (freqs_all, mean_all, sd_all)

        y_min = min(y_min, np.min(mean_all - sd_all))
        y_max = max(y_max, np.max(mean_all + sd_all))

    # -----------------------------------------------
    # Cria figura
    # -----------------------------------------------
    fig, axes = plt.subplots(
        1, len(canais),
        figsize=(14, 4),
        sharey=True
    )

    if len(canais) == 1:
        axes = [axes]

    for ax, canal in zip(axes, canais):
        if canal not in stats_dict or 'GERAL' not in stats_dict[canal]:
            continue

        # --- 1) Curva GERAL (cinza tracejada) ---
        freqs_all, mean_all, sd_all = stats_dict[canal]['GERAL']

        if freq_max is not None:
            mask_freq_all = freqs_all <= freq_max
        else:
            mask_freq_all = slice(None)

        f_all = freqs_all[mask_freq_all]
        m_all = mean_all[mask_freq_all]
        sd_all_plot = sd_all[mask_freq_all]

        ax.plot(
            f_all,
            m_all,
            color='gray',
            linewidth=2,
            linestyle='--',
            label='Geral'
        )

        ax.fill_between(
            f_all,
            m_all - sd_all_plot,
            m_all + sd_all_plot,
            color='gray',
            alpha=0.25
        )

        # --- 2) Curvas dos grupos ---
        for g in group_order:
            if g not in stats_dict[canal]:
                continue

            freqs, mean_psd, sd_psd = stats_dict[canal][g]

            if freq_max is not None:
                mask_freq = freqs <= freq_max
            else:
                mask_freq = slice(None)

            f_plot = freqs[mask_freq]
            m_plot = mean_psd[mask_freq]
            sd_plot = sd_psd[mask_freq]

            ax.plot(
                f_plot,
                m_plot,
                label=g,
                color=palette_group[g],
                linewidth=2
            )

            ax.fill_between(
                f_plot,
                m_plot - sd_plot,
                m_plot + sd_plot,
                color=palette_group[g],
                alpha=0.2
            )

        ax.set_title(canal, fontsize=13)
        ax.set_xlabel('Frequência (Hz)', fontsize=12)
        ax.grid(axis='y', linestyle='--', linewidth=0.6, alpha=0.4)

    axes[0].set_ylabel('Potência Espectral (dB)', fontsize=12)

    # Mesma escala de Y em todos os canais
    if y_max > y_min:
        margin = 0.05 * (y_max - y_min)
        for ax in axes:
            ax.set_ylim(y_min - margin, y_max + margin)

    ax.set_ylim(-40)

    # Título geral
    fig.suptitle(f'PSD média ± 1 SD por canal – {protocolo_label}', fontsize=15)

    # Legenda única (no último eixo)
    axes[-1].legend(frameon=False, fontsize=11, loc='upper right')

    plt.tight_layout()
    plt.show()


# ======================================================
# 3) Chamadas para Protocolo A e Protocolo B
# ======================================================

# ---------- Protocolo A ----------
order_A = ['CV', 'SV']
palette_A = {
    'CV': '#4C72B0',   # azul
    'SV': '#DD8452'    # laranja
}

plot_psd_figure1(
    df_A_final,
    protocolo_label='Protocolo A',
    group_order=order_A,
    palette_group=palette_A,
    canais=('CZ', 'C3', 'C4'),
    freq_max=40  # ou outro limite em Hz que faça sentido
)

# ---------- Protocolo B ----------
order_B = ['CF', 'SF']
palette_B = {
    'CF': '#4C72B0',   # azul
    'SF': '#DD8452'    # laranja
}

plot_psd_figure1(
    df_B_final,
    protocolo_label='Protocolo B',
    group_order=order_B,
    palette_group=palette_B,
    canais=('CZ', 'C3', 'C4'),
    freq_max=40
)



#%% Salvando algunsa arquivos para mandar para o Jean
# Cria a nova coluna 'CZ'
df_A_final['CZ'] = df_A_final['psd_trecho'].apply(lambda x: x.get('CZ')[1])

# Cria a nova coluna 'C3'
df_A_final['C3'] = df_A_final['psd_trecho'].apply(lambda x: x.get('C3')[1])

# Cria a nova coluna 'C4'
df_A_final['C4'] = df_A_final['psd_trecho'].apply(lambda x: x.get('C4')[1])

# Visualizar as colunas resultantes
print(df_A_final[['psd_trecho', 'CZ', 'C3', 'C4', 'grupo']].head())

df_CZ_psd = pd.DataFrame(df_A_final['CZ'].tolist())
df_CZ_psd['grupo'] = df_A_final['grupo']

df_C3_psd = pd.DataFrame(df_A_final['C3'].tolist())
df_C3_psd['grupo'] = df_A_final['grupo']

df_C4_psd = pd.DataFrame(df_A_final['C4'].tolist())
df_C4_psd['grupo'] = df_A_final['grupo']


# 1. Crie um dicionário mapeando o 'sufixo' do nome ao DataFrame
dict_canais = {
    'CZ': df_CZ_psd,
    'C3': df_C3_psd,
    'C4': df_C4_psd
}

grupos = ['CV', 'SV']

# 2. Itere sobre a chave (nome) e o valor (df)
for nome_canal, df_canal in dict_canais.items():
    for grupo in grupos:
        # Monta o nome do arquivo dinamicamente
        nome_arquivo = f'X_{grupo}_psd_{nome_canal}.csv'
        
        # Filtra e salva
        # Nota: Adicionei a lógica de f-string correta no to_csv
        df_canal[df_canal['grupo'] == grupo][df_canal.columns[:-1]].to_csv(nome_arquivo)
        
        print(f"Salvo: {nome_arquivo}")



#%% === UI interativa para plotar resultados da PLSC (Jupyter) ===
# Requisitos: ipywidgets, matplotlib
# Se precisar: !pip install ipywidgets && jupyter nbextension enable --py widgetsnbextension

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import traceback

# ---- verificação ipywidgets ----
try:
    import ipywidgets as w
except ImportError as e:
    print("⚠️ ipywidgets não encontrado.\n"
          "Instale e habilite com:\n"
          "  pip install ipywidgets\n"
          "  jupyter nbextension enable --py widgetsnbextension\n"
          "Depois, reinicie o kernel e execute esta célula novamente.")
    raise

# ========= helpers =========
def _looks_like_plsc_dict(d):
    return isinstance(d, dict) and {"U","V","S","LX","LY","explained_cov","Xz","Yz","R"}.issubset(d.keys())

def _discover_plsc_dicts(ns):
    """Retorna {nome: dict} para todos os dicionários PLSC no escopo."""
    return {n: o for n, o in ns.items() if not n.startswith("_") and _looks_like_plsc_dict(o)}

def _find_groups_for_dict(dct, ns):
    """Tenta achar o vetor 'grupo' compatível com o n de amostras do dict."""
    n = dct["LX"].shape[0]
    for name, obj in ns.items():
        try:
            import pandas as pd
            if isinstance(obj, pd.DataFrame) and len(obj) == n and "grupo" in obj.columns:
                return obj["grupo"].to_numpy()
        except Exception:
            pass
    return np.array(["G"] * n)  # fallback

def _get_x_labels(ns, dct):
    if "psd_canais" in ns and isinstance(ns["psd_canais"], (list, tuple)):
        return list(ns["psd_canais"])
    return [f"x{i+1}" for i in range(dct["V"].shape[0])]

def _get_y_labels(dct):
    return [f"y{i+1}" for i in range(dct["U"].shape[0])]

def _call_plot(func_name, dict_name, dct, controls, ns):
    """Encapsula as chamadas dos gráficos e faz validações básicas."""
    lv  = int(controls["lv"].value)
    lvx = int(controls["lvx"].value)
    lvy = int(controls["lvy"].value)

    max_lv = dct["V"].shape[1] - 1
    lv  = max(0, min(lv,  max_lv))
    lvx = max(0, min(lvx, max_lv))
    lvy = max(0, min(lvy, max_lv))

    xlabels = _get_x_labels(ns, dct)
    ylabels = _get_y_labels(dct)
    groups  = _find_groups_for_dict(dct, ns)

    g = globals()
    if func_name not in g:
        raise NameError(f"Função '{func_name}' não encontrada no escopo.")

    fn = g[func_name]

    if func_name == "plot_saliences_bars":
        return fn(dct["V"], xlabels, lv=lv, title=f"{dict_name} – X saliences (LV{lv+1})")
    elif func_name == "plot_saliences_bars_Y":
        return fn(dct["U"], ylabels, lv=lv, title=f"{dict_name} – Y saliences (LV{lv+1})")
    elif func_name == "plot_scores_coupled":
        return fn(dct["LX"], dct["LY"], groups, lv=lv, title=f"{dict_name} – LX vs LY (LV{lv+1})")
    elif func_name == "plot_pca_style":
        return fn(dct["LX"], groups, lvx=lvx, lvy=lvy, title=f"{dict_name} – LX LV{lvx+1} vs LV{lvy+1}")
    elif func_name == "plot_permutation_scree":
        perm_name = f"perm_sing_vals_{dict_name}"
        if perm_name not in g:
            raise NameError(f"Permutações '{perm_name}' não encontradas.")
        return fn(dct["S"], g[perm_name], title=f"{dict_name} – Permutation Scree")
    elif func_name == "plot_bootstrap_ratios":
        boot_name = f"boot_se_{dict_name}"
        if boot_name not in g:
            raise NameError(f"Bootstrap SE '{boot_name}' ausente.")
        return fn(dct["V"], g[boot_name], xlabels, lv=lv, title=f"{dict_name} – Bootstrap Ratios (LV{lv+1})")
    elif func_name == "plot_xy_correlation_heatmap":
        return fn(dct["Xz"], dct["Yz"], xlabels, ylabels, title=f"{dict_name} – Corr(Y,X)")
    elif func_name == "plot_biplot_V":
        return fn(dct["V"], dct["LX"], xlabels, lvx=lvx, lvy=lvy, title=f"{dict_name} – Biplot V & LX")
    else:
        raise ValueError(f"Gráfico '{func_name}' não suportado.")

# ========= descobrir os dicts disponíveis no seu notebook =========
_plsc_dicts = _discover_plsc_dicts(globals())    # {nome: dict}
_dict_options = sorted(_plsc_dicts.keys())

# ========= opções de gráficos (funções que você forneceu) =========
_graph_options = [
    "plot_saliences_bars",
    "plot_saliences_bars_Y",
    "plot_scores_coupled",
    "plot_pca_style",
    "plot_permutation_scree",
    "plot_bootstrap_ratios",
    "plot_xy_correlation_heatmap",
    "plot_biplot_V",
]

# ========= widgets =========
dd_dicts = w.SelectMultiple(options=_dict_options, description="Protocolos", layout=w.Layout(width="320px"))
dd_graph = w.Dropdown(options=_graph_options, description="Gráfico", layout=w.Layout(width="320px"))

lv  = w.BoundedIntText(value=0, min=0, max=99, description="LV")
lvx = w.BoundedIntText(value=0, min=0, max=99, description="LVx")
lvy = w.BoundedIntText(value=1, min=0, max=99, description="LVy")

btn = w.Button(description="Plotar", button_style="primary")
out = w.Output(layout=w.Layout(border="1px solid #777", padding="10px", min_height="380px"))

_controls = {"lv": lv, "lvx": lvx, "lvy": lvy}

def _on_plot(_):
    clear_output(wait=True)
    display(ui)
    out.clear_output()
    with out:
        sel = list(dd_dicts.value)
        if not sel:
            print("Selecione ao menos um protocolo/dict (esquerda).")
            return
        graph = dd_graph.value
        errors = []
        for name in sel:
            dct = _plsc_dicts.get(name)
            if dct is None:
                print(f"[AVISO] Dict '{name}' indisponível.")
                continue
            try:
                fig = _call_plot(graph, name, dct, _controls, globals())
                #display(fig)
                plt.show(fig)
            except Exception as e:
                errors.append((name, e, traceback.format_exc()))
        if errors:
            print("\n— Ocorreram erros —")
            for name, e, tb in errors:
                print(f"[{name}] Falhou '{graph}': {repr(e)}\n{tb}")

btn.on_click(_on_plot)

top = w.HBox([dd_dicts, dd_graph, w.VBox([lv, lvx, lvy, btn])])
ui = w.VBox([top, out])
display(ui)
#%% Interface interativa (Notebook) para explorar PSDs normalizadas
# ===============================================================
import ipywidgets as widgets
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
import numpy as np

# --- IMPORTANTE ---
# As variáveis abaixo DEVEM existir no notebook:
# df_especifico_norm, df_geral_norm
# e as funções: plot_bandas_psd, plot_psd_media_psd_canais, plot_psd_media_individuos

# ===============================================================
#   1. Estrutura base
# ===============================================================
fontes = {
    "Específico (normalizado)": df_especifico_norm,
    "Geral (normalizado)": df_geral_norm
}

# Elementos da UI
fonte_dd = widgets.Dropdown(options=list(fontes.keys()), description='Fonte:')
dataset_dd = widgets.Dropdown(description='Dataset:')
modo_dd = widgets.Dropdown(
    options=[
        "Bandas por indivíduo (vários canais)",
        "Média por indivíduo (todos os canais)",
        "Média por canal (entre indivíduos)"
    ],
    description='Modo:'
)
individuo_dd = widgets.Dropdown(description='Indivíduo:')
canais_select = widgets.SelectMultiple(description='Canais:', rows=6)
plot_btn = widgets.Button(description='Plotar', button_style='success')
output = widgets.Output()

# ===============================================================
#   2. Funções auxiliares
# ===============================================================
def inferir_canais(dfm):
    """Lê os nomes dos canais do primeiro indivíduo válido."""
    for _, linha in dfm.iterrows():
        psd = linha.get('psds', None)
        if psd is not None:
            return [str(c).strip() for c in psd.index.tolist()]
    return []

def atualizar_datasets(change=None):
    """Atualiza lista de datasets quando muda a fonte."""
    fonte_nome = fonte_dd.value
    datasets = sorted(list(fontes[fonte_nome].keys()))
    dataset_dd.options = datasets
    if datasets:
        dataset_dd.value = datasets[0]
        atualizar_individuos(None)

def atualizar_individuos(change=None):
    """Atualiza lista de indivíduos e canais conforme dataset."""
    df_master = fontes[fonte_dd.value][dataset_dd.value]
    individuo_dd.options = list(df_master.index)
    canais_select.options = inferir_canais(df_master)
    if len(canais_select.options) >= 3:
        canais_select.value = tuple(canais_select.options[:3])

def on_plot(_):
    """Executa o plot de acordo com o modo selecionado."""
    with output:
        clear_output(wait=True)
        df_master = fontes[fonte_dd.value][dataset_dd.value]
        modo = modo_dd.value
        canais = list(canais_select.value)
        ind = individuo_dd.value

        if not canais:
            print("⚠️ Selecione ao menos um canal.")
            return

        if modo.startswith("Bandas"):
            # plota as curvas de bandas para 1 indivíduo e vários canais
            _ = plot_bandas_psd(
                df_master,
                ind,
                canais=canais,
                faixa_total=(0.5, 100),
                mostrar_relativo=True,
                escala_db=False
            )

        elif modo.startswith("Média por indivíduo"):
            _ = plot_psd_media_canais(
                df_master,
                ind,
                faixa_total=(0.5, 100),
                escala_db=False,
                erro_padrao_habilitado=True
            )

        else:  # Média por canal (entre indivíduos)
            for ch in canais:
                _ = plot_psd_media_individuos(
                    df_master,
                    ch,
                    faixa_total=(0.5, 100),
                    escala_db=False,
                    erro_padrao_habilitado=True
                )
        plt.show()

# ===============================================================
#   3. Ligações de eventos
# ===============================================================
fonte_dd.observe(atualizar_datasets, names='value')
dataset_dd.observe(atualizar_individuos, names='value')
plot_btn.on_click(on_plot)

# Inicialização
atualizar_datasets()

# ===============================================================
#   4. Layout do painel
# ===============================================================
controls_left = widgets.VBox([
    fonte_dd,
    dataset_dd,
    modo_dd,
    individuo_dd,
    canais_select,
    plot_btn
])

ui = widgets.HBox([controls_left, output])
display(ui)



# %% Plotar os topoplot (rever) 

import numpy as np
import matplotlib.pyplot as plt
import mne

# Bandas de interesse
bands = {
    'Total': (1, 80),
    'Delta': (1, 4),
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta':  (13, 30),
    'Gamma': (30, 80)
}

# ---------- 1. Selecionar os dados ----------
df = df_especifico_norm['psd_ProtA_CV_especifico_df_norm']
row = df.iloc[0]  # <- escolha o sujeito desejado

psd_df = row['psds']
freqs = row['freqs']
ch_names = row['ch_labels']
subject_id = row['ind']

# ---------- 2. Extrair potência média por banda ----------
def compute_band_power(psd_df, band):
    fmin, fmax = band

    # ignora a coluna "canal" se existir (aparentemente agora não tem mais)
    freq_cols = [col for col in psd_df.columns if isinstance(col, str)]
    
    # converte os nomes das colunas para float para comparação
    float_cols = [float(c) for c in freq_cols]
    
    # máscara booleana
    mask = [(f >= fmin) and (f <= fmax) for f in float_cols]
    
    # seleciona colunas dentro da banda (volta para string)
    selected_cols = [f"{f:.1f}" for f, m in zip(float_cols, mask) if m]
    
    return psd_df[selected_cols].mean(axis=1).values

# ---------- 3. Criar objeto MNE info ----------
info = mne.create_info(ch_names=['Fp1', 'Fp2', 'Fz', 'F3', 'F4', 'F7', 'F8', 'Cz', 'C3', 'C4', 'T7', 'T8',
 'P7', 'P8', 'Pz', 'P3', 'P4', 'O1', 'O2', 'FCz', 'FC1', 'FC2', 'FC3',
 'Oz', 'C2', 'CP1', 'CP3', 'CP4', 'C1', 'FC4', 'CPz', 'CP2'], sfreq=1000.0, ch_types='eeg')



# Aplica a montagem padrão
info.set_montage('standard_1020')


# ---------- 4. Gerar os topoplots ----------
fig, axes = plt.subplots(2, 3, figsize=(12, 6))
axes = axes.ravel()

for i, (band_name, band_range) in enumerate(bands.items()):
    powers = compute_band_power(psd_df, band_range)
    mne.viz.plot_topomap(powers, info, axes=axes[i], show=False)
    axes[i].set_title(band_name)

fig.suptitle(f'Topoplots - Sujeito {subject_id}')
plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import mne

def topomaps_bandas_individuo(df_master, ind,
                              bands=None,
                              faixa_total=(1, 80),
                              titulo_prefixo=None,
                              cmap='viridis',
                              vlim_auto=True):
    """
    Plota 6 topoplots (Total, Delta, Theta, Alpha, Beta, Gamma) para um indivíduo
    a partir do DataFrame mestre (colunas 'freqs' e 'psds': DF canais x freqs).
    """
    if bands is None:
        bands = {
            'Total': (1, 80),
            'Delta': (1, 4),
            'Theta': (4, 8),
            'Alpha': (8, 13),
            'Beta' : (13, 30),
            'Gamma': (30, 80),
        }

    # --- pegar linha do indivíduo ---
    label = _resolver_indice(df_master, ind)
    row   = df_master.loc[label]

    psd_df = row['psds'].copy()
    freqs  = np.asarray(row['freqs'], dtype=float)
    # garantir colunas numéricas = frequências reais
    try:
        psd_df.columns = np.asarray(psd_df.columns, dtype=float)
    except Exception:
        pass

    # limitar às frequências de interesse (faixa_total ∩ colunas existentes)
    fmin, fmax = faixa_total
    mask_total = (freqs >= fmin) & (freqs <= fmax)
    freqs_use  = freqs[mask_total]
    psd_df     = psd_df.loc[:, psd_df.columns.intersection(freqs_use)]

    # mapear nomes para o padrão do MNE (Fp1, Fp2, Fz, Cz, C3, C4, …)
    def _to_mne_name(ch):
        return str(ch).strip().title().replace('Pz','Pz').replace('Cz','Cz')  # .title() já resolve 99%

    psd_df.index = [_to_mne_name(ch) for ch in psd_df.index]

    # --- montar lista de canais que o MNE conhece e que existem no DF ---
    montage = mne.channels.make_standard_montage('standard_1020')
    ch_valid = [ch for ch in psd_df.index if ch in montage.ch_names]
    if not ch_valid:
        raise ValueError("Nenhum canal com posição conhecida no montage 10-20.")

    # restringir DF aos canais válidos e ordenar alfabeticamente (evita surpresas)
    psd_df = psd_df.loc[ch_valid]

    # criar Info com exatamente estes canais (ordem do vetor deve bater com info.ch_names)
    info = mne.create_info(ch_names=list(psd_df.index), sfreq=1000.0, ch_types='eeg')
    info.set_montage(montage)

    # ---- função de potência por integração (melhor que média) ----
    def _potencia(y, x):
        return float(np.trapezoid(y, x)) if y.size and x.size else np.nan

    # calcular potência por canal para cada banda
    def _band_power_matrix(psd_df, freqs):
        chans = list(psd_df.index)
        # potência total (na faixa_total efetiva)
        m_tot = (freqs >= fmin) & (freqs <= fmax)
        total_vec = np.array([_potencia(psd_df.loc[ch, m_tot].values, freqs[m_tot]) for ch in chans])

        band_vecs = {}
        for nome, (lo, hi) in bands.items():
            lo_eff, hi_eff = max(lo, fmin), min(hi, fmax, freqs.max())
            m = (freqs >= lo_eff) & (freqs <= hi_eff)
            if not np.any(m):
                band_vecs[nome] = np.full(len(chans), np.nan)
            else:
                band_vecs[nome] = np.array([_potencia(psd_df.loc[ch, m].values, freqs[m]) for ch in chans])
        return chans, total_vec, band_vecs

    chans, total_vec, band_vecs = _band_power_matrix(psd_df, psd_df.columns.values.astype(float))

    # mesma escala entre mapas?
    if vlim_auto:
        vmin = vmax = None
    else:
        all_vals = np.concatenate([total_vec] + [band_vecs[k] for k in ('Delta','Theta','Alpha','Beta','Gamma') if k in band_vecs])
        vmin, vmax = np.nanpercentile(all_vals, [5, 95])

    # plotar 6 mapas
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    axes = axes.ravel()

    # helper p/ compatibilidade de versões do MNE (vmin/vmax pode não existir)
    def _plot_topomap(vals, ax, title):
        try:
            mne.viz.plot_topomap(vals, info, axes=ax, contours=0, cmap=cmap,
                                 show=False)
        except TypeError:
            mne.viz.plot_topomap(vals, info, axes=ax, contours=0, cmap=cmap, show=False)
        ax.set_title(title)

    _plot_topomap(total_vec, axes[0], 'Total')

    order = ['Delta','Theta','Alpha','Beta','Gamma']
    for ax, nome in zip(axes[1:], order):
        _plot_topomap(band_vecs[nome], ax, nome)

    tit = f"{titulo_prefixo} — " if titulo_prefixo else ""
    fig.suptitle(f"{tit}Indivíduo {ind}")
    plt.tight_layout()
    plt.show()