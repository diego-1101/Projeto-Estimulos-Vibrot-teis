# Antigravity — Implementar v2 do EEG PSD Dashboard (Prot A/B/C + CDA MATLAB-like + X/Y + Domain Both)

# Importante
Este markdown foi gerado a partir deste prompt abaxio. Entenda a necessidade inicial e foque principalmente no que será escrito depois do bloco abaixo.

```text
Okay. Agora eu estou escrevendo um prompt para deixar em .md para que o agente do Antigravity faça um dashboard para que eu consiga fazer deploy desse dashboard e eu, meu orientador e minha co-orinetadora consigamos interagir e visualizar os dados de forma prática e online. Os arquivos que eu te mandei são o começo desse dashboard (leia-os com atenção para entender a estrutura do dashboard e faça perguntas pertinente se eu esquecer de citar algo crucial que quero manter ou modificar) que já está em funcionamento com deploy pela Render, só que eu preciso fazer algumas alterações para que eu consiga principalmente implementar os plots que este codigo em matlab implemente. Dito isso, eu vou mandar aqui nesta mensagem tudo o que eu quero editar e você vai organizar as ideias e escrever o prompt para que eu envie para o antigravity, implemente isso e o dashboard já esteja funcionando em menos de 30 min. 1. Agora, temos os dados de três protocolos: Protocolo A (que possui dois grupos- CV e SV), Protocolo B (que possui dois grupos CF e SF) e o Protocolo C (que não possui distinção de grupos). Todos os dados necessários estão no formato .csv na pasta "data". Para cada protocolo, temos um Data Frame nomeado como "analise_df_{letra_protocolo}_final.csv", ex o protocolo A tem nome "analise_df_A_final.csv". As colunas de cada data frame são: - df_A_final.columns: ['Tempo 1', 'Tempo 2', 'Tempo 3', 'ID', 'grupo', 'Desempenho', 'Acuracia', 'Similaridade', 'Especificidade', 'Protocolo', 'Complexidade', 'Tempo_inicio', 'Delta_t1', 'Delta_t2', 'Delta_t3', 'd1_s', 'd2_s', 'd3_s', 'Trecho_eeg', 'idx_ini', 'idx_fim', 'n_amostras', '_trecho_info', 'psd_delta_CZ', 'psd_theta_CZ', 'psd_alfa_CZ', 'psd_beta_CZ', 'psd_gamma_CZ', 'psd_delta_C3', 'psd_theta_C3', 'psd_alfa_C3', 'psd_beta_C3', 'psd_gamma_C3', 'psd_delta_C4', 'psd_theta_C4', 'psd_alfa_C4', 'psd_beta_C4', 'psd_gamma_C4', 'psd_trecho', 'psd_norm_delta_CZ', 'psd_norm_theta_CZ', 'psd_norm_alfa_CZ', 'psd_norm_beta_CZ', 'psd_norm_gamma_CZ', 'psd_norm_delta_C3', 'psd_norm_theta_C3', 'psd_norm_alfa_C3', 'psd_norm_beta_C3', 'psd_norm_gamma_C3', 'psd_norm_delta_C4', 'psd_norm_theta_C4', 'psd_norm_alfa_C4', 'psd_norm_beta_C4', 'psd_norm_gamma_C4', 'CZ', 'C3', 'C4'] - df_B_final: ['Tempo 1', 'Tempo 2', 'Tempo 3', 'ID', 'grupo', 'Desempenho', 'Acuracia', 'Similaridade', 'Especificidade', 'Protocolo', 'Complexidade', 'Tempo_inicio', 'Delta_t1', 'Delta_t2', 'Delta_t3', 'd1_s', 'd2_s', 'd3_s', 'Trecho_eeg', 'idx_ini', 'idx_fim', 'n_amostras', '_trecho_info', 'psd_delta_CZ', 'psd_theta_CZ', 'psd_alfa_CZ', 'psd_beta_CZ', 'psd_gamma_CZ', 'psd_delta_C3', 'psd_theta_C3', 'psd_alfa_C3', 'psd_beta_C3', 'psd_gamma_C3', 'psd_delta_C4', 'psd_theta_C4', 'psd_alfa_C4', 'psd_beta_C4', 'psd_gamma_C4', 'psd_trecho', 'psd_norm_delta_CZ', 'psd_norm_theta_CZ', 'psd_norm_alfa_CZ', 'psd_norm_beta_CZ', 'psd_norm_gamma_CZ', 'psd_norm_delta_C3', 'psd_norm_theta_C3', 'psd_norm_alfa_C3', 'psd_norm_beta_C3', 'psd_norm_gamma_C3', 'psd_norm_delta_C4', 'psd_norm_theta_C4', 'psd_norm_alfa_C4', 'psd_norm_beta_C4', 'psd_norm_gamma_C4', 'CZ', 'C3', 'C4'] - df_C_final: ['Tempo 1', 'Tempo 2', 'ID', 'Desempenho', 'Acuracia', 'Similaridade', 'Especificidade', 'Protocolo', 'Complexidade', 'Tempo_inicio', 'd1_s', 'd2_s', 'Trecho_eeg', 'idx_ini', 'idx_fim', 'n_amostras', '_trecho_info', 'psd_delta_CZ', 'psd_theta_CZ', 'psd_alfa_CZ', 'psd_beta_CZ', 'psd_gamma_CZ', 'psd_delta_C3', 'psd_theta_C3', 'psd_alfa_C3', 'psd_beta_C3', 'psd_gamma_C3', 'psd_delta_C4', 'psd_theta_C4', 'psd_alfa_C4', 'psd_beta_C4', 'psd_gamma_C4', 'psd_trecho', 'n_traj', 'novo_tempo_inicio', 'tamanho_original_trial', 'ordem_trial', 't_inicial', 't_final', 'Problema', 'psd_norm_delta_CZ', 'psd_norm_theta_CZ', 'psd_norm_alfa_CZ', 'psd_norm_beta_CZ', 'psd_norm_gamma_CZ', 'psd_norm_delta_C3', 'psd_norm_theta_C3', 'psd_norm_alfa_C3', 'psd_norm_beta_C3', 'psd_norm_gamma_C3', 'psd_norm_delta_C4', 'psd_norm_theta_C4', 'psd_norm_alfa_C4', 'psd_norm_beta_C4', 'psd_norm_gamma_C4', 'CZ', 'C3', 'C4'] Perceba que tudo o que for necessário plotar, está nas colunas desses data frames. 2. Eu quero que a CDA seja feita exatamente como a função do matlab faz. Assim, quero que você escreva no prompt a maneira que é feito para que ele implemente isso corretamente e consiga ter o exato resultado (tanto numérico quanto gráfico) da função do matlab. Coloque o codigo ou a documentação se for necessário. 3. O esquema dos gráficos interativos em plotly devem se manter os mesmos. Inclusive a opção de poder escolher ter dois gráficos independentes lado a lado para comparação (Enable Comparasion). 4. Quero que adicione o protocolo C na escolha dos protocolos e mude o color by de acordo com cada protocolo (o protocolo C só pode ser colorido por complexidade, o B só pode ser colorido por grupo e/ou complexidade e o A pode ser colorido por grupo e/ou complexidade e/ou overlap, ou seja, é preciso que tenha uma seleção inteligente e, nos que tem mais de uma possibilidade, a possibilidade de colorir conjuntamente a duas possibilidades) 5. Quero que tire toda a lógica do Covariance Mode. Quando o método de análise for escolhido, só será computado de acordo com as funções normais que geralmente usam autocovariância. 6. Mantenha os seguintes métodos de Análise: PCA, LDA, CDA (com a alteração para seguir exatamente o que o Matlab faz) e PLS. 7. Quero que adicione duas novas opções obrigatória que serão para escolher X e Y: - para X as seguintes opções: -- PSD trecho completo: aqui X será composto pelas colunas 'CZ', 'C3' e 'C4' do data frame daquele protocolo uma do lado da outra. -- PSD estratificada por bandas de frequência: aqui X será composto pelas colunas 'psd_delta_CZ', 'psd_theta_CZ', 'psd_alfa_CZ', 'psd_beta_CZ', 'psd_gamma_CZ', 'psd_delta_C3', 'psd_theta_C3', 'psd_alfa_C3', 'psd_beta_C3', 'psd_gamma_C3', 'psd_delta_C4', 'psd_theta_C4', 'psd_alfa_C4', 'psd_beta_C4' e 'psd_gamma_C4' do data frame daquele protocolo uma do lado da outra. -- PSD estratificada por bandas de frequência e normalizada: aqui X será composto pelas colunas 'psd_norm_delta_CZ','psd_norm_theta_CZ', 'psd_norm_alfa_CZ', 'psd_norm_beta_CZ', 'psd_norm_gamma_CZ', 'psd_norm_delta_C3', 'psd_norm_theta_C3', 'psd_norm_alfa_C3', 'psd_norm_beta_C3', 'psd_norm_gamma_C3', 'psd_norm_delta_C4', 'psd_norm_theta_C4', 'psd_norm_alfa_C4', 'psd_norm_beta_C4' e 'psd_norm_gamma_C4' do data frame daquele protocolo uma do lado da outra. - para Y eu quero que seja uma checkbox em que a pessoa irá checar as caixas que ela quer que Y contenha: -- Caixa Desempenho: coluna 'Desempenho' do data frame daquele protocolo. -- Caixa Acurácia: coluna 'Acuracia' do data frame daquele protocolo. -- Caixa Similaridade: coluna 'Similaridade' do data frame daquele protocolo. -- Caixa Especificidade: coluna 'Especificidade' do data frame daquele protocolo. -- Caixa Proporção Espacial x: 'Proporção espacial x' data frame daquele protocolo. -- Caixa Proporção Espacial y: 'Proporção espacial y' data frame daquele protocolo. As checkboxes que forem escolhidas irão compor o Y final, onde cada check será uma coluna de Y. 8. Na opção Data Domain é o que mais vai ter alteração e que preciso que tenha atenção e, caso fique confuso o que eu expliquei você me pergunta para poder explicar perfeitamente para a AI. Aqui eu quero que seja possível ver no domínio do X-> PSD's (PSD Feature), no domínio do Y -> Desempenho e em ambos. Quando eu colocar ambos, é preciso que apareça uma outra selection box para que eu selecione o que vai aparecer em cada eixo. Ou seja, vamos supor que eu escolhi um gráfico de 3 dimensões, com PCA sendo a análise, o X sendo a PSD do trecho completo, o Y sendo o Desempenho, Acuracia e Sensibilidade e o Data Domain eu coloco Ambos. Deve aparecer uma selection box para cada eixo onde eu posso escolher entre "PC1 de X", "PC2 de X", "PC3 de X" (sempre limite até a terceira PC/LD/PLS/Canônica), "PC1 de Y", "PC2 de Y" e PC3 de Y" (note que a depender do tipo de análise e do Y escolhido, podemos não ter mais de 1 ou de 2 eixos , então deve ser feita uma verificação de dimensão de cada componente para que informe no selecionador apenas as existentes para não dar erro depois). Se eu selecionei, por exemplo, PC1 de X para o eixo 1, PC2 de X para o eixo 2 e PC1 de Y para o eixo 3, o gráfico plotado deve ser com essas exatas três dimensões, ou seja, os dados de X projetados em PC1 do X no primeiro eixo, os dados de X projetados em PC2 do X no segundo eixo e os dados de Y projetados em PC1 do Y no terceiro eixo, de tal forma que as coordenadas sejam as combinações destes tres eixos escolhidos. 9. Não se esqueça que eu não quero que nada na interface alem disso que eu comentei mude. Então, tem que manter a lógica dos dois gráficos independentes para comparar, a lógica de escolher entre tema claro e escuro, etc.
```


Você está trabalhando em um app Dash (Plotly) já em produção (Render) com comparação lado-a-lado, tema claro/escuro e um botão "Run Analysis". **Não altere nada na UI além do que for explicitamente pedido aqui.** Mantenha o Comparison Mode, o tema e o layout geral.

## Contexto do código atual (para manter compatibilidade)
- `run_single_analysis` hoje carrega dados via `load_and_preprocess_data(protocol)` e escolhe `X` pelo `domain` (`psd` ou `bx`) e usa `Y_labels = meta['grupo']`. Depois chama `compute_embedding` e concatena `embedding + meta` para plotar (2D/3D) :contentReference[oaicite:6]{index=6}.
- A lógica de cor atual cria `color_label` por grupo/complexidade/overlap e combinações :contentReference[oaicite:7]{index=7}.
- O “Data Domain” atual só tem PSD vs Behavioral no dropdown :contentReference[oaicite:8]{index=8}.
- O loader atual exige `ID` e `grupo` :contentReference[oaicite:9]{index=9} e monta PSD com `psd_norm_*` e fallback `psd_` numérico :contentReference[oaicite:10]{index=10}.
- O engine atual tem `compute_embedding(X, Y_labels, Y_continuous, method, covariance_mode, n_components)` e implementa PCA / LDA / PLS / "CDA" (auto=LDA, cross=PLSCanonical/CCA) :contentReference[oaicite:11]{index=11}.

## Objetivo geral da v2
1) Suportar **3 protocolos**:
   - Prot A: grupos CV/SV; pode colorir por grupo, complexidade, overlap e combinações.
   - Prot B: grupos CF/SF; pode colorir por grupo, complexidade e combinações.
   - Prot C: sem grupos; só pode colorir por complexidade (e se não houver complexidade em alguma linha, lidar com fallback robusto).
2) Remover totalmente a lógica de **Covariance Mode** (UI + engine + callbacks).
3) Manter os métodos: **PCA, LDA, CDA, PLS**.
4) Implementar **CDA exatamente como MATLAB `manova1`** (mesmo resultado numérico e comportamento).
5) Adicionar seleções obrigatórias para **X** e **Y**:
   - X (dropdown):
     a) "PSD trecho completo" => concat colunas `CZ`, `C3`, `C4` (cada uma contém um vetor/array serializado; precisamos parsear e concatenar ao longo das features).
     b) "PSD estratificada por bandas" => usar colunas psd_* por bandas: 
        `psd_delta_CZ`, `psd_theta_CZ`, `psd_alfa_CZ`, `psd_beta_CZ`, `psd_gamma_CZ`,
        `psd_delta_C3`, `psd_theta_C3`, `psd_alfa_C3`, `psd_beta_C3`, `psd_gamma_C3`,
        `psd_delta_C4`, `psd_theta_C4`, `psd_alfa_C4`, `psd_beta_C4`, `psd_gamma_C4`.
     c) "PSD estratificada por bandas (normalizada)" => usar colunas `psd_norm_*` equivalentes.
   - Y (checkboxes que compõem colunas do Y final):
     - Desempenho => `Desempenho`
     - Acurácia => `Acuracia`
     - Similaridade => `Similaridade`
     - Especificidade => `Especificidade`
     - Proporção Espacial x => **confirmar nome exato da coluna no CSV** (ver pergunta no final)
     - Proporção Espacial y => **confirmar nome exato da coluna no CSV**

6) Alterar “Data Domain” para suportar: **X only**, **Y only**, **Both (mix axes)**.
   - Se Both: mostrar 3 dropdowns (Axis 1/2/3) (apenas os necessários para 2D ou 3D).
   - Cada eixo permite escolher: `C1 de X`, `C2 de X`, `C3 de X`, `C1 de Y`, `C2 de Y`, `C3 de Y`, mas **apenas os que existem** (ex.: LDA pode ter só 1 componente; se Y tiver 1 variável talvez só 1 componente para PCA de Y, etc.).
   - O plot final deve usar exatamente os eixos selecionados (ex.: eixo1=PC1 de X, eixo2=PC2 de X, eixo3=PC1 de Y).

7) Manter o esquema de gráficos interativos Plotly e o Comparison Mode (dois painéis independentes).

---

# A) Atualizações de DATA LOADING (data_loader.py)

## A1) Arquivos e nomes
Hoje o loader procura `data/df_{protocol}_final.csv` :contentReference[oaicite:12]{index=12}.
Atualize para aceitar ambos padrões:
- `data/analise_df_{protocol}_final.csv` (novo padrão informado)
- fallback: `data/df_{protocol}_final.csv` (padrão antigo)
Use o primeiro que existir.

## A2) Protocolo C sem coluna `grupo`
O loader atual falha se não existir `grupo` :contentReference[oaicite:13]{index=13}.
Regras:
- Prot A e B: manter `grupo` do CSV.
- Prot C: criar `df['grupo'] = 'ALL'` (constante) para manter compatibilidade de meta/hover.
- Remover `dropna(subset=['grupo'])` ou adaptar: se Prot C, não drop por grupo; se A/B, manter.

## A3) Metadados extras
Hoje carrega Complexidade/Overlap de CSVs externos para A/B :contentReference[oaicite:14]{index=14}.
Regras novas:
- Preferir usar as colunas já existentes no `analise_df_*_final.csv` (você disse que tudo já está no DF).
- Só usar os CSVs auxiliares como fallback (se não existir coluna no DF).
- Para Prot C: se `Complexidade` existe, incluir; se não existir, seguir robusto (sem quebrar).

## A4) Construção das features X e Y (NOVA API DO LOADER)
Troque `load_and_preprocess_data` para retornar:
- `df` completo (limpo) + `meta` (ID, grupo, Complexidade, Overlap quando existir)
- Um helper que constrói `X` a partir de `x_mode`
- Um helper que constrói `Y` a partir das checkboxes

Sugestão:
- `load_data(protocol) -> df_clean, meta`
- `build_X(df_clean, x_mode) -> X (pd.DataFrame or np.ndarray) + feature_names`
- `build_Y(df_clean, y_cols_selected) -> Y (pd.DataFrame)`

### X mode details
1) PSD trecho completo: colunas `CZ`, `C3`, `C4` são arrays serializados (string). Parse:
   - aceitar formatos: JSON-like (`[1,2,3]`), python-list string, ou string com espaços.
   - converter para `np.array(float)` e concatenar `CZ|C3|C4` -> vetor final por amostra.
   - retorno: `X` shape `(n_samples, 3*Nfreq)` com nomes tipo `CZ_f001`, ..., `C3_f001`, ..., `C4_f001` (ou nomes genéricos se Nfreq desconhecido).
2) PSD bandas: usar colunas listadas (psd_delta_CZ ... psd_gamma_C4) como features escalares.
3) PSD bandas normalizada: usar colunas psd_norm_* equivalentes.

### Y details
- Y é DataFrame com colunas na ordem das checkboxes marcadas.
- Validar: pelo menos 1 checkbox marcada; caso contrário, mostrar erro amigável no plot.

### Normalização
No loader atual vocês fazem z-score dos PSD e dos comportamentais :contentReference[oaicite:15]{index=15}.
Agora:
- Para PCA/LDA/PLS, padronize com StandardScaler por default (igual atual).
- Para CDA MATLAB-like, siga o MATLAB: ele centraliza (mean global) internamente e usa SSCP; **não** precisa z-score obrigatório (mas pode oferecer opção? NÃO adicionar UI nova — então: padronize somente para PCA/PLS, e para CDA use só centralização, para bater MATLAB).

---

# B) Remover “Covariance Mode” (app.py + analysis_engine.py)

## B1) UI
Remover completamente o bloco “Covariance Mode” do `create_analysis_controls` (RadioItems) e qualquer callback que dependa disso. Hoje existe e influencia disable_domains :contentReference[oaicite:16]{index=16}.

## B2) Engine
- Remover o argumento `covariance_mode` de `compute_embedding` e eliminar caminhos "cross/auto".
- PLS: sempre será **X -> Y** (PLSRegression), usando `Y` montado pelas checkboxes.
- CDA: sempre será **MATLAB-like manova1** (ver seção C).
- LDA: supervisionado por labels (ver seção D).
- PCA: normal.

---

# C) Implementar CDA exatamente como MATLAB `manova1`

## C1) Exigência
CDA deve produzir:
- `canon` (scores canônicos) como `C1..Ck` para plot.
- `stats` compatíveis: Wilks lambda, chisq, p-values por dimensão, eigenvalues, eigenvectors, etc.
- O objetivo é reproduzir o resultado numérico/geométrico do MATLAB `manova1`.

## C2) Implementação Python (use SciPy)
Crie um módulo `manova1_matlab.py` e implemente:

- Entrada:
  - `X` (np.ndarray)
  - `group` (array-like labels)
  - `alpha=0.05`

- Saída:
  - `d` (int)
  - `p` (np.ndarray)
  - `stats` (dict) com:
    - `W`, `B`, `T`, `dfW`, `dfB`, `dfT`
    - `lambda`, `chisq`, `chisqdf`, `eigenval`, `eigenvec`, `canon`, `mdist`, `gmdist`, `gnames`

Use este código (não inventar; copiar fielmente):

```python
import numpy as np
from scipy.linalg import cholesky, solve_triangular, eigh
from scipy.stats import chi2

def manova1_like_matlab(X, group, alpha=0.05):
    X = np.asarray(X, dtype=float)
    g = np.asarray(list(group), dtype=object).reshape(-1)

    # remove NaNs em X
    no_nan_X = ~np.isnan(X).any(axis=1)
    is_nan_mask_original = ~no_nan_X
    X2 = X[no_nan_X, :]
    g2 = g[no_nan_X]

    # grp2idx estável (ordem de aparição)
    label_to_idx = {}
    group_names = []
    group_idx = np.empty(X2.shape[0], dtype=int)
    for i, lab in enumerate(g2):
        if lab not in label_to_idx:
            label_to_idx[lab] = len(group_names)
            group_names.append(lab)
        group_idx[i] = label_to_idx[lab]

    nsample, nvar = X2.shape
    ngroups = len(group_names)

    xm = X2.mean(axis=0)
    x_centered = X2 - xm
    TSSP = x_centered.T @ x_centered

    WSSP = np.zeros((nvar, nvar), dtype=float)
    for j in range(ngroups):
        rows = np.where(group_idx == j)[0]
        if rows.size > 1:
            gx = x_centered[rows, :]
            gx = gx - gx.mean(axis=0)
            WSSP += gx.T @ gx

    BSSP = TSSP - WSSP

    R = cholesky(WSSP, lower=False, check_finite=False)
    S = solve_triangular(R.T, BSSP, lower=True, check_finite=False)
    S = solve_triangular(R, S.T, lower=False, check_finite=False).T
    S = 0.5 * (S + S.T)

    evals, evecs = eigh(S, check_finite=False)  # asc
    e = evals
    vv = evecs

    v = solve_triangular(R, vv, lower=False, check_finite=False)

    ei = np.argsort(e)
    e = e[ei]
    v = v[:, ei]
    if np.min(e) <= -1:
        raise ValueError("singular sum of squares (min eigen <= -1)")

    maxdim = min(ngroups - 1, nvar)
    dims = np.arange(0, maxdim, dtype=int)

    lam_all = np.flip(1.0 / np.cumprod(e + 1.0))
    lam = lam_all[dims]

    chistat = -(nsample - 1.0 - (ngroups + nvar)/2.0) * np.log(lam)
    chisqdf = (nvar - dims) * (ngroups - 1 - dims)
    pp = 1.0 - chi2.cdf(chistat, chisqdf)

    idx_ok = np.where(pp > alpha)[0]
    d = int(dims[idx_ok[0]]) if idx_ok.size > 0 else int(dims.max() + 1)

    # reorder DESC
    e_desc = np.flip(e)
    v_desc = v[:, np.flip(np.arange(v.shape[1]))]

    # rescale so within-group var = 1
    vs = np.diag(v_desc.T @ WSSP @ v_desc) / (nsample - ngroups)
    vs = np.where(vs <= 0, 1.0, vs)
    v_desc = v_desc / np.sqrt(vs)[None, :]

    # flip sign for consistency
    neg = (v_desc.sum(axis=0) < 0)
    v_desc[:, neg] *= -1

    canon = x_centered @ v_desc

    gmean = np.full((ngroups, canon.shape[1]), np.nan, dtype=float)
    for j in range(ngroups):
        rows = np.where(group_idx == j)[0]
        gmean[j, :] = canon[rows, :].mean(axis=0)

    mdist = np.sum((canon - gmean[group_idx, :])**2, axis=1)
    diff = gmean[:, None, :] - gmean[None, :, :]
    gmdist = np.sum(diff**2, axis=2)

    # reinsert NaNs
    if np.any(is_nan_mask_original):
        canon_full = np.full((X.shape[0], canon.shape[1]), np.nan, dtype=float)
        mdist_full = np.full((X.shape[0],), np.nan, dtype=float)
        kept_rows = np.where(~is_nan_mask_original)[0]
        canon_full[kept_rows, :] = canon
        mdist_full[kept_rows] = mdist
        canon = canon_full
        mdist = mdist_full

    stats = {
        "W": WSSP, "B": BSSP, "T": TSSP,
        "dfW": int(nsample - ngroups),
        "dfB": int(ngroups - 1),
        "dfT": int(nsample - 1),
        "lambda": lam,
        "chisq": chistat,
        "chisqdf": chisqdf,
        "eigenval": e_desc,
        "eigenvec": v_desc,
        "canon": canon,
        "mdist": mdist,
        "gmdist": gmdist,
        "gnames": group_names
    }

    return d, pp, stats
```

## C3) Como o dashboard deve usar CDA

- Para CDA, o “embedding” a plotar é `stats['canon'][:, :n_components]` renomeado para `C1, C2, C3`.
- Em `stats` mostrado no painel:
    - exibir `p` (p-values por dimensão), `lambda` (Wilks), `chisq` etc.
- Não use sklearn LDA para CDA.

---

# D) Definir “labels” supervisionados (para LDA e CDA)

Crie uma função central no app (ou módulo helper):

`build_supervision_labels(meta, protocol, color_by_mode) -> labels (categorical string)`

Regras:

- Protocol A:
    - allowed: group, complexity, overlap, group+complexity, group+overlap, complexity+overlap, all
- Protocol B:
    - allowed: group, complexity, group+complexity
- Protocol C:
    - allowed: complexity apenas

Quando o usuário selecionar uma combinação (ex. group+complexity), o label vira string concatenada, como já é feito na `color_label` (ex.: `CV_C6`) .

Use as mesmas strings do `color_label` para:

- `color_label` (plot)
- `labels` de LDA/CDA (supervisão)

Se o usuário escolher um color_by inválido para o protocolo, force para o default permitido e informe "Data Warning" na legenda/título (padrão que vocês já fazem no fallback) .

---

# E) Nova UI: Protocol C + X selector + Y checkboxes + Data Domain Both

## E1) Protocol dropdown

Adicionar “Protocol C” em `protocol-dropdown` (hoje só A e B) .

## E2) Remover Covariance Mode

Remover o bloco do radio (hoje existe) e remover callback `disable_domains` associado .

## E3) Adicionar X selector (obrigatório) nos controles de cada painel

Em `create_analysis_controls(panel_id)` adicionar dropdown:

- id: `{'type': 'x-mode-dropdown', 'index': panel_id}`
- options:
    - 'PSD trecho completo' => 'psd_full'
    - 'PSD estratificada por bandas' => 'psd_bands'
    - 'PSD estratificada por bandas (normalizada)' => 'psd_bands_norm'
- default: 'psd_bands_norm' (ou qualquer default, mas consistente)

## E4) Adicionar Y checkboxes (obrigatório)

Adicionar checklist:

- id: `{'type': 'y-checklist', 'index': panel_id}`
- options: Desempenho/Acuracia/Similaridade/Especificidade/PropX/PropY
- default: ['Desempenho'] (marcado)

## E5) Data Domain agora tem 3 opções

Trocar o dropdown que hoje tem PSD vs Behavioral por:

- 'X (PSD Feature)' => 'x'
- 'Y (Behavior)' => 'y'
- 'Both (Mix Axes)' => 'both'

## E6) Axis selectors quando domain='both'

Quando domain == 'both':

- mostrar dropdowns `axis-1`, `axis-2` e (se 3D) `axis-3`:
    - ids: `{'type': 'axis-select', 'index': panel_id, 'axis': 1}` etc.
- options dinâmicas baseadas em:
    - quais componentes existem em embedding de X
    - quais componentes existem em embedding de Y
- labels:
    - `C1 de X`, `C2 de X`, `C3 de X`
    - `C1 de Y`, `C2 de Y`, `C3 de Y`
- default:
    - 2D: eixo1=C1 de X, eixo2=C2 de X (se existir; senão fallback seguro)
    - 3D: eixo3=C1 de Y (se existir; senão fallback para C3 de X, etc.)

---

# F) Engine v2: compute_embedding_v2

Substitua `compute_embedding` por algo mais claro:

`compute_embeddings(X, Y, labels, method, n_components) -> dict`

que retorna:

- `X_scores`: DataFrame com colunas C1..Ck
- `Y_scores`: DataFrame com colunas C1..Ck (para PCA; para LDA/CDA só faz sentido se supervisionar Y também; ver abaixo)
- `stats`: dict com stats do método (variance, canonical correlations, wilks, p, etc.)

Regras por método:

1. PCA:
    - X_scores = PCA(X)
    - Y_scores = PCA(Y)
2. LDA:
    - LDA precisa de labels; aplique LDA em X -> X_scores
    - Para Y_scores:
        - se domain='both' e o usuário selecionar eixos de Y: compute também LDA em Y usando os mesmos labels
        - se labels não tiverem >=2 classes, retornar erro amigável
3. CDA:
    - use `manova1_like_matlab` em X com labels -> X_scores = canon
    - e, se domain='both' e usuário quiser eixos de Y: rode `manova1_like_matlab` em Y com labels -> Y_scores
4. PLS:
    - PLSRegression entre X e Y:
        - X_scores = model.x_scores_
        - Y_scores = model.y_scores_ (expor também)
        - stats: canonical correlations corr(x_scores_i, y_scores_i) (igual já fazem no PLS cross atual) .

Obs: garanta que `n_components` seja limitado por dimensões e por número de classes quando necessário (LDA/CDA).

---

# G) Plotting v2: suportar domain x/y/both

Refatore `run_single_analysis` (que hoje escolhe X pelo domain e sempre usa meta['grupo'] como label) para:

Inputs adicionais:

- x_mode
- y_cols checklist
- domain ('x'|'y'|'both')
- axis selections quando both

Pipeline:

1. `df, meta = load_data(protocol)`
2. `X = build_X(df, x_mode)`
3. `Y = build_Y(df, y_cols)`
4. `labels = build_supervision_labels(meta, protocol, color_by)`
5. `emb = compute_embeddings(X, Y, labels, method, n_components)`
6. Construir `plot_df = meta + color_label + coords`
    - domain=='x': usar coords = X_scores[['C1','C2','C3']]
    - domain=='y': usar coords = Y_scores
    - domain=='both': montar coords eixo a eixo, conforme axis selectors:
        - ex.: axis1='C1_X' axis2='C2_X' axis3='C1_Y'
7. Plot 2D/3D com px.scatter/px.scatter_3d (manter estética atual: marker size, theme, hover) .

Hover:

- manter `ID`, `grupo`, e incluir `Complexidade`, `Overlap` se existirem .
- adicional: incluir Y raw selecionado (Desempenho etc.) se existir (sem poluir demais).

Stats panel:

- PCA: explained variance
- PLS: canonical correlations
- CDA: mostrar lambda/p/chisq por dimensão + d estimado

---

# H) Comparison Mode (não mudar)

O modo comparação atual cria 2 conjuntos de controles e plota left/right com configs independentes .

Mantenha isso, apenas adicionando os novos estados (x_mode, y_checklist, axis selectors, domain) para cada painel.

---

# I) Critérios de robustez e UX

- Se Prot C não tem `grupo`, não quebrar: `grupo='ALL'`.
- Se usuário escolher color_by inválido para o protocolo, fazer fallback automático.
- Se `domain='both'` e Y não tem componentes suficientes para o eixo escolhido, não quebrar:
    - restrinja opções no dropdown para só as existentes.
- Se Y checklist vazio: lançar erro amigável no plot ("Select at least one Y variable").

---

# J) Checklist de mudanças em arquivos

1. app.py
- Protocol dropdown: incluir C.
- create_analysis_controls: remover covariance-mode; adicionar X selector; adicionar Y checklist; mudar Data Domain; adicionar axis selectors condicionais quando both.
- callbacks: remover disable_domains; atualizar update_single/update_comparison para passar novos States.
- run_single_analysis: refatorar para usar df/meta + build_X/build_Y + compute_embeddings + axis mixing.
1. data_loader.py
- refatorar para `load_data`, `build_X`, `build_Y`.
- suportar Prot C sem grupo.
- suportar nomes `analise_df_{prot}_final.csv` e fallback `df_{prot}_final.csv`.
1. analysis_engine.py
- substituir compute_embedding por compute_embeddings (X_scores, Y_scores, stats).
- incorporar manova1_like_matlab em CDA.
- PLS sempre X->Y (PLSRegression) e exportar x_scores e y_scores.
1. novo: manova1_matlab.py
- conter `manova1_like_matlab` acima.