#%% Funções e imports

from scipy.io import loadmat
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
import os
import re
# Função para transformar os dados retirados de .mat para data frame organizado
def montar_df_psd(dados, individuos=None):
    """
    Constrói um DataFrame "mestre" a partir da estrutura vinda do MATLAB
    (ex.: loadmat()['dados_combinados']), onde cada linha representa um
    indivíduo e contém, em colunas, objetos com os dados de PSD.

    Parâmetros
    ----------
    dados : np.ndarray
        Estrutura do MATLAB (geralmente `arquivo['dados_combinados']`) em que:
        - dados[:, 0] contém os identificadores dos indivíduos (strings/objetos).
        - dados[:, 1] é uma célula/struct por indivíduo com 4 elementos na ordem:
          [freqs, psds, srate, ch_labels], onde:
            * freqs: array 1D de frequências (n_freq,)
            * psds : array 2D com PSD (n_canais, n_freq)
            * srate: taxa de amostragem (escalar)
            * ch_labels: lista/array de nomes de canais (n_canais,)
    individuos : list[str] | None, opcional
        Lista de IDs dos indivíduos. Se None, é inferida de `dados[:, 0]`.

    Retorno
    -------
    pandas.DataFrame
        DataFrame com uma linha por indivíduo. Colunas:
          - 'freqs'     : np.ndarray 1D (frequências do indivíduo)
          - 'psds'      : pandas.DataFrame (linhas=canais, colunas=freqs)
          - 'srate'     : float (taxa de amostragem do indivíduo)
          - 'ch_labels' : list[str] (nomes dos canais, já com strip)
          - 'ind'       : str (ID do indivíduo; duplicado do índice)
        O índice do DataFrame final é a lista de `individuos`.

    Observações
    -----------
    - Cada célula de 'psds' contém um DataFrame (dtype=object). Isso é útil
      para manter a estrutura por indivíduo, mas operações vetoriais globais
      não funcionam diretamente (é preciso iterar linha a linha).
    - A função faz `np.squeeze` para remover dimensões unitárias vindas do MATLAB
      e normaliza `ch_labels` para strings sem espaços extras.
    """
    import numpy as np
    import pandas as pd

    # 1) Inferir lista de IDs, se necessário
    if individuos is None:
        individuos = [str(x[0]) if isinstance(x, np.ndarray) else str(x)
                      for x in dados[:, 0]]

    linhas = []

    # 2) Percorrer indivíduos e montar as linhas
    for i, ind in enumerate(individuos):
        # cada célula em dados[i,1] costuma ser um array com shape (1,1)
        bloco = dados[i, 1][0][0]

        freqs = np.squeeze(bloco[0])            # (n_freq,)
        psds  = np.squeeze(bloco[1])            # (n_chan, n_freq)
        srate = float(np.squeeze(bloco[2]))     # escalar
        chraw = np.squeeze(bloco[3])            # nomes de canais (obj)

        # normalizar labels vindos do MATLAB (podem vir como arrays de objetos)
        ch_labels = []
        for x in chraw:
            # x pode ser np.ndarray(['C3'], dtype='<U2') ou já string
            if isinstance(x, np.ndarray):
                x = x[0]
            ch_labels.append(str(x).strip())

        # DataFrame de PSDs por indivíduo
        df_psd = pd.DataFrame(psds, index=ch_labels, columns = freqs)
        df_psd.index.name = 'canal'

        linhas.append({
            'freqs': freqs,
            'psds': df_psd,
            'srate': srate,
            'ch_labels': ch_labels,
            'ind': ind
        })

    # 3) Montar o DataFrame final de uma vez (mais eficiente que concatenar no loop)
    df_final = pd.DataFrame(linhas)
    df_final.index = individuos

    return df_final

def extrair_bandpowers(conjunto_df,
                       canais=('C3','C4','CZ'),
                       bandas=None,
                       metodo='trapz',
                       faixa_total=(0.5, 45),
                       retornar_relativo=True):
    """
    Percorre um dicionário de DataFrames 'mestres' (como os criados pela sua função),
    e calcula a potência por banda de frequência para canais específicos (C3, C4, CZ).

    Parâmetros
    ----------
    conjunto_df : dict[str, pandas.DataFrame]
        Dicionário onde cada valor é um DF com colunas:
        - 'freqs' (np.ndarray 1D)
        - 'psds'  (pandas.DataFrame: linhas=canais, colunas=freqs)
        - 'ind'   (id do indivíduo; também está no índice)
    canais : tuple[str], default ('C3','C4','CZ')
        Quais canais extrair.
    bandas : dict[str, tuple[float,float]] | None
        Faixas de frequência (Hz). Se None, usa:
        {'delta':(0.5,4), 'theta':(4,8), 'alpha':(8,13), 'beta':(13,30), 'gamma':(30,45)}
    metodo : {'trapz','sum','mean'}, default 'trapz'
        Como agregar a PSD dentro da banda:
        - 'trapz' integra por regra do trapézio (recomendado p/ potência)
        - 'sum'  soma simples dos bins
        - 'mean' média dos bins
    faixa_total : tuple[float,float], default (0.5,45)
        Janela para potência total (usada no cálculo relativo).
    retornar_relativo : bool, default True
        Se True, inclui coluna 'power_rel' = banda/total (na faixa_total).

    Retorna
    -------
    pandas.DataFrame
        Colunas: ['dataset','ind','canal','banda','power','power_rel'(opcional)]
    """
    if bandas is None:
        bandas = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta' : (13, 30),
            'gamma': (30, 45)
        }

    registros = []

    for dataset_nome, df_master in conjunto_df.items():
        # percorre indivíduos (linhas do DF-mestre)
        for idx, linha in df_master.iterrows():
            freqs = np.asarray(linha['freqs']).astype(float)
            psd_df = linha['psds']  # DF (canais x freqs)

            # garantir interseção de canais existentes
            canais_existentes = [c for c in canais if c in psd_df.index]
            if not canais_existentes:
                continue

            # máscara para potência total (se for calcular relativo)
            if retornar_relativo:
                m_total = (freqs >= faixa_total[0]) & (freqs < faixa_total[1])

            for canal in canais_existentes:
                y_all = np.asarray(psd_df.loc[canal, :]).astype(float)

                # potência total na janela definida
                if retornar_relativo:
                    if metodo == 'trapz':
                        p_total = np.trapz(y_all[m_total], freqs[m_total])
                    elif metodo == 'sum':
                        p_total = y_all[m_total].sum()
                    else:
                        p_total = y_all[m_total].mean()
                    # evitar divisão por zero
                    p_total = float(p_total) if p_total != 0 else np.nan

                # por banda
                for nome_banda, (lo, hi) in bandas.items():
                    m = (freqs >= lo) & (freqs < hi)
                    if not np.any(m):
                        power = np.nan
                    else:
                        if metodo == 'trapz':
                            power = float(np.trapz(y_all[m], freqs[m]))
                        elif metodo == 'sum':
                            power = float(y_all[m].sum())
                        else:
                            power = float(y_all[m].mean())

                    registro = {
                        'dataset': dataset_nome,
                        'ind'    : linha['ind'] if 'ind' in df_master.columns else idx,
                        'canal'  : canal,
                        'banda'  : nome_banda,
                        'power'  : power
                    }
                    if retornar_relativo:
                        registro['power_rel'] = power / p_total if (p_total and not np.isnan(p_total)) else np.nan

                    registros.append(registro)

    return pd.DataFrame(registros)
# gera a tabela tidy com todas as bandas para C3/C4/CZ em todos os datasets

# --- HELPERS ---------------------------------------------------------------

def _id_variants(texto: str):
    """
    Dada uma string de ID (ex.: 'ID07', '07', '7'), gera variações
    que serão usadas para casar com rótulos vindos do DF.
    """
    s = str(texto).strip()
    # pega só os dígitos; se não houver, fica string vazia
    digits = ''.join(re.findall(r'\d+', s))
    if digits == '':
        return {s}  # não há dígitos; devolve só a forma crua

    # normalizações úteis
    v = set()
    v.add(s)                      # como veio
    v.add(digits)                 # só números (ex.: '7')
    v.add(digits.lstrip('0') or '0')   # sem zeros à esquerda
    for z in (2, 3):
        v.add(digits.zfill(z))         # zero-padded '07', '007'
        v.add('ID' + digits.zfill(z))  # 'ID07', 'ID007'
    v.add('ID' + digits)               # 'ID7'
    return v

def _resolver_indice(df_master: pd.DataFrame, ind):
    """
    Resolve o índice da linha do indivíduo aceitando variações:
    '02', 2, 'ID02', 'ID2', etc. Compara pela parte numérica do ID.
    Retorna a *label* correta para usar em df_master.loc[...].
    """
    candidatos = _id_variants(ind)

    # 1) tentar casar no INDEX
    idx_labels = df_master.index.tolist()
    # monta mapa: para cada label do índice, todas as suas variantes apontam para a label original
    mapa = {}
    for lab in idx_labels:
        for var in _id_variants(lab):
            mapa.setdefault(var, lab)

    for c in candidatos:
        if c in mapa:
            return mapa[c]

    # 2) se existir coluna 'ind', repetir o processo nas linhas
    if 'ind' in df_master.columns:
        col_vals = df_master['ind'].tolist()
        for pos, val in enumerate(col_vals):
            for var in _id_variants(val):
                if var in candidatos:
                    return df_master.index[pos]

    # não achou: informar exemplos
    exemplos = []
    for lab in idx_labels[:15]:
        exemplos.append(str(lab))
    raise ValueError(
        f"ID '{ind}' não encontrado.\n"
        f"Tente passar apenas o número (ex.: '2' ou '02') ou com prefixo 'ID'.\n"
        f"IDs vistos no índice (amostra): {exemplos}"
    )

# --- FUNÇÃO PRINCIPAL para plotar as bandas -----

def plot_bandas_psd(df_master,
                    ind,
                    canais=('C3','C4','CZ'),
                    bandas=None,
                    metodo='trapezoid',
                    faixa_total=(0.5, 45),
                    mostrar_relativo=False,
                    titulo_prefixo=None,
                    escala_db = False):
    
    """
    Plota as curvas de Potência Espectral Densidade (PSD) para canais específicos de um indivíduo,
    destacando as bandas de frequência clássicas do EEG (delta, theta, alpha, beta, gamma) e
    exibindo as respectivas potências absolutas e relativas.

    Parâmetros
    ----------
    df_master : pandas.DataFrame
        DataFrame mestre que contém as informações de PSDs para cada indivíduo.
        Deve incluir colunas 'freqs' (vetor de frequências) e 'psds' (DataFrame com canais x frequências).

    ind : str ou int
        Identificador do indivíduo a ser plotado. Pode estar no formato 'ID02', '02' ou '2'.

    canais : tuple of str, opcional
        Lista ou tupla com os nomes dos canais EEG a serem plotados (ex.: ('C3', 'C4', 'CZ')).

    bandas : dict, opcional
        Dicionário com as bandas de frequência e seus intervalos em Hz.
        Exemplo padrão: {'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30), 'gamma': (30, 60)}.

    metodo : str, opcional
        Método para cálculo da potência dentro das bandas:
        'trapezoid' (padrão), 'sum' ou 'mean'.

    faixa_total : tuple, opcional
        Intervalo total de frequências (Hz) exibido no gráfico (ex.: (0.5, 45)).

    mostrar_relativo : bool, opcional
        Se True, exibe também a potência relativa (% da potência total) de cada banda.

    titulo_prefixo : str, opcional
        Texto a ser exibido antes do título principal do gráfico (ex.: 'Baseline' ou 'Cond. Visual').

    escala_db : bool, opcional
        Se True, converte a PSD para escala logarítmica (dB) apenas para exibição.
        As potências integradas continuam sendo calculadas no domínio linear.

    Retorna
    -------
    resultados : dict
        Dicionário contendo as potências absolutas e relativas por banda para cada canal.
        Estrutura:
            resultados[canal][banda] = {'abs': potência_absoluta, 'rel': potência_relativa}

    Descrição geral
    ---------------
    - Calcula a potência por banda usando integração trapezoidal (ou soma/média, conforme 'metodo').
    - Permite plotar a PSD em escala linear (uV²/Hz) ou logarítmica (dB).
    - Destaca graficamente as regiões das bandas com cores fixas:
    delta=azul, theta=laranja, alpha=verde, beta=vermelho, gamma=roxo.
    - Exibe legendas automáticas com valores de potência e porcentagens.
    - Ajusta automaticamente a faixa de exibição e organiza múltiplos canais em subplots verticais.
    """

    
    if bandas is None:
        bandas = {'delta': (0.5, 4), 'theta': (4, 8),
                  'alpha': (8, 13), 'beta': (13, 30), 'gamma': (30, 60)}

    # paleta fixa p/ cada banda (C0..C4 = paleta default do Matplotlib)
    band_colors = {'delta':'C0', 'theta':'C1', 'alpha':'C2', 'beta':'C3', 'gamma':'C4'}

    # localizar linha
    label = _resolver_indice(df_master, ind)
    linha = df_master.loc[label]

    freqs = np.asarray(linha['freqs']).astype(float)
    psd_df = linha['psds'].copy()
  
    try:
        psd_df.columns = np.asarray(psd_df.columns, dtype=float)
    except Exception:
        pass

    if isinstance(canais, str):
        canais = (canais,)
    canais_exist = [c for c in canais if c in psd_df.index]
    if not canais_exist:
        raise ValueError(f"Nenhum dos canais {canais} existe. Disponíveis: {list(psd_df.index)}")

    def potencia(y, x):
        if metodo == 'trapezoid':
            return float(np.trapezoid(y, x)) if y.size and x.size else np.nan
        elif metodo == 'sum':
            return float(np.sum(y)) if y.size else np.nan
        else:
            return float(np.mean(y)) if y.size else np.nan

    freqs_alinh = np.asarray(psd_df.columns, dtype=float)

    n = len(canais_exist)
    fig, axes = plt.subplots(n, 1, figsize=(9, 3.2*n), sharex=True)
    if n == 1:
        axes = [axes]

    resultados = {}
    m_total = (freqs_alinh >= faixa_total[0]) & (freqs_alinh < faixa_total[1])
    eps = 1e-15  # evita log10(0)

    for ax, canal in zip(axes, canais_exist):
        # y_all SEMPRE linear para cálculo de potência
        y_all = np.asarray(psd_df.loc[canal, freqs_alinh]).astype(float)
        p_total = potencia(y_all[m_total], freqs_alinh[m_total]) if mostrar_relativo else None

        # y_plot é a série a ser exibida (linear OU dB)
        if escala_db:
            y_plot = 10.0 * np.log10(np.maximum(y_all, eps))
            ylabel = 'PSD (dB)'
            # linha-base para preencher as bandas: usa o mínimo da curva em dB
            y_base = np.nanmin(y_plot)
        else:
            y_plot = y_all
            ylabel = 'PSD (uV²/Hz)'
            y_base = 0.0

        # curva PSD
        h_psd, = ax.plot(freqs_alinh, y_plot, linewidth=1.2, label=f'PSD {canal}', zorder=3)

        resultados[canal] = {}
        band_handles, band_labels = [], []

        for nome_b, (lo, hi) in bandas.items():
            hi_eff = min(hi, float(freqs_alinh.max()))
            m = (freqs_alinh >= lo) & (freqs_alinh < hi_eff)

            if not np.any(m):
                p_abs, p_rel, patch = np.nan, None, None
            else:
                # potência ABS/REL calculada em linear
                p_abs = potencia(y_all[m], freqs_alinh[m])
                p_rel = (p_abs / p_total) if (mostrar_relativo and p_total and not np.isnan(p_total)) else None

                # preenchimento na escala exibida (y_plot)
                patch = ax.fill_between(freqs_alinh[m], y_plot[m], y_base,
                                        step='pre', alpha=0.30,
                                        color=band_colors.get(nome_b, None),
                                        zorder=1)

            lbl = f"{nome_b} (P={p_abs:.3g}" + (f", rel={p_rel:.2%}" if (mostrar_relativo and p_rel is not None) else "") + ")"
            if patch is not None:
                band_handles.append(patch)
                band_labels.append(lbl)

            resultados[canal][nome_b] = {'abs': p_abs, 'rel': (p_rel if mostrar_relativo else None)}

        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        handles = [h_psd] + band_handles
        labels  = [h_psd.get_label()] + band_labels
        ax.legend(handles=handles, labels=labels, loc='upper right', fontsize=9, frameon=True)

    axes[-1].set_xlabel('Frequência (Hz)')
    tprefix = f"{titulo_prefixo} — " if titulo_prefixo else ""
    fig.suptitle(f"{tprefix}Indivíduo {ind}", y=1.02, fontsize=12)
    plt.xlim((0, 100))
    plt.tight_layout()
    plt.show()

    return resultados

# Plot com média + std o erro padrão

def plot_psd_media_canais(df_master,
                   ind,
                   bandas=None,
                   escala_db=False,
                   faixa_total=(0.5, 100),
                   alpha_bandas = 0.125,
                   alpha_desvio = 0.5,
                   erro_padrao_habilitado = True,
                   titulo_prefixo=None):
    """
    Plota a média e o desvio padrão das PSDs de todos os canais de um indivíduo.

    Parâmetros:
    ------------
    df_master : DataFrame
        DataFrame mestre com colunas 'freqs' e 'psds' (iguais à função plot_bandas_psd).
    ind : str
        Identificador do indivíduo a ser plotado (ex.: '02' ou 'ID02').
    bandas : dict, opcional
        Dicionário com bandas de frequência e intervalos (Hz).
        Exemplo: {'delta': (0.5,4), 'theta': (4,8), 'alpha': (8,13), 'beta': (13,30), 'gamma': (30,60)}
    escala_db : bool, opcional
        Se True, converte a PSD média e o desvio para escala dB (10*log10).
    faixa_total : tuple, opcional
        Limite inferior e superior de frequência exibida.
    erro_padrao_habilitado: bool, opcional
        Se False, o desvio padrão será plotado ao invés do erro padrão
    titulo_prefixo : str, opcional
        Texto a ser exibido antes do título do gráfico.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    def _potencia(y, x):
        return float(np.trapezoid(y, x)) if y.size and x.size else np.nan

    if bandas is None:
        bandas = {'delta': (0.5, 4), 'theta': (4, 8),
                  'alpha': (8, 13), 'beta': (13, 30), 'gamma': (30, 60)}

    band_colors = {'delta':'C0', 'theta':'C1', 'alpha':'C2', 'beta':'C3', 'gamma':'C4'}

    # localizar linha do indivíduo
    label = _resolver_indice(df_master, ind)
    linha = df_master.loc[label]

    freqs = np.asarray(linha['freqs']).astype(float)
    psd_df = linha['psds'].copy()

    # Garantir que colunas correspondam às frequências reais
    cols = np.asarray(psd_df.columns)
    if len(cols) == len(freqs) and np.array_equal(cols, np.arange(len(freqs))):
        psd_df.columns = freqs
    else:
        try:
            psd_df.columns = np.asarray(psd_df.columns, dtype=float)
        except Exception:
            pass

    freqs_alinh = np.asarray(psd_df.columns, dtype=float)

    # --- média e desvio entre canais (domínio linear) ---
    psd_vals = psd_df.values.astype(float)         # shape: (n_canais, n_freqs)
    media_lin = np.nanmean(psd_vals, axis=0)       # (n_freqs,)
    dp_lin    = np.nanstd(psd_vals,  axis=0)
    n = len(psd_vals)
    erro_padrao = dp_lin/(n**(1/2))

    # Mostrará o erro padrão e não o desvio padrão
    if erro_padrao_habilitado:
        dp_lin= erro_padrao
        
    # --- converter para dB se pedido ---
    eps = 1e-15  # para evitar log10(0)
    if escala_db:
        media_db = 10.0 * np.log10(np.maximum(media_lin, eps))
        upper_db = 10.0 * np.log10(np.maximum(media_lin + dp_lin, eps))
        lower_db = 10.0 * np.log10(np.maximum(media_lin - dp_lin, eps))
        curva    = media_db
        faixa_lo = lower_db
        faixa_hi = upper_db
        ylabel   = 'PSD (dB)'
    else:
        curva    = media_lin
        faixa_lo = np.maximum(media_lin - dp_lin, 0.0)  # nada negativo
        faixa_hi = media_lin + dp_lin
        ylabel   = 'PSD (uV²/Hz)'

    

    m_total = (freqs_alinh >= faixa_total[0]) & (freqs_alinh < faixa_total[1])
    p_total = _potencia(media_lin[m_total], freqs_alinh[m_total])


    # --- plot ---
    fig, ax = plt.subplots(figsize=(10, 5))

    # desvio (faixa cinza)
    ax.fill_between(freqs_alinh, faixa_lo, faixa_hi, color='gray', alpha=alpha_desvio,
                    label='Desvio padrão', zorder=1)

    # média (linha preta)
    linha_media, = ax.plot(freqs_alinh, curva, color='black', linewidth=1.6,
                           label='Média PSD', zorder=2)

    # bandas (sombras suaves; mesmas cores da outra função)
    band_handles, band_labels = [], []
    for nome_b, (lo, hi) in bandas.items():
        hi_eff = min(hi, float(freqs_alinh.max()))
        m = (freqs_alinh >= lo) & (freqs_alinh < hi_eff)
        if np.any(m):
            
            # potência ABSOLUTA da banda usando a média linear
            p_abs = _potencia(media_lin[m], freqs_alinh[m])
            # em dB ou linear, preenche entre os mesmos envelopes da faixa cinza
            patch = ax.fill_between(freqs_alinh[m],
                                    faixa_lo[m], faixa_hi[m],
                                    color=band_colors.get(nome_b, None),
                                    alpha= alpha_bandas, zorder=0)
            band_handles.append(patch)
            band_labels.append(f"{nome_b} (P={p_abs:.3g})")

    # legenda consistente: curva preta + patches das bandas
    handles = [linha_media] + band_handles
    labels  = ['Média PSD'] + band_labels
    ax.legend(handles=handles, labels=labels, loc='upper right',
              fontsize=9, frameon=True)

    ax.set_xlim(faixa_total)
    ax.set_xlabel('Frequência (Hz)')
    ax.set_ylabel(ylabel)
    tprefix = f"{titulo_prefixo} — " if titulo_prefixo else ""
    ax.set_title(f"{tprefix}Indivíduo {ind} — Média e Desvio Padrão entre Canais")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_psd_media_individuos(df_master,
                   ch,
                   bandas=None,
                   escala_db=False,
                   faixa_total=(0.5, 100),
                   alpha_bandas = 0.125,
                   alpha_desvio = 0.5,
                   erro_padrao_habilitado = True,
                   titulo_prefixo=None):
    """
    Plota a média e o desvio padrão das PSDs de todos os canais de um indivíduo.

    Parâmetros:
    ------------
    df_master : DataFrame
        DataFrame mestre com colunas 'freqs' e 'psds' (iguais à função plot_bandas_psd).
    ch : list of str or str
        Identificador do canal a ser plotado (ex.: 'CZ').
    bandas : dict, opcional
        Dicionário com bandas de frequência e intervalos (Hz).
        Exemplo: {'delta': (0.5,4), 'theta': (4,8), 'alpha': (8,13), 'beta': (13,30), 'gamma': (30,60)}
    escala_db : bool, opcional
        Se True, converte a PSD média e o desvio para escala dB (10*log10).
    faixa_total : tuple, opcional
        Limite inferior e superior de frequência exibida.
    erro_padrao_habilitado: bool, opcional
        Se False, o desvio padrão será plotado ao invés do erro padrão
    titulo_prefixo : str, opcional
        Texto a ser exibido antes do título do gráfico.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    def _potencia(y, x):
        return float(np.trapezoid(y, x)) if y.size and x.size else np.nan

    if bandas is None:
        bandas = {'delta': (0.5, 4), 'theta': (4, 8),
                  'alpha': (8, 13), 'beta': (13, 30), 'gamma': (30, 60)}

    band_colors = {'delta':'C0', 'theta':'C1', 'alpha':'C2', 'beta':'C3', 'gamma':'C4'}

    # --- normalizar 'ch' para uma lista de canais ---
    if isinstance(ch, str):
        ch = [ch]
    else:
        ch = list(ch)  # aceita set/tuple/np.array e converte para list

    # --- obter canais disponíveis de forma robusta ---
    canais_disponiveis = df_master['psds'].iloc[0].index.astype(str).str.strip().tolist()
    faltantes = [c for c in ch if c not in canais_disponiveis]
    if faltantes:
        raise ValueError(f"Canais inexistentes: {faltantes}. Disponíveis: {canais_disponiveis}")

    freqs_alinh = np.asarray(df_master['freqs'].iloc[0], dtype=float)

    for channel in ch:
        # --- média e desvio entre indivíduos (domínio linear) ---
        lista = []
        for ind in df_master.index:
            lista.append(df_master['psds'][ind].loc[channel])
        
        media_lin = np.mean(lista, axis=0)       # (n_freqs,)
        dp_lin    = np.std(lista,  axis=0)
        n = len(lista)
        erro_padrao = dp_lin/np.sqrt(n)

        # Mostrará o erro padrão e não o desvio padrão
        if erro_padrao_habilitado:
            dp_lin= erro_padrao
            
        # --- converter para dB se pedido ---
        eps = 1e-15  # para evitar log10(0)
        if escala_db:
            media_db = 10.0 * np.log10(np.maximum(media_lin, eps))
            upper_db = 10.0 * np.log10(np.maximum(media_lin + dp_lin, eps))
            lower_db = 10.0 * np.log10(np.maximum(media_lin - dp_lin, eps))
            curva    = media_db
            faixa_lo = lower_db
            faixa_hi = upper_db
            ylabel   = 'PSD (dB)'
        else:
            curva    = media_lin
            faixa_lo = np.maximum(media_lin - dp_lin, 0.0)  # nada negativo
            faixa_hi = media_lin + dp_lin
            ylabel   = 'PSD (uV²/Hz)'

        m_total = (freqs_alinh >= faixa_total[0]) & (freqs_alinh < faixa_total[1])
        p_total = _potencia(media_lin[m_total], freqs_alinh[m_total])


        # --- plot ---
        fig, ax = plt.subplots(figsize=(10, 5))

        # desvio (faixa cinza)
        ax.fill_between(freqs_alinh, faixa_lo, faixa_hi, color='gray', alpha=alpha_desvio,
                        label='Desvio padrão', zorder=1)

        # média (linha preta)
        linha_media, = ax.plot(freqs_alinh, curva, color='black', linewidth=1.6,
                            label='Média PSD', zorder=2)

        # bandas (sombras suaves; mesmas cores da outra função)
        band_handles, band_labels = [], []
        for nome_b, (lo, hi) in bandas.items():
            hi_eff = min(hi, float(freqs_alinh.max()))
            m = (freqs_alinh >= lo) & (freqs_alinh < hi_eff)
            if np.any(m):
                
                # potência ABSOLUTA da banda usando a média linear
                p_abs = _potencia(media_lin[m], freqs_alinh[m])
                # em dB ou linear, preenche entre os mesmos envelopes da faixa cinza
                patch = ax.fill_between(freqs_alinh[m],
                                        faixa_lo[m], faixa_hi[m],
                                        color=band_colors.get(nome_b, None),
                                        alpha= alpha_bandas, zorder=0)
                band_handles.append(patch)
                band_labels.append(f"{nome_b} (P={p_abs:.3g})")

        # legenda consistente: curva preta + patches das bandas
        handles = [linha_media] + band_handles
        labels  = ['Média PSD'] + band_labels
        ax.legend(handles=handles, labels=labels, loc='upper right',
                fontsize=9, frameon=True)

        ax.set_xlim(faixa_total)
        ax.set_xlabel('Frequência (Hz)')
        ax.set_ylabel(ylabel)
        tprefix = f"{titulo_prefixo} — " if titulo_prefixo else ""
        ax.set_title(f"{tprefix}Canal {channel} — Média e Erro Padrão entre Indivíduos" if erro_padrao_habilitado else f"{tprefix}Canal {channel} — Média e Desvio Padrão entre Indivíduos")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

#%% Lendo os arquivos .mat e já transformando em Data Frame
#Pasta dos arquivos
pasta = r"Arquivos Auxiliares\PSD"

#varrendo todos os arquivos e criando suas respectivas data frames
variaveis = {}

for nome_arquivo in os.listdir(pasta):
    if nome_arquivo.endswith('.mat'):
        nome_var = os.path.splitext(nome_arquivo)[0]  # tira o ".mat"
        caminho_completo = os.path.join(pasta, nome_arquivo)
        variaveis[nome_var] = loadmat(caminho_completo)

conjunto_df = {}
for nome, conteudo in variaveis.items():
    dados = conteudo['dados_combinados']
    individuos = [str(ind[0]) for ind in dados[:, 0]]
    df = montar_df_psd(dados, individuos)
    conjunto_df[nome + '_df'] = df

#separando o dicionário conjunto em dois dicionários com a remoção de média geral e com remoção de média dos canais específicos

df_especifico = {}
df_geral ={}
for chaves in conjunto_df.keys():
    if "geral" in chaves:
        df_geral[chaves] = conjunto_df[chaves]
    elif 'especifico' in chaves:
        df_especifico[chaves] = conjunto_df[chaves]

# Ajustando os indices de indivíduos dos data frames das baselines
lista_baselines = ['psd_Baseline_OA_geral_df', 'psd_Baseline_OF_geral_df',
                   'psd_Baseline_OA_especifico_df', 'psd_Baseline_OF_especifico_df']
for baseline in lista_baselines:
    if "geral" in baseline:
        df_geral[baseline].index = [f'ID{ID}' for ID in df_geral[baseline].index]
        df_geral[baseline]['ind'] = [f'ID{IND}' for IND in df_geral[baseline]['ind']]
    elif "especifico" in baseline:
        df_especifico[baseline].index = [f'ID{ID}' for ID in df_especifico[baseline].index]
        df_especifico[baseline]['ind'] = [f'ID{IND}' for IND in df_especifico[baseline]['ind']]

#%% Normalizando pelo base line

'''
Cada protocolo tem sua especificidade.
A CV -> normaliza com a psd de olhos abertos
A SV -> normalizar com a psd de olhos fechados
B CF -> normaliza com a psd de olhos fechados
B SF -> normalizar com a psd de olhos fechados
C -> normalizar com a psd de olhos abertos

'''

def normalizar_psd_por_baseline_interp(df_tarefa, df_base, ind, modo='ratio', eps=1e-15):
    """
    Normaliza a PSD da tarefa pela baseline, mesmo com frequências diferentes.
    Alinha as frequências via interpolação.
    """
    import numpy as np
    import pandas as pd

    label_t = _resolver_indice(df_tarefa, ind)
    label_b = _resolver_indice(df_base,   ind)

    freqs_t = np.asarray(df_tarefa.loc[label_t, 'freqs'], dtype=float)
    freqs_b = np.asarray(df_base.loc[label_b, 'freqs'], dtype=float)
    psd_t = df_tarefa.loc[label_t, 'psds'].copy().astype(float)
    psd_b = df_base.loc[label_b, 'psds'].copy().astype(float)

    # Ajustar colunas para serem floats (frequências)
    psd_t.columns = freqs_t
    psd_b.columns = freqs_b

    canais = psd_t.index.intersection(psd_b.index)
    psd_t = psd_t.loc[canais]

    psd_norm = pd.DataFrame(index=canais, columns=freqs_t, dtype=float)

    # Interpolação e normalização canal a canal
    for canal in canais:
        base_interp = np.interp(freqs_t, freqs_b, psd_b.loc[canal].values)
        tarefa = psd_t.loc[canal].values

        if modo == 'ratio':
            norm = tarefa / base_interp
        elif modo == 'percent':
            norm = (tarefa - base_interp) / np.maximum(base_interp, eps) * 100
        elif modo == 'diff':
            norm = tarefa - base_interp
        else:
            raise ValueError("modo deve ser 'ratio_db', 'percent' ou 'diff'.")

        psd_norm.loc[canal] = norm

    return psd_norm, freqs_t

#fazendo um novo conjunto de data frames com a psd já normalizada
df_especifico_norm = {}
df_geral_norm = {}

for chaves in conjunto_df.keys():
    if not (chaves in lista_baselines):


        if "geral" in chaves:
            df_geral_norm[f'{chaves}_norm'] = conjunto_df[chaves].copy()
        elif 'especifico' in chaves:
            df_especifico_norm[f'{chaves}_norm']  = conjunto_df[chaves].copy()

# Normalizando os referenciados com os canais específicos
modo ='ratio' # 'diff', 'ratio_db', 'percent'
for chave in df_especifico_norm:
    # Depende de qual protocolo vamos fazer, SV, SF e CF são todos com a baseline de olhos fechados 
    if 'SV' in chave or 'SF'in chave or 'CF' in chave: 
        temp_psd=[]
        temp_freqs=[]
        for indiv in df_especifico_norm[chave].index:
            psd_norm_df, freqs = normalizar_psd_por_baseline_interp(
                df_tarefa = df_especifico_norm[chave], 
                df_base= conjunto_df['psd_Baseline_OA_especifico_df'], #Olhos abertos 
                ind = indiv,
                modo= modo)
            temp_psd.append(psd_norm_df)
            temp_freqs.append(freqs)
        df_especifico_norm[chave]['psds'] = temp_psd
        df_especifico_norm[chave]['freqs'] = temp_freqs
    else:
        temp_psd=[]
        temp_freqs=[]
        for indiv in df_especifico_norm[chave].index:
            psd_norm_df, freqs = normalizar_psd_por_baseline_interp(
                df_tarefa = df_especifico_norm[chave], 
                df_base= conjunto_df['psd_Baseline_OF_especifico_df'], #Olhos abertos 
                ind = indiv,
                modo= modo #modo em que tiramos a diferença
                ) 
            temp_psd.append(psd_norm_df)
            temp_freqs.append(freqs)
        df_especifico_norm[chave]['psds'] = temp_psd
        df_especifico_norm[chave]['freqs'] = temp_freqs

# Normalizando os referenciados com a média de todos os canais
for chave in df_geral_norm:
    # Depende de qual protocolo vamos fazer, SV, SF e CF são todos com a baseline de olhos fechados 
    if 'SV' in chave or 'SF'in chave or 'CF' in chave: 
        temp_psd=[]
        temp_freqs=[]
        for indiv in df_geral_norm[chave].index:
            psd_norm_df, freqs = normalizar_psd_por_baseline_interp(
                df_tarefa = df_geral_norm[chave], 
                df_base= conjunto_df['psd_Baseline_OA_especifico_df'], #Olhos abertos 
                ind = indiv,
                modo= 'diff')
            temp_psd.append(psd_norm_df)
            temp_freqs.append(freqs)
        df_geral_norm[chave]['psds'] = temp_psd
        df_geral_norm[chave]['freqs'] = temp_freqs
    else:
        temp_psd=[]
        temp_freqs=[]
        for indiv in df_geral_norm[chave].index:
            psd_norm_df, freqs = normalizar_psd_por_baseline_interp(
                df_tarefa = df_geral_norm[chave], 
                df_base= conjunto_df['psd_Baseline_OF_especifico_df'], #Olhos abertos 
                ind = indiv,
                modo= 'diff' #modo em que tiramos a diferença
                ) 
            temp_psd.append(psd_norm_df)
            temp_freqs.append(freqs)
        df_geral_norm[chave]['psds'] = temp_psd
        df_geral_norm[chave]['freqs'] = temp_freqs


'''#%% Se precisar para a reunião do dia 16/10

# Plotar as medias
# Especifico
for ind in df_especifico['psd_ProtA_CV_especifico_df'].index:
    plot_psd_media(df_master = df_especifico['psd_ProtA_CV_especifico_df'], ind  = ind,faixa_total =(0.5,50), titulo_prefixo='Protocolo A CV especifico')
# Geral
for ind in df_geral['psd_protA_CV_geral_df'].index:
    plot_psd_media(df_master = df_geral['psd_protA_CV_geral_df'], ind  = ind,faixa_total =(0.5,50), titulo_prefixo='Protocolo A CV geral')
# Especifico Normalizado
for ind in df_especifico_norm['psd_ProtA_CV_especifico_df_norm'].index:
    plot_psd_media(df_master = df_especifico_norm['psd_ProtA_CV_especifico_df_norm'], ind  = ind,faixa_total =(0.5,50), titulo_prefixo='Protocolo A CV especifico normalizado')
# Geral Normalizado
for ind in df_geral_norm['psd_protA_CV_geral_df_norm'].index:
    plot_psd_media(df_master = df_geral_norm['psd_protA_CV_geral_df_norm'], ind  = ind,faixa_total =(0.5,50), titulo_prefixo='Protocolo A CV geral normalizado')

# Plotar as badas especificas
%% Especifico
for ind in df_especifico_norm['psd_ProtA_CV_especifico_df_norm'].index:
    plot_bandas_psd(df_master = df_especifico_norm['psd_ProtA_CV_especifico_df_norm'], ind  = ind,faixa_total =(0,100), titulo_prefixo='Protocolo A CV especifico',escala_db=False)
%% Geral
for ind in df_especifico['psd_ProtA_CV_especifico_df'].index:
    plot_bandas_psd(df_master = df_especifico['psd_ProtA_CV_especifico_df'], ind  = ind,faixa_total =(0,100), titulo_prefixo='Protocolo A CV geral',escala_db=False)'''

#%% Fazendo a PLSC

# 1) Pegando os dados de desempenho
df_protA = pd.read_csv('df_protA.csv')
df_protB = pd.read_csv('df_protB.csv')
df_protC = pd.read_csv('df_protC.csv')

df_A = df_protA[['Tempo 1','Tempo 2','Tempo 3','ID','grupo','Desempenho']]
df_A['ID'] = df_A['ID'].str.replace('df_', '', regex=False)

df_B = df_protB[['Tempo 1','Tempo 2','Tempo 3','ID','grupo','Desempenho']]
df_B['ID'] = df_B['ID'].str.replace('df_', '', regex=False)

df_C = df_protC[df_protC['Fase'] == 'Fase Execucao']
df_C = df_C[['Tempo 1','Tempo 2', 'ID','Desempenho']]
df_C['ID'] = df_C['ID'].str.replace('df_', '', regex=False)

# 2) Estruturando X e Y

# 2.1 Fazendo os cortes nos EEG's de acordo com o tempo
#2.1.1. Convertendo os tempos para segundos
# Fórmula para converter número MATLAB em datetime Python
import datetime as dt
def matlab_datenum_to_datetime(datenum):
    # O datenum do MATLAB começa em 0000-01-00
    # Ajuste de offset para o epoch do Python
    python_datetime = dt.datetime.fromordinal(int(datenum)) \
                      + dt.timedelta(days=datenum % 1) \
                      - dt.timedelta(days=366)
    return python_datetime

cols_tempo = ['Tempo 1','Tempo 2','Tempo 3']
for cols in cols_tempo:
    df_A[cols] = df_A[cols].apply(matlab_datenum_to_datetime) #em datetime
    df_B[cols] = df_B[cols].apply(matlab_datenum_to_datetime) #em datetime

    #df_A[f'{cols} readable'] = df_A[cols].dt.strftime("%d-%m-%Y %H:%M:%S") #formato legível
    #df_B[f'{cols} readable'] = df_B[cols].dt.strftime("%d-%m-%Y %H:%M:%S") #formato legível
    if cols != 'Tempo 3':
        df_C[cols] = df_C[cols].apply(matlab_datenum_to_datetime) #em datetime
        #df_C[f'{cols} readable'] = df_C[cols].dt.strftime("%d-%m-%Y %H:%M:%S") #formato legível

#2.1.2. Pegando os tempos iniciais de cada EEG
todas_as_pastas = [
    # Oitava Leva
    "2019-02-07_09-03-52_ID44_Sara_ProtC", "2019-02-07_08-53-40_ID44_Sara_OF", "2019-02-07_08-51-21_ID44_Sara_OA",
    "2019-02-06_11-45-59_ID43_Anna_ProtB_SF", "2019-02-06_11-42-04_ID43_Anna_OF", "2019-02-06_11-39-56_ID43_Anna_OA",
    "2019-02-06_09-18-58_ID42_Fabiana_ProtB_CF", "2019-02-06_09-06-05_ID42_Fabiana_OF", "2019-02-06_09-03-54_ID42_Fabiana_OA",
    "2019-02-04_11-54-15_ID41_Fernanda_ProtC", "2019-02-04_11-40-54_ID41_Fernanda_OF", "2019-02-04_11-38-47_ID41_Fernanda_OA",
    "2019-02-04_09-27-54_ID40_Gabriel_ProtB_SF", "2019-02-04_09-19-28_ID40_Gabriel_OF", "2019-02-04_09-17-18_ID40_Gabriel_OA",

    # Sétima Leva
    "2019-02-01_11-48-15_ID39_Raquel_ProtA_CV", "2019-02-01_11-34-35_ID39_Raquel_OF", "2019-02-01_11-32-25_ID39_Raquel_OA",
    "2019-02-01_09-05-36_ID38_Lara_ProtB_CF", "2019-02-01_08-53-20_ID38_Lara_OF", "2019-02-01_08-51-09_ID38_Lara_OA",
    "2019-01-31_09-34-32_ID37_Leandro_ProtA_SV", "2019-01-31_09-31-58_ID37_Leandro_of", "2019-01-31_09-29-39_ID37_Leandro_oa",
    "2019-01-30_11-43-42_ID36_Rodrigo_ProtA_CV", "2019-01-30_11-29-21_ID36_Rodrigo_of", "2019-01-30_11-27-08_ID36_Rodrigo_oa",
    "2019-01-30_09-04-51_ID35_Joao_ProtC", "2019-01-30_08-52-05_ID35_Joao_of", "2019-01-30_08-49-44_ID35_Joao_oa",
    "2019-01-29_17-12-39_ID34_Renan_protA_SV", "2019-01-29_17-09-50_ID34_Renan_of",

    # Sexta Leva
    "2019-01-29_17-07-34_ID34_Renan_oa", "2019-01-29_11-38-04_ID33_Gabriel_protB_CF", "2019-01-29_11-33-47_ID33_Gabriel_of",
    "2019-01-29_11-31-29_ID33_Gabriel_oa", "2019-01-29_09-54-31_ID32_Larissa_protB_SF", "2019-01-29_09-50-17_ID32_Larissa_of",
    "2019-01-29_09-47-20_ID32_Larissa_oa", "2019-01-25_09-56-28_ID31_DanielAp_protA_CV", "2019-01-25_09-54-12_ID31_DanielAp_of",
    "2019-01-25_09-51-33_ID31_DanielAp_oa", "2019-01-23_11-13-47_ID30_Andre_protC", "2019-01-23_11-10-03_ID30_Andre_of",
    "2019-01-23_11-07-50_ID30_Andre_oa", "2019-01-23_08-49-19_ID29_Claudia_protB_SF", "2019-01-23_08-47-00_ID29_Claudia_of",
    "2019-01-23_08-42-45_ID29_Claudia_oa", "2019-01-22_16-32-53_ID28_Leticia_protB_SF",

    # Quinta Leva
    "2019-01-22_16-27-40_ID28_Leticia_of", "2019-01-22_16-25-26_ID28_Leticia_oa", "2019-01-22_14-35-51_ID28_Luisa_protA_SV",
    "2019-01-22_14-29-07_ID28_Luisa_of", "2019-01-22_14-26-53_ID28_Luisa_oa", "2019-01-22_11-59-38_ID26_Yuri_protB_CF",
    "2019-01-22_11-56-41_ID26_Yuri_of", "2019-01-22_11-54-18_ID26_Yuri_oa", "2019-01-21_17-48-01_ID25_Nathalia_protB_CF",
    "2019-01-21_17-45-14_ID25_Nathalia_of", "2019-01-21_17-42-22_ID25_Nathalia_oa", "2019-01-21_11-51-56_ID24_Evelyn_protA_CV",
    "2019-01-21_11-49-38_ID24_Evelyn_of", "2019-01-21_11-47-24_ID24_Evelyn_oa", "2019-01-21_09-21-55_ID23_Lucas_protA_SV",
    "2019-01-21_09-16-38_ID23_Lucas_of",

    # Quarta Leva
    "2019-01-21_09-14-16_ID23_Lucas_oa", "2019-01-18_12-24-05_ID22_Noemi_protC", "2019-01-18_12-21-21_ID22_Noemi_of",
    "2019-01-18_12-18-56_ID22_Noemi_oa", "2019-01-18_10-12-53_ID21_AnaPaula_ProtA_CV", "2019-01-18_10-10-04_ID21_AnaPaula_of",
    "2019-01-18_10-07-47_ID21_AnaPaula_oa", "2019-01-17_11-50-34_ID20_Douglas_ProtC", "2019-01-17_11-48-05_ID20_Douglas_of",
    "2019-01-17_11-45-52_ID20_Douglas_oa", "2019-01-17_09-29-27_ID19_Otavio_protA_SV", "2019-01-17_09-26-32_ID19_Otavio_of",
    "2019-01-17_09-24-13_ID19_Otavio_oa", "2019-01-16_10-27-13_ID18_Mariana_protB_CF", "2019-01-16_10-23-03_ID18_Mariana_of",
    "2019-01-16_10-20-03_ID18_Mariana_oa", "2019-01-15_17-28-08_ID17_GabrielFreitas_protA_CV",

    # Terceira Leva
    "2019-01-15_17-25-51_ID17_GabrielFreitas_of", "2019-01-15_17-23-34_ID17_GabrielFreitas_oa", "2019-01-15_14-52-32_ID16_Patricia_ProtA_SV",
    "2019-01-15_14-47-13_ID16_Patricia_of", "2019-01-15_14-44-29_ID16_Patricia_oa", "2019-01-14_21-19-06_ID15_Douglas_protB_SF",
    "2019-01-14_21-16-07_ID15_Douglas_oa", "2019-01-14_21-13-50_ID15_Douglas_of", "2019-01-14_19-54-43_ID14_Valeria_protC",
    "2019-01-14_19-52-12_ID14_Valeria_of", "2019-01-14_19-49-53_ID14_Valeria_oa", "2019-01-11_13-54-59_ID13_Allan_protB_SF",
    "2019-01-11_13-52-09_ID13_Allan_of", "2019-01-11_13-40-32_ID13_Allan_oa", "2019-01-11_11-15-00_ID12_FelipeRufino_protB_CF",
    "2019-01-11_11-08-09_ID12_FelipeRufino_of",

    # Segunda Leva (removendo duplicatas visíveis na terceira)
    "2019-01-11_11-05-52_ID12_FelipeRufino_oa", "2019-01-11_09-11-56_ID11_Elton_protC", "2019-01-11_09-08-12_ID11_Elton_of",
    "2019-01-11_09-05-59_ID11_Elton_oa", "2019-01-09_19-27-40_ID10_Bernardo_protA_CV", "2019-01-09_19-22-58_ID10_Bernardo_of",
    "2019-01-09_19-20-46_ID10_Bernardo_oa", "2019-01-09_15-29-54_ID09_Elaine_protA_SV", "2019-01-09_14-47-31_ID09_Elaine_of",
    "2019-01-09_14-45-14_ID09_Elaine_oa", "2019-01-09_12-29-50_ID08_Catharina_protC", "2019-01-09_12-25-57_ID08_Catharina_of",
    "2019-01-09_12-23-42_ID08_Catharina_oa", "2019-01-09_10-34-46_ID07_AnaBeatriz_ProtA_CV", "2019-01-09_10-26-37_ID07_AnaBeatriz_of",

    # Primeira Leva
    "2019-01-09_10-24-17_ID07_AnaBeatriz_oa", "2019-01-08_19-19-22_ID06_Priscila_protA_SV", "2019-01-08_19-13-47_ID06_Priscila_of",
    "2019-01-08_19-06-55_ID06_Priscila_oa", "2019-01-08_17-55-57_ID05_Eric_protB_CF", "2019-01-08_17-52-29_ID05_Eric_of",
    "2019-01-08_17-50-17_ID05_Eric_oa", "2019-01-08_16-27-37_ID04_Narana_protB_CF", "2019-01-08_16-25-01_ID04_Narana_of",
    "2019-01-08_16-22-45_ID04_Narana_oa", "2019-01-08_13-58-17_ID03_Alessandra_protB_SF", "2019-01-08_13-55-20_ID03_Alessandra_of",
    "2019-01-08_13-53-01_ID03_Alessandra_oa", "2019-01-08_08-54-32_ID02_Matheus_protB_SF", "2019-01-08_08-42-17_ID02_Matheus_of",
    "2019-01-08_08-39-55_ID02_Matheus_oa"
]

import pandas as pd
import re

dados = []

# Regex para extrair as partes: Data/Hora, ID, Nome e o Protocolo Final
# Captura: (Data_Hora) _ (ID) _ (Nome_Aluno) _ (Protocolo)
PADRAO_PASTA = re.compile(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_(ID\d+)_([A-Za-z]+)_([a-zA-Z0-9_]+)")

for nome_arquivo in todas_as_pastas:
    match = PADRAO_PASTA.match(nome_arquivo)

    if match:
        tempo_inicio_str, id_aluno_num, nome_aluno_bruto, protocolo_final = match.groups()
        
        # 1. Tratamento da coluna ID
        # Garante que o ID fique no formato "ID_XX"
        id_final = id_aluno_num.replace('ID', 'ID_')

        # 2. Tratamento da coluna Tempo_inicio
        # Apenas pega a string da data/hora no formato 'YYYY-MM-DD_HH-MM-SS'
        # A conversão para pd.to_datetime será feita no DataFrame.

        # 3. Tratamento da coluna Protocolo
        # Padroniza para maiúsculas e lida com as variações 'oa'/'of'
        print(protocolo_final)
        if (protocolo_final == 'oa') or (protocolo_final == 'of'):
            protocolo = protocolo_final.upper()
        elif protocolo_final.startswith('P'): 
            protocolo = protocolo_final.replace('P','p')
        else: protocolo = protocolo_final


        dados.append({
            'ID': id_final,
            'Tempo_inicio_str': tempo_inicio_str, # String temporária
            'Protocolo': protocolo,
            'Nome_arquivo': nome_arquivo
        })

# Criação do DataFrame com os 
df_nome_pastas = pd.DataFrame(dados)

# Conversão da coluna Tempo_inicio para pd.datetime
# O pandas é inteligente e lida com o formato 'YYYY-MM-DD_HH-MM-SS'
df_nome_pastas['Tempo_inicio'] = pd.to_datetime(df_nome_pastas['Tempo_inicio_str'], format='%Y-%m-%d_%H-%M-%S')

# 6. Reordenação e limpeza das colunas
df_nome_pastas = df_nome_pastas[['ID', 'Tempo_inicio', 'Protocolo', 'Nome_arquivo']]

# 7. Colocando em ordem crescente
df_nome_pastas= df_nome_pastas.sort_values(
                                        by= ['Protocolo','ID'],  # Colunas usadas para ordenação
                                        ascending=[True, True]       # Ordem: Protocolo (A-Z), ID_Num (0-9)
                                        )

df_A['Protocolo'] = 'protA_' + df_A['grupo'].astype(str)
df_B['Protocolo'] = 'protB_' + df_B['grupo'].astype(str) 
df_C['Protocolo'] = 'protC'

# --- Protocolo A
'''
Descobri que na verdade não existe ID_27 no procolo A SV
Por isso, estou trocando esse ID para o ID CORRETO
ID_27 -> ID_28
'''
#df_A.loc[df['ID'] == 'ID_27', 'ID'] = 'ID_28'
df_A['ID'] = df_A['ID'].replace('ID_27','ID_28')

tempos_inicio = []
for _, linha in df_A.iterrows():
    ind = linha['ID']
    prot = linha['Protocolo']
    tempos_inicio.append(df_nome_pastas.loc[(df_nome_pastas['ID'] == ind) & (df_nome_pastas['Protocolo'] == prot),'Tempo_inicio'].iloc[0])  # pega o ESCALAR
df_A['Tempo_inicio'] = pd.to_datetime(tempos_inicio)

#Colunas de variação de cada tempo em um formato legível
df_A['Delta_t1'] = (df_A['Tempo 1'] - df_A['Tempo_inicio'])
df_A['Delta_t2'] = df_A['Tempo 2'] - df_A['Tempo_inicio']
df_A['Delta_t3'] = df_A['Tempo 3'] - df_A['Tempo_inicio']
#Delta em segundos
for i, col in enumerate([c for c in df_A.columns if c.startswith('Delta_')]):
    df_A[f'd{i+1}_s'] = df_A[col].dt.total_seconds()


# --- Protocolo B
tempos_inicio = []
for _, linha in df_B.iterrows():
    ind = linha['ID']
    prot = linha['Protocolo']
    tempos_inicio.append(df_nome_pastas.loc[(df_nome_pastas['ID'] == ind) & (df_nome_pastas['Protocolo'] == prot),'Tempo_inicio'].iloc[0])  # pega o ESCALAR
df_B['Tempo_inicio'] = pd.to_datetime(tempos_inicio)

#Colunas de variação de cada tempo em um formato legível
df_B['Delta_t1'] = (df_B['Tempo 1'] - df_B['Tempo_inicio'])
df_B['Delta_t2'] = df_B['Tempo 2'] - df_B['Tempo_inicio']
df_B['Delta_t3'] = df_B['Tempo 3'] - df_B['Tempo_inicio']
#Delta em segundos
for i, col in enumerate([c for c in df_B.columns if c.startswith('Delta_')]):
    df_B[f'd{i+1}_s'] = df_B[col].dt.total_seconds()


# --- Protocolo C
tempos_inicio = []
for _, linha in df_C.iterrows():
    ind = linha['ID']
    prot = linha['Protocolo']
    tempos_inicio.append(df_nome_pastas.loc[(df_nome_pastas['ID'] == ind) & (df_nome_pastas['Protocolo'] == prot),'Tempo_inicio'].iloc[0])  # pega o ESCALAR
df_C['Tempo_inicio'] = pd.to_datetime(tempos_inicio)

#Colunas de variação de cada tempo em um formato legível
df_C['Delta_t1'] = (df_C['Tempo 1'] - df_C['Tempo_inicio'])
df_C['Delta_t2'] = df_C['Tempo 2'] - df_C['Tempo_inicio']
#Delta em segundos
for i, col in enumerate([c for c in df_C.columns if c.startswith('Delta_')]):
    df_C[f'd{i+1}_s'] = df_C[col].dt.total_seconds()


#%% Cortando os dados 
import os, re
import numpy as np
import pandas as pd
from scipy.io import loadmat

FS = 1000  # Hz (ajuste se necessário)

# ---------------- helpers ----------------
def ensure_output_cols(df: pd.DataFrame):
    if 'Trecho_eeg' not in df.columns:
        df['Trecho_eeg'] = None
    for c in ['idx_ini','idx_fim','n_amostras','_trecho_info']:
        if c not in df.columns:
            df[c] = np.nan

def cut_and_fill(df: pd.DataFrame, id_value: str, protocolo: str,
                 eeg_dict: dict, fs: int, start_col: str, end_col: str):
    """
    Procura todas as linhas de df com (ID, Protocolo) e grava o corte do EEG em 'Trecho_eeg'.
    start_col/end_col são nomes de colunas com tempos em segundos (float).
    """
    mask = (df['ID'] == id_value) & (df['Protocolo'] == protocolo)
    if not mask.any():
        print(f'DEU MERDA! ID: {id_value}, protocolo: {protocolo}')
        return

    n = eeg_dict['CZ'].shape[-1]  # nº de amostras do arquivo atual
    for idx, row in df.loc[mask].iterrows():
        t0 = row.get(start_col, np.nan)
        t1 = row.get(end_col, np.nan)

        if pd.isna(t0) or pd.isna(t1):
            df.at[idx, 'Trecho_eeg'] = None
            df.at[idx, '_trecho_info'] = f'faltou {start_col} ou {end_col}'
            continue

        i0 = max(0, int(round(t0 * fs)))
        i1 = min(n, int(round(t1 * fs)))

        if i1 <= i0:
            df.at[idx, 'Trecho_eeg'] = None
            df.at[idx, '_trecho_info'] = f'intervalo inválido ({i0}, {i1})'
            #print(f'd1_s: {int(round(t0 * fs))}, d2_s: {int(round(t1 * fs))}, n ={n}\n i0: {i0}, i1:{i1}')
            continue

        trecho = {ch: sig[i0:i1].copy() for ch, sig in eeg_dict.items()}
        df.at[idx, 'Trecho_eeg']  = trecho
        df.at[idx, 'idx_ini']     = i0
        df.at[idx, 'idx_fim']     = i1
        df.at[idx, 'n_amostras']  = i1 - i0
        df.at[idx, '_trecho_info'] = 'ok'

# ---------------- preparar DFs ----------------
ensure_output_cols(df_A)
ensure_output_cols(df_B)
ensure_output_cols(df_C)

# mapeamento de como cortar por protocolo
PROTO_RULES = {
    'A': {'df': df_A, 'start_col': 'd2_s', 'end_col': 'd3_s'},        # d2 -> d3
    'B': {'df': df_B, 'start_col': 'd1_s', 'end_col': 'd2_s'},        # d1 -> d2
    'C': {'df': df_C, 'start_col': 'd1_s', 'end_col': 'd2_s'},        # d1 -> d2 (df_C não tem d3_s)
}

# ---------------- regex ----------------
PADRAO_ID_ARQUIVO = re.compile(r"ID_?(\d+)") 
PADRAO_PROTOCOLO_PASTA = re.compile(r".*(prot[A-Za-z]_[A-Z]{2,2}|prot[C,c,c])_mat$") 

# ---------------- varrer as pastas/arquivos ----------------
lista_geral = [l for l in os.listdir(r'D:\dados_pro_diego\arquivos_filtrados_mat')
               if l.startswith('filtrado geral_p')]


for pasta in lista_geral:
    m_prot = PADRAO_PROTOCOLO_PASTA.match(pasta)
    if not m_prot:
        continue
    protocolo = m_prot.groups()[0]          # ex: 'protA_CV', 'protB_SV', 'protC'
    prot_key  = protocolo[4].upper()        # 'A' | 'B' | 'C'

    if prot_key not in PROTO_RULES:
        print(f'Protocolo não mapeado: {protocolo}')
        continue

    rule = PROTO_RULES[prot_key]
    df_target   = rule['df']
    start_col   = rule['start_col']
    end_col     = rule['end_col']

    print(f"\n--- Pasta: {pasta} | Protocolo: {protocolo} -> DF alvo: {['A','B','C'][['A','B','C'].index(prot_key)]} ---")

    file_names = os.listdir(rf'D:\dados_pro_diego\arquivos_filtrados_mat\{pasta}')
    for arquivo in file_names:
        m_id = PADRAO_ID_ARQUIVO.search(arquivo)
        if not m_id:
            continue
        ind = f'ID_{m_id.group(1)}'

        # carrega EEG do .mat
        data = loadmat(rf'D:\dados_pro_diego\arquivos_filtrados_mat\{pasta}\{arquivo}')
        eeg  = data['eeg_data']  # ajuste se o nome do campo for outro

        for i in range(0,data['chanlocs'].shape[-1]):
            if str(data['chanlocs'][0][i][0][0]) == 'CZ':
                idx_CZ = i
            elif str(data['chanlocs'][0][i][0][0]) == 'C3':
                idx_C3 = i
            elif str(data['chanlocs'][0][i][0][0]) == 'C4':
                idx_C4 = i

        # canais (ajuste os índices se necessário)
        eeg_dict = {'CZ': eeg[idx_CZ], 'C3': eeg[idx_C3], 'C4': eeg[idx_C4]}

        # corta e preenche as linhas correspondentes no DF correto
        cut_and_fill(df_target, ind, protocolo, eeg_dict, FS, start_col, end_col)

from scipy import signal #para subamostragem

lista_baseline_geral = [l for l in os.listdir(r'D:\dados_pro_diego\arquivos_filtrados_mat')
               if l.startswith('filtrado geral_B')]
PADRAO_BASELINE_PASTA = re.compile(r"^filtrado geral_(Baseline (OF|OA))_mat$")

ind = []
eeg_signal =[]
grupo =[]
for pasta in lista_baseline_geral:
    m_baseline = PADRAO_BASELINE_PASTA.match(pasta)
    if not m_baseline:
        continue
    protocolo = m_baseline.groups()[0]      # ex: 'OF' ou 'OA'
    file_names = os.listdir(rf'D:\dados_pro_diego\arquivos_filtrados_mat\{pasta}')
    
    for arquivo in file_names:
        m_id = f'ID_{arquivo[:2]}'
    
        # carrega EEG do .mat
        data = loadmat(rf'D:\dados_pro_diego\arquivos_filtrados_mat\{pasta}\{arquivo}')
        eeg  = data['eeg_data']

        for i in range(0,data['chanlocs'].shape[-1]):
            if str(data['chanlocs'][0][i][0][0]) == 'CZ':
                idx_CZ = i
            elif str(data['chanlocs'][0][i][0][0]) == 'C3':
                idx_C3 = i
            elif str(data['chanlocs'][0][i][0][0]) == 'C4':
                idx_C4 = i
        #Downsample para 1000Hz
        eeg = signal.resample_poly(
                    eeg, 
                    up=1, 
                    down=2, 
                    axis=1)  # <-- eixo dos sinais
        # pegando EEG dos canais CZ C3 e C4
        eeg_dict = {'CZ': eeg[idx_CZ], 'C3': eeg[idx_C3], 'C4': eeg[idx_C4]}
        ind.append(m_id)
        eeg_signal.append(eeg_dict)
        grupo.append(protocolo)

df_baseline = {
        'ID':ind,
        'grupo': grupo,
        'Trecho_eeg':eeg_signal,
    }
df_baseline = pd.DataFrame(df_baseline)  


#%% Calculando a PSD 
#Adicionando mais colunas so para um teste
df_A['Acuracia'] = df_protA['Acuracia']
df_A['Especificidade'] = df_protA['Especificidade']
df_A['Similaridade'] = df_protA['Similaridade']
df_B['Acuracia'] = df_protB['Acuracia']
df_B['Especificidade'] = df_protB['Especificidade']
df_B['Similaridade'] = df_protB['Similaridade']

# Removendo as que deram problemas 
erro_A = df_A[df_A['_trecho_info']!='ok']['ID'].unique()

df_A_final = df_A[~df_A['ID'].isin(erro_A)] #Pego todos as linhas que não tem problema "~" serve para eu pegar ao contrário dos que estão dentro dos erros

erro_B = df_B[df_B['_trecho_info']!='ok']['ID'].unique()

df_B_final = df_B[~df_B['ID'].isin(erro_B)]

erro_C = df_C[df_C['_trecho_info']!='ok']['ID'].unique()

df_C_final = df_C[~df_C['ID'].isin(erro_C)]


#%% Calculo da psd dos trechos e já normalizando pela baseline

from scipy.signal import welch, get_window

def add_psd_column(df: pd.DataFrame,
                   fs: float = 1000.0,
                   method: str = "welch",
                   window: str = "hann",
                   nperseg: int = 2*1000,
                   noverlap: int = 1000,
                   detrend: str = "constant",
                   scaling: str = "density",
                   channels=("CZ", "C3", "C4"),
                   coluna_trecho: str = "Trecho_eeg",
                   coluna_saida: str = "psd_trecho") -> pd.DataFrame:
    """
    Calcula a PSD (Welch) dos trechos em `coluna_trecho` e salva em `coluna_saida`.
    Cada célula de `coluna_trecho` deve ser um dict {canal: array_1d}.
    Saída por linha: dict {canal: (freq, psd)} com arrays numpy.

    Parâmetros padrão:
      fs=1000 Hz, window='hann', nperseg=1024, noverlap=512, detrend='constant', scaling='density'
    """
    if coluna_saida not in df.columns:
        df[coluna_saida] = None

    win = get_window(window, nperseg)  # pré-cria janela

    def _psd_de_uma_linha(trecho_dict):
        if not isinstance(trecho_dict, dict):
            return None

        out = {}
        for ch in channels:
            sig = trecho_dict.get(ch, None)
            if sig is None:
                out[ch] = None
                continue

            sig = np.asarray(sig).ravel()
            if sig.size < 4:  # muito curto
                out[ch] = None
                continue

            # Ajusta nperseg para não exceder o tamanho do trecho
            nseg = min(nperseg, sig.size)
            novl = min(noverlap, max(0, nseg // 2))

            f, Pxx = welch(
                sig,
                fs=fs,
                window=win if nseg == nperseg else get_window(window, nseg),
                nperseg=nseg,
                noverlap=novl,
                detrend=detrend,
                scaling=scaling,
                return_onesided=True
            )
            out[ch] = (f, Pxx)
        return out

    # aplica linha a linha
    df[coluna_saida] = df[coluna_trecho].apply(_psd_de_uma_linha)
    return df

def add_bandpowers_per_channel(
    df: pd.DataFrame,
    bands: dict | None = None,
    channels: tuple[str, ...] = ("CZ", "C3", "C4"),
    psd_col: str = "psd_trecho",
    prefix: str = "psd",                 # prefixo das colunas criadas
    normalize_by_bandwidth: bool = False # se True, divide pela largura da banda (média por Hz)
) -> pd.DataFrame:
    """
    Cria colunas com potência por banda PARA CADA CANAL, a partir de `psd_col`.

    Espera por linha: df[psd_col] = {'CZ': (f, Pxx), 'C3': (f, Pxx), 'C4': (f, Pxx)}.
    Cria colunas no padrão: {prefix}_{banda}_{canal}, ex.: psd_delta_CZ, psd_theta_C3, ...

    bands: dict como {"delta": (0.5,4), "theta": (4,8), "alfa": (8,13), "beta": (13,30), "gamma": (30,60)}
    channels: canais presentes no dict da PSD (keys do psd_trecho)
    normalize_by_bandwidth: se True, retorna potência média por Hz (bandpower/Δf).
    """
    # bandas padrão (Hz)
    if bands is None:
        bands = {
            "delta": (0.5, 4),
            "theta": (4, 8),
            "alfa":  (8, 13),
            "beta":  (13, 30),
            "gamma": (30, 60),
        }

    # garantir que as colunas existem (preenche com NaN)
    for banda in bands.keys():
        for ch in channels:
            colname = f"{prefix}_{banda}_{ch}"
            if colname not in df.columns:
                df[colname] = np.nan

    def _bandpower(sig_psd_item, f_lo, f_hi):
        """Calcula potência (ou potência média por Hz) em [f_lo, f_hi] para um (f,Pxx)."""
        if sig_psd_item is None:
            return np.nan
        f, Pxx = sig_psd_item
        if f is None or Pxx is None:
            return np.nan
        f = np.asarray(f).ravel()
        Pxx = np.asarray(Pxx).ravel()

        mask = (f >= f_lo) & (f <= f_hi) & np.isfinite(Pxx)
        if mask.sum() < 2:
            return np.nan

        bp = np.trapz(Pxx[mask], f[mask])  # integral da PSD na banda
        if normalize_by_bandwidth:
            bw = (f_hi - f_lo)
            if bw > 0:
                bp = bp / bw
        return float(bp)

    # preencher linha a linha
    for idx, psd_dict in df[psd_col].items():
        if not isinstance(psd_dict, dict):
            # deixa NaN nas colunas já criadas
            continue
        for banda, (f_lo, f_hi) in bands.items():
            for ch in channels:
                colname = f"{prefix}_{banda}_{ch}"
                val = _bandpower(psd_dict.get(ch, None), f_lo, f_hi)
                df.at[idx, colname] = val

    return df

nperseg = 2048 #(potencia de dois mais proxima de FS*2)
noverlap=1024 #nperseg//2

df_A_final = add_psd_column(df_A_final, fs=1000, window="hann", nperseg = nperseg, noverlap=noverlap)
df_B_final =  add_psd_column(df_B_final, fs=1000, window="hann", nperseg = nperseg, noverlap=noverlap)  
df_baseline = add_psd_column(df_baseline, fs=1000, window="hann", nperseg = nperseg, noverlap=noverlap)

df_A_final = add_bandpowers_per_channel(df_A_final)
df_B_final = add_bandpowers_per_channel(df_B_final)
df_baseline = add_bandpowers_per_channel(df_baseline)

#Reordenando as colunas
ordem_colunas = ['Tempo 1', 'Tempo 2', 'Tempo 3', 'ID', 'grupo', 'Desempenho','Acuracia',
                 'Similaridade', 'Especificidade',
                 'Protocolo','Complexidade', 'Tempo_inicio', 'Delta_t1', 'Delta_t2', 'Delta_t3', 
                 'd1_s','d2_s', 'd3_s', 'Trecho_eeg', 'idx_ini', 'idx_fim', 
                 'n_amostras','_trecho_info',
                'psd_delta_CZ', 'psd_theta_CZ', 'psd_alfa_CZ', 'psd_beta_CZ', 'psd_gamma_CZ',
                'psd_delta_C3', 'psd_theta_C3', 'psd_alfa_C3', 'psd_beta_C3', 'psd_gamma_C3',
                'psd_delta_C4', 'psd_theta_C4', 'psd_alfa_C4', 'psd_beta_C4', 'psd_gamma_C4',
                'psd_trecho']
df_A_final = df_A_final[ordem_colunas]
df_B_final = df_B_final[ordem_colunas]
df_baseline = df_baseline[['ID','grupo','Trecho_eeg', 'psd_trecho','psd_delta_CZ', 'psd_theta_CZ', 'psd_alfa_CZ', 'psd_beta_CZ', 'psd_gamma_CZ',
                'psd_delta_C3', 'psd_theta_C3', 'psd_alfa_C3', 'psd_beta_C3', 'psd_gamma_C3',
                'psd_delta_C4', 'psd_theta_C4', 'psd_alfa_C4', 'psd_beta_C4', 'psd_gamma_C4',]]

#%%Normalizando pela baseline
relacoes_A = {
    'CV': 'Baseline OA',
    'SV': 'Baseline OF'
}
relacoes_B = {
    'CF': 'Baseline OF',
    'SF': 'Baseline OF'
}
def normalizar_bandas(df_a_normalizar, df_baseline, relacoes):
    #df_a_normalizar = df_A_final.copy()
    """
    Normaliza as potências de bandas de EEG de um DataFrame de tarefa (df_a_normalizar)
    em relação aos valores correspondentes de baseline (df_baseline), criando novas
    colunas com os valores normalizados.

    A função percorre cada linha de df_a_normalizar, identifica o participante (ID)
    e o grupo experimental, encontra a baseline correspondente (de acordo com o dicionário
    `relacoes` que mapeia grupo de tarefa → condição de baseline), e divide a potência
    de cada banda e canal do EEG da tarefa pela potência da respectiva banda e canal
    da baseline do mesmo participante. O resultado é gravado em novas colunas que
    começam com o prefixo `psd_norm_`.

    Parâmetros
    ----------
    df_a_normalizar : pandas.DataFrame
        DataFrame contendo os valores de potência espectral (PSD) de cada banda e canal
        durante a tarefa. Deve conter as colunas:
            - 'ID' (identificador do participante)
            - 'grupo' (nome do grupo, ex: 'CF' ou 'SF')
            - colunas de PSD no formato 'psd_<banda>_<canal>' (ex: 'psd_delta_C3')
    df_baseline : pandas.DataFrame
        DataFrame contendo as potências de baseline de cada participante nas condições
        correspondentes ('Baseline OA' ou 'Baseline OF'), com as mesmas colunas de PSD
        do DataFrame de tarefa.
    relacoes : dict
        Dicionário que relaciona o grupo experimental de df_a_normalizar à condição
        de baseline correspondente em df_baseline.
        Exemplo: {'CF': 'Baseline OF', 'SF': 'Baseline OF'}

    Retorna
    -------
    pandas.DataFrame
        O mesmo DataFrame df_a_normalizar, acrescido de novas colunas contendo os valores
        normalizados das potências PSD (uma para cada coluna de PSD original),
        nomeadas como `psd_norm_<banda>_<canal>`.

    Notas
    -----
    - A normalização é feita elemento a elemento: valor_tarefa / valor_baseline.
    - Se o mesmo participante não possuir baseline correspondente, a função lançará
    um erro ao tentar acessar `.iloc[0]` — recomenda-se garantir previamente a correspondência.
    - A coluna 'psd_trecho' é ignorada no processo de normalização.

    Exemplo de uso
    --------------
    >>> relacoes = {'CF': 'Baseline OF', 'SF': 'Baseline OF'}
    >>> df_norm = normalizar_bandas(df_A_final, df_baseline, relacoes)
    >>> df_norm.filter(like='psd_norm_').head()
    """

    cols_norm = [f'psd_norm_{c[4:]}' for c in df_a_normalizar.columns if (c.startswith('psd_')) & (c != 'psd_trecho')]
    df_a_normalizar[cols_norm] = 0
    for idx, row in df_a_normalizar.iterrows():
        ind = row['ID']
        grupo = relacoes[row['grupo']]
        cols = [c for c in df_a_normalizar.columns if (c.startswith('psd_')) & (c != 'psd_trecho') & (not c.startswith('psd_norm_'))]
        for col in cols:
            mask = (df_baseline['ID'] == ind) & (df_baseline['grupo'] == grupo)
            # valor da baseline 
            psd_baseline_trecho = df_baseline[mask][col].iloc[0]
            # valor da ser normalizado 
            psd_atual_trecho = row[col]
            valor_normalizado = psd_atual_trecho/psd_baseline_trecho
            #Atribuir esse valor para a df original
            df_a_normalizar.at[idx, f'psd_norm_{col[4:]}'] = valor_normalizado

    return df_a_normalizar
df_A_final = normalizar_bandas(df_a_normalizar= df_A_final, 
                               df_baseline= df_baseline, relacoes= relacoes_A)
df_B_final = normalizar_bandas(df_a_normalizar= df_B_final, 
                               df_baseline= df_baseline, relacoes= relacoes_B)          
 

#%% 3) Finalmente Fazendo a PLSC
from numpy.linalg import svd

def plsc_basic(X, Y, standardize=True):
    """     
    Perform Partial Least Squares Correlation (PLSC) between two data matrices.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features_x)
        Brain/EEG data matrix. Each row is ONE observation (e.g. one subject,
        or one trial), and each column is ONE EEG feature (e.g. channel power,
        channel×frequency, etc.).
        
        Example for your case (after we reshape):
            - n_samples = number of subjects            (subject-level PLSC)
              or
            - n_samples = total number of trials across subjects (trial-level PLSC)
            - n_features_x = 32 * n_freqs   (if you flatten 32×freq PSD into 1D)

    Y : array-like, shape (n_samples, n_features_y)
        Behavioral / performance / design matrix. Must have the SAME number of
        rows (observations) as X. Columns can be:
            - one performance value per observation        -> shape (n_samples, 1)
            - several behavioral variables (RT, error, ...) -> shape (n_samples, k)

    standardize : bool, default=True
        If True, each column of X and Y is z-scored (mean=0, std=1) before PLSC.
        This is the typical "correlation" style PLS in neuroimaging.

    Returns
    -------
    results : dict
        A dictionary with the main PLSC elements:

        - 'U' : array, shape (n_features_y, n_lv)
            Saliences (weights) for Y variables.
        - 'V' : array, shape (n_features_x, n_lv)
            Saliences (weights) for X variables.
        - 'S' : array, shape (n_lv,)
            Singular values. Each value reflects the covariance captured by
            that latent variable (LV).
        - 'LX' : array, shape (n_samples, n_lv)
            Latent "brain" scores: projection of X onto V  -> LX = Xz @ V
        - 'LY' : array, shape (n_samples, n_lv)
            Latent "behavior" scores: projection of Y onto U -> LY = Yz @ U
        - 'explained_cov' : array, shape (n_lv,)
            Fraction of total cross-covariance explained by each LV.
        - 'Xz', 'Yz' : arrays
            Standardized versions of X and Y (helpful for debugging).

    Notes
    -----
    - This is the core of PLSC as used in neuro papers: compute cross-covariance
      (or cross-correlation, if standardized) between Y and X, do SVD, get LVs.
    - This function ASSUMES your data is already in tabular (2D) form.
      Your current data is nested (each row has a 32×L PSD). In the next step
      we will write helper functions to FLATTEN those structures into the X, Y
      that this function expects.
    """
    # Convert to arrays
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    # Basic shape checks
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"X and Y must have the same number of rows (observations). "
                         f"Got X.shape={X.shape}, Y.shape={Y.shape}")

    # 1. Standardize columns (correlation PLS)
    if standardize:
        Xz = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, ddof=1, keepdims=True) + 1e-12)
        Yz = (Y - Y.mean(axis=0, keepdims=True)) / (Y.std(axis=0, ddof=1, keepdims=True) + 1e-12)
    else:
        Xz = X - X.mean(axis=0, keepdims=True)
        Yz = Y - Y.mean(axis=0, keepdims=True)

    # 2. Cross-covariance / correlation matrix
    #    R has shape (n_features_y, n_features_x)
    R = Yz.T @ Xz

    # 3. SVD of the cross-block
    #    R = U S V^T
    U, S, Vt = svd(R, full_matrices=False)
    V = Vt.T

    # 4. Project to get latent scores
    LX = Xz @ V   # "brain" scores
    LY = Yz @ U   # "behavior" scores

    # 5. Explained covariance (like in the article)
    total = (S ** 2).sum()
    explained_cov = (S ** 2) / (total + 1e-15)

    return {
        "U": U,
        "V": V,
        "S": S,
        "LX": LX,
        "LY": LY,
        "explained_cov": explained_cov,
        "Xz": Xz,
        "Yz": Yz,
        "R": R,
    }

#psd_canais = ['psd_delta_CZ', 'psd_theta_CZ', 'psd_alfa_CZ', 'psd_beta_CZ', 'psd_gamma_CZ',
#       'psd_delta_C3', 'psd_theta_C3', 'psd_alfa_C3', 'psd_beta_C3', 'psd_gamma_C3',
#       'psd_delta_C4', 'psd_theta_C4', 'psd_alfa_C4', 'psd_beta_C4', 'psd_gamma_C4']
psd_canais = ['psd_norm_delta_CZ', 'psd_norm_theta_CZ', 'psd_norm_alfa_CZ',
       'psd_norm_beta_CZ', 'psd_norm_gamma_CZ', 'psd_norm_delta_C3',
       'psd_norm_theta_C3', 'psd_norm_alfa_C3', 'psd_norm_beta_C3',
       'psd_norm_gamma_C3', 'psd_norm_delta_C4', 'psd_norm_theta_C4',
       'psd_norm_alfa_C4', 'psd_norm_beta_C4', 'psd_norm_gamma_C4']

# ------ Protocolo A
X1 = df_A_final[df_A_final['grupo' == 'CV']][psd_canais].to_numpy()
Y = df_A_final['Desempenho'].to_numpy()
Y = Y[:,np.newaxis]
#Y = df_A_final[['Acuracia','Similaridade','Especificidade']].to_numpy()

# 1. Standardize columns (correlation PLS)
Xz = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, ddof=1, keepdims=True) + 1e-12)
Yz = (Y - Y.mean(axis=0, keepdims=True)) / (Y.std(axis=0, ddof=1, keepdims=True) + 1e-12)

R = Yz.T@X


#%%
X1 = df_A_final[df_A_final['grupo']=='CV'][psd_canais].to_numpy()
Y1 = df_A_final[df_A_final['grupo']=='CV']['Desempenho'].to_numpy()
Y1 = Y1[:,np.newaxis]
#Y = df_A_final[df_A_final['grupo']=='CV'][['Acuracia','Similaridade','Especificidade']].to_numpy()
#dict_A_CV = plsc_basic(X,Y)

# 1. Standardize columns (correlation PLS)
Xz1 = (X1 - X1.mean(axis=0, keepdims=True)) / (X1.std(axis=0, ddof=1, keepdims=True) + 1e-12)
Yz1 = (Y1 - Y1.mean(axis=0, keepdims=True)) / (Y1.std(axis=0, ddof=1, keepdims=True) + 1e-12)

R1 = Yz1.T@Xz1

R1 = Y1.T@X1

X2 = df_A_final[df_A_final['grupo']=='SV'][psd_canais].to_numpy()
Y2 = df_A_final[df_A_final['grupo']=='SV']['Desempenho'].to_numpy()
Y2 = Y2[:,np.newaxis]
Y = df_A_final[df_A_final['grupo']=='CV'][['Acuracia','Similaridade','Especificidade']].to_numpy()

# 1. Standardize columns (correlation PLS)
Xz2 = (X2 - X2.mean(axis=0, keepdims=True)) / (X2.std(axis=0, ddof=1, keepdims=True) + 1e-12)
Yz2 = (Y2 - Y2.mean(axis=0, keepdims=True)) / (Y2.std(axis=0, ddof=1, keepdims=True) + 1e-12)

R2 = Yz2.T @ Xz2

R2 = Y2.T @ X2

R = np.vstack([R1, R2])
Xz = np.vstack([Xz1,Xz2])
Yz = np.vstack([Yz1,Yz2])

'''#sem centralizar
Xz = np.vstack([X1,X2])
Yz = np.vstack([Y1,Y2])'''

# 3. SVD of the cross-block
#    R = U S V^T
U, S, Vt = svd(R, full_matrices=False)
V = Vt.T

# 4. Project to get latent scores
LX = Xz @ V   # "brain" scores
#LY = Yz @ U   # "behavior" scores

# 5. Explained covariance (like in the article)
total = (S ** 2).sum()
explained_cov = (S ** 2) / (total + 1e-15)


dict_A = {
        "U": U,
        "V": V,
        "S": S,
        "LX": LX,
        "LY": LY,
        "explained_cov": explained_cov,
        "Xz": Xz,
        "Yz": Yz,
        "R": R,
    }

#%% Teste com PCA para os dados do protocolo A
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

pca = PCA()
pca_CV = pca.fit_transform(Xz1)

print(f"\n--- Dados CV Após PCA ---")
print(pca_CV[:5]) # Mostra as primeiras 5 linhas
print(f"Shape Após PCA: {pca_CV.shape}")


# 4. Plotagem dos Resultados da PCA
plt.figure(figsize=(10, 8))

# Itera sobre os grupos únicos nos rótulos (y)
y = ['CV' for i in range(Xz1.shape[0])]
unique_labels = np.unique(y)
colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan'] # Cores para os grupos

for i, label in enumerate(unique_labels):
    # Seleciona apenas as amostras que pertencem ao rótulo atual
    indices_to_plot = y == label
    plt.scatter(pca_CV[indices_to_plot, 0], # Primeiro Componente Principal (eixo X)
                pca_CV[indices_to_plot, 1], # Segundo Componente Principal (eixo Y)
                label=label,
                color=colors[i % len(colors)], # Garante que as cores se repetem se houver muitos grupos
                s=100, # Tamanho do ponto
                alpha=0.7, # Transparência
                edgecolors='w') # Borda branca

plt.xlabel(f'Primeiro Componente Principal (Variância Explicada: {pca.explained_variance_ratio_[0]*100:.2f}%)')
plt.ylabel(f'Segundo Componente Principal (Variância Explicada: {pca.explained_variance_ratio_[1]*100:.2f}%)')
plt.title('Dados Projetados no Espaço PCA (2D)')
plt.grid(True)
plt.legend(title='Grupos')
plt.axhline(0, color='gray', linewidth=0.5) # Linha horizontal em y=0
plt.axvline(0, color='gray', linewidth=0.5) # Linha vertical em x=0
plt.show()

# Opcional: Visualização da variância explicada cumulativa
# É muito útil para decidir quantos componentes reter
plt.figure(figsize=(8, 5))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--')
plt.xlabel('Número de Componentes')
plt.ylabel('Variância Explicada Cumulativa')
plt.title('Curva de Variância Explicada Cumulativa')
plt.grid(True)
plt.show()


#%% ------ Protocolo B
X = df_B_final[psd_canais].to_numpy()
Y = df_B_final['Desempenho'].to_numpy()
Y = Y[:,np.newaxis]
#Y = df_B_final[['Acuracia','Similaridade','Especificidade']].to_numpy()

dict_B = plsc_basic(X,Y)

X = df_B_final[df_B_final['grupo']=='CF'][psd_canais].to_numpy()
Y = df_B_final[df_B_final['grupo']=='CF']['Desempenho'].to_numpy()
Y = Y[:,np.newaxis]
#Y = df_B_final[df_B_final['grupo']=='CV'][['Acuracia','Similaridade','Especificidade']].to_numpy()
dict_B_CF = plsc_basic(X,Y)

X = df_B_final[df_B_final['grupo']=='SF'][psd_canais].to_numpy()
Y = df_B_final[df_B_final['grupo']=='SF']['Desempenho'].to_numpy()
Y = Y[:,np.newaxis]
#Y = df_B_final[df_B_final['grupo']=='SV'][['Acuracia','Similaridade','Especificidade']].to_numpy()
dict_B_SF = plsc_basic(X,Y)

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


#%% --------- Protocolo B --------- 

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

#%% PLOTS DAS DISTRIBUIÇÕES DOS DADOS DE DESEMPENHO
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Copia de segurança 
df = df_A_final.copy() 
#  Garante ordem fixa dos grupos 
order = ['CV', 'SV']
# Paleta de cores para cada grupo (ex.: azul e laranja) 
palette_violin = { 
    'CV': '#4C72B0', # azul suave 
    'SV': '#DD8452' # laranja suave 
    }

# -------------------------------------------------------
# Cálculo das médias e IC95%
# -------------------------------------------------------
means = []
ci_low = []
ci_high = []

for g in order:
    data = df[df['grupo'] == g]['Desempenho'].dropna()
    m = data.mean()
    sem = stats.sem(data)
    ci = 1.96 * sem  # IC 95% aproximado

    means.append(m)
    ci_low.append(m - ci)
    ci_high.append(m + ci)

means = np.array(means)
ci_low = np.array(ci_low)
ci_high = np.array(ci_high)


# -------------------------------------------------------
# Figura com subplots
# -------------------------------------------------------
# Cálculo da distribuição geral (todos os ensaios juntos)
data_all = df['Desempenho'].dropna()
m_all = data_all.mean()
sem_all = stats.sem(data_all)
ci_all = 1.96 * sem_all  # IC95% geral

# Cria figura com 2 subplots lado a lado, o da esquerda mais estreito
fig, axes = plt.subplots(
    1, 2,
    figsize=(9, 6),
    gridspec_kw={'width_ratios': [1, 2.2]},
    sharey=True
)

# -------------------------------------------------------
# SUBPLOT ESQUERDO: distribuição geral
# -------------------------------------------------------
df_all = df[['Desempenho']].copy()
df_all['cond'] = 'Geral'

sns.violinplot(
    data=df_all,
    x='cond',
    y='Desempenho',
    inner=None,
    cut=0,
    linewidth=1,
    color='#B0B0B0',   # cinza neutro para o geral
    ax=axes[0]
)

sns.stripplot(
    data=df_all,
    x='cond',
    y='Desempenho',
    color='k',
    size=3,
    jitter=0.12,
    alpha=0.25,
    ax=axes[0]
)

# Média + IC95% gerais
axes[0].scatter(
    0, m_all,
    s=30,
    color='white',
    edgecolor='black',
    zorder=10
)

axes[0].errorbar(
    0, m_all,
    yerr=ci_all,
    fmt='none',
    ecolor='black',
    elinewidth=1,
    capsize=6,
    capthick=1.5,
    zorder=9
)

axes[0].set_xlabel('Geral', fontsize=13)
axes[0].set_title('Distribuição Geral', fontsize=14)

# -------------------------------------------------------
# SUBPLOT DIREITO: distribuição por grupo (seu gráfico)
# -------------------------------------------------------
sns.violinplot(
    data=df,
    x='grupo',
    y='Desempenho',
    order=order,
    inner=None,
    cut=0,
    linewidth=1,
    palette=palette_violin,
    ax=axes[1]
)

sns.stripplot(
    data=df,
    x='grupo',
    y='Desempenho',
    order=order,
    color='k',
    size=3,
    jitter=0.12,
    alpha=0.35,
    ax=axes[1]
)

axes[1].scatter(
    x=np.arange(len(order)),
    y=means,
    s=30,
    color='white',
    edgecolor='black',
    zorder=10,
    label='Média'
)

yerr = np.vstack([means - ci_low, ci_high - means])

axes[1].errorbar(
    x=np.arange(len(order)),
    y=means,
    yerr=yerr,
    fmt='none',
    ecolor='black',
    elinewidth=1,
    capsize=6,
    capthick=1.5,
    zorder=9,
    label='IC 95%'
)

axes[1].set_xlabel('Grupo', fontsize=13)
axes[1].set_title('Desempenho por Grupo (Protocolo A)', fontsize=14)

# -------------------------------------------------------
# Estética comum aos dois
# -------------------------------------------------------
for ax in axes:
    ax.set_ylabel('Desempenho', fontsize=13)
    ax.set_yticks(np.arange(0, 1.05, 0.05))
    ax.set_ylim(0.2, 1.1)
    ax.grid(axis='y', linestyle='--', linewidth=0.6, alpha=0.4)
    sns.despine(ax=ax)

# Legenda só no da direita
axes[1].legend(frameon=False, fontsize=11, loc='lower right')

fig.suptitle('Desempenho – Distribuição Geral e por Grupo (Protocolo A)', fontsize=15, y=0.98)

plt.tight_layout()
plt.show()


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

#%% 2. Itere sobre a chave (nome) e o valor (df)
for nome_canal, df_canal in dict_canais.items():
    for grupo in grupos:
        # Monta o nome do arquivo dinamicamente
        nome_arquivo = f'X_{grupo}_psd_{nome_canal}.csv'
        
        # Filtra e salva
        # Nota: Adicionei a lógica de f-string correta no to_csv
        df_canal[df_canal['grupo'] == grupo][df_canal.columns[:-1]].to_csv(nome_arquivo)
        
        print(f"Salvo: {nome_arquivo}")



#%% Fazendo os plots do PLSC de acordo com o paper
COL_GROUPS = None  # deixe None para cores automáticas do Matplotlib
MARKERS = ["o","s","^","D","P","X"]

# =========================
# 1) Barras de saliências
# =========================
def plot_saliences_bars(V, labels, lv=0, title="EEG Saliences (X) – LV1"):
    """
    Plot EEG saliences (V) for a given latent variable (LV) as a sorted bar plot.

    Parameters
    ----------
    V : ndarray, shape (n_features_x, n_lv)
        Saliences/loadings of the X block returned by PLSC.
    labels : list of str
        Names of each X variable (e.g., "alpha_C3").
    lv : int, default=0
        Latent variable index to plot.
    title : str
        Figure title.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object containing the bar plot.
    """
    import numpy as np
    coeff = V[:, lv]
    #idx = np.argsort(coeff)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.bar(range(len(coeff)), coeff)
    ax.set_xticks(range(len(coeff)))
    ax.set_xticklabels(np.array(labels), rotation=90)
    ax.axhline(0, lw=0.8, color="k")
    ax.set_yticks(np.arange(-0.6,0.66,.05))
    ax.set_ylabel("Salience (loading)")
    ax.set_title(title)
    ax.grid(True,axis='y')
    plt.tight_layout()
    return fig

def plot_saliences_bars_Y(U, y_labels, lv=0, title="Behavior/Design Saliences (Y) – LV1"):
    """
    Plot behavioral/design saliences (U) for a given LV as a bar plot.

    Parameters
    ----------
    U : ndarray, shape (n_features_y, n_lv)
        Saliences/loadings of the Y block from PLSC.
    y_labels : list of str
        Labels of behavioral/design variables.
    lv : int, default=0
        Latent variable index.
    title : str
        Title of the figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure with saliences for Y variables.
    """

    coeff = U[:, lv]
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.bar(range(len(coeff)), coeff)
    ax.set_xticks(range(len(coeff)))
    ax.set_xticklabels(y_labels, rotation=0)
    ax.axhline(0, lw=0.8, color="k")
    ax.set_ylabel("Salience (loading)")
    ax.set_title(title)
    plt.tight_layout()
    return fig

# =============================================
# 2) Scores (LX vs LY) – scatter acoplado LV1
# =============================================
def plot_scores_coupled(LX, LY, groups, lv=0, title="Brain vs Behavior Scores – LV1"):
    """
    Scatter plot of brain (LX) versus behavior (LY) scores for a given LV,
    optionally grouped by condition/group.

    Parameters
    ----------
    LX : ndarray, shape (n_samples, n_lv)
        Brain scores from PLSC.
    LY : ndarray, shape (n_samples, n_lv)
        Behavior/design scores from PLSC.
    groups : array-like
        Group label per observation (e.g., ['CV','SV',...]).
    lv : int, default=0
        Latent variable to plot.
    title : str
        Plot title.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Scatter plot of LX vs LY with group separation and correlation r.
    """
    
    x = LX[:, lv]; y = LY[:, lv]
    group_names = np.unique(groups)
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    for i, g in enumerate(group_names):
        m = groups == g
        ax.scatter(x[m], y[m], label=str(g), s=45, marker=MARKERS[i % len(MARKERS)])
    ax.set_xlabel("Brain score (LX{})".format(lv+1))
    ax.set_ylabel("Behavior/Design score (LY{})".format(lv+1))
    ax.set_title(title)
    ax.legend(frameon=False, title="Group")
    # correlação LX-LY para a LV:
    r = np.corrcoef(x, y)[0,1]
    ax.text(0.02, 0.98, f"r = {r:.2f}", transform=ax.transAxes, ha="left", va="top")
    plt.tight_layout()
    return fig

# ======================================================
# 3) PCA-style de LX (ou LY) em 2D para separação grupos
# ======================================================
def plot_pca_style(scores, groups, lvx=0, lvy=1, title="Brain scores (LX) – LV1 vs LV2"):
    """
    2D PCA-style scatter plot of PLS scores (LX or LY) using two LVs,
    highlighting separation across groups.

    Parameters
    ----------
    scores : ndarray, shape (n_samples, n_lv)
        LX or LY scores.
    groups : array-like
        Group identifiers for coloring/markers.
    lvx : int, default=0
        LV on the x-axis.
    lvy : int, default=1
        LV on the y-axis.
    title : str
        Figure title.

    Returns
    -------
    fig : matplotlib.figure.Figure
        2D scatter plot of scores by group.
    """
    
    group_names = np.unique(groups)
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for i, g in enumerate(group_names):
        m = groups == g
        ax.scatter(scores[m, lvx], scores[m, lvy], label=str(g),
                   s=45, marker=MARKERS[i % len(MARKERS)])
    ax.axhline(0, lw=0.6, color="k"); ax.axvline(0, lw=0.6, color="k")
    ax.set_xlabel(f"LV{lvx+1}")
    ax.set_ylabel(f"LV{lvy+1}")
    ax.set_title(title)
    ax.legend(frameon=False, title="Group")
    plt.tight_layout()
    return fig

# ==========================================
# 4) Scree/permutações (significância de LV)
# ==========================================
def plot_permutation_scree(sing_vals, perm_sing_vals, title="Permutation Scree (Singular Values)"):
    """
    Plot permutation-based scree test showing significance of latent variables.

    Parameters
    ----------
    sing_vals : array-like, shape (n_lv,)
        Observed singular values from PLSC.
    perm_sing_vals : ndarray, shape (n_perm, n_lv)
        Singular values computed from permuted data.
    title : str
        Plot title.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Permutation scree plot with observed, mean permuted, and 95th percentile.
    """
    
    obs = np.array(sing_vals)
    perm = np.array(perm_sing_vals)  # shape: B x L
    L = obs.size
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    # média e p95 dos permutados por LV
    mu = perm.mean(axis=0)
    p95 = np.percentile(perm, 95, axis=0)
    ax.plot(range(1, L+1), obs, marker="o", label="Observed δ")
    ax.plot(range(1, L+1), mu, marker="s", label="Permuted mean")
    ax.plot(range(1, L+1), p95, marker="^", label="Permuted 95th pct")
    ax.set_xlabel("Latent Variable (LV)")
    ax.set_ylabel("Singular value (δ)")
    ax.set_title(title)
    ax.legend(frameon=False)
    plt.tight_layout()
    return fig

# ===================================
# 5) Bootstrap ratios (estabilidade)
# ===================================
def plot_bootstrap_ratios(loadings, boot_se, labels, lv=0, thr=2.0,
                          title="Bootstrap Ratios – X saliences (LV1)"):
    """
    Plot bootstrap ratios (loading / bootstrap SE) for a given LV, sorted,
    with stability threshold lines.

    Parameters
    ----------
    loadings : ndarray, shape (n_features_x, n_lv)
        Saliences (V or U) for X or Y.
    boot_se : ndarray, shape (n_features_x, n_lv)
        Bootstrap standard errors.
    labels : list of str
        Variable names.
    lv : int, default=0
        Latent variable to consider.
    thr : float, default=2.0
        Stability threshold (±2 ≈ p < 0.05).
    title : str
        Title of the graph.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Bar plot of bootstrap ratios.
    """    
    br = loadings[:, lv] / (boot_se[:, lv] + 1e-12)
    idx = np.argsort(br)
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    ax.bar(range(len(br)), br[idx])
    ax.axhline(thr, color="k", ls="--", lw=1)
    ax.axhline(-thr, color="k", ls="--", lw=1)
    ax.set_xticks(range(len(br)))
    ax.set_xticklabels(np.array(labels)[idx], rotation=90)
    ax.set_ylabel("Bootstrap ratio (≈ Z)")
    ax.set_title(title + f"  |  |BR|>{thr}")
    plt.tight_layout()
    return fig

# ======================================================
# 6) Heatmap simples X–Y (correlações por condição)
# ======================================================
def plot_xy_correlation_heatmap(Xz, Yz, x_labels, y_labels, title="Corr(X,Y) por variável"):
    """
    Plot a heatmap of correlations (or covariances) between standardized X and Y blocks.

    Parameters
    ----------
    Xz : ndarray, shape (n_samples, n_features_x)
        Standardized X block.
    Yz : ndarray, shape (n_samples, n_features_y)
        Standardized Y block.
    x_labels : list of str
        Labels for X variables.
    y_labels : list of str
        Labels for Y variables.
    title : str
        Heatmap title.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Heatmap of Y×X correlations.
    """

    # Xz: I x J (padronizado dentro de condição); Yz: I x K
    C = Yz.T @ Xz / (Xz.shape[0] - 1)
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    im = ax.imshow(C, aspect="auto")
    ax.set_yticks(range(len(y_labels))); ax.set_yticklabels(y_labels)
    ax.set_xticks(range(len(x_labels))); ax.set_xticklabels(x_labels, rotation=90)
    ax.set_title(title); fig.colorbar(im, ax=ax, label="corr/cov")
    plt.tight_layout()
    return fig

# ======================================================
# 7) Biplot 2D das variáveis de X (saliências V) + dados
# ======================================================
def plot_biplot_V(V, LX, x_labels, lvx=0, lvy=1, title="Biplot – X saliences & scores"):
    """
    Plot a 2D biplot combining PLS brain scores (LX) and EEG variable saliences (V).

    Parameters
    ----------
    V : ndarray, shape (n_features_x, n_lv)
        X-block saliences.
    LX : ndarray, shape (n_samples, n_lv)
        Brain scores.
    x_labels : list of str
        Names of X variables.
    lvx : int, default=0
        LV on the x-axis.
    lvy : int, default=1
        LV on the y-axis.
    title : str
        Figure title.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Biplot showing arrows for variable saliences and points for scores.
    """

    
    # seta das variáveis (saliências) e pontos (scores LX) na LVx × LVy
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax.scatter(LX[:, lvx], LX[:, lvy], s=20, alpha=0.6)
    for j, lbl in enumerate(x_labels):
        ax.arrow(0, 0, V[j, lvx], V[j, lvy], head_width=0.03, length_includes_head=True)
        ax.text(V[j, lvx]*1.06, V[j, lvy]*1.06, lbl, fontsize=9)
    ax.axhline(0, lw=0.6, color="k"); ax.axvline(0, lw=0.6, color="k")
    ax.set_xlabel(f"LV{lvx+1}"); ax.set_ylabel(f"LV{lvy+1}")
    ax.set_title(title)
    plt.tight_layout()
    return fig

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

#%%
df_bandas = extrair_bandpowers(conjunto_df)

# exemplo: ver médias por dataset/canal/banda
resumo = (df_bandas
          .groupby(['dataset','canal','banda'], as_index=False)[['power','power_rel']]
          .mean())

# %%
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