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


#%% Se precisar para a reunião do dia 16/10

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
#%% Especifico
for ind in df_especifico_norm['psd_ProtA_CV_especifico_df_norm'].index:
    plot_bandas_psd(df_master = df_especifico_norm['psd_ProtA_CV_especifico_df_norm'], ind  = ind,faixa_total =(0,100), titulo_prefixo='Protocolo A CV especifico',escala_db=False)
#%% Geral
for ind in df_especifico['psd_ProtA_CV_especifico_df'].index:
    plot_bandas_psd(df_master = df_especifico['psd_ProtA_CV_especifico_df'], ind  = ind,faixa_total =(0,100), titulo_prefixo='Protocolo A CV geral',escala_db=False)
#%%
df_bandas = extrair_bandpowers(conjunto_df)

# exemplo: ver médias por dataset/canal/banda
resumo = (df_bandas
          .groupby(['dataset','canal','banda'], as_index=False)[['power','power_rel']]
          .mean())
