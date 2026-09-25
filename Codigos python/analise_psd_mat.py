#%% Funções e imports

from scipy.io import loadmat
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
from pathlib import Path
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

#%% NÃO USAR Lendo os arquivos .mat e já transformando em Data Frame
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

#%% NÃO USAR Normalizando pelo base line

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

#%% Preparando os dados para as análises multivariadas feitas no site do dashboard e através da PLS

# Escolhendo se vamos fazer com a fases de Estimulção(protA e protB)/Exploracao(ProtC) ou Execução (protA, protB e protC) 
fase = 1 # 1-> Estimulação/Exploracao, 2-> Execucao 

# 1) Pegando os dados de desempenho
df_protA = pd.read_csv('df_protA.csv')
df_A = pd.read_csv('df_protocoloA_tempos_com_t1_corrigido.csv',parse_dates=['Tempo 1 Corrigido'])['Tempo 1 Corrigido'] # correção do tempo 1
df_protA = pd.concat([df_protA,df_A], axis = 1)
df_protB = pd.read_csv('df_protB.csv')
df_protC = pd.read_csv('df_protC.csv')

df_A = df_protA[['Tempo 1', 'Tempo 1 Corrigido', 'Tempo 2',
                 'Tempo 3','ID','grupo','Desempenho', 'Complexidade', 'Overlap', 
                 'Proporção espacial x', 'Proporção espacial y']]
df_A['ID'] = df_A['ID'].str.replace('df_', '', regex=False)

df_B = df_protB[['Tempo 1','Tempo 2','Tempo 3','ID','grupo','Desempenho', 'Complexidade',
                 'Proporção espacial x', 'Proporção espacial y']]
df_B['ID'] = df_B['ID'].str.replace('df_', '', regex=False)

if fase == 1:
    df_C = df_protC[df_protC['Fase'] == 'Fase Exploracao']
else:
    df_C = df_protC[df_protC['Fase'] == 'Fase Execucao']
col_traj = [c for c in df_C.columns if 'ero da Traj' in c][0]
df_C = df_C[['Tempo 1','Tempo 2', 'ID','Desempenho', 'Complexidade', 'Proporção espacial x', 
             'Proporção espacial y', col_traj]]
df_C.rename(columns={col_traj: 'n_traj'}, inplace=True)
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
        #print(protocolo_final)
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
#df_A['Delta_t1'] = (df_A['Tempo 1'] - df_A['Tempo_inicio'])
df_A['Delta_t1'] = (df_A['Tempo 1 Corrigido'] - df_A['Tempo_inicio'])
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


''' Seria isso caso a Bruna não tivesse feito alguns cortes no sinal original
Como ela precisou fazer uns cortes e concatenou um atrás do outro, 
    a lógica do tempo 1 e do tempo 2 se alteraram, por isso 
    eu estou comentando essa parte.
Caso queria com os valores reais (sem nehum corte, descomentar este trecho)

#Colunas de variação de cada tempo em um formato legível
df_C['Delta_t1'] = (df_C['Tempo 1'] - df_C['Tempo_inicio'])
df_C['Delta_t2'] = df_C['Tempo 2'] - df_C['Tempo_inicio']
#Delta em segundos
for i, col in enumerate([c for c in df_C.columns if c.startswith('Delta_')]):
    df_C[f'd{i+1}_s'] = df_C[col].dt.total_seconds()
'''
primeiros_tempos = df_C.groupby('ID')['Tempo 1'].first().reset_index()

#criando um dicionário para mapear cada ID ao seu primeiro tempo
mapa_primeiro_tempo = primeiros_tempos.set_index('ID')['Tempo 1'].to_dict()
#aplicar o mapeamento à coluna ID do df_C
df_C['novo_tempo_inicio'] = df_C['ID'].map(mapa_primeiro_tempo)
df_C['d1_s'] = 0
df_C['d2_s'] = 0

# Pegando o tamanho de cada trial
df_C['tamanho_original_trial'] = df_C['Tempo 2'] - df_C['Tempo 1']
df_C['tamanho_original_trial'] = df_C['tamanho_original_trial'].dt.total_seconds()

# Corrigindo o protocolo C
#lendo os arquivos que diz o que foi cortado
cortes = pd.read_csv(r'Arquivos Auxiliares\cortes_por_ID_trial_protC.csv')

#pegando apenas as linhas da fase de execucao
cortes = cortes[cortes['ID'].str.contains('_execucao')]
cortes['ID'] = cortes['ID'].str.extract(r'(ID\d+)')

# padroniza para o formato ID_08 (com underline)
cortes['ID'] = cortes['ID'].str.replace(r'ID(\d+)', r'ID_\1', regex=True)

#padroniza o numero dos trials
cortes['Trial'] = cortes['Trial'].str.extract(r'(\d+)')[0].astype(int)

# 1) cortar conforme a lógica considerando os cortes dos ruídos 
PADRAO_ID_ARQUIVO = re.compile(r"ID_?(\d+)") 


# 1. Criar a coluna de ordem sequencial dos trials por ID no df_C
df_C['ordem_trial'] = df_C.groupby('ID').cumcount() + 1

# 2. Agrupar os cortes por (ID, Trial) para capturar múltiplos cortes no mesmo trial
cortes_grouped = cortes.groupby(['ID', 'Trial']).agg(
    t_inicial_list=('t_inicial', list),
    t_final_list=('t_final', list)
).reset_index()

# 3. Renomear a coluna 'Trial' do cortes_grouped para 'ordem_trial' para facilitar o merge
cortes_grouped.rename(columns={'Trial': 'ordem_trial'}, inplace=True)

# 4. Fazer merge left com df_C usando ID e ordem_trial
df_C = df_C.merge(
    cortes_grouped,
    how='left',
    on=['ID', 'ordem_trial']
)

# 5. Criar a coluna Problema: 1 se houve corte (ou seja, se t_inicial_list não for nulo)
df_C['Problema'] = df_C['t_inicial_list'].notna().astype(int)

# 6. (Opcional) Renomear as colunas de listas para t_inicial e t_final
df_C.rename(columns={
    't_inicial_list': 't_inicial',
    't_final_list': 't_final'
}, inplace=True)

# Trocando os missing values de t_inicial e t_final por listas vazias

df_C['t_inicial'] = df_C['t_inicial'].apply(lambda x: x if isinstance(x,list) else [])
df_C['t_final'] = df_C['t_final'].apply(lambda x: x if isinstance(x,list) else [])

# Corrigindo os tempos para depois cortar certo

id_anterior=''
for idx, row in df_C.iterrows():
    print(f"ID: {row['ID']}, Trial: {row['ordem_trial']}")
    #  resetando as variáveis auxiliares para  proximo ID
    if row['ID'] != id_anterior:
        acumulo = 0
        d2_anterior =0
        print('reset')

    if (row['ordem_trial'] == 1) & (row['Problema'] != 1):
        df_C.at[idx,"d1_s"]= 0
        df_C.at[idx,"d2_s"]= float(df_C.iloc[idx]['d1_s']) + float(row['tamanho_original_trial'])
        d2_anterior = float(df_C.iloc[idx]['d1_s']) + float(row['tamanho_original_trial'])

    elif (row['ordem_trial'] == 1) & (row['Problema'] == 1):
        for t_ini, t_fim in zip(row['t_inicial'], row['t_final']):
            #print(f"  Corte: {t_ini} - {t_fim}")
            d_corte = t_fim - t_ini #(segundos)
            acumulo += d_corte
            #print('tamanho corte:',d_corte)
            #print('acumulo:',acumulo)
        df_C.at[idx,"d1_s"]= 0
        #print(float(df_C.iloc[idx]['d1_s']) + float(row['tamanho_original_trial']) - acumulo)
        df_C.at[idx,"d2_s"]= float(df_C.iloc[idx]['d1_s']) + float(row['tamanho_original_trial']) - acumulo
        d2_anterior =  float(df_C.iloc[idx]['d1_s']) + float(row['tamanho_original_trial']) - acumulo
        acumulo =0
    else:
        df_C.at[idx,"d1_s"]= d2_anterior
        if row['Problema'] == 1:
            for t_ini, t_fim in zip(row['t_inicial'], row['t_final']):
                #print(f"  Corte: {t_ini} - {t_fim}")
                d_corte = t_fim - t_ini #(segundos)
                acumulo += d_corte
                #print('tamanho corte:',d_corte)
                #print('acumulo:',acumulo)
            df_C.at[idx,"d2_s"]= float(df_C.iloc[idx]['d1_s']) + float(row['tamanho_original_trial']) - acumulo 
            d2_anterior = float(df_C.iloc[idx]['d1_s']) +  float(row['tamanho_original_trial']) - acumulo
            acumulo = 0
        else:
            df_C.at[idx,"d2_s"] = float(df_C.iloc[idx]['d1_s']) + float(row['tamanho_original_trial']) 
            d2_anterior = float(df_C.iloc[idx]['d1_s']) +  float(row['tamanho_original_trial']) 
    id_anterior = row['ID']
    
#%% Cortando os dados 
FS = 1000  # Hz 

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
            print(f'd1_s: {int(round(t0 * fs))}, d2_s: {int(round(t1 * fs))}, n ={n}\n i0: {i0}, i1:{i1}')
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
if fase == 1:
    PROTO_RULES = {
        'A': {'df': df_A, 'start_col': 'd1_s', 'end_col': 'd2_s'},        # d1 -> d2
        'B': {'df': df_B, 'start_col': 'd1_s', 'end_col': 'd2_s'},        # d1 -> d2
        'C': {'df': df_C, 'start_col': 'd1_s', 'end_col': 'd2_s'},        # d1 -> d2 (df_C não tem d3_s)
    }
else:
    PROTO_RULES = {
        'A': {'df': df_A, 'start_col': 'd2_s', 'end_col': 'd3_s'},        # d2 -> d3
        'B': {'df': df_B, 'start_col': 'd2_s', 'end_col': 'd3_s'},        # d2 -> d3
        'C': {'df': df_C, 'start_col': 'd1_s', 'end_col': 'd2_s'},        # d1 -> d2 (df_C não tem d3_s, só muda a fase)
    }

# ---------------- regex ----------------
PADRAO_ID_ARQUIVO = re.compile(r"ID_?(\d+)") 
PADRAO_PROTOCOLO_PASTA = re.compile(r".*(prot[A-Za-z]_[A-Z]{2,2}|prot[C,c,c])_mat$") 

# ---------------- varrer as pastas/arquivos ----------------
lista_geral = [l for l in os.listdir(r'D:\dados_pro_diego\arquivos_filtrados_mat')
               if l.startswith('filtrado geral_p')]

tres_canais = False #variável que controla se queremos todos os canais ou apenas CZ, C3 e C4
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
        
        if tres_canais:
            for i in range(0,data['chanlocs'].shape[-1]):
                if str(data['chanlocs'][0][i][0][0]) == 'CZ':
                    idx_CZ = i
                elif str(data['chanlocs'][0][i][0][0]) == 'C3':
                    idx_C3 = i
                elif str(data['chanlocs'][0][i][0][0]) == 'C4':
                    idx_C4 = i

            # canais (ajuste os índices se necessário)
            eeg_dict = {'CZ': eeg[idx_CZ], 'C3': eeg[idx_C3], 'C4': eeg[idx_C4]}
        else: 
            nome_canais=[]
            for i in range(0,data['chanlocs'].shape[-1]):
                nome_canais.append(str(data['chanlocs'][0][i][0][0]).strip())

            eeg_dict = {canal: eeg[i].copy() for i, canal in enumerate(nome_canais)}
            
        # corta e preenche as linhas correspondentes no DF correto
        cut_and_fill(df_target, ind, protocolo, eeg_dict, FS, start_col, end_col)

from scipy import signal #para subamostragem

pastas_baseline = {
    'Baseline OA': r'D:\Dados Bruna\TCC\Refazendo IC\filtrado - geral\arq_mat_OA',
    'Baseline OF': r'D:\Dados Bruna\TCC\Refazendo IC\filtrado - geral\arq_mat_OF'
}

ind = []
eeg_signal = []
grupo = []
for protocolo, caminho_pasta in pastas_baseline.items():
    file_names = [f for f in os.listdir(caminho_pasta) if f.endswith('.mat')]
    
    for arquivo in file_names:
        m_id = f'ID_{arquivo[:2]}'
    
        # carrega EEG do .mat
        data = loadmat(os.path.join(caminho_pasta, arquivo))
        eeg  = data['eeg_data']

        # Nota: Os dados reprocessados pela IC já estão corretamente em 1000 Hz.
        # Não aplicar downsample (resample_poly down=2), pois reduziria incorretamente para 500 Hz.
        
        if tres_canais:
            # pegando EEG dos canais CZ C3 e C4
            for i in range(0,data['chanlocs'].shape[-1]):
                if str(data['chanlocs'][0][i][0][0]) == 'CZ':
                    idx_CZ = i
                elif str(data['chanlocs'][0][i][0][0]) == 'C3':
                    idx_C3 = i
                elif str(data['chanlocs'][0][i][0][0]) == 'C4':
                    idx_C4 = i
            eeg_dict = {'CZ': eeg[idx_CZ], 'C3': eeg[idx_C3], 'C4': eeg[idx_C4]}
        else: 
            nome_canais=[]
            for i in range(0,data['chanlocs'].shape[-1]):
                nome_canais.append(str(data['chanlocs'][0][i][0][0]).strip())

            eeg_dict = {canal: eeg[i].copy() for i, canal in enumerate(nome_canais)}
        
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
if fase == 2:
    #Adicionando as colunas que Y pode assumir alem do desempenho
    df_A['Acuracia'] = df_protA['Acuracia']
    df_A['Especificidade'] = df_protA['Especificidade']
    df_A['Similaridade'] = df_protA['Similaridade']
    df_B['Acuracia'] = df_protB['Acuracia']
    df_B['Especificidade'] = df_protB['Especificidade']
    df_B['Similaridade'] = df_protB['Similaridade']
    df_C['Acuracia'] =[c for c in df_protC[df_protC['Fase']== 'Fase Execucao']["Acuracia"]]
    df_C['Similaridade'] = [c for c in df_protC[df_protC['Fase']== 'Fase Execucao']["Similaridade"]]
    df_C['Especificidade'] = [c for c in 1 - df_protC[df_protC['Fase']== 'Fase Execucao']["Taxa de Falsos Positivos"]]

# Removendo as que deram problemas no corte 
erro_A = df_A[df_A['_trecho_info']!='ok']['ID'].unique()

df_A_final = df_A[~df_A['ID'].isin(erro_A)] #Pego todos as linhas que não tem problema "~" serve para eu pegar ao contrário dos que estão dentro dos erros

erro_B = df_B[df_B['_trecho_info']!='ok']['ID'].unique()

df_B_final = df_B[~df_B['ID'].isin(erro_B)]

erro_C = df_C[df_C['_trecho_info']!='ok']['ID'].unique()

df_C_final = df_C[~df_C['ID'].isin(erro_C)]


#Calculo da psd dos trechos e já normalizando pela baseline

'''
Cada protocolo tem sua especificidade.
A CV -> normaliza com a psd de olhos abertos
A SV -> normalizar com a psd de olhos fechados
B CF -> normaliza com a psd de olhos fechados
B SF -> normalizar com a psd de olhos fechados
C -> normalizar com a psd de olhos abertos

'''

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
            vals = trecho_dict.get(ch, None)
            if vals is None:
                out[ch] = None
                continue

            if isinstance(vals, list):
                segs = [np.asarray(s).ravel() for s in vals]
            else:
                segs = [np.asarray(vals).ravel()]

            f_final = None
            pxx_sum = None
            weight_sum = 0

            for sig in segs:
                if sig.size < 4:
                    continue

                nseg = min(nperseg, sig.size)
                novl = min(noverlap, max(0, nseg // 2))

                f, Pxx = welch(
                    sig,
                    fs=fs,
                    window=win if nseg == nperseg else get_window(window, nseg),
                    nperseg=nseg,
                    noverlap=novl,
                    nfft=nperseg,
                    detrend=detrend,
                    scaling=scaling,
                    return_onesided=True
                )
                
                if f_final is None:
                    f_final = f
                    pxx_sum = np.zeros_like(Pxx)

                peso = sig.size
                if f_final.shape == f.shape:
                    pxx_sum += Pxx * peso
                    weight_sum += peso

            if weight_sum > 0:
                out[ch] = (f_final, pxx_sum / weight_sum)
            else:
                out[ch] = None
        return out

    # aplica linha a linha
    df[coluna_saida] = df[coluna_trecho].apply(_psd_de_uma_linha)
    return df

def add_bandpowers_per_channel(
    df: pd.DataFrame,
    bands: dict | None = None,
    channels: tuple[str, ...] = ("CZ", "C3", "C4"),
    psd_col: str = "psd_trecho",
    prefix: str = "psd",
    normalize_by_bandwidth: bool = False
) -> pd.DataFrame:
    """
    Cria colunas com potência total e potência por banda PARA CADA CANAL,
    a partir de `psd_col`.

    Espera por linha:
        df[psd_col] = {'CZ': (f, Pxx), 'C3': (f, Pxx), 'C4': (f, Pxx), ...}

    Cria colunas no padrão:
        Potência total:
            {prefix}_total_{canal}
            ex.: psd_total_CZ, psd_total_C3, psd_total_C4

        Potência por banda:
            {prefix}_{banda}_{canal}
            ex.: psd_delta_CZ, psd_theta_C3, ...

    As colunas de potência total são inseridas ANTES das colunas de potência por banda.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de entrada.
    bands : dict | None
        Ex.: {"delta": (0.5,4), "theta": (4,8), "alfa": (8,13), "beta": (13,30), "gamma": (30,60)}
    channels : tuple[str, ...]
        Canais esperados no dicionário de PSD.
    psd_col : str
        Nome da coluna que contém os dicionários com PSD por canal.
    prefix : str
        Prefixo das colunas criadas.
    normalize_by_bandwidth : bool
        Se True, divide a potência da banda pela largura da banda (média por Hz).
        Para a potência total, divide pela largura total do espectro disponível.

    Returns
    -------
    pd.DataFrame
        DataFrame com as novas colunas adicionadas e reordenadas.
    """
    df = df.copy()

    if bands is None:
        bands = {
            "delta": (0.5, 4),
            "theta": (4, 8),
            "alfa":  (8, 13),
            "beta":  (13, 30),
            "gamma": (30, 60),
        }

    # ---------------- helpers ----------------
    def _integrate_psd(sig_psd_item, f_lo=None, f_hi=None):
        """
        Calcula potência integrando a PSD.
        Se f_lo/f_hi forem None, integra toda a faixa disponível.
        """
        if sig_psd_item is None:
            return np.nan

        f, Pxx = sig_psd_item
        if f is None or Pxx is None:
            return np.nan

        f = np.asarray(f).ravel()
        Pxx = np.asarray(Pxx).ravel()

        valid = np.isfinite(f) & np.isfinite(Pxx)
        if valid.sum() < 2:
            return np.nan

        f = f[valid]
        Pxx = Pxx[valid]

        if f_lo is None and f_hi is None:
            mask = np.ones_like(f, dtype=bool)
        else:
            mask = np.ones_like(f, dtype=bool)
            if f_lo is not None:
                mask &= (f >= f_lo)
            if f_hi is not None:
                mask &= (f <= f_hi)

        if mask.sum() < 2:
            return np.nan

        f_sel = f[mask]
        Pxx_sel = Pxx[mask]

        power = np.trapz(Pxx_sel, f_sel)

        if normalize_by_bandwidth:
            bw = f_sel.max() - f_sel.min()
            if bw > 0:
                power = power / bw

        return float(power)

    # ---------------- criar colunas ----------------
    total_cols = []
    band_cols = []

    # colunas de potência total primeiro
    for ch in channels:
        colname = f"{prefix}_total_{ch}"
        total_cols.append(colname)
        if colname not in df.columns:
            df[colname] = np.nan

    # depois colunas por banda
    for banda in bands.keys():
        for ch in channels:
            colname = f"{prefix}_{banda}_{ch}"
            band_cols.append(colname)
            if colname not in df.columns:
                df[colname] = np.nan

    # ---------------- preencher linha a linha ----------------
    for idx, psd_dict in df[psd_col].items():
        if not isinstance(psd_dict, dict):
            continue

        # potência total por canal
        for ch in channels:
            total_col = f"{prefix}_total_{ch}"
            val_total = _integrate_psd(psd_dict.get(ch, None), f_lo=None, f_hi=None)
            df.at[idx, total_col] = val_total

        # potência por banda por canal
        for banda, (f_lo, f_hi) in bands.items():
            for ch in channels:
                band_col = f"{prefix}_{banda}_{ch}"
                val_band = _integrate_psd(psd_dict.get(ch, None), f_lo=f_lo, f_hi=f_hi)
                df.at[idx, band_col] = val_band

    # ---------------- reordenar colunas ----------------
    # mantém a ordem original das demais colunas, mas coloca as novas
    # em bloco: total primeiro, bandas depois
    created_cols = total_cols + band_cols
    other_cols = [c for c in df.columns if c not in created_cols]
    df = df[other_cols + total_cols + band_cols]

    return df

'''
nperseg = int(2**(np.ceil(np.log2(2*FS))))# 2048 (potencia de dois mais proxima de FS*2)
noverlap= int(nperseg//2) # 1024 (nperseg//2)
'''
nperseg = int(2**(np.ceil(np.log2(4*FS))))# (potencia de dois mais proxima de FS*4)
porcentagem_overlap = 8
noverlap = int(nperseg*(porcentagem_overlap/100))

# 1) Bandas clássicas de EEG (usadas nos Topoplots)
bandas_classicas = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alfa":  (8, 13),
    "beta":  (13, 30),
    "gamma": (30, 60),
}

# 2) Bandinhas consecutivas de 2 em 2 Hz de 0 a 50 Hz (sugeridas pelo Jean para a CDA)
bandas_jean = {}
for f_ini in range(0, 50, 2):
    f_fim = f_ini + 2
    bandas_jean[f"{f_ini}_{f_fim}Hz"] = (f_ini, f_fim)

# 3) Dicionário unificado com todas as bandas
bandas_totais = {**bandas_classicas, **bandas_jean}

channels_list= ['FP1', 'FP2', 'FZ', 'F3', 'F4', 'F7', 'F8', 'CZ', 'C3', 'C4', 'T7', 'T8', 'P7', 
                'P8', 'PZ', 'P3', 'P4', 'O1', 'O2', 'FCZ', 'FC1', 'FC2', 'FC3', 'OZ', 'C2', 'CP1', 
                'CP3', 'CP4', 'C1', 'FC4', 'CPZ', 'CP2']

# Fazendo o calculo da psd em cima dos protocolos 
df_A_final = add_psd_column(df_A_final, fs=1000, window="hann", channels=channels_list, nperseg = nperseg, noverlap=noverlap)
df_B_final =  add_psd_column(df_B_final, fs=1000, window="hann",channels=channels_list, nperseg = nperseg, noverlap=noverlap)  
df_C_final = add_psd_column(df_C_final, fs=1000, window="hann", channels=channels_list,nperseg = nperseg, noverlap=noverlap)
df_baseline = add_psd_column(df_baseline, fs=1000, window="hann", channels=channels_list,nperseg = nperseg, noverlap=noverlap)

# Fazendo o cálculo da potência total e por bandas para cada canal (tanto clássicas quanto de 2 em 2 Hz)
df_A_final = add_bandpowers_per_channel(df_A_final, channels=channels_list, bands=bandas_totais)
df_B_final = add_bandpowers_per_channel(df_B_final, channels=channels_list, bands=bandas_totais)
df_C_final = add_bandpowers_per_channel(df_C_final, channels=channels_list, bands=bandas_totais)
df_baseline = add_bandpowers_per_channel(df_baseline, channels=channels_list, bands=bandas_totais)

#%% Normalizando pela baseline
if fase == 1:
    relacoes_A = {
        'CV': 'Baseline OF', # Na estimulação -> OF, na execução -> OA
        'SV': 'Baseline OF'
    }
    relacoes_B = {
    'CF': 'Baseline OA', # Na estimulação -> OF, na execução -> OF
    'SF': 'Baseline OA' # Na estimulação -> OA, na execução -> OF
    }
else:
    relacoes_A = {
        'CV': 'Baseline OA', # Na estimulação -> OF, na execução -> OA
        'SV': 'Baseline OF'
    }
    relacoes_B = {
        'CF': 'Baseline OF', # Na estimulação -> OA, na execução -> OF
        'SF': 'Baseline OF' # Na estimulação -> OA, na execução -> OF
    }

relacao_C = 'Baseline OA'


def normalizar_bandas(df_a_normalizar, df_baseline, relacoes):
    """
    Normaliza colunas PSD de df_a_normalizar pelas respectivas colunas PSD
    de df_baseline, criando novas colunas no formato:

        psd_norm_{banda}_{canal}

    Exemplos:
        psd_total_C3  -> psd_norm_total_C3
        psd_delta_C3  -> psd_norm_delta_C3
        psd_beta_CZ   -> psd_norm_beta_CZ
    """
    df_a_normalizar = df_a_normalizar.copy()

    # Colunas PSD que serão normalizadas
    cols_psd = [
        c for c in df_a_normalizar.columns
        if c.startswith('psd_')
        and c != 'psd_trecho'
        and not c.startswith('psd_norm_')
    ]

    # Criar colunas normalizadas com NaN inicialmente
    for col in cols_psd:
        norm_col = f'psd_norm_{col[4:]}'
        if norm_col not in df_a_normalizar.columns:
            df_a_normalizar[norm_col] = np.nan

    # Loop linha a linha
    for idx, row in df_a_normalizar.iterrows():
        ind = row['ID']

        # Define qual baseline usar
        if 'grupo' in df_a_normalizar.columns:
            grupo_tarefa = row['grupo']
            grupo_baseline = relacoes[grupo_tarefa]
        else:
            grupo_baseline = relacoes  # string, ex: "Baseline OF"

        # Máscara para encontrar a linha da baseline correspondente
        if 'grupo' in df_baseline.columns:
            mask = (df_baseline['ID'] == ind) & (df_baseline['grupo'] == grupo_baseline)
        else:
            mask = (df_baseline['ID'] == ind)

        baseline_rows = df_baseline.loc[mask]

        if baseline_rows.empty:
            # Sem baseline correspondente: deixa NaN
            continue

        baseline_row = baseline_rows.iloc[0]

        # Normalizar cada coluna PSD
        for col in cols_psd:
            norm_col = f'psd_norm_{col[4:]}'

            psd_atual = row[col]
            psd_base = baseline_row[col] if col in baseline_row.index else np.nan

            if pd.isna(psd_atual) or pd.isna(psd_base) or psd_base == 0:
                valor_normalizado = np.nan
            else:
                valor_normalizado = psd_atual / psd_base

            df_a_normalizar.at[idx, norm_col] = valor_normalizado

    return df_a_normalizar
df_A_final = normalizar_bandas(df_a_normalizar= df_A_final, 
                               df_baseline= df_baseline, relacoes= relacoes_A)
df_B_final = normalizar_bandas(df_a_normalizar= df_B_final, 
                               df_baseline= df_baseline, relacoes= relacoes_B)

df_C_final = normalizar_bandas(df_a_normalizar= df_C_final, 
                               df_baseline= df_baseline, relacoes= relacao_C)     

# Normalização do trecho completo
def normalizar_trecho(df_a_normalizar,df_baseline,relacoes,tamanho_trecho = 110):
    """
    Normaliza a PSD de um trecho de EEG em relação à PSD da baseline correspondente,
    criando novas colunas com os valores normalizados para cada canal.

    A função percorre cada linha de `df_a_normalizar`, identifica o participante (ID)
    e, se existir, o grupo experimental, encontra a linha correspondente em
    `df_baseline` e normaliza a PSD do trecho atual pela PSD da baseline do mesmo
    participante e canal. A normalização é feita ponto a ponto, considerando apenas
    os primeiros `tamanho_trecho` pontos do vetor de PSD. O resultado é armazenado
    em novas colunas nomeadas como `psd_<canal>_norm`.

    Parâmetros
    ----------
    df_a_normalizar : pandas.DataFrame
        DataFrame contendo os trechos cuja PSD já foi calculada e armazenada na
        coluna `'psd_trecho'`. Deve conter:
            - 'ID' (identificador do participante)
            - opcionalmente 'grupo' (grupo experimental, ex: 'CF' ou 'SF')
            - 'Trecho_eeg' (usada para extrair os nomes dos canais)
            - 'psd_trecho', em que cada célula deve conter um dicionário no formato
            `{canal: (freq, psd)}`, por exemplo:
            `{'CZ': (freq_array, psd_array), 'C3': (...), 'C4': (...)}`

    df_baseline : pandas.DataFrame
        DataFrame contendo a PSD da baseline correspondente de cada participante.
        Deve conter:
            - 'ID'
            - opcionalmente 'grupo'
            - 'psd_trecho', no mesmo formato de `df_a_normalizar`

    relacoes : dict ou str
        Se `df_a_normalizar` possuir a coluna `'grupo'`, deve ser um dicionário que
        relaciona cada grupo experimental à condição de baseline correspondente em
        `df_baseline`.
        Exemplo: `{'CF': 'Baseline OF', 'SF': 'Baseline OF'}`

        Se `df_a_normalizar` não possuir a coluna `'grupo'`, `relacoes` deve ser uma
        string com o nome fixo da condição de baseline a ser usada.

    tamanho_trecho : int, default=100
        Número de pontos iniciais do vetor de PSD a serem considerados na
        normalização. A função utiliza apenas os índices `0:tamanho_trecho` tanto da
        PSD atual quanto da PSD da baseline.

    Retorna
    -------
    pandas.DataFrame
        O mesmo DataFrame `df_a_normalizar`, acrescido de novas colunas contendo os
        vetores de PSD normalizados para cada canal, nomeadas como
        `psd_<canal>_norm`.

    Notas
    -----
    - A normalização é feita ponto a ponto:
    `psd_normalizado = psd_atual_trecho / psd_baseline_trecho`
    - Cada nova célula das colunas `psd_<canal>_norm` contém um vetor numpy com os
    valores normalizados da PSD.
    - Se o participante não possuir baseline correspondente, a função lançará erro
    ao tentar acessar `.iloc[0]`.
    - Se houver valores zero na PSD da baseline, podem surgir `inf` ou `nan` na
    normalização.
    - A função usa a coluna `'Trecho_eeg'` apenas para obter a lista de canais, mas
    a normalização em si é feita com base na coluna `'psd_trecho'`.

    Exemplo de uso
    --------------
    >>> relacoes = {'CF': 'Baseline OF', 'SF': 'Baseline OF'}
    >>> df_norm = normalizar_trecho(df_A_final, df_baseline, relacoes, tamanho_trecho=100)
    >>> df_norm[['psd_CZ_norm', 'psd_C3_norm', 'psd_C4_norm']].head()
"""
    
    channels = list(df_a_normalizar['Trecho_eeg'].iloc[0].keys()) #vetor de canais
    cols_norm = [f'psd_{c}_norm' for c in channels]
    df_a_normalizar[cols_norm] = None
    
    for idx, row in df_a_normalizar.iterrows():
        ind=row['ID']
        # Se o df não tiver coluna 'grupo', usa uma baseline fixa (passada em relacoes como string)
        if 'grupo' in df_a_normalizar.columns:
            grupo = relacoes[row['grupo']]
        else:
            grupo = relacoes  # aqui relacoes vira tipo: "Baseline OF" (string)
        
        for ch in channels:
            # Mask da linha da baseline
            if 'grupo' in df_baseline.columns:
                mask = (df_baseline['ID'] == ind) & (df_baseline['grupo'] == grupo)
            else:
                mask = (df_baseline['ID'] == ind)

            baseline_data = df_baseline[mask]['psd_trecho'].iloc[0].get(ch)
            row_data = row['psd_trecho'].get(ch)
            # 2. Verificamos se o canal existe e não é None
            if (baseline_data is not None) and (row_data is not None):
                psd_baseline_trecho = baseline_data[1] # Acessa o array de PSD
                
                # Cortando o trecho apenas em frequências de interesse
                psd_baseline_trecho = psd_baseline_trecho[0:tamanho_trecho]
                
                # Valor da psd da linha atual daquele canal
                psd_atual_trecho = row['psd_trecho'][ch][1][0:tamanho_trecho]    
                
                # Cálculo da normalização
                psd_normalizado = psd_atual_trecho / psd_baseline_trecho
                
            else:
                # Caso o canal seja None (como o CP2 no seu print)
                print(f'Psd do Canal {ch} não encontrado ou é None, provavelmente removido.')
                psd_normalizado = None
            
            df_a_normalizar.at[idx,f'psd_{ch}_norm'] = psd_normalizado
        
    return df_a_normalizar
df_A_final = normalizar_trecho(df_a_normalizar=df_A_final,
                            df_baseline=df_baseline,
                            relacoes= relacoes_A)
df_B_final = normalizar_trecho(df_a_normalizar=df_B_final,
                            df_baseline=df_baseline,
                            relacoes= relacoes_B)
df_C_final = normalizar_trecho(df_a_normalizar=df_C_final,
                            df_baseline=df_baseline,
                            relacoes= relacao_C)

#%% Exportando as colunas da potencia total e em bandas

def limpar_e_salvar_topomap_de_colunas_filtradas(
    df_filtrado: pd.DataFrame,
    nome_base_arquivo: str,
    group_series: pd.Series | None = None,
    protocolo: str | None = None,
    prefix: str = "psd_norm_",
    pasta_saida: str = ".",
    eps: float = 1e-12,
    salvar_csv: bool = True,
    index: bool = False
):
    """
    Recebe diretamente um dataframe já filtrado contendo apenas colunas PSD,
    por exemplo:
        df_A_final[[c for c in df_A_final.columns if c.startswith('psd_norm') and c != 'psd_trecho']]

    A função:
    1) Detecta canais com qualquer NaN em qualquer coluna/trial
    2) Remove TODAS as colunas desses canais
    3) Salva um CSV limpo (formato largo)
    4) Monta e salva um CSV longo para dashboard:
           grupo | banda | canal | psd_mean | psd_db_mean
       e, se `protocolo` for informado:
           protocolo | grupo | banda | canal | psd_mean | psd_db_mean
    5) Salva um CSV relatório com canais e colunas removidos

    Parameters
    ----------
    df_filtrado : pd.DataFrame
        DataFrame contendo apenas colunas do tipo psd_norm_{banda}_{canal}.
    nome_base_arquivo : str
        Nome base dos arquivos. Ex.: "A", "protA", "df_A".
    group_series : pd.Series | None
        Série com os grupos correspondentes às linhas de df_filtrado.
        Ex.: df_A_final['grupo']
        Se None, cria grupo único = "all".
    protocolo : str | None
        Nome do protocolo para incluir no CSV longo.
    prefix : str
        Prefixo esperado nas colunas.
    pasta_saida : str
        Pasta de saída.
    eps : float
        Proteção para log10.
    salvar_csv : bool
        Se True, salva os CSVs.
    index : bool
        Se True, salva índice nos CSVs.

    Returns
    -------
    dict
        {
            'df_limpo_largo': ...,
            'df_topomap_longo': ...,
            'df_relatorio': ...,
            'canais_removidos': ...,
            'colunas_removidas': ...,
            'n_colunas_removidas': ...
        }
    """
    df_filtrado = df_filtrado.copy()

    # Garantir grupos
    if group_series is None:
        grupos = pd.Series(["all"] * len(df_filtrado), index=df_filtrado.index, name="grupo")
    else:
        grupos = pd.Series(group_series, index=df_filtrado.index, name="grupo")

    # 1) mapear canal -> colunas
    mapa_canal_cols = {}
    for col in df_filtrado.columns:
        canal = col.rsplit("_", 1)[-1]
        mapa_canal_cols.setdefault(canal, []).append(col)

    # 2) detectar canais ruins
    canais_removidos = []
    colunas_removidas = []

    for canal, cols_do_canal in mapa_canal_cols.items():
        if df_filtrado[cols_do_canal].isna().any().any():
            canais_removidos.append(canal)
            colunas_removidas.extend(cols_do_canal)

    canais_removidos = sorted(canais_removidos)
    colunas_removidas = sorted(colunas_removidas)
    n_colunas_removidas = len(colunas_removidas)

    # 3) remover colunas ruins
    df_limpo_largo = df_filtrado.drop(columns=colunas_removidas, errors="ignore").copy()

    # 4) relatório
    relatorio = []
    for canal in canais_removidos:
        cols_canal = [c for c in colunas_removidas if c.endswith(f"_{canal}")]
        relatorio.append({
            "canal_removido": canal,
            "n_colunas_removidas_do_canal": len(cols_canal),
            "colunas_removidas": "; ".join(cols_canal)
        })
    df_relatorio = pd.DataFrame(relatorio)

    # 5) montar dataframe longo
    pattern = re.compile(rf"^{re.escape(prefix)}(.+?)_(.+)$")
    df_aux = pd.concat([grupos, df_limpo_largo], axis=1)

    registros = []
    for col in df_limpo_largo.columns:
        m = pattern.match(col)
        if not m:
            continue

        banda = m.group(1)
        canal = m.group(2)

        for grupo, subdf in df_aux.groupby("grupo", dropna=False):
            vals = pd.to_numeric(subdf[col], errors="coerce").dropna()

            if len(vals) == 0:
                psd_mean = np.nan
                psd_db_mean = np.nan
            else:
                psd_mean = float(vals.mean())
                psd_std = float(vals.std())
                psd_db_mean = float((10 * np.log10(vals.clip(lower=eps))).mean())
                psd_db_std = float((10 * np.log10(vals.clip(lower=eps))).std())

            registro = {
                "grupo": grupo,
                "banda": banda,
                "canal": canal,
                "psd_mean": psd_mean,
                "psd_std": psd_std,
                "psd_db_mean": psd_db_mean,
                "psd_db_std": psd_db_std
            }

            if protocolo is not None:
                registro["protocolo"] = protocolo

            registros.append(registro)

    df_topomap_longo = pd.DataFrame(registros)

    # ordenar colunas
    if not df_topomap_longo.empty:
        ordem_bandas = {
            "total": 0,
            "delta": 1,
            "theta": 2,
            "alfa": 3,
            "alpha": 3,
            "beta": 4,
            "gamma": 5
        }
        df_topomap_longo["_ordem_banda"] = df_topomap_longo["banda"].map(
            lambda x: ordem_bandas.get(str(x).lower(), 999)
        )

        cols_ordem = ["grupo", "banda", "canal", "psd_mean", "psd_std", "psd_db_mean","psd_db_std"]
        if protocolo is not None:
            cols_ordem = ["protocolo"] + cols_ordem

        df_topomap_longo = (
            df_topomap_longo
            .sort_values(([ "protocolo"] if protocolo is not None else []) + ["grupo", "_ordem_banda", "canal"])
            .drop(columns="_ordem_banda")
            .reset_index(drop=True)
        )[cols_ordem]

    # 6) salvar
    if salvar_csv:
        Path(pasta_saida).mkdir(parents=True, exist_ok=True)

        #caminho_largo = Path(pasta_saida) / f"{nome_base_arquivo}_limpo_largo.csv"
        caminho_longo = Path(pasta_saida) / f"{nome_base_arquivo}.csv"
        #caminho_rel = Path(pasta_saida) / f"{nome_base_arquivo}_relatorio_remocao.csv"

        #df_limpo_largo.to_csv(caminho_largo, index=index)
        df_topomap_longo.to_csv(caminho_longo, index=index)
        #df_relatorio.to_csv(caminho_rel, index=index)

    return {
        #"df_limpo_largo": df_limpo_largo,
        "df_topomap_longo": df_topomap_longo,
        #"df_relatorio": df_relatorio,
        "canais_removidos": canais_removidos,
        "colunas_removidas": colunas_removidas,
        "n_colunas_removidas": n_colunas_removidas
    }

if fase ==1: tipo ='estimulacao'
else: tipo ='execucao'

# Bandas clássicas esperadas pelo Dashboard para os mapas topográficos
bandas_topo_nomes = ['total', 'delta', 'theta', 'alfa', 'beta', 'gamma']

# Protocolo A
col_number_norm = [
    c for c in df_A_final.columns 
    if any(c.startswith(f'psd_norm_{b}_') for b in bandas_topo_nomes)
    and not c.endswith('_norm') and not c.startswith('psd_trecho')
]
col_number = [
    c for c in df_A_final.columns 
    if any(c.startswith(f'psd_{b}_') for b in bandas_topo_nomes)
    and not c.startswith('psd_norm') and not c.endswith('_norm') and not c.startswith('psd_trecho')
]

res_A = limpar_e_salvar_topomap_de_colunas_filtradas(
    df_filtrado=df_A_final[col_number_norm],
    nome_base_arquivo=f"topoplot_protA_{tipo}_norm",
    group_series=df_A_final["grupo"],
    protocolo=None,
    pasta_saida="saida_dashboard"
)

print('Protocolo A normalizado')
print(res_A["canais_removidos"])
print(res_A["n_colunas_removidas"])
print('---'*100)

# Protocolo B
col_number_norm = [
    c for c in df_B_final.columns 
    if any(c.startswith(f'psd_norm_{b}_') for b in bandas_topo_nomes)
    and not c.endswith('_norm') and not c.startswith('psd_trecho')
]
col_number = [
    c for c in df_B_final.columns 
    if any(c.startswith(f'psd_{b}_') for b in bandas_topo_nomes)
    and not c.startswith('psd_norm') and not c.endswith('_norm') and not c.startswith('psd_trecho')
]

res_B = limpar_e_salvar_topomap_de_colunas_filtradas(
    df_filtrado=df_B_final[col_number_norm],
    nome_base_arquivo=f"topoplot_protB_{tipo}_norm",
    group_series=df_B_final["grupo"],
    protocolo=None,
    pasta_saida="saida_dashboard"
)

print('Protocolo B normalizado')
print(res_B["canais_removidos"])
print(res_B["n_colunas_removidas"])
print('---'*100)

# Protocolo C
col_number_norm = [
    c for c in df_C_final.columns 
    if any(c.startswith(f'psd_norm_{b}_') for b in bandas_topo_nomes)
    and not c.endswith('_norm') and not c.startswith('psd_trecho')
]
col_number = [
    c for c in df_C_final.columns 
    if any(c.startswith(f'psd_{b}_') for b in bandas_topo_nomes)
    and not c.startswith('psd_norm') and not c.endswith('_norm') and not c.startswith('psd_trecho')
]

res_C = limpar_e_salvar_topomap_de_colunas_filtradas(
    df_filtrado=df_C_final[col_number_norm],
    nome_base_arquivo=f"topoplot_protC_{tipo}_norm",
    group_series=None,
    protocolo=None,
    pasta_saida="saida_dashboard"
)

print('Protocolo C normalizado')
print(res_C["canais_removidos"])
print(res_C["n_colunas_removidas"])
print('---'*100)

# Protocolo C sem normalizar
res_C = limpar_e_salvar_topomap_de_colunas_filtradas(
    df_filtrado=df_C_final[col_number],
    nome_base_arquivo=f"topoplot_protC_{tipo}",
    group_series=None,
    protocolo=None,
    prefix='psd_',
    pasta_saida="saida_dashboard"
)

print('Protocolo C sem normalizar')
print(res_C["canais_removidos"])
print(res_C["n_colunas_removidas"])
print('---'*100)

# Para os topoplots, vamos comparar o protocolo C com a baseline, uma vez que ele não tem divisão de grupos
baseline_C = df_baseline[(df_baseline['grupo']== 'Baseline OA') & 
                         (df_baseline['ID'].isin(df_C_final['ID'].unique()))]

col_number = [
    c for c in baseline_C.columns 
    if any(c.startswith(f'psd_{b}_') for b in bandas_topo_nomes)
    and not c.startswith('psd_norm') 
    and not c.endswith('_norm') 
    and not c.startswith('psd_trecho')
    and all(canal not in c for canal in res_C["canais_removidos"])
]
res_baseline = limpar_e_salvar_topomap_de_colunas_filtradas(
    df_filtrado=baseline_C[col_number],
    nome_base_arquivo="topoplot_baseline_olhosAbertos_protC",
    group_series=None,
    protocolo=None,
    prefix='psd_',
    pasta_saida="saida_dashboard"
)
print('Baseline Protocolo C para comparação')
print(res_baseline["canais_removidos"])
print(res_baseline["n_colunas_removidas"])
print('---'*100)
# %% Exportar os dados do trecho da psd normalizado em colunas
def exportar_psd_normalizado_csv(df,
                                 colunas_psd,
                                 incluir_meta=True,
                                 caminho_csv='psd_normalizado_expandido.csv'):
    """
    Expande colunas contendo vetores/arrays de PSD em colunas numéricas
    e salva em CSV. Trata células None preenchendo com NaN.
    """

    df = df.copy()
    partes = []

    if incluir_meta:
        cols_meta = [c for c in df.columns if c not in colunas_psd]
        partes.append(df[cols_meta].reset_index(drop=True))

    for col in colunas_psd:
        serie = df[col]

        # acha o primeiro array válido para definir o tamanho esperado
        primeiro_valido = next(
            (x for x in serie if isinstance(x, (list, np.ndarray))),
            None
        )

        if primeiro_valido is None:
            print(f'Coluna {col} sem nenhum array válido. Pulando.')
            continue

        tamanho = len(primeiro_valido)

        # substitui None por vetor de NaN
        dados_corrigidos = [
            np.asarray(x) if isinstance(x, (list, np.ndarray))
            else np.full(tamanho, np.nan)
            for x in serie
        ]

        expandido = pd.DataFrame(dados_corrigidos, index=df.index)

        nome_base = col.replace('psd_', '').replace('_norm', '')
        expandido.columns = [f'{nome_base}_{i}' for i in range(expandido.shape[1])]

        partes.append(expandido.reset_index(drop=True))

    df_final = pd.concat(partes, axis=1)
    df_final.to_csv(caminho_csv, index=False)
    return df_final

if fase == 1:
    fase_protocolo = 'estimulacao' 
else: 
    fase_protocolo = 'execucao'
    
colunas_todos_canais = [c for c in df_A_final.columns if c.endswith('_norm')] 
df_csv = exportar_psd_normalizado_csv(
    df_A_final,
    colunas_psd=colunas_todos_canais,
    incluir_meta=False,
    caminho_csv=f'protA_X_psd_norm_completo_{fase_protocolo}.csv'   
)

colunas_todos_canais = [c for c in df_B_final.columns if c.endswith('_norm')] 
df_csv = exportar_psd_normalizado_csv(
    df_B_final,
    colunas_psd=colunas_todos_canais,
    incluir_meta=False,
    caminho_csv=f'protB_X_psd_norm_completo_{fase_protocolo}.csv'   
)

colunas_todos_canais = [c for c in df_C_final.columns if c.endswith('_norm')] 
df_csv = exportar_psd_normalizado_csv(
    df_C_final,
    colunas_psd=colunas_todos_canais,
    incluir_meta=False,
    caminho_csv=f'protC_X_psd_norm_completo_{fase_protocolo}.csv'   
)

#%% Exportando X composto pela potencia de cada banda de 0 a 50 tomada de 2 em 2

def exportar_histograma_2hz_csv(df, protocolo, fase, incluir_meta=True):
    """Filtra as colunas no formato 'psd_norm_{f_ini}_{f_fim}Hz_{canal}' de 0 a

    50 Hz e as exporta em formato de matriz de características (Features) para o
    CDA, mantendo as linhas (trials) idênticas.
    """
    df = df.copy()

    # 1. Determina o tipo de fase do protocolo (estimulacao ou execucao)
    tipo = "estimulacao" if fase == 1 else "execucao"

    # 2. Define o nome do arquivo seguindo o padrão solicitado
    nome_arquivo = f"prot{protocolo}_X_psd_norm_2em2_{tipo}.csv"

    # 3. Rastreia as colunas do histograma via Regex para o canal/frequência
    # Captura tanto formatos com zero à esquerda (ex: 02_04Hz) quanto normais (ex: 2_4Hz)
    padrao_regex = re.compile(r"^psd_norm_(\d+)_(\d+)Hz_(.+)$")

    colunas_info = []
    for col in df.columns:
        match = padrao_regex.match(col)
        if match:
            f_ini = int(match.group(1))
            f_fim = int(match.group(2))
            canal = match.group(3)

            # Filtra apenas o limite proposto pelo Jean (de 0 até 50 Hz)
            if f_ini < 50:
                colunas_info.append((f_ini, canal, col))

    # 4. Ordena as colunas de forma lógica (frequência inicial crescente e canal)
    # Garante que fique: 0_2Hz_FP1, 0_2Hz_FP2... 2_4Hz_FP1...
    colunas_info.sort(key=lambda x: (x[0], x[1]))
    colunas_histograma = [item[2] for item in colunas_info]

    if not colunas_histograma:
        print(
            f"⚠️ Nenhuma coluna de 2 Hz encontrada para o protocolo {protocolo} na fase de {tipo}."
        )
        return None

    # 5. Estrutura o DataFrame final de exportação
    colunas_finais = []
    if incluir_meta:
        # Mantém metadados importantes para você identificar os trials se precisar
        cols_meta_existentes = [
            c for c in ["ID", "grupo", "Complexidade"] if c in df.columns
        ]
        colunas_finais.extend(cols_meta_existentes)

    colunas_finais.extend(colunas_histograma)
    df_exportar = df[colunas_finais]

    # 6. Salva na pasta atual de execução (mesma pasta do código)
    caminho_salvamento = os.path.join(os.getcwd(), nome_arquivo)
    df_exportar.to_csv(caminho_salvamento, index=False)

    print(
        f"✅ Matriz exportada: {nome_arquivo} | Shape: {df_exportar.shape} ({df_exportar.shape[0]} trials)"
    )
    return df_exportar

# --- EXPORTAÇÃO DOS HISTOGRAMAS DE 2 Hz PARA O CDA ---
print("\n--- Iniciando exportação dos histogramas de 2 Hz (Plano Jean) ---")

# Executa a exportação para o Protocolo A
df_export_A = exportar_histograma_2hz_csv(
    df=df_A_final,
    protocolo="A",
    fase=fase,
    incluir_meta=False,  # Altere para False se quiser estritamente APENAS os sinais elétricos
)

# Executa a exportação para o Protocolo B
df_export_B = exportar_histograma_2hz_csv(
    df=df_B_final, protocolo="B", fase=fase, incluir_meta=False
)

# Executa a exportação para o Protocolo C
df_export_C = exportar_histograma_2hz_csv(
    df=df_C_final, protocolo="C", fase=fase, incluir_meta=False
)
#%% Salvando o Y e os filtros de cada protocolo
if fase == 2:
    df_A_final[['ID','Complexidade','grupo','Overlap','Desempenho', 'Acuracia','Similaridade', 'Especificidade', 'Proporção espacial x', 'Proporção espacial y' ]].to_csv('analise_df_A_final.csv')
    df_B_final[['ID','Complexidade','grupo','Desempenho', 'Acuracia','Similaridade', 'Especificidade', 'Proporção espacial x', 'Proporção espacial y' ]].to_csv('analise_df_B_final.csv')
    df_C_final[['ID','Complexidade','Desempenho', 'Acuracia','Similaridade', 'Especificidade', 'Proporção espacial x', 'Proporção espacial y' ]].to_csv('analise_df_C_final.csv')
