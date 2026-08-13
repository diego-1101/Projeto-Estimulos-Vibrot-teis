import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import t as t_dist
from scipy.stats import shapiro, kstest
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import MultiComparison
from statsmodels.stats.libqsturng import psturng, qsturng
from itertools import combinations

def run_normality_test(data_series, alpha=0.05):
    """
    Runs Shapiro-Wilk and Kolmogorov-Smirnov normality tests.
    """
    s = data_series.dropna()
    if len(s) < 3:
        return None, "Dados insuficientes para teste de normalidade (n < 3)."
        
    w, p_sw = shapiro(s)
    d, p_ks = kstest(s, 'norm', args=(s.mean(), s.std()))
    
    return {
        'Shapiro-Wilk': {'stat': w, 'p_value': p_sw, 'is_normal': p_sw > alpha},
        'Kolmogorov-Smirnov': {'stat': d, 'p_value': p_ks, 'is_normal': p_ks > alpha}
    }, None

def run_anova_typ3(df, target, independents):
    """
    Runs Type 3 ANOVA using statsmodels with interaction between all independents.
    """
    if not independents:
        return None, "Nenhuma variável independente selecionada."
    
    data = df.dropna(subset=[target] + independents).copy()
    
    rhs = " * ".join([f"C({var})" for var in independents])
    formula = f"Q('{target}') ~ {rhs}"
    
    try:
        model = ols(formula, data=data).fit()
        anova_res = sm.stats.anova_lm(model, typ=3)
        return anova_res, None
    except Exception as e:
        return None, str(e)

def tukey_hsd_from_model(data, response_col, group_col, anova_table, alpha=0.05):
    """
    Post-hoc de Tukey usando o MSE residual do modelo fatorial.
    Compatível com statsmodels anova_lm tipo III.
    """
    if anova_table is None:
        return None
    
    # anova_table can be a DataFrame or a list of dicts (from Dash DataTable)
    if isinstance(anova_table, list):
        temp_df = pd.DataFrame(anova_table)
        # Procura linha de Resíduo de forma mais flexível
        res_mask = temp_df.iloc[:, 0].astype(str).str.contains('Residual|Resid|Resíduo', case=False, na=False)
        if not any(res_mask): 
            print("DEBUG TUKEY: Linha 'Residual' não encontrada na tabela.")
            return None
        res_row = temp_df[res_mask].iloc[0]
        
        def to_float(x):
            if pd.isna(x) or x == "": return 0.0
            try: return float(str(x).replace(',', '.'))
            except: return 0.0
            
        sum_sq = to_float(res_row.get('Soma dos Quadrados', res_row.get('sum_sq', 0)))
        df_resid = to_float(res_row.get('df', 0))
        ms_residual = sum_sq / df_resid if df_resid > 0 else 0
        print(f"DEBUG TUKEY (List): MSE={ms_residual:.6f}, DF={df_resid}")
    else:
        if 'Residual' not in anova_table.index:
            print("DEBUG TUKEY: 'Residual' não está no índice do DataFrame.")
            return None
        res_row = anova_table.loc['Residual']
        df_resid = res_row['df']
        if 'mean_sq' in anova_table.columns:
            ms_residual = res_row['mean_sq']
        else:
            ms_residual = res_row['sum_sq'] / df_resid if df_resid > 0 else 0
        print(f"DEBUG TUKEY (DF): MSE={ms_residual:.6f}, DF={df_resid}")

    if ms_residual <= 0 or df_resid <= 0:
        print("DEBUG TUKEY: MSE ou DF inválidos (<= 0). Verifique a tabela da ANOVA.")
        return None

    grupos = sorted(data[group_col].astype(str).unique())
    k = len(grupos)
    resultados = []
    
    for g1, g2 in combinations(grupos, 2):
        y1 = data[data[group_col].astype(str) == g1][response_col].dropna().values
        y2 = data[data[group_col].astype(str) == g2][response_col].dropna().values
        if len(y1) == 0 or len(y2) == 0: continue
        
        n1, n2 = len(y1), len(y2)
        mean_diff = np.mean(y1) - np.mean(y2)
        
        # Erro padrão para Tukey (Studentized Range)
        se = np.sqrt(ms_residual / 2.0 * (1.0/n1 + 1.0/n2))
        
        # Estatística q
        q_stat = abs(mean_diff) / se if se > 0 else 0
        
        # p-value (psturng espera q, k, df)
        p_val_raw = psturng(q_stat, k, df_resid)
        # Garantir que p_val seja um float escalar puro do Python
        p_val = float(np.atleast_1d(p_val_raw)[0])
        p_val = min(p_val, 1.0)
        
        is_sig = p_val < alpha
        stars = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else 'ns'))
        
        resultados.append({
            'group1': g1,
            'group2': g2,
            'p-adj': p_val,
            'stars': stars,
            'reject': bool(is_sig)
        })
            
    return pd.DataFrame(resultados)

def plot_interactive_dot_sig(df, x_col, y_col, order=None, alpha=0.05, title=None, show_sig_bars=True, anova_table=None):
    """
    Creates an interactive dot plot with means, 95% CI, and significance brackets.
    """
    data = df[[x_col, y_col]].dropna().copy()
    if data.empty:
        return go.Figure(), [], "Sem dados válidos para plotar."
        
    if order is None:
        order = sorted(data[x_col].unique().tolist())

    g = data.groupby(x_col)[y_col].agg(['mean', 'std', 'count']).reindex(order).reset_index()

    def ci95(s, n):
        if n and n > 1 and pd.notnull(s):
            return t_dist.ppf(1 - alpha/2, df=n-1) * (s / np.sqrt(n))
        return np.nan
        
    g['ci95'] = [ci95(s, n) for s, n in zip(g['std'], g['count'])]

    fig = go.Figure()

    # Base Box/Strip plot for jitter points
    for lvl in order:
        lvl_data = data[data[x_col] == lvl]
        fig.add_trace(go.Box(
            y=lvl_data[y_col],
            name=str(lvl),
            boxpoints='all',
            jitter=0.5,
            pointpos=0,
            fillcolor='rgba(255,255,255,0)',
            line=dict(color='rgba(0,0,0,0)'),
            marker=dict(color='gray', opacity=0.5, size=6),
            showlegend=False,
            hoverinfo='y'
        ))

    # Means and CI95
    fig.add_trace(go.Scatter(
        x=g[x_col].astype(str),
        y=g['mean'],
        error_y=dict(type='data', array=g['ci95'], visible=True, color='blue', thickness=2, width=6),
        mode='markers',
        marker=dict(color='blue', size=10, symbol='circle'),
        name=f'Média ± IC{(1-alpha)*100:.0f}%',
        hovertemplate='Grupo: %{x}<br>Média: %{y:.3f}<br>IC: ±%{error_y.array:.3f}<extra></extra>'
    ))

    sig_results = []
    if show_sig_bars and len(order) >= 2:
        try:
            if anova_table is not None:
                # Use model-based Tukey
                res_df = tukey_hsd_from_model(data, y_col, x_col, anova_table, alpha=alpha)
            else:
                # Fallback to standard One-Way Tukey
                mc = MultiComparison(data[y_col], data[x_col])
                res = mc.tukeyhsd(alpha=alpha)
                res_df = pd.DataFrame(data=res._results_table.data[1:], columns=res._results_table.data[0])
            
            if res_df is not None and not res_df.empty:
                sig_pairs = []
                for _, row in res_df.iterrows():
                    # check for 'reject' or 'p-adj' < alpha
                    is_rejected = row.get('reject', False) or row.get('p-adj', 1.0) < alpha
                    if is_rejected:
                        p_val = row.get('p-adj', alpha)
                        # Generate stars if not present
                        stars = row.get('stars', '***' if p_val < 0.001 else ('**' if p_val < 0.01 else '*'))
                        sig_pairs.append((str(row['group1']), str(row['group2']), stars, p_val))
                        sig_results.append({'group1': row['group1'], 'group2': row['group2'], 'p-adj': p_val, 'stars': stars})

            # Draw brackets
            if sig_pairs:
                y_max = data[y_col].max()
                y_range = y_max - data[y_col].min()
                if y_range == 0: y_range = 1.0
                
                step = y_range * 0.08
                cap = y_range * 0.02
                
                levels = []
                x_pos = {str(lvl): i for i, lvl in enumerate(order)}
                
                for g1, g2, stars, _ in sig_pairs:
                    if g1 not in x_pos or g2 not in x_pos: continue
                    i1, i2 = x_pos[g1], x_pos[g2]
                    if i1 > i2: i1, i2 = i2, i1
                    
                    my_level = 0
                    for lvl_idx, intervals in enumerate(levels):
                        overlap = False
                        for (start, end) in intervals:
                            if not (i2 < start or i1 > end):
                                overlap = True
                                break
                        if not overlap:
                            my_level = lvl_idx
                            break
                    else:
                        my_level = len(levels)
                        levels.append([])
                    
                    levels[my_level].append((i1, i2))
                    y0 = y_max + step * (my_level + 1)
                    
                    fig.add_shape(type="path",
                        path=f"M {i1} {y0-cap} L {i1} {y0} L {i2} {y0} L {i2} {y0-cap}",
                        line=dict(color="black", width=1.5),
                        xref="x", yref="y"
                    )
                    
                    fig.add_annotation(
                        x=(i1+i2)/2,
                        y=y0 + (step * 0.2),
                        text=stars,
                        showarrow=False,
                        font=dict(size=14, color="black"),
                        xref="x", yref="y"
                    )
        except Exception as e:
            print(f"Tukey Error: {e}")

    fig.update_layout(
        title=title if title else f"Dotplot + IC{(1-alpha)*100:.0f}%",
        xaxis_title=x_col,
        yaxis_title=y_col,
        template='plotly_white',
        margin=dict(t=60, b=40, l=40, r=40)
    )
    
    return fig, sig_results, None

def plot_interactive_interaction(df, x_col, line_col, y_col, facet_col=None, ylim=None, title=None, alpha=0.05):
    """
    Creates an interactive Plotly interaction plot.
    """
    group_cols = [x_col, line_col]
    if facet_col:
        group_cols.append(facet_col)
    
    data = df[group_cols + [y_col]].dropna().copy()
    if data.empty:
        return go.Figure(), "Sem dados válidos para plotar."
        
    g = data.groupby(group_cols)[y_col].agg(['mean', 'std', 'count']).reset_index()

    def ci95(s, n):
        if n and n > 1 and pd.notnull(s):
            return t_dist.ppf(1 - alpha/2, df=n-1) * (s / np.sqrt(n))
        return 0.0
    g['ci95'] = [ci95(s, n) for s, n in zip(g['std'], g['count'])]

    # Ensure everything is casted nicely
    g[x_col] = g[x_col].astype(str)
    g[line_col] = g[line_col].astype(str)
    if facet_col:
        g[facet_col] = g[facet_col].astype(str)

    fig = px.line(
        g,
        x=x_col,
        y='mean',
        color=line_col,
        facet_col=facet_col,
        error_y='ci95',
        markers=True,
        title=title if title else f"Interação: {y_col} ~ {x_col} × {line_col}",
        labels={'mean': y_col}
    )

    fig.update_traces(marker=dict(size=8), error_y=dict(thickness=1.5, width=4))
    
    if ylim and isinstance(ylim, (list, tuple)) and len(ylim) == 2:
        fig.update_yaxes(range=ylim)
        
    fig.update_layout(template='plotly_white', margin=dict(t=60, b=40, l=40, r=40))
    
    return fig, None

def plot_interactive_hybrid(df, x_col, line_col, y_col, facet_col=None, ylim=None, title=None, alpha=0.05):
    """
    Creates an interactive hybrid Plotly figure combining Boxplot + Stripplot (Dispersão)
    with Interaction Lines (Means + IC95%) in foreground, perfectly aligned over boxplot centers.
    """
    group_cols = [x_col, line_col]
    if facet_col and facet_col != 'None':
        group_cols.append(facet_col)
    else:
        facet_col = None
        
    data = df[group_cols + [y_col]].dropna().copy()
    if data.empty:
        return go.Figure(), "Sem dados válidos para plotar."

    # Standardize string representations
    data[x_col] = data[x_col].astype(str)
    data[line_col] = data[line_col].astype(str)
    if facet_col:
        data[facet_col] = data[facet_col].astype(str)
        if 'overlap' in facet_col.lower():
            def fmt_ov(val):
                try:
                    f = float(val)
                    return f"Overlap {int(round(f * 100))}%"
                except:
                    return f"Overlap {val}"
            data[facet_col] = data[facet_col].apply(fmt_ov)
        facet_vals = sorted(data[facet_col].unique().tolist())
    else:
        facet_vals = [None]

    line_vals = sorted(data[line_col].unique().tolist())
    x_vals = sorted(data[x_col].unique().tolist())
    num_x = len(x_vals)
    K = len(line_vals)
    
    # Calculate exact horizontal offsets for grouped boxplots & lines
    if K > 1:
        box_width = 0.28
        spacing = 0.36
        offsets = [(-spacing/2) + idx * (spacing / (K - 1)) for idx in range(K)]
    else:
        box_width = 0.4
        offsets = [0.0]
        
    PALETTE_GRUPO = {'CV': '#0d6efd', 'SV': '#dc3545', 'CF': '#0d6efd', 'SF': '#dc3545'}
    COLOR_MAP = {}
    for idx, l_val in enumerate(line_vals):
        COLOR_MAP[l_val] = PALETTE_GRUPO.get(l_val, px.colors.qualitative.Set1[idx % len(px.colors.qualitative.Set1)])
        
    from plotly.subplots import make_subplots
    
    if len(facet_vals) > 1:
        fig = make_subplots(
            rows=1, cols=len(facet_vals),
            subplot_titles=[str(fv) for fv in facet_vals],
            shared_yaxes=True,
            horizontal_spacing=0.04
        )
    else:
        fig = go.Figure()

    for f_idx, f_val in enumerate(facet_vals, start=1):
        if f_val is not None:
            sub_data = data[data[facet_col] == f_val].copy()
        else:
            sub_data = data.copy()

        # 1. Background Boxplots & Scatter Points (Aligned at numeric x + offset)
        for k_idx, l_val in enumerate(line_vals):
            l_data = sub_data[sub_data[line_col] == l_val].copy()
            if l_data.empty: continue
            
            color = COLOR_MAP[l_val]
            offset = offsets[k_idx]
            
            # Map categorical x to numeric position + group offset
            l_data['x_num'] = l_data[x_col].apply(lambda x: x_vals.index(x) + offset if x in x_vals else 0.0)
            
            box_trace = go.Box(
                x=l_data['x_num'],
                y=l_data[y_col],
                name=f"{line_col}: {l_val}",
                legendgroup=f"{l_val}",
                showlegend=False,
                boxpoints='all',
                jitter=0.35,
                pointpos=0,
                width=box_width,
                fillcolor=color,
                opacity=0.3,
                line=dict(color=color, width=1.5),
                marker=dict(color=color, size=4, opacity=0.5),
                hoverinfo='y+name'
            )
            if len(facet_vals) > 1:
                fig.add_trace(box_trace, row=1, col=f_idx)
            else:
                fig.add_trace(box_trace)

        # 2. Foreground Interaction Lines (Means + IC95%) centered EXACTLY over boxes
        grouped = sub_data.groupby([x_col, line_col])[y_col].agg(['mean', 'std', 'count']).reset_index()
        
        def calc_ci95(s, n):
            if n and n > 1 and pd.notnull(s):
                return t_dist.ppf(1 - alpha/2, df=n-1) * (s / np.sqrt(n))
            return 0.0
        grouped['ci95'] = [calc_ci95(s, n) for s, n in zip(grouped['std'], grouped['count'])]
        
        for k_idx, l_val in enumerate(line_vals):
            g_sub = grouped[grouped[line_col] == l_val].copy()
            if g_sub.empty: continue
            
            offset = offsets[k_idx]
            g_sub['x_order'] = g_sub[x_col].apply(lambda x: x_vals.index(x) if x in x_vals else 99)
            g_sub = g_sub.sort_values('x_order')
            g_sub['x_num'] = g_sub['x_order'] + offset
            
            color = COLOR_MAP[l_val]
            
            line_trace = go.Scatter(
                x=g_sub['x_num'],
                y=g_sub['mean'],
                error_y=dict(type='data', array=g_sub['ci95'], visible=True, color=color, thickness=2, width=6),
                mode='lines+markers',
                marker=dict(size=9, symbol='circle', color='white', line=dict(color=color, width=2.5)),
                line=dict(color=color, width=3),
                name=f"{line_col}: {l_val}",
                legendgroup=f"{l_val}",
                showlegend=(f_idx == 1),
                hovertemplate=f"Grupo: {l_val}<br>Complexidade: %{{text}}<br>Média: %{{y:.3f}}<br>IC95%: ±%{{error_y.array:.3f}}<extra></extra>",
                text=g_sub[x_col]
            )
            if len(facet_vals) > 1:
                fig.add_trace(line_trace, row=1, col=f_idx)
            else:
                fig.add_trace(line_trace)

    fig.update_layout(
        title=title if title else f"Gráfico Híbrido: {y_col} ~ {x_col} × {line_col}",
        template='plotly_white',
        margin=dict(t=60, b=40, l=50, r=40)
    )
    
    # Configure custom ticks on X axis to show categorical labels at 0, 1, 2...
    tick_vals = list(range(num_x))
    tick_text = [str(x) for x in x_vals]
    
    if len(facet_vals) == 1:
        fig.update_xaxes(title_text=x_col, tickvals=tick_vals, ticktext=tick_text)
        fig.update_yaxes(title_text=y_col)
    else:
        fig.update_yaxes(title_text=y_col, col=1)
        for c in range(1, len(facet_vals) + 1):
            fig.update_xaxes(title_text=x_col, tickvals=tick_vals, ticktext=tick_text, col=c)
            
    if ylim and isinstance(ylim, (list, tuple)) and len(ylim) == 2:
        fig.update_yaxes(range=ylim)

    return fig, None

def plot_interactive_significance_heatmap(data, response_col, group_col, anova_table=None, alpha=0.05):
    """
    Creates an interactive Plotly heatmap matrix of Tukey HSD post-hoc p-values
    using a single-color sequential scale (Blues) and crisp contrasting text.
    """
    res_df = tukey_hsd_from_model(data, response_col, group_col, anova_table, alpha=alpha)
    if res_df is None or res_df.empty:
        return go.Figure(), "Sem resultados de Post-Hoc disponíveis para montar o Heatmap."
        
    grupos = sorted(list(set(res_df['group1'].astype(str)).union(set(res_df['group2'].astype(str)))))
    k = len(grupos)
    
    p_matrix = np.ones((k, k))
    text_matrix = np.full((k, k), "", dtype=object)
    hover_text_matrix = np.full((k, k), "", dtype=object)
    
    g_to_idx = {g: i for i, g in enumerate(grupos)}
    
    only_stars = (k > 10)
    
    for _, row in res_df.iterrows():
        g1, g2 = str(row['group1']), str(row['group2'])
        p_val = float(row['p-adj'])
        stars = str(row.get('stars', 'ns'))
        if g1 in g_to_idx and g2 in g_to_idx:
            i, j = g_to_idx[g1], g_to_idx[g2]
            p_matrix[i, j] = p_val
            p_matrix[j, i] = p_val
            
            # Hover text: always full detail
            hover_label = f"p = {p_val:.4f} ({stars})" if p_val < 0.001 else f"p = {p_val:.3f} ({stars})"
            hover_text_matrix[i, j] = hover_label
            hover_text_matrix[j, i] = hover_label
            
            # Cell text: stars if large k, full label if small k
            if only_stars:
                cell_label = stars
            else:
                cell_label = f"p = {p_val:.4f}<br>({stars})" if p_val < 0.001 else f"p = {p_val:.3f}<br>({stars})"
            text_matrix[i, j] = cell_label
            text_matrix[j, i] = cell_label
            
    for i in range(k):
        text_matrix[i, i] = "—"
        hover_text_matrix[i, i] = "Diagonal"
        p_matrix[i, i] = 1.0
        
    for i in range(k):
        for j in range(k):
            if i != j and not text_matrix[i, j]:
                text_matrix[i, j] = "ns" if only_stars else "p = 1.000<br>(ns)"
                hover_text_matrix[i, j] = "p = 1.000 (ns)"

    # Scale: 0 for diagonal/ns, higher values for significant differences
    log_p = -np.log10(np.clip(p_matrix, 1e-4, 1.0))
    np.fill_diagonal(log_p, 0)
    
    # Custom single-color scale (Monochromatic Blues)
    mono_blues = [
        [0.0, '#f4f8ff'],      # ns / diagonal -> Ice white-blue
        [0.25, '#c6e0fe'],     # p ~ 0.05 (*) -> Soft light blue
        [0.6, '#3d8bfd'],      # p ~ 0.01 (**) -> Medium blue
        [1.0, '#0a58ca']       # p < 0.001 (***) -> Deep navy blue
    ]

    if k <= 6:
        font_size = 12
        tick_size = 12
    elif k <= 10:
        font_size = 10
        tick_size = 10
    elif k <= 14:
        font_size = 8
        tick_size = 9
    else:
        font_size = 7
        tick_size = 8

    fig = go.Figure(data=go.Heatmap(
        z=log_p,
        x=grupos,
        y=grupos,
        text=text_matrix,
        customdata=hover_text_matrix,
        texttemplate="%{text}",
        textfont={"size": font_size, "color": "#111827"},
        colorscale=mono_blues,
        colorbar=dict(title="-log10(p-adj)"),
        hovertemplate="Grupo 1: %{y}<br>Grupo 2: %{x}<br>%{customdata}<extra></extra>"
    ))
    
    fig.update_layout(
        title=f"Heatmap de Significância (Post-Hoc Tukey HSD: {group_col})",
        xaxis=dict(title="Grupo / Condição", tickfont=dict(size=tick_size)),
        yaxis=dict(title="Grupo / Condição", autorange="reversed", tickfont=dict(size=tick_size)),
        template="plotly_white",
        margin=dict(t=60, b=40, l=60, r=40)
    )
    
    return fig, None


