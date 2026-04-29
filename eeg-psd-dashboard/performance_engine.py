import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import t as t_dist
from scipy.stats import shapiro, kstest
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import MultiComparison

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
    formula = f"{target} ~ {rhs}"
    
    try:
        model = ols(formula, data=data).fit()
        anova_res = sm.stats.anova_lm(model, typ=3)
        return anova_res, None
    except Exception as e:
        return None, str(e)

def plot_interactive_dot_sig(df, x_col, y_col, order=None, alpha=0.05, title=None, show_sig_bars=True):
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
            mc = MultiComparison(data[y_col], data[x_col])
            res = mc.tukeyhsd(alpha=alpha)
            # Create a dataframe from tukey results
            res_df = pd.DataFrame(data=res._results_table.data[1:], columns=res._results_table.data[0])
            
            sig_pairs = []
            for _, row in res_df.iterrows():
                if row['reject']:
                    p_val = row.get('p-adj', 0.05) # if it doesn't exist, we assume it's under alpha due to reject=True
                    stars = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else '*')
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
