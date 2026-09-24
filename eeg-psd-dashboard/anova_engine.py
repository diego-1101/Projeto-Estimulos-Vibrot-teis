import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def compute_anova_and_plot(df, value_col, group_col, alpha=0.05):
    """
    Computes One-Way ANOVA and Tukey HSD post-hoc test, then generates a Plotly figure
    with means, 95% CI, and standardized significance brackets.
    """
    # 1. Clean data: drop NaNs in relevant columns
    data = df[[value_col, group_col]].dropna().copy()
    
    if data.empty or data[group_col].nunique() < 2:
        return go.Figure().update_layout(title="Not enough data or groups for ANOVA"), {}

    # Helper function to map complexities to English and format group labels
    comp_map = {
        '1': 'Easy', '4': 'Easy', '2': 'Medium', '6': 'Medium', '3': 'Hard', '8': 'Hard',
        'Fácil': 'Easy', 'Médio': 'Medium', 'Difícil': 'Hard'
    }
    def format_anova_label(lbl):
        s = str(lbl).replace('.0', '').strip()
        parts = s.split('_')
        return "_".join([comp_map.get(p, p) for p in parts])

    # Standardize group_col values
    data['group_disp'] = data[group_col].map(format_anova_label)
    actual_group_col = 'group_disp'

    # 2. ANOVA
    groups = [group[value_col].values for name, group in data.groupby(actual_group_col)]
    f_stat, p_val = stats.f_oneway(*groups)
    
    stats_dict = {
        'F': f_stat,
        'p_value': p_val,
        'significant': p_val < alpha
    }

    # 3. Post-Hoc Tukey HSD
    tukey = pairwise_tukeyhsd(endog=data[value_col], groups=data[actual_group_col], alpha=alpha)
    tukey_df = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])
    significant_pairs = tukey_df[tukey_df['reject'] == True]
    
    stats_dict['tukey_summary'] = tukey_df.to_dict('records')

    # 4. Plotly Visualization
    fig = go.Figure()
    
    # Sort groups with semantic awareness (Easy -> Medium -> Hard, etc.)
    def sort_key(g):
        gs = str(g)
        rank = 99
        if 'Easy' in gs: rank = 1
        elif 'Medium' in gs: rank = 2
        elif 'Hard' in gs: rank = 3
        return (rank, gs)

    group_names = sorted(data[actual_group_col].unique(), key=sort_key)
    
    # Calculate Mean and 95% CI for each group
    means = []
    ci_lower = []
    ci_upper = []
    
    for g in group_names:
        g_data = data[data[actual_group_col] == g][value_col]
        n = len(g_data)
        mean = np.mean(g_data)
        std_err = stats.sem(g_data) if n > 1 else 0
        ci = std_err * stats.t.ppf((1 + 1 - alpha) / 2., n - 1) if n > 1 else 0
        
        means.append(mean)
        ci_lower.append(mean - ci)
        ci_upper.append(mean + ci)

    # Determine marker colors based on group presence:
    # CV/CF -> Light Blue (#5dade2), SV/SF -> Dark Blue (#0d47a1), fallback -> #0d6efd
    def get_marker_color(cat_str):
        parts = str(cat_str).split('_')
        for p in parts:
            if p in ['CV', 'CF']:
                return '#5dade2'
            elif p in ['SV', 'SF']:
                return '#0d47a1'
        return '#0d6efd'

    group_colors = [get_marker_color(g) for g in group_names]
    
    has_cv_cf = any(c == '#5dade2' for c in group_colors)
    has_sv_sf = any(c == '#0d47a1' for c in group_colors)
    has_cf = any('CF' in str(g) for g in group_names)
    has_sf = any('SF' in str(g) for g in group_names)

    # 4.1 Dummy legend traces for groups
    if has_cv_cf:
        cv_label = "Grupo CF" if (has_cf and not any('CV' in str(g) for g in group_names)) else ("Grupo CV/CF" if has_cf else "Grupo CV")
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(color='#5dade2', size=10, symbol='circle'),
            name=cv_label,
            showlegend=True
        ))
    if has_sv_sf:
        sv_label = "Grupo SF" if (has_sf and not any('SV' in str(g) for g in group_names)) else ("Grupo SV/SF" if has_sf else "Grupo SV")
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(color='#0d47a1', size=10, symbol='circle'),
            name=sv_label,
            showlegend=True
        ))
    if not has_cv_cf and not has_sv_sf:
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(color='#0d6efd', size=10, symbol='circle'),
            name=f'Média ± IC{int((1-alpha)*100)}%',
            showlegend=True
        ))

    # 4.2 Add Means and CI bounds (without individual data points dispersion)
    for i, g in enumerate(group_names):
        m = means[i]
        c = group_colors[i]
        ci_err = ci_upper[i] - m
        fig.add_trace(go.Scatter(
            x=[str(g)],
            y=[m],
            error_y=dict(
                type='data',
                symmetric=True,
                array=[ci_err],
                visible=True,
                color=c,
                thickness=2,
                width=6
            ),
            mode='markers',
            marker=dict(color=c, size=10, symbol='circle'),
            showlegend=False,
            hovertemplate=f"Grupo: {g}<br>Média: %{{y:.3f}}<br>IC: ±{ci_err:.3f}<extra></extra>"
        ))
    
    # Add text annotations for Means
    for i, m in enumerate(means):
        ci_val = ci_upper[i] - m
        fig.add_annotation(
            x=str(group_names[i]),
            y=m + ci_val,
            text=f"M: {m:.3f}<br>CI: ±{ci_val:.3f}",
            showarrow=False,
            font=dict(size=9, color='black'),
            yshift=12
        )

    # 4.3 Add Significance Brackets (using paper coordinates with packed levels)
    levels = []
    used_bracket_colors = set()
    
    if not significant_pairs.empty:
        step_paper = 0.04
        cap_paper = 0.012
        x_pos = {str(g): i for i, g in enumerate(group_names)}
        
        for _, row in significant_pairs.iterrows():
            g1, g2 = str(row['group1']), str(row['group2'])
            if g1 not in x_pos or g2 not in x_pos:
                continue
            i1, i2 = x_pos[g1], x_pos[g2]
            if i1 > i2:
                i1, i2 = i2, i1
                
            # Level packing
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
            y0 = 1.02 + step_paper * my_level
            
            # Determine bracket color:
            # Red (#dc3545) for inter-group (CV/CF vs SV/SF)
            # Green (#198754) for intra-group (CV vs CV, SV vs SV)
            # Dark grey (#333333) for comparisons without group factor
            def get_group_part(cat_str):
                for p in str(cat_str).split('_'):
                    if p in ['CV', 'SV', 'CF', 'SF']:
                        return p
                return None
                
            gp1 = get_group_part(g1)
            gp2 = get_group_part(g2)
            
            bracket_color = "#333333"
            if gp1 and gp2:
                is_cv1 = gp1 in ['CV', 'CF']
                is_cv2 = gp2 in ['CV', 'CF']
                if is_cv1 != is_cv2:
                    bracket_color = '#dc3545'  # Red for inter-group
                else:
                    bracket_color = '#198754'  # Green for intra-group
            
            used_bracket_colors.add(bracket_color)
            
            # Stars
            p_adj = row['p-adj']
            stars = "***" if p_adj < 0.001 else ("**" if p_adj < 0.01 else ("*" if p_adj < 0.05 else "ns"))
            
            fig.add_shape(
                type="path",
                path=f"M {i1} {y0-cap_paper} L {i1} {y0} L {i2} {y0} L {i2} {y0-cap_paper}",
                line=dict(color=bracket_color, width=1.5),
                xref="x", yref="paper"
            )
            fig.add_annotation(
                x=(i1 + i2) / 2,
                y=y0 + 0.012,
                text=stars,
                showarrow=False,
                font=dict(size=14, color=bracket_color),
                xref="x", yref="paper"
            )

    # Legend for significance brackets matching Performance Evaluation
    if '#dc3545' in used_bracket_colors:
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='lines',
            line=dict(color='#dc3545', width=2),
            name="Sig. Entre Grupos (CV/CF ↔ SV/SF)",
            showlegend=True
        ))
    if '#198754' in used_bracket_colors:
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='lines',
            line=dict(color='#198754', width=2),
            name="Sig. Intra-Grupo (CV ↔ CV / SV ↔ SV)",
            showlegend=True
        ))
    if '#333333' in used_bracket_colors:
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='lines',
            line=dict(color='#333333', width=2),
            name="Diferença Significativa",
            showlegend=True
        ))

    num_levels = len(levels)
    top_margin = max(80, int(60 + num_levels * 18))
    fig_height = 420 + top_margin + 40

    fig.update_xaxes(
        type='category',
        categoryorder='array',
        categoryarray=[str(g) for g in group_names],
        tickangle=0 if len(group_names) <= 4 else 45
    )

    fig.update_layout(
        title=f"ANOVA: {value_col} by {group_col} (F={f_stat:.2f}, p={p_val:.4f})",
        yaxis_title=value_col,
        plot_bgcolor='white',
        template='plotly_white',
        margin=dict(t=top_margin, b=40, l=40, r=40),
        height=fig_height,
        xaxis_showgrid=True,
        yaxis_showgrid=True,
        xaxis_gridcolor='lightgray',
        yaxis_gridcolor='lightgray',
        xaxis_gridwidth=0.5,
        yaxis_gridwidth=0.5,
        xaxis_griddash='dash',
        yaxis_griddash='dash'
    )
    
    return fig, stats_dict
