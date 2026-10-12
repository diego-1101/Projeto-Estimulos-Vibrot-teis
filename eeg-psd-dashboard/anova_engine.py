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

    # Helper function to map complexities and groups to English
    comp_map = {
        '1': 'Easy', '4': 'Easy', '2': 'Medium', '6': 'Medium', '3': 'Hard', '8': 'Hard',
        'Fácil': 'Easy', 'Médio': 'Medium', 'Difícil': 'Hard'
    }
    grp_map = {'CV': 'V+', 'SV': 'V-', 'CF': 'F+', 'SF': 'F-'}

    def format_anova_label(lbl):
        s = str(lbl).replace('.0', '').strip()
        parts = s.split('_')
        return "_".join([grp_map.get(comp_map.get(p, p), comp_map.get(p, p)) for p in parts])

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
    
    # Sort groups: Group first (V+/F+, then V-/F-), then Complexity (Easy -> Medium -> Hard)
    def sort_key(g):
        gs = str(g)
        # 1. Group Rank (V+/F+ first = 1, V-/F- second = 2, others = 3)
        group_rank = 3
        if any(grp in gs for grp in ['V+', 'F+', 'CV', 'CF']):
            group_rank = 1
        elif any(grp in gs for grp in ['V-', 'F-', 'SV', 'SF']):
            group_rank = 2

        # 2. Complexity Rank (Easy -> Medium -> Hard)
        comp_rank = 99
        if 'Easy' in gs:
            comp_rank = 1
        elif 'Medium' in gs:
            comp_rank = 2
        elif 'Hard' in gs:
            comp_rank = 3

        return (group_rank, comp_rank, gs)

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

    # Standardized color map for complexity: Easy (#2ecc71), Medium (#f39c12), Hard (#e74c3c)
    has_complexity = any(any(c in str(g) for c in ['Easy', 'Medium', 'Hard']) for g in group_names)

    def get_marker_color(cat_str):
        for p in str(cat_str).split('_'):
            if p in ['V+', 'F+', 'CV', 'CF']:
                return '#5dade2'  # Light blue for V+ / F+
            elif p in ['V-', 'F-', 'SV', 'SF']:
                return '#0d47a1'  # Dark blue for V- / F-
        return '#0d6efd'  # Standard blue if no group factor (e.g. Protocol C)

    def get_marker_symbol(cat_str):
        for p in str(cat_str).split('_'):
            if p in ['V+', 'F+', 'CV', 'CF']:
                return 'x'
            elif p in ['V-', 'F-', 'SV', 'SF']:
                return 'circle'
        return 'circle'

    group_colors = [get_marker_color(g) for g in group_names]
    group_symbols = [get_marker_symbol(g) for g in group_names]
    
    has_v_f_plus = any(s == 'x' for s in group_symbols)
    has_v_f_minus = any(any(grp in str(g).split('_') for grp in ['V-', 'F-', 'SV', 'SF']) for g in group_names)
    has_f_plus = any(any(grp in str(g).split('_') for grp in ['F+', 'CF']) for g in group_names)
    has_f_minus = any(any(grp in str(g).split('_') for grp in ['F-', 'SF']) for g in group_names)
    has_v_plus = any(any(grp in str(g).split('_') for grp in ['V+', 'CV']) for g in group_names)
    has_v_minus = any(any(grp in str(g).split('_') for grp in ['V-', 'SV']) for g in group_names)

    # 4.1 Dummy legend traces matching Performance Evaluation
    if has_v_f_plus or has_v_f_minus:
        if has_v_f_plus:
            plus_label = "Group F+ (X)" if (has_f_plus and not has_v_plus) else ("Group V+/F+ (X)" if has_f_plus else "Group V+ (X)")
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(color='#5dade2', size=11, symbol='x', line=dict(width=2, color='#5dade2')),
                name=plus_label,
                showlegend=True
            ))
        if has_v_f_minus:
            minus_label = "Group F- (Circle)" if (has_f_minus and not has_v_minus) else ("Group V-/F- (Circle)" if has_f_minus else "Group V- (Circle)")
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(color='#0d47a1', size=10, symbol='circle'),
                name=minus_label,
                showlegend=True
            ))
    else:
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(color='#0d6efd', size=10, symbol='circle'),
            name=f'Mean ± CI{int((1-alpha)*100)}%',
            showlegend=True
        ))

    # 4.2 Add Means and CI bounds
    for i, g in enumerate(group_names):
        m = means[i]
        c = group_colors[i]
        sym = group_symbols[i]
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
            marker=dict(
                color=c,
                size=11 if sym == 'x' else 10,
                symbol=sym,
                line=dict(width=2, color=c) if sym == 'x' else dict(width=1, color='white')
            ),
            showlegend=False,
            hovertemplate=f"Condition: {g}<br>Mean: {m:.3f}<br>CI: ±{ci_err:.3f}<extra></extra>"
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
            # Red (#dc3545) for inter-group (V+/F+ vs V-/F-)
            # Green (#198754) for intra-group (V+ vs V+, V- vs V-)
            # Dark grey (#333333) for comparisons without group factor
            def get_group_part(cat_str):
                for p in str(cat_str).split('_'):
                    if p in ['V+', 'V-', 'F+', 'F-', 'CV', 'SV', 'CF', 'SF']:
                        return p
                return None
                
            gp1 = get_group_part(g1)
            gp2 = get_group_part(g2)
            
            bracket_color = "#333333"
            if gp1 and gp2:
                is_cv1 = gp1 in ['V+', 'F+', 'CV', 'CF']
                is_cv2 = gp2 in ['V+', 'F+', 'CV', 'CF']
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
            name="Between-Group Sig. (V+/F+ <-> V-/F-)",
            showlegend=True
        ))
    if '#198754' in used_bracket_colors:
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='lines',
            line=dict(color='#198754', width=2),
            name="Intra-Group Sig. (V+ <-> V+ / V- <-> V-)",
            showlegend=True
        ))
    if '#333333' in used_bracket_colors:
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='lines',
            line=dict(color='#333333', width=2),
            name="Significant Difference",
            showlegend=True
        ))

    num_levels = len(levels)
    top_margin = max(90, int(60 + num_levels * 18))
    fig_height = 400 + top_margin + 50

    fig.update_xaxes(
        type='category',
        categoryorder='array',
        categoryarray=[str(g) for g in group_names],
        tickangle=0 if len(group_names) <= 4 else 45
    )

    VAR_NAME_EN = {
        'Desempenho': 'Performance',
        'Acuracia': 'Accuracy',
        'Acurácia': 'Accuracy',
        'Similaridade': 'Similarity',
        'Especificidade': 'Specificity',
        'Proporção espacial x': 'Spatial Proportion X',
        'Proporção espacial y': 'Spatial Proportion Y',
        'Proporcao espacial x': 'Spatial Proportion X',
        'Proporcao espacial y': 'Spatial Proportion Y',
        'grupo': 'Group',
        'complexidade': 'Complexity',
        'grupo_complexidade': 'Group & Complexity'
    }
    val_disp = VAR_NAME_EN.get(value_col, value_col)
    grp_disp_name = VAR_NAME_EN.get(group_col, group_col)

    fig.update_layout(
        title=dict(
            text=f"ANOVA: {val_disp} by {grp_disp_name} (F={f_stat:.2f}, p={p_val:.4f})",
            y=0.98,
            x=0.5,
            xanchor='center',
            yanchor='top',
            yref='container'
        ),
        xaxis_title=grp_disp_name,
        yaxis_title=val_disp,
        plot_bgcolor='white',
        template='plotly_white',
        margin=dict(t=top_margin, b=80, l=50, r=40),
        height=fig_height,
        autosize=True,
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
