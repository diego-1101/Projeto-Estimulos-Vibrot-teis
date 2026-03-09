import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def compute_anova_and_plot(df, value_col, group_col, alpha=0.05):
    """
    Computes One-Way ANOVA and Tukey HSD post-hoc test, then generates a Plotly figure
    with data points, mean, 95% CI, and significance brackets.
    """
    # 1. Clean data: drop NaNs in relevant columns
    data = df[[value_col, group_col]].dropna()
    
    if data.empty or data[group_col].nunique() < 2:
        return go.Figure().update_layout(title="Not enough data or groups for ANOVA"), {}

    # 2. ANOVA
    groups = [group[value_col].values for name, group in data.groupby(group_col)]
    # Use standard ANOVA
    f_stat, p_val = stats.f_oneway(*groups)
    
    stats_dict = {
        'F': f_stat,
        'p_value': p_val,
        'significant': p_val < alpha
    }

    # 3. Post-Hoc Tukey HSD
    tukey = pairwise_tukeyhsd(endog=data[value_col], groups=data[group_col], alpha=alpha)
    tukey_df = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])
    significant_pairs = tukey_df[tukey_df['reject'] == True]
    
    stats_dict['tukey_summary'] = tukey_df.to_dict('records')

    # 4. Plotly Visualization
    fig = go.Figure()
    
    group_names = sorted(data[group_col].unique())
    
    # Calculate Mean and 95% CI for each group
    means = []
    ci_lower = []
    ci_upper = []
    
    for g in group_names:
        g_data = data[data[group_col] == g][value_col]
        n = len(g_data)
        mean = np.mean(g_data)
        std_err = stats.sem(g_data)
        ci = std_err * stats.t.ppf((1 + 1 - alpha) / 2., n - 1) if n > 1 else 0
        
        means.append(mean)
        ci_lower.append(mean - ci)
        ci_upper.append(mean + ci)
        
        # Add jittered scatter points
        jitter = np.random.uniform(-0.1, 0.1, size=n)
        fig.add_trace(go.Scatter(
            x=[group_names.index(g) + j for j in jitter],
            y=g_data,
            mode='markers',
            marker=dict(color='lightgray', size=5, opacity=0.7),
            showlegend=False,
            hoverinfo='y+name',
            name=str(g)
        ))

    # Add Means and CI bounds
    fig.add_trace(go.Scatter(
        x=list(range(len(group_names))),
        y=means,
        error_y=dict(
            type='data',
            symmetric=False,
            array=[ci_upper[i] - means[i] for i in range(len(means))],
            arrayminus=[means[i] - ci_lower[i] for i in range(len(means))],
            visible=True,
            color='blue',
            thickness=1.5,
            width=5
        ),
        mode='markers',
        marker=dict(color='blue', size=10, symbol='circle'),
        name=f'Mean \u00B1 CI{int((1-alpha)*100)}%'
    ))
    
    # Add text annotations for Means
    for i, m in enumerate(means):
        ci_val = ci_upper[i] - m
        fig.add_annotation(
            x=i,
            y=m + ci_val + (np.max(data[value_col]) * 0.05), # slightly above CI
            text=f"M: {m:.3f}<br>CI: \u00B1{ci_val:.3f}",
            showarrow=False,
            font=dict(size=9, color='black'),
            yshift=10
        )

    # Add Significance Brackets
    if not significant_pairs.empty:
        max_y = data[value_col].max()
        y_range = max_y - data[value_col].min()
        bracket_height = max_y + y_range * 0.05
        step = y_range * 0.05
        
        for idx, row in significant_pairs.iterrows():
            g1_idx = group_names.index(row['group1'])
            g2_idx = group_names.index(row['group2'])
            
            # Simple bracket
            fig.add_shape(
                type="path",
                path=f"M {g1_idx},{bracket_height - step/3} L {g1_idx},{bracket_height} L {g2_idx},{bracket_height} L {g2_idx},{bracket_height - step/3}",
                line=dict(color="black", width=1.5)
            )
            # Add star
            # Define star size based on p-adj
            p_adj = row['p-adj']
            stars = "*" if p_adj < 0.05 else ""
            stars = "**" if p_adj < 0.01 else stars
            stars = "***" if p_adj < 0.001 else stars
            
            fig.add_annotation(
                x=(g1_idx + g2_idx) / 2,
                y=bracket_height,
                text=stars,
                showarrow=False,
                yshift=5,
                font=dict(color="black", size=12)
            )
            bracket_height += step

    fig.update_layout(
        title=f"ANOVA: {value_col} by {group_col} (F={f_stat:.2f}, p={p_val:.4f})",
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(len(group_names))),
            ticktext=[str(g) for g in group_names],
            tickangle=45
        ),
        yaxis_title=value_col,
        plot_bgcolor='white',
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
