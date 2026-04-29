import sys

def patch_app():
    with open('app.py', 'r', encoding='utf-8') as f:
        content = f.read()
        
    # We need to replace the decorator of run_performance_analysis
    old_dec = """@app.callback(
    [Output('perf-normality-result-card', 'style'),
     Output('perf-normality-output', 'children'),
     Output('perf-anova-result-card', 'style'),
     Output('perf-anova-table-output', 'children'),
     Output('perf-anova-dotplot', 'figure'),
     Output('perf-anova-posthoc-output', 'children'),
     Output('perf-interaction-result-card', 'style'),
     Output('perf-interaction-plot', 'figure'),
     Output('perf-controls-error', 'children')],
    [Input('perf-run-btn', 'n_clicks')],"""

    new_dec = """@app.callback(
    [Output('perf-normality-result-card', 'style'),
     Output('perf-normality-output', 'children'),
     Output('perf-anova-result-card', 'style'),
     Output('perf-anova-table-output', 'children'),
     Output('perf-interaction-result-card', 'style'),
     Output('perf-interaction-plot', 'figure'),
     Output('perf-controls-error', 'children')],
    [Input('perf-run-btn', 'n_clicks')],"""

    content = content.replace(old_dec, new_dec)
    
    # We need to replace the ANOVA Post-Hoc section in run_performance_analysis
    old_anova = """                # Combine independents for post-hoc
                if len(anova_indeps) > 0:
                    group_col = "_".join(anova_indeps)
                    df[group_col] = df[anova_indeps].astype(str).agg('_'.join, axis=1)
                    
                    fig, sig_results, err = plot_interactive_dot_sig(df, group_col, anova_var, alpha=alpha, title=f"Post-Hoc: {group_col} ({anova_var})")
                    anova_fig = fig
                    
                    if err:
                        posthoc_out = html.Div(err, className="alert alert-warning")
                    elif not sig_results:
                        posthoc_out = html.Div("Nenhuma diferença significativa encontrada no post-hoc.", className="alert alert-info")
                    else:
                        sig_df = pd.DataFrame(sig_results)
                        posthoc_out = dash_table.DataTable(
                            data=sig_df.to_dict('records'),
                            columns=[{'name': i, 'id': i} for i in sig_df.columns],
                            style_cell={'textAlign': 'left', 'padding': '5px'},
                            style_header={'fontWeight': 'bold', 'backgroundColor': '#f8f9fa'}
                        )
                else:
                    posthoc_out = html.Div("Selecione pelo menos uma variável independente.", className="alert alert-warning")"""
    
    new_anova = """                anova_tbl = html.Div([
                    html.P("👆 Clique em uma linha da tabela acima para atualizar o gráfico e os resultados do Post-Hoc abaixo.", className="text-muted small mb-2"),
                    dash_table.DataTable(
                        id='perf-anova-data-table',
                        data=anova_res.to_dict('records'),
                        columns=[{'name': i, 'id': i} for i in anova_res.columns],
                        style_cell={'textAlign': 'left', 'padding': '10px'},
                        style_header={'fontWeight': 'bold', 'backgroundColor': '#f8f9fa'},
                        style_data_conditional=[
                            {
                                'if': {'filter_query': '{PR(>F)} < '+str(alpha), 'column_id': 'PR(>F)'},
                                'backgroundColor': '#d4edda',
                                'fontWeight': 'bold'
                            }
                        ]
                    )
                ])"""

    # We also need to fix the anova_tbl creation that was right above this
    # Wait, the old code had:
    #                 from dash import dash_table
    #                 anova_tbl = dash_table.DataTable(
    #                     data=anova_res.to_dict('records'),
    old_table_creation = """                from dash import dash_table
                anova_tbl = dash_table.DataTable(
                    data=anova_res.to_dict('records'),
                    columns=[{'name': i, 'id': i} for i in anova_res.columns],
                    style_cell={'textAlign': 'left', 'padding': '10px'},
                    style_header={'fontWeight': 'bold', 'backgroundColor': '#f8f9fa'},
                    style_data_conditional=[
                        {
                            'if': {'filter_query': '{PR(>F)} < '+str(alpha), 'column_id': 'PR(>F)'},
                            'backgroundColor': '#d4edda',
                            'fontWeight': 'bold'
                        }
                    ]
                )"""
    
    content = content.replace(old_table_creation, "                from dash import dash_table")
    content = content.replace(old_anova, new_anova)
    
    # We need to replace `anova_style, anova_tbl, anova_fig, posthoc_out = {'display': 'none'}, "", go.Figure(), ""`
    # with `anova_style, anova_tbl = {'display': 'none'}, ""`
    content = content.replace(
        "anova_style, anova_tbl, anova_fig, posthoc_out = {'display': 'none'}, \"\", go.Figure(), \"\"",
        "anova_style, anova_tbl = {'display': 'none'}, \"\""
    )
    
    # We need to replace the return statement
    content = content.replace(
        "return norm_style, norm_out, anova_style, anova_tbl, anova_fig, posthoc_out, inter_style, inter_fig, error_msg",
        "return norm_style, norm_out, anova_style, anova_tbl, inter_style, inter_fig, error_msg"
    )

    new_callback = """
@app.callback(
    [Output('perf-anova-dotplot', 'figure'),
     Output('perf-anova-posthoc-output', 'children')],
    [Input('perf-anova-data-table', 'data'),
     Input('perf-anova-data-table', 'active_cell')],
    [State('perf-protocol-dropdown', 'value'),
     State('perf-anova-target', 'value'),
     State('perf-alpha-input', 'value')]
)
def update_posthoc_plot(table_data, active_cell, prot, target_var, alpha):
    if not table_data:
        return go.Figure(), "Execute a ANOVA primeiro."
        
    alpha = float(alpha) if alpha else 0.05
    
    # Identify which row to use
    selected_row_idx = None
    if active_cell:
        selected_row_idx = active_cell['row']
    else:
        # Default to first significant row
        for i, row in enumerate(table_data):
            if row.get('Source') not in ['Intercept', 'Residual']:
                try:
                    p_val = float(row.get('PR(>F)', 1.0))
                    if p_val < alpha:
                        selected_row_idx = i
                        break
                except: pass
        if selected_row_idx is None:
            # default to first effect
            for i, row in enumerate(table_data):
                if row.get('Source') not in ['Intercept', 'Residual']:
                    selected_row_idx = i
                    break
                    
    if selected_row_idx is None:
        return go.Figure(), "Nenhum fator analisável encontrado."
        
    source_str = table_data[selected_row_idx]['Source']
    
    # Parse source_str: 'C(Complexidade):C(grupo)' -> ['Complexidade', 'grupo']
    import re
    factors = re.findall(r'C\((.*?)\)', source_str)
    if not factors:
        return go.Figure(), f"Não é possível fazer Post-Hoc para a linha selecionada: {source_str}."
        
    import os
    import pandas as pd
    from performance_engine import plot_interactive_dot_sig
    
    base_path = os.path.dirname(__file__)
    csv_path = os.path.join(base_path, 'data', f'analise_df_{prot}_final.csv')
    df = pd.read_csv(csv_path, index_col=0)
    
    if prot == 'C' and 'nivel' not in df.columns and 'Complexidade' in df.columns:
        df['nivel'] = df['Complexidade']
    
    if 'Desempenho ponderado com proporção' in df.columns:
        df.rename(columns={'Desempenho ponderado com proporção': 'Desempenho_ponderado'}, inplace=True)
        
    group_col = "_".join(factors)
    df[group_col] = df[factors].astype(str).agg('_'.join, axis=1)
    
    title = f"Post-Hoc: {group_col} ({target_var}) [Fonte selecionada: {source_str}]"
    
    fig, sig_results, err = plot_interactive_dot_sig(df, group_col, target_var, alpha=alpha, title=title)
    
    if err:
        posthoc_out = html.Div(err, className="alert alert-warning")
    elif not sig_results:
        posthoc_out = html.Div("Nenhuma diferença significativa encontrada no post-hoc.", className="alert alert-info")
    else:
        sig_df = pd.DataFrame(sig_results)
        from dash import dash_table
        posthoc_out = dash_table.DataTable(
            data=sig_df.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in sig_df.columns],
            style_cell={'textAlign': 'left', 'padding': '5px'},
            style_header={'fontWeight': 'bold', 'backgroundColor': '#f8f9fa'}
        )
        
    return fig, posthoc_out
"""
    # Append the new callback right before `if __name__ == '__main__':`
    content = content.replace("if __name__ == '__main__':", new_callback + "\nif __name__ == '__main__':")
    
    with open('app.py', 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == '__main__':
    patch_app()
    print("Done")
