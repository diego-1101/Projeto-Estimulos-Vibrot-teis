import sys

callbacks = """
# ==============================================================================
# PERFORMANCE EVALUATION CALLBACKS
# ==============================================================================

@app.callback(
    [Output('perf-normality-controls', 'style'),
     Output('perf-anova-controls', 'style'),
     Output('perf-interaction-controls', 'style'),
     Output('perf-alpha-container', 'style')],
    [Input('perf-normality-check', 'value'),
     Input('perf-anova-check', 'value'),
     Input('perf-interaction-check', 'value')]
)
def toggle_performance_controls(norm_val, anova_val, inter_val):
    norm_style = {'display': 'block'} if 'yes' in (norm_val or []) else {'display': 'none'}
    anova_style = {'display': 'block'} if 'yes' in (anova_val or []) else {'display': 'none'}
    inter_style = {'display': 'block'} if 'yes' in (inter_val or []) else {'display': 'none'}
    
    alpha_style = {'display': 'block'} if ('yes' in (norm_val or []) or 'yes' in (anova_val or [])) else {'display': 'none'}
    
    return norm_style, anova_style, inter_style, alpha_style

@app.callback(
    [Output('perf-normality-group', 'options'),
     Output('perf-normality-group-container', 'style'),
     Output('perf-anova-independents', 'options'),
     Output('perf-anova-independents', 'value'),
     Output('perf-interaction-x', 'options'),
     Output('perf-interaction-x', 'value'),
     Output('perf-interaction-line', 'options'),
     Output('perf-interaction-line', 'value'),
     Output('perf-interaction-facet', 'options'),
     Output('perf-interaction-facet', 'value')],
    [Input('perf-protocol-dropdown', 'value')]
)
def update_perf_options(prot):
    if prot == 'A':
        norm_opts = [{'label': 'CV', 'value': 'CV'}, {'label': 'SV', 'value': 'SV'}, {'label': 'Ambos', 'value': 'Ambos'}]
        norm_style = {'display': 'block'}
        indep_opts = [
            {'label': 'Complexidade', 'value': 'Complexidade'},
            {'label': 'Grupo', 'value': 'grupo'},
            {'label': 'Overlap', 'value': 'Overlap'}
        ]
        indep_val = ['Complexidade', 'grupo', 'Overlap']
    elif prot == 'B':
        norm_opts = [{'label': 'CF', 'value': 'CF'}, {'label': 'SF', 'value': 'SF'}, {'label': 'Ambos', 'value': 'Ambos'}]
        norm_style = {'display': 'block'}
        indep_opts = [
            {'label': 'Complexidade', 'value': 'Complexidade'},
            {'label': 'Grupo', 'value': 'grupo'}
        ]
        indep_val = ['Complexidade', 'grupo']
    else: # C
        norm_opts = []
        norm_style = {'display': 'none'}
        indep_opts = [
            {'label': 'Complexidade', 'value': 'nivel'} # No csv prot C eh nivel
        ]
        indep_val = ['nivel']

    # For interaction plot, same independent options + None for facet
    ix_opts = indep_opts
    if len(ix_opts) > 0:
        ix_val = ix_opts[0]['value']
        il_val = ix_opts[1]['value'] if len(ix_opts) > 1 else ix_opts[0]['value']
    else:
        ix_val, il_val = None, None
        
    if prot == 'C':
         if_opts = [{'label': 'None', 'value': 'None'}] + indep_opts
    else:
         if_opts = indep_opts
    if_val = if_opts[0]['value'] if len(if_opts) > 0 else None

    return norm_opts, norm_style, indep_opts, indep_val, ix_opts, ix_val, ix_opts, il_val, if_opts, if_val

@app.callback(
    [Output('perf-normality-result-card', 'style'),
     Output('perf-normality-output', 'children'),
     Output('perf-anova-result-card', 'style'),
     Output('perf-anova-table-output', 'children'),
     Output('perf-anova-dotplot', 'figure'),
     Output('perf-anova-posthoc-output', 'children'),
     Output('perf-interaction-result-card', 'style'),
     Output('perf-interaction-plot', 'figure'),
     Output('perf-controls-error', 'children')],
    [Input('perf-run-btn', 'n_clicks')],
    [State('perf-protocol-dropdown', 'value'),
     State('perf-normality-check', 'value'),
     State('perf-normality-var', 'value'),
     State('perf-normality-group', 'value'),
     State('perf-anova-check', 'value'),
     State('perf-anova-target', 'value'),
     State('perf-anova-independents', 'value'),
     State('perf-alpha-input', 'value'),
     State('perf-interaction-check', 'value'),
     State('perf-interaction-x', 'value'),
     State('perf-interaction-y', 'value'),
     State('perf-interaction-line', 'value'),
     State('perf-interaction-facet', 'value'),
     State('perf-interaction-ylim', 'value')]
)
def run_performance_analysis(n_clicks, prot, norm_chk, norm_var, norm_group,
                            anova_chk, anova_var, anova_indeps, alpha,
                            inter_chk, inter_x, inter_y, inter_line, inter_facet, inter_ylim):
    if not n_clicks:
        raise PreventUpdate

    try:
        import os
        import ast
        from performance_engine import run_normality_test, run_anova_typ3, plot_interactive_dot_sig, plot_interactive_interaction
        
        # Determine CSV path
        base_path = os.path.dirname(__file__)
        csv_path = os.path.join(base_path, 'data', f'analise_df_{prot}_final.csv')
        
        if not os.path.exists(csv_path):
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, f"Arquivo não encontrado: {csv_path}"
            
        df = pd.read_csv(csv_path, index_col=0)
        
        if prot == 'C' and 'nivel' not in df.columns and 'Complexidade' in df.columns:
            df['nivel'] = df['Complexidade']
        
        if 'Desempenho ponderado com proporção' in df.columns:
            df.rename(columns={'Desempenho ponderado com proporção': 'Desempenho_ponderado'}, inplace=True)
            
        alpha = float(alpha) if alpha else 0.05
        
        # Returns
        norm_style, norm_out = {'display': 'none'}, ""
        anova_style, anova_tbl, anova_fig, posthoc_out = {'display': 'none'}, "", go.Figure(), ""
        inter_style, inter_fig = {'display': 'none'}, go.Figure()
        error_msg = ""
        
        # 1. Normality Test
        if 'yes' in (norm_chk or []):
            df_norm = df.copy()
            if prot in ['A', 'B'] and norm_group != 'Ambos':
                df_norm = df_norm[df_norm['grupo'] == norm_group]
                
            norm_res, norm_err = run_normality_test(df_norm[norm_var], alpha=alpha)
            norm_style = {'display': 'block'}
            
            if norm_err:
                norm_out = html.Div(norm_err, className="alert alert-danger")
            else:
                shapiro_res = norm_res['Shapiro-Wilk']
                ks_res = norm_res['Kolmogorov-Smirnov']
                
                def make_badge(is_norm):
                     return html.Span("✅ Normal", className="badge bg-success ms-2") if is_norm else html.Span("❌ Não Normal", className="badge bg-danger ms-2")
                     
                norm_out = html.Div([
                    html.H5(f"Testando: {norm_var} | Grupo: {norm_group if prot in ['A','B'] else 'N/A'}", className="mb-3"),
                    html.Div([
                        html.Strong("Shapiro-Wilk:"),
                        html.Span(f" W = {shapiro_res['stat']:.4f}, p = {shapiro_res['p_value']:.4e}", className="ms-2"),
                        make_badge(shapiro_res['is_normal'])
                    ], className="mb-2"),
                    html.Div([
                        html.Strong("Kolmogorov-Smirnov:"),
                        html.Span(f" D = {ks_res['stat']:.4f}, p = {ks_res['p_value']:.4e}", className="ms-2"),
                        make_badge(ks_res['is_normal'])
                    ])
                ])
                
        # 2. ANOVA and Post-Hoc
        if 'yes' in (anova_chk or []):
            anova_res, anova_err = run_anova_typ3(df, anova_var, anova_indeps)
            anova_style = {'display': 'block'}
            
            if anova_err:
                anova_tbl = html.Div(anova_err, className="alert alert-danger")
            elif anova_res is not None:
                # Format ANOVA table
                anova_res = anova_res.reset_index().rename(columns={'index': 'Source'})
                
                # Format float columns
                for col in ['sum_sq', 'df', 'F', 'PR(>F)']:
                    if col in anova_res.columns:
                        anova_res[col] = anova_res[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "")
                        
                from dash import dash_table
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
                )
                
                # Combine independents for post-hoc
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
                    posthoc_out = html.Div("Selecione pelo menos uma variável independente.", className="alert alert-warning")

        # 3. Interaction Plot
        if 'yes' in (inter_chk or []):
            try:
                parsed_ylim = ast.literal_eval(inter_ylim) if inter_ylim else None
            except:
                parsed_ylim = None
                
            facet = inter_facet if inter_facet and inter_facet != 'None' else None
            
            fig, err = plot_interactive_interaction(df, inter_x, inter_line, inter_y, facet, parsed_ylim, alpha=alpha)
            inter_style = {'display': 'block'}
            inter_fig = fig
            
            if err:
                error_msg += f"Interaction Plot Error: {err}\n"
                
        return norm_style, norm_out, anova_style, anova_tbl, anova_fig, posthoc_out, inter_style, inter_fig, error_msg

    except Exception as e:
        import traceback
        traceback.print_exc()
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, f"Erro Crítico: {str(e)}"
"""

def patch_callbacks():
    with open('app.py', 'r', encoding='utf-8') as f:
        content = f.read()
        
    if "def toggle_performance_controls(" not in content:
        content += "\n" + callbacks
        with open('app.py', 'w', encoding='utf-8') as f:
            f.write(content)
            
if __name__ == '__main__':
    patch_callbacks()
    print("Callbacks patched.")
