import os

def patch_app_v2():
    with open('app.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 1. Add imports at the top
    if 'from performance_engine import' not in "".join(lines):
        lines.insert(25, "from performance_engine import run_normality_test, run_anova_typ3, plot_interactive_dot_sig, plot_interactive_interaction\n")

    content = "".join(lines)

    # 2. Add Tab to layout
    if "'tab-performance'" not in content:
        content = content.replace(
            "dcc.Tab(label='Topoplot', value='tab-topoplot', className='custom-tab', selected_className='custom-tab--selected')",
            "dcc.Tab(label='Topoplot', value='tab-topoplot', className='custom-tab', selected_className='custom-tab--selected'),\n                dcc.Tab(label='Performance Evaluation', value='tab-performance', className='custom-tab', selected_className='custom-tab--selected')"
        )

    # 3. Add to render_content callback
    if "elif tab == 'tab-performance':" not in content:
        content = content.replace(
            "elif tab == 'tab-topoplot':",
            "elif tab == 'tab-performance':\n        return get_performance_layout()\n    elif tab == 'tab-topoplot':"
        )

    # 4. Add get_performance_layout function
    if 'def get_performance_layout():' not in content:
        layout_func = """
def get_performance_layout():
    return html.Div([
        html.Div([
            html.H2("EEG PSD Dashboard", className="text-primary mb-4"),
            html.H5("v2 Performance Evaluation", className="text-muted mb-4"),
            html.Hr(),
            
            html.Label("Protocol", className="control-label"),
            dcc.Dropdown(
                id='perf-protocol-dropdown',
                options=[
                    {'label': 'Protocol A', 'value': 'A'},
                    {'label': 'Protocol B', 'value': 'B'},
                    {'label': 'Protocol C', 'value': 'C'}
                ],
                value='A',
                className="mb-3 dash-dropdown"
            ),
            
            html.Div(id='perf-fase-container', style={'display': 'none'}, children=[
                html.Label("Fase", className="control-label"),
                dcc.Dropdown(id='perf-fase-dropdown', className="mb-3 dash-dropdown"),
            ]),
            
            html.Hr(),
            
            dcc.Checklist(
                id='perf-normality-check',
                options=[{'label': ' Testar Normalidade', 'value': 'yes'}],
                value=[],
                className="mb-2"
            ),
            html.Div(id='perf-normality-controls', style={'display': 'none'}, children=[
                html.Label("Variável de Teste", className="control-label"),
                dcc.Dropdown(id='perf-normality-var', options=Y_VARIABLES, value='Desempenho', className="mb-2 dash-dropdown"),
                html.Div(id='perf-normality-group-container', children=[
                    html.Label("Grupo", className="control-label"),
                    dcc.Dropdown(id='perf-normality-group', options=[{'label': 'CV', 'value': 'CV'}, {'label': 'SV', 'value': 'SV'}, {'label': 'Ambos', 'value': 'Ambos'}], value='Ambos', className="mb-2 dash-dropdown")
                ])
            ]),
            
            html.Hr(),
            
            dcc.Checklist(
                id='perf-anova-check',
                options=[{'label': ' Executar ANOVA Typ=3 & Post-Hoc', 'value': 'yes'}],
                value=[],
                className="mb-2"
            ),
            html.Div(id='perf-anova-controls', style={'display': 'none'}, children=[
                html.Label("Variável Dependente (Y)", className="control-label"),
                dcc.Dropdown(id='perf-anova-target', options=Y_VARIABLES, value='Desempenho', className="mb-2 dash-dropdown"),
                html.Label("Variáveis Independentes (X)", className="control-label"),
                dcc.Checklist(id='perf-anova-independents', options=[
                    {'label': 'Complexidade', 'value': 'Complexidade'},
                    {'label': 'Grupo', 'value': 'grupo'},
                    {'label': 'Overlap', 'value': 'Overlap'}
                ], value=['Complexidade'], className="mb-2 list-style-none", labelStyle={'display': 'block', 'marginBottom': '5px'}),
            ]),
            
            html.Hr(),
            
            html.Div(id='perf-alpha-container', style={'display': 'none'}, children=[
                html.Label("Nível de Significância (Alpha)", className="control-label"),
                dcc.Input(id='perf-alpha-input', type='number', value=0.05, min=0.001, max=0.1, step=0.01, className="form-control mb-3")
            ]),

            dcc.Checklist(
                id='perf-interaction-check',
                options=[{'label': ' Plotar Interaction Plot', 'value': 'yes'}],
                value=[],
                className="mb-2"
            ),
            html.Div(id='perf-interaction-controls', style={'display': 'none'}, children=[
                html.Label("Eixo X", className="control-label"),
                dcc.Dropdown(id='perf-interaction-x', options=[], className="mb-2 dash-dropdown"),
                html.Label("Eixo Y", className="control-label"),
                dcc.Dropdown(id='perf-interaction-y', options=Y_VARIABLES, value='Desempenho', className="mb-2 dash-dropdown"),
                html.Label("Linha", className="control-label"),
                dcc.Dropdown(id='perf-interaction-line', options=[], className="mb-2 dash-dropdown"),
                html.Label("Facet", className="control-label"),
                dcc.Dropdown(id='perf-interaction-facet', options=[{'label': 'None', 'value': 'None'}], value='None', className="mb-2 dash-dropdown"),
                html.Label("Escala Y (ex: 0.2, 1.0)", className="control-label"),
                dcc.Input(id='perf-interaction-ylim', type='text', value='(0.2, 1.0)', className="form-control mb-2")
            ]),
            
            html.Hr(),
            html.Button("Executar Análise", id='perf-run-btn', className="btn btn-primary w-100 mt-2"),
            html.Div(id='perf-controls-error', className="text-danger mt-2")
        ], className="sidebar"),
        
        html.Div([
            html.Div(id='perf-normality-result-card', style={'display': 'none'}, className="card p-3 mb-3 shadow-sm", children=[
                html.H4("Teste de Normalidade", className="text-info"),
                html.Div(id='perf-normality-output')
            ]),
            
            html.Div(id='perf-anova-result-card', style={'display': 'none'}, className="card p-3 mb-3 shadow-sm", children=[
                html.H4("ANOVA e Post-Hoc", className="text-warning"),
                html.Div(id='perf-anova-table-output', style={'overflowX': 'auto'}),
                dcc.Graph(id='perf-anova-dotplot', style={'height': '600px', 'marginTop': '15px'}),
                html.Details([
                    html.Summary("Tabela de Resultados Post-Hoc (Significativos)", style={'cursor': 'pointer', 'fontWeight': 'bold', 'color': '#0d6efd', 'marginTop': '15px'}),
                    html.Div(id='perf-anova-posthoc-output', className="mt-2 bg-light p-2 rounded", style={'overflowX': 'auto'})
                ])
            ]),
            
            html.Div(id='perf-interaction-result-card', style={'display': 'none'}, className="card p-3 mb-3 shadow-sm", children=[
                html.H4("Interaction Plot", className="text-success"),
                dcc.Graph(id='perf-interaction-plot', style={'height': '500px'})
            ])
            
        ], className="main-content")
    ])
"""
        content = content.replace("# --- Layout ---", layout_func + "\n# --- Layout ---")

    # 5. Callbacks
    if 'toggle_performance_controls' not in content:
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
     Output('perf-interaction-facet', 'value'),
     Output('perf-fase-dropdown', 'options'),
     Output('perf-fase-dropdown', 'value'),
     Output('perf-fase-container', 'style')],
    [Input('perf-protocol-dropdown', 'value')]
)
def update_perf_options(prot):
    norm_style = {'display': 'block'}
    if prot == 'A':
        norm_opts = [{'label': 'CV', 'value': 'CV'}, {'label': 'SV', 'value': 'SV'}, {'label': 'Ambos', 'value': 'Ambos'}]
        indep_opts = [{'label': 'Complexidade', 'value': 'Complexidade'}, {'label': 'Grupo', 'value': 'grupo'}, {'label': 'Overlap', 'value': 'Overlap'}]
        indep_val = ['Complexidade', 'grupo', 'Overlap']
    elif prot == 'B':
        norm_opts = [{'label': 'CF', 'value': 'CF'}, {'label': 'SF', 'value': 'SF'}, {'label': 'Ambos', 'value': 'Ambos'}]
        indep_opts = [{'label': 'Complexidade', 'value': 'Complexidade'}, {'label': 'Grupo', 'value': 'grupo'}]
        indep_val = ['Complexidade', 'grupo']
    else:
        norm_opts = [{'label': 'N/A', 'value': 'ALL'}]
        norm_style = {'display': 'none'}
        indep_opts = [{'label': 'Complexidade', 'value': 'Complexidade'}]
        indep_val = ['Complexidade']
        
    ix_opts = indep_opts
    ix_val = indep_opts[0]['value']
    il_val = indep_opts[1]['value'] if len(indep_opts) > 1 else indep_opts[0]['value']
    if_opts = [{'label': 'None', 'value': 'None'}] + indep_opts
    if_val = 'None'
    
    fase_opts = []
    fase_val = None
    fase_style = {'display': 'none'}
    if prot == 'A' or prot == 'B':
        fase_opts = [{'label': 'N/A', 'value': 'ALL'}]
        fase_val = 'ALL'
    elif prot == 'C':
        fase_opts = [{'label': 'Estimulação', 'value': 'Fase Estimulacao'}, {'label': 'Execução', 'value': 'Fase Execucao'}]
        fase_val = 'Fase Execucao'
        fase_style = {'display': 'block'}
        
    return norm_opts, norm_style, indep_opts, indep_val, ix_opts, ix_val, ix_opts, il_val, if_opts, if_val, fase_opts, fase_val, fase_style

@app.callback(
    [Output('perf-normality-result-card', 'style'), Output('perf-normality-output', 'children'),
     Output('perf-anova-result-card', 'style'), Output('perf-anova-table-output', 'children'),
     Output('perf-interaction-result-card', 'style'), Output('perf-interaction-plot', 'figure'),
     Output('perf-controls-error', 'children')],
    [Input('perf-run-btn', 'n_clicks')],
    [State('perf-protocol-dropdown', 'value'), State('perf-normality-check', 'value'),
     State('perf-normality-var', 'value'), State('perf-normality-group', 'value'),
     State('perf-anova-check', 'value'), State('perf-anova-target', 'value'),
     State('perf-anova-independents', 'value'), State('perf-alpha-input', 'value'),
     State('perf-interaction-check', 'value'), State('perf-interaction-x', 'value'),
     State('perf-interaction-y', 'value'), State('perf-interaction-line', 'value'),
     State('perf-interaction-facet', 'value'), State('perf-interaction-ylim', 'value'),
     State('perf-fase-dropdown', 'value')]
)
def run_performance_analysis(n_clicks, prot, norm_chk, norm_var, norm_group,
                            anova_chk, anova_var, anova_indeps, alpha,
                            inter_chk, inter_x, inter_y, inter_line, inter_facet, inter_ylim, fase_val):
    if not n_clicks: raise PreventUpdate
    try:
        import os, ast
        from performance_engine import run_normality_test, run_anova_typ3, plot_interactive_dot_sig, plot_interactive_interaction
        csv_path = os.path.join(os.path.dirname(__file__), 'data', f'df_prot{prot}_performance.csv')
        if not os.path.exists(csv_path): return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, f"Arquivo não encontrado: {csv_path}"
        df = pd.read_csv(csv_path)
        if 'Fase' in df.columns and fase_val and fase_val != 'ALL': df = df[df['Fase'] == fase_val].copy()
        alpha = float(alpha) if alpha else 0.05
        norm_style, norm_out = {'display': 'none'}, ""
        anova_style, anova_tbl = {'display': 'none'}, ""
        inter_style, inter_fig = {'display': 'none'}, go.Figure()
        
        if 'yes' in (norm_chk or []):
            df_norm = df.copy()
            if prot in ['A', 'B'] and norm_group != 'Ambos': df_norm = df_norm[df_norm['grupo'] == norm_group]
            norm_res, norm_err = run_normality_test(df_norm[norm_var], alpha=alpha)
            norm_style = {'display': 'block'}
            if norm_err: norm_out = html.Div(norm_err, className="alert alert-danger")
            else:
                s, k = norm_res['Shapiro-Wilk'], norm_res['Kolmogorov-Smirnov']
                def b(is_norm): return html.Span("✅ Normal", className="badge bg-success ms-2") if is_norm else html.Span("❌ Não Normal", className="badge bg-danger ms-2")
                norm_out = html.Div([
                    html.H5(f"Testando: {norm_var} | Grupo: {norm_group if prot in ['A','B'] else 'N/A'}", className="mb-3"),
                    html.Div([html.Strong("Shapiro-Wilk:"), html.Span(f" W = {s['stat']:.4f}, p = {s['p_value']:.4e}", className="ms-2"), b(s['is_normal'])], className="mb-2"),
                    html.Div([html.Strong("Kolmogorov-Smirnov:"), html.Span(f" D = {k['stat']:.4f}, p = {k['p_value']:.4e}", className="ms-2"), b(k['is_normal'])], className="mb-2")
                ])

        if 'yes' in (anova_chk or []):
            anova_res, anova_err = run_anova_typ3(df, anova_var, anova_indeps)
            anova_style = {'display': 'block'}
            if anova_err: anova_tbl = html.Div(anova_err, className="alert alert-danger")
            elif anova_res is not None:
                anova_res = anova_res.reset_index().rename(columns={'index': 'Termo', 'sum_sq': 'Soma dos Quadrados', 'PR(>F)': 'p-valor'})
                for col in ['Soma dos Quadrados', 'df', 'F', 'p-valor']:
                    if col in anova_res.columns: anova_res[col] = anova_res[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "")
                from dash import dash_table
                anova_tbl = html.Div([
                    html.P("👆 Clique em uma linha da tabela acima para atualizar o gráfico e os resultados do Post-Hoc abaixo.", className="text-muted small mb-2"),
                    dash_table.DataTable(
                        id='perf-anova-data-table', data=anova_res.to_dict('records'),
                        columns=[{'name': i, 'id': i} for i in anova_res.columns],
                        style_cell={'textAlign': 'left', 'padding': '10px'},
                        style_header={'fontWeight': 'bold', 'backgroundColor': '#f8f9fa'},
                        style_data_conditional=[{'if': {'filter_query': '{p-valor} < '+str(alpha), 'column_id': 'p-valor'}, 'backgroundColor': '#d4edda', 'fontWeight': 'bold'}]
                    )
                ])

        if 'yes' in (inter_chk or []):
            try: parsed_ylim = ast.literal_eval(inter_ylim) if inter_ylim else None
            except: parsed_ylim = None
            facet = inter_facet if inter_facet and inter_facet != 'None' else None
            fig, err = plot_interactive_interaction(df, inter_x, inter_line, inter_y, facet, parsed_ylim, alpha=alpha)
            inter_style, inter_fig = {'display': 'block'}, fig

        return norm_style, norm_out, anova_style, anova_tbl, inter_style, inter_fig, ""
    except Exception as e:
        import traceback; traceback.print_exc()
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, f"Erro Crítico: {str(e)}"

@app.callback(
    [Output('perf-anova-dotplot', 'figure'), Output('perf-anova-posthoc-output', 'children')],
    [Input('perf-anova-data-table', 'data'), Input('perf-anova-data-table', 'active_cell')],
    [State('perf-protocol-dropdown', 'value'), State('perf-anova-target', 'value'), State('perf-alpha-input', 'value'), State('perf-fase-dropdown', 'value')]
)
def update_posthoc_plot(table_data, active_cell, prot, target_var, alpha, fase_val):
    if not table_data: return go.Figure(), "Execute a ANOVA primeiro."
    alpha = float(alpha) if alpha else 0.05
    idx = active_cell['row'] if active_cell else next((i for i, r in enumerate(table_data) if r.get('Termo') not in ['Intercept', 'Residual'] and float(r.get('p-valor', 1.0)) < alpha), next((i for i, r in enumerate(table_data) if r.get('Termo') not in ['Intercept', 'Residual']), None))
    if idx is None: return go.Figure(), "Nenhum fator analisável encontrado."
    source_str = table_data[idx]['Termo']
    import re; clean = source_str.replace("Q('", "").replace("')", ""); factors = re.findall(r'C\\((.*?)\\)', clean)
    if not factors: return go.Figure(), f"Não é possível fazer Post-Hoc para a linha selecionada: {source_str}."
    import os, pandas as pd; from performance_engine import plot_interactive_dot_sig
    csv_path = os.path.join(os.path.dirname(__file__), 'data', f'df_prot{prot}_performance.csv')
    df = pd.read_csv(csv_path)
    if 'Fase' in df.columns and fase_val and fase_val != 'ALL': df = df[df['Fase'] == fase_val].copy()
    gc = "_".join(factors); df[gc] = df[factors].astype(str).agg('_'.join, axis=1)
    fig, sig, err = plot_interactive_dot_sig(df, gc, target_var, alpha=alpha, title=f"Post-Hoc: {gc} ({target_var}) [Fonte: {source_str}]")
    if err: return fig, html.Div(err, className="alert alert-warning")
    if not sig: return fig, html.Div("Nenhuma diferença significativa encontrada.", className="alert alert-info")
    from dash import dash_table; sdf = pd.DataFrame(sig)
    return fig, dash_table.DataTable(data=sdf.to_dict('records'), columns=[{'name': i, 'id': i} for i in sdf.columns], style_cell={'textAlign': 'left', 'padding': '5px'}, style_header={'fontWeight': 'bold', 'backgroundColor': '#f8f9fa'})
"""
        main_block = "if __name__ == '__main__':"
        content = content.replace(main_block, callbacks + "\n\n" + main_block)

    with open('app.py', 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == '__main__':
    patch_app_v2()
    print("Re-applied performance tab to app.py surgically.")
