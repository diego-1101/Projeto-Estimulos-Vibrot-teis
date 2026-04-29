import sys

def patch_app():
    with open('app.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 1. Add imports at the top
    if 'from performance_engine import' not in content:
        import_stmt = "\nfrom performance_engine import run_normality_test, run_anova_typ3, plot_interactive_dot_sig, plot_interactive_interaction\n"
        content = content.replace("from topoplot_engine import generate_topoplot_grid_base64", "from topoplot_engine import generate_topoplot_grid_base64" + import_stmt)
    
    # 2. Add get_performance_layout() function right before app.layout
    layout_func = """
def get_performance_layout():
    return html.Div([
        html.Div([
            html.H2("EEG PSD Dashboard", className="text-primary mb-4"),
            html.H5("Performance Evaluation", className="text-muted mb-4"),
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
            
            html.Hr(),
            
            # --- Normality Test Config ---
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
            
            # --- ANOVA Config ---
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
            
            # --- Common Alpha ---
            html.Div(id='perf-alpha-container', style={'display': 'none'}, children=[
                html.Label("Nível de Significância (Alpha)", className="control-label"),
                dcc.Input(id='perf-alpha-input', type='number', value=0.05, min=0.001, max=0.1, step=0.01, className="form-control mb-3")
            ]),

            # --- Interaction Plot Config ---
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
            # Results Panel
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

# --- Layout ---
"""
    if "def get_performance_layout():" not in content:
        content = content.replace("# --- Layout ---", layout_func, 1)

    # 3. Add Tab
    if "dcc.Tab(label='Performance Evaluation'" not in content:
        tab_str = "                dcc.Tab(label='Topoplot', value='tab-topoplot', className='custom-tab', selected_className='custom-tab--selected'),\n                dcc.Tab(label='Performance Evaluation', value='tab-performance', className='custom-tab', selected_className='custom-tab--selected')"
        content = content.replace("                dcc.Tab(label='Topoplot', value='tab-topoplot', className='custom-tab', selected_className='custom-tab--selected')", tab_str)
        
    # 4. Handle render_content
    if "elif tab == 'tab-performance':" not in content:
        render_append = """    elif tab == 'tab-topoplot':
        return get_topoplot_layout()
    elif tab == 'tab-performance':
        return get_performance_layout()"""
        content = content.replace("    elif tab == 'tab-topoplot':\n        return get_topoplot_layout()", render_append)
        
    with open('app.py', 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == '__main__':
    patch_app()
    print("Done patching.")
