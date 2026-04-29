import sys
import re

def patch_app_final():
    with open('app.py', 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Update get_performance_layout to include Fase dropdown
    fase_dropdown = """
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
            
            html.Div(id='perf-fase-container', children=[
                html.Label("Fase", className="control-label"),
                dcc.Dropdown(id='perf-fase-dropdown', className="mb-3 dash-dropdown"),
            ]),
"""
    old_prot_dropdown = """
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
"""
    content = content.replace(old_prot_dropdown, fase_dropdown)

    # 2. Update update_perf_options to handle Fase options
    # We need to find the update_perf_options function and modify its Outputs and logic
    old_perf_options_dec = """@app.callback(
    [Output('perf-normality-group', 'options'),
     Output('perf-normality-group-container', 'style'),
     Output('perf-anova-independents', 'options'),
     Output('perf-anova-independents', 'value'),
     Output('perf-interaction-x', 'options'),
     Output('perf-interaction-x', 'value'),
     Output('perf-interaction-line', 'options'),
     Output('perf-interaction-line', 'value'),
     Output('perf-interaction-facet', 'options'),
     Output('perf-interaction-facet', 'value')],"""
    
    new_perf_options_dec = """@app.callback(
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
     Output('perf-fase-container', 'style')],"""
    
    content = content.replace(old_perf_options_dec, new_perf_options_dec)

    old_perf_options_body = """    return norm_opts, norm_style, indep_opts, indep_val, ix_opts, ix_val, ix_opts, il_val, if_opts, if_val"""
    new_perf_options_body = """    
    fase_opts = []
    fase_val = None
    fase_style = {'display': 'none'}
    
    if prot == 'A':
        fase_opts = [{'label': 'N/A', 'value': 'ALL'}]
        fase_val = 'ALL'
    elif prot == 'B':
        fase_opts = [{'label': 'N/A', 'value': 'ALL'}]
        fase_val = 'ALL'
    elif prot == 'C':
        fase_opts = [
            {'label': 'Estimulação', 'value': 'Fase Estimulacao'},
            {'label': 'Execução', 'value': 'Fase Execucao'}
        ]
        fase_val = 'Fase Execucao'
        fase_style = {'display': 'block'}
        
    return norm_opts, norm_style, indep_opts, indep_val, ix_opts, ix_val, ix_opts, il_val, if_opts, if_val, fase_opts, fase_val, fase_style"""
    
    content = content.replace(old_perf_options_body, new_perf_options_body)

    # 3. Update run_performance_analysis to include Fase in State and filtering
    old_run_analysis_dec = """def run_performance_analysis(n_clicks, prot, norm_chk, norm_var, norm_group,
                            anova_chk, anova_var, anova_indeps, alpha,
                            inter_chk, inter_x, inter_y, inter_line, inter_facet, inter_ylim):"""
    
    # Wait, the decorator also needs to change to include Fase State
    # Searching for the Input/State section
    content = content.replace(
        "     State('perf-interaction-ylim', 'value')]",
        "     State('perf-interaction-ylim', 'value'),\n     State('perf-fase-dropdown', 'value')]"
    )
    
    new_run_analysis_sig = """def run_performance_analysis(n_clicks, prot, norm_chk, norm_var, norm_group,
                            anova_chk, anova_var, anova_indeps, alpha,
                            inter_chk, inter_x, inter_y, inter_line, inter_facet, inter_ylim, fase_val):"""
    
    content = content.replace(old_run_analysis_dec, new_run_analysis_sig)

    # Filtering logic inside run_performance_analysis
    filtering_logic = """
        df = pd.read_csv(csv_path, index_col=0)
        
        if 'Fase' in df.columns and fase_val and fase_val != 'ALL':
            df = df[df['Fase'] == fase_val].copy()
"""
    content = content.replace("        df = pd.read_csv(csv_path, index_col=0)", filtering_logic)

    # 4. Update update_posthoc_plot to include Fase in State and better regex
    content = content.replace(
        "     State('perf-alpha-input', 'value')]",
        "     State('perf-alpha-input', 'value'),\n     State('perf-fase-dropdown', 'value')]"
    )
    
    old_update_posthoc_sig = "def update_posthoc_plot(table_data, active_cell, prot, target_var, alpha):"
    new_update_posthoc_sig = "def update_posthoc_plot(table_data, active_cell, prot, target_var, alpha, fase_val):"
    content = content.replace(old_update_posthoc_sig, new_update_posthoc_sig)

    # Update regex and column rename
    old_factors_logic = """    # Parse source_str: 'C(Complexidade):C(grupo)' -> ['Complexidade', 'grupo']
    import re
    factors = re.findall(r'C\((.*?)\)', source_str)"""
    
    new_factors_logic = """    # Parse source_str: 'C(Complexidade):C(grupo)' -> ['Complexidade', 'grupo']
    import re
    # Remove Q('...') if present and extract content of C(...)
    clean_source = source_str.replace("Q('", "").replace("')", "")
    factors = re.findall(r'C\((.*?)\)', clean_source)"""
    
    content = content.replace(old_factors_logic, new_factors_logic)

    # Add filtering to update_posthoc_plot
    content = content.replace(
        "    df = pd.read_csv(csv_path, index_col=0)",
        """    df = pd.read_csv(csv_path, index_col=0)
    if 'Fase' in df.columns and fase_val and fase_val != 'ALL':
        df = df[df['Fase'] == fase_val].copy()"""
    )
    
    # Rename columns in ANOVA table to match user preference
    content = content.replace(
        "anova_res = anova_res.reset_index().rename(columns={'index': 'Source'})",
        "anova_res = anova_res.reset_index().rename(columns={'index': 'Termo', 'sum_sq': 'Soma dos Quadrados', 'PR(>F)': 'p-valor'})"
    )
    # Update conditionals for the renamed column
    content = content.replace("PR(>F)", "p-valor")
    content = content.replace("'Source'", "'Termo'")

    with open('app.py', 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == '__main__':
    patch_app_final()
    print("Final patch applied.")
