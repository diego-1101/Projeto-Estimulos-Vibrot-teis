import sys
import re

def patch_app_correctly():
    with open('app.py', 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Ensure the layout has the Fase dropdown for Performance
    if 'perf-fase-dropdown' not in content:
        # We need to insert it.
        old_prot_dropdown = """            html.Label("Protocol", className="control-label"),
            dcc.Dropdown(
                id='perf-protocol-dropdown',
                options=[
                    {'label': 'Protocol A', 'value': 'A'},
                    {'label': 'Protocol B', 'value': 'B'},
                    {'label': 'Protocol C', 'value': 'C'}
                ],
                value='A',
                className="mb-3 dash-dropdown"
            ),"""
        new_prot_dropdown = """            html.Label("Protocol", className="control-label"),
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
            ]),"""
        content = content.replace(old_prot_dropdown, new_prot_dropdown)

    # 2. Update update_perf_options callback
    # We need to find the decorator and add the Fase outputs
    if 'Output(\'perf-fase-dropdown\', \'options\')' not in content:
        old_dec = "     Output('perf-interaction-facet', 'value')],"
        new_dec = "     Output('perf-interaction-facet', 'value'),\n     Output('perf-fase-dropdown', 'options'),\n     Output('perf-fase-dropdown', 'value'),\n     Output('perf-fase-container', 'style')],"
        content = content.replace(old_dec, new_dec)
        
        old_body = "    return norm_opts, norm_style, indep_opts, indep_val, ix_opts, ix_val, ix_opts, il_val, if_opts, if_val"
        new_body = """    
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
        content = content.replace(old_body, new_body)

    # 3. Update run_performance_analysis callback
    if 'fase_val' not in content.split('def run_performance_analysis')[1].split(':')[0]:
        content = content.replace(
            "State('perf-interaction-ylim', 'value')]",
            "State('perf-interaction-ylim', 'value'),\n     State('perf-fase-dropdown', 'value')]"
        )
        content = content.replace(
            "def run_performance_analysis(n_clicks, prot, norm_chk, norm_var, norm_group,",
            "def run_performance_analysis(n_clicks, prot, norm_chk, norm_var, norm_group,\n                            anova_chk, anova_var, anova_indeps, alpha,\n                            inter_chk, inter_x, inter_y, inter_line, inter_facet, inter_ylim, fase_val):"
        )
        # Note: I need to make sure I don't duplicate arguments.
        # Let's use a regex to be safer.
        content = re.sub(
            r"def run_performance_analysis\(n_clicks, prot, norm_chk, norm_var, norm_group,\s+anova_chk, anova_var, anova_indeps, alpha,\s+inter_chk, inter_x, inter_y, inter_line, inter_facet, inter_ylim\):",
            "def run_performance_analysis(n_clicks, prot, norm_chk, norm_var, norm_group,\n                            anova_chk, anova_var, anova_indeps, alpha,\n                            inter_chk, inter_x, inter_y, inter_line, inter_facet, inter_ylim, fase_val):",
            content
        )

    # 4. Use the correct CSV file and filter by Fase
    content = content.replace("analise_df_{prot}_final.csv", "df_prot{prot}_performance.csv")
    
    # 5. Fix the regex in update_posthoc_plot
    content = content.replace(
        "    factors = re.findall(r'C\\((.*?)\\)', source_str)",
        "    clean_source = source_str.replace(\"Q('\", \"\").replace(\"')\", \"\")\n    factors = re.findall(r'C\\((.*?)\\)', clean_source)"
    )

    # 6. Ensure update_posthoc_plot also has fase_val State
    if 'State(\'perf-fase-dropdown\', \'value\')]' not in content.split('def update_posthoc_plot')[0]:
         content = content.replace(
            "State('perf-alpha-input', 'value')]",
            "State('perf-alpha-input', 'value'),\n     State('perf-fase-dropdown', 'value')]"
        )
         content = content.replace(
             "def update_posthoc_plot(table_data, active_cell, prot, target_var, alpha):",
             "def update_posthoc_plot(table_data, active_cell, prot, target_var, alpha, fase_val):"
         )

    # 7. Add filtering by Fase inside the functions if not present
    filter_block = """
        df = pd.read_csv(csv_path)
        if 'Fase' in df.columns and fase_val and fase_val != 'ALL':
            df = df[df['Fase'] == fase_val].copy()
"""
    # Wait, I need to be careful with where I insert this.
    # I'll just replace the existing loading line.
    content = re.sub(
        r"df = pd\.read_csv\(csv_path, index_col=0\)",
        "df = pd.read_csv(csv_path)\n        if 'Fase' in df.columns and fase_val and fase_val != 'ALL':\n            df = df[df['Fase'] == fase_val].copy()",
        content
    )

    with open('app.py', 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == '__main__':
    patch_app_correctly()
    print("App fixed and synchronized with performance CSVs.")
