import sys
import re

def patch_app_v3():
    with open('app.py', 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Remove Fase dropdown from layout
    fase_container_regex = r"html\.Div\(id='perf-fase-container'.*?Dropdown\(id='perf-fase-dropdown'.*?\),?\s*\]\),"
    content = re.sub(fase_container_regex, "", content, flags=re.DOTALL)

    # 2. Update update_perf_options callback to remove Fase outputs
    old_perf_opts_dec = """     Output('perf-interaction-facet', 'value'),
     Output('perf-fase-dropdown', 'options'),
     Output('perf-fase-dropdown', 'value'),
     Output('perf-fase-container', 'style')],"""
    new_perf_opts_dec = """     Output('perf-interaction-facet', 'value')],"""
    content = content.replace(old_perf_opts_dec, new_perf_opts_dec)

    old_perf_opts_body = """    
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
        
    return norm_opts, norm_style, indep_opts, indep_val, ix_opts, ix_val, ix_opts, il_val, if_opts, if_val, fase_opts, fase_val, fase_style"""
    
    new_perf_opts_body = """    return norm_opts, norm_style, indep_opts, indep_val, ix_opts, ix_val, ix_opts, il_val, if_opts, if_val"""
    content = content.replace(old_perf_opts_body, new_perf_opts_body)

    # 3. Update run_performance_analysis callback signature and State
    content = content.replace(
        "     State('perf-interaction-ylim', 'value'),\n     State('perf-fase-dropdown', 'value')]",
        "     State('perf-interaction-ylim', 'value')]"
    )
    
    old_run_sig = "def run_performance_analysis(n_clicks, prot, norm_chk, norm_var, norm_group,\n                            anova_chk, anova_var, anova_indeps, alpha,\n                            inter_chk, inter_x, inter_y, inter_line, inter_facet, inter_ylim, fase_val):"
    new_run_sig = "def run_performance_analysis(n_clicks, prot, norm_chk, norm_var, norm_group,\n                            anova_chk, anova_var, anova_indeps, alpha,\n                            inter_chk, inter_x, inter_y, inter_line, inter_facet, inter_ylim):"
    content = content.replace(old_run_sig, new_run_sig)

    # 4. Update filtering logic in run_performance_analysis
    old_filtering = """        df = pd.read_csv(csv_path)
        if 'Fase' in df.columns and fase_val and fase_val != 'ALL':
            df = df[df['Fase'] == fase_val].copy()"""
    new_filtering = """        df = pd.read_csv(csv_path)
        if prot == 'C' and 'Fase' in df.columns:
            df = df[df['Fase'] == 'Fase Execucao'].copy()"""
    content = content.replace(old_filtering, new_filtering)

    # 5. Update update_posthoc_plot callback signature and State
    content = content.replace(
        "     State('perf-alpha-input', 'value'),\n     State('perf-fase-dropdown', 'value')]",
        "     State('perf-alpha-input', 'value')]"
    )
    
    old_posthoc_sig = "def update_posthoc_plot(table_data, active_cell, prot, target_var, alpha, fase_val):"
    new_posthoc_sig = "def update_posthoc_plot(table_data, active_cell, prot, target_var, alpha):"
    content = content.replace(old_posthoc_sig, new_posthoc_sig)

    # 6. Update filtering in update_posthoc_plot and fix p-value parsing
    old_posthoc_filtering = """    csv_path = os.path.join(os.path.dirname(__file__), 'data', f'df_prot{prot}_performance.csv')
    df = pd.read_csv(csv_path)
    if 'Fase' in df.columns and fase_val and fase_val != 'ALL': df = df[df['Fase'] == fase_val].copy()"""
    
    new_posthoc_filtering = """    csv_path = os.path.join(os.path.dirname(__file__), 'data', f'df_prot{prot}_performance.csv')
    df = pd.read_csv(csv_path)
    if prot == 'C' and 'Fase' in df.columns:
        df = df[df['Fase'] == 'Fase Execucao'].copy()"""
    content = content.replace(old_posthoc_filtering, new_posthoc_filtering)

    # Robust p-value parsing in update_posthoc_plot
    old_p_parsing = """                try:
                    p_val = float(row.get('p-valor', 1.0))
                    if p_val < alpha:
                        selected_row_idx = i
                        break
                except: pass"""
    new_p_parsing = """                try:
                    p_str = row.get('p-valor')
                    if p_str and p_str.strip():
                        p_val = float(p_str)
                        if p_val < alpha:
                            selected_row_idx = i
                            break
                except: pass"""
    content = content.replace(old_p_parsing, new_p_parsing)

    # 7. Improve Factor Extraction (handle both C(...) and raw names)
    old_factor_regex = "factors = re.findall(r'C\\\\((.*?)\\\\)', clean)"
    new_factor_regex = """factors = re.findall(r'C\\((.*?)\\)', clean)
    if not factors:
        # Fallback for interactions like Var1:Var2 or simple Var
        factors = [f.strip() for f in clean.split(':') if f.strip() not in ['Intercept', 'Residual']]"""
    
    # Wait, the reapply script used double backslashes in the string literal. 
    # Let's be careful.
    content = content.replace(old_factor_regex, new_factor_regex)

    with open('app.py', 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == '__main__':
    patch_app_v3()
    print("App fixed: Fase removed for C, auto-filtered, and callback robustness improved.")
