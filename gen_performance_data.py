import pandas as pd
import os

def gen_perf(src, dst, extra_vars):
    if not os.path.exists(src): return
    # Use latin1 to read if there are special chars
    try:
        df = pd.read_csv(src, encoding='latin1')
    except:
        df = pd.read_csv(src)
    
    new_cols = {}
    for c in df.columns:
        cl = str(c).lower()
        if 'desempenho' in cl and 'ponderado' in cl: new_cols[c] = 'Desempenho_ponderado'
        elif 'desempenho' in cl: new_cols[c] = 'Desempenho'
        elif 'acuracia' in cl or 'acurácia' in cl: new_cols[c] = 'Acuracia'
        elif 'similaridade' in cl: new_cols[c] = 'Similaridade'
        elif 'especificidade' in cl: new_cols[c] = 'Especificidade'
        elif 'propor' in cl and ' x' in cl: new_cols[c] = 'Proporção espacial x'
        elif 'propor' in cl and ' y' in cl: new_cols[c] = 'Proporção espacial y'
        elif 'complexidade' in cl: new_cols[c] = 'Complexidade'
        elif 'grupo' in cl: new_cols[c] = 'grupo'
        elif 'overlap' in cl: new_cols[c] = 'Overlap'
        elif 'fase' in cl: new_cols[c] = 'Fase'
    
    df.rename(columns=new_cols, inplace=True)
    
    target_cols = [
        'Desempenho', 'Acuracia', 'Similaridade', 'Especificidade', 
        'Proporção espacial x', 'Proporção espacial y', 'Desempenho_ponderado', 'Fase'
    ] + extra_vars
    
    available = []
    seen = set()
    for c in target_cols:
        if c in df.columns and c not in seen:
            available.append(c)
            seen.add(c)
            
    print(f"Saving {len(available)} columns to {dst}")
    # Save as UTF-8
    df[available].to_csv(dst, index=False, encoding='utf-8')

base = 'Codigos python/'
data_dir = 'eeg-psd-dashboard/data/'
gen_perf(base + 'df_protA.csv', data_dir + 'df_protA_performance.csv', ['Complexidade', 'grupo', 'Overlap'])
gen_perf(base + 'df_protB.csv', data_dir + 'df_protB_performance.csv', ['Complexidade', 'grupo'])
gen_perf(base + 'df_protC.csv', data_dir + 'df_protC_performance.csv', ['Complexidade', 'Fase'])
