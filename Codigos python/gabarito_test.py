#%% 
import pandas as pd
import re

# Ler os arquivos CSV como strings
gab_seq_df = pd.read_csv('gab_seq.csv', dtype=str)  # Substitua pelo caminho correto
gab_seq_completa_df = pd.read_csv('gab_seq_completa.csv', dtype=str)  # Substitua pelo caminho correto

# Função para converter strings de listas para listas numéricas
def string_to_list(string):
    if pd.isna(string):  # Verificar se a célula está vazia ou é NaN
        return None
    try:
        # Usar regex para extrair números e converter para lista
        return [int(x) for x in re.findall(r'\d+', string)]
    except Exception:
        return None  # Retorna None se houver erro

# Aplicar a conversão em todas as colunas
for col in gab_seq_df.columns:
    gab_seq_df[col] = gab_seq_df[col].apply(string_to_list)

for col in gab_seq_completa_df.columns:
    gab_seq_completa_df[col] = gab_seq_completa_df[col].apply(string_to_list)

# Exibir os DataFrames convertidos
print("Gab Seq:")
print(gab_seq_df)

print("\nGab Seq Completa:")
print(gab_seq_completa_df)


# Salvar os DataFrames convertidos para verificar
gab_seq_df.to_csv('gab_seq_converted.csv', index=False)
gab_seq_completa_df.to_csv('gab_seq_completa_converted.csv', index=False)


#%%%
print(gab_seq_completa_df['1'][0][3]) 
 # %%
