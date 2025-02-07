#%%importanto as bibliotecas
import numpy as np
import pandas as pd

#%% Loading template ("gabarito") data
# Load gabarito data (converted from MATLAB)
gab_seq_df = pd.read_csv('gab_seq_converted.csv', dtype=str)
gab_seq_completa_df = pd.read_csv('gab_seq_completa_converted.csv', dtype=str)

#%% Loading the patiente file

# Patient IDs
ids = ['07', '10', '17', '21', '24', '31', '36', '39']

#Pegando o primeiro id primeiro para depois fazer um loop, lembrando que foi feito no protocolo A_CV
metadado_df = pd.read_csv(fr'C:\Users\diego\OneDrive\Documents\GitHub\Projeto-Estimulos-Vibrot-teis\Dados\Protocolo_A_CV\expA_CV_{ids[0]}\expA_CV_{ids[0]}_metadata.csv')
index_df = pd.read_csv(fr'C:\Users\diego\OneDrive\Documents\GitHub\Projeto-Estimulos-Vibrot-teis\Dados\Protocolo_A_CV\expA_CV_{ids[0]}\expA_CV_{ids[0]}_index.csv',header = None)
overlap_df = pd.read_csv(fr'C:\Users\diego\OneDrive\Documents\GitHub\Projeto-Estimulos-Vibrot-teis\Dados\Protocolo_A_CV\expA_CV_{ids[0]}\expA_CV_{ids[0]}_overlap.csv', header = None)
res_df = pd.read_csv(fr'C:\Users\diego\OneDrive\Documents\GitHub\Projeto-Estimulos-Vibrot-teis\Dados\Protocolo_A_CV\expA_CV_{ids[0]}\expA_CV_{ids[0]}_res_certo.csv')
tempo_df = pd.read_csv(fr'C:\Users\diego\OneDrive\Documents\GitHub\Projeto-Estimulos-Vibrot-teis\Dados\Protocolo_A_CV\expA_CV_{ids[0]}\expA_CV_{ids[0]}_tempo.csv',header=None)
traj_df = pd.read_csv(fr'C:\Users\diego\OneDrive\Documents\GitHub\Projeto-Estimulos-Vibrot-teis\Dados\Protocolo_A_CV\expA_CV_{ids[0]}\expA_CV_{ids[0]}_traj.csv',header=None)

# replacing F/M to 1/2
metadado_df['Genero Discretizado'] = metadado_df['Genero'].replace({'F': 1, 'f': 1, 'M': 2, 'm': 2})
# Reording the columns 
metadado_df = metadado_df[['Nome', 'Idade', 'Peso', 'Altura', 'Genero', 'Genero Discretizado', 'Tipo']]

#%% # converting time DF into readable time
from datetime import datetime, timedelta

# MATLAB starts in "0000-01-01", while Python starts in "1970-01-01"
def matlab_to_datetime(matlab_date):
    base_date = datetime(1, 1, 1)
    adjusted_date = matlab_date - 1  # Ajustar pela diferença de 1 ano
    full_datetime = base_date + timedelta(days=adjusted_date)
    return full_datetime.time()
rTempo_df = tempo_df.applymap(matlab_to_datetime)

rTempo_df.columns = ['Início estímulo (rep 1)',
                    'Fim Estímulo (rep 1)',
                    'Começo da resposta  do participante (rep 1)',
                        'Início estímulo (rep 2)',
                        'Fim Estímulo (rep 2)',
                        'Começo da resposta  do participante (rep 2)',
                            'Início estímulo (rep 3)',
                            'Fim Estímulo (rep 3)', 
                            'Começo da resposta  do participante (rep 3)']

#%% treating res_df

rep1 = res_df[res_df['Repetition'] == 1] #catching only the first repetition
rep2 = res_df[res_df['Repetition'] == 2] #catching only the second repetition
rep3 = res_df[res_df['Repetition'] == 3] #catching only the third repetition

# %% My code

for j in range(len(res_df)): #for to iterate over trials (draws)
    num_traj = traj_df.iloc[0][j]
    overlap_val = overlap_df.iloc[0,j]
    index_val = index_df.iloc[0,j]
    print(index_val)
    for w in range(3): #for to iterate over repetitions
        # Extract temporal components
        t1 = tempo_df.iloc[j,w*3]
        t2 = tempo_df.iloc[j,w*3 + 1]
        t3 = tempo_df.iloc[j,w*3 + 2]
        
        #search on index_df the position of this j trajectory 
        pos = list(zip(*index_df.eq(j).to_numpy().nonzero()))[0][1]
        if(max(res_df[(res_df['Trajectory']== pos) & (res_df['Repetition']== w)]['X_Position']))





# %%
