# Objetivo
Modificar o dashboard (Dash/Plotly) para também **plotar resultados de CDA calculados no MATLAB** (via `manova1`) sem recalcular CDA em Python. Ou seja, adicionar ao dashboard um novo plot de CDA chamado CDA Matlab.

# Nova fonte de dados
Existe uma pasta local (ou no repositório) chamada:

`dados_dataframe_final/cda_results_matlab/`

Dentro dela, há subpastas:
- `protA/`
- `protB/`
- `protC/`

Cada subpasta contém vários CSVs exportados pelo MATLAB. Cada CSV já vem com:
1) todas as colunas de `protX_labels.csv` (metadados por trial: grupo/complexidade/overlap)         
2) colunas de coordenadas canônicas: `CAN1`, `CAN2`, `CAN3` (quando existirem)

Esses CANs correspondem a `stats.canon(:,1:3)` do MATLAB `manova1`.

# O que o dashboard deve fazer
## 1) Adicionar um modo “CDA (MATLAB precomputed)”
No seletor de método de análise, adicionar uma opção nova:
- `CDA (MATLAB precomputed)` (ou algo equivalente)

Quando essa opção estiver selecionada:
- o dashboard **não roda** PCA/LDA/PLS/CDA em Python
- ele apenas **lê um CSV** da pasta `cda_results_matlab` e plota CAN1/CAN2/CAN3

## 2) Adicionar seletores para localizar o CSV correto
Criar seletores (em cada painel, mantendo o Comparison Mode):

### a) Protocol
- A / B / C

### b) Domain
- EEG (X)
- Behavior (Y)

### c) Behavior Combo (só quando Domain = Behavior)
Dropdown com:
- ACS_plus_PropX
- ACS_plus_PropY
- ACS_plus_PropXY
- Des_plus_PropX
- Des_plus_PropY
- Des_plus_PropXY
- Des_plus_ACS_plus_PropXY

### d) Scenario
Dropdown com:
- G1
- G2
- ALL
Regras:
- Prot C só tem ALL (desabilitar G1/G2)
- Prot A e B têm G1/G2/ALL

### e) Factor (Color/Test factor)
Dropdown depende do protocolo:
- Prot A: Overlap / Complexidade / Grupo
- Prot B: Complexidade / Grupo
- Prot C: Complexidade

## 3) Mapeamento do arquivo CSV (regra de nome)
Os CSVs seguem o padrão:

### EEG:
`EEG_<SCENARIO>_CDA_<FACTOR>.csv`
ex: `EEG_ALL_CDA_Complexidade.csv`

### Behavior:
`BEHAV_<COMBO>_<SCENARIO>_CDA_<FACTOR>.csv`
ex: `BEHAV_Des_plus_PropXY_G1_CDA_Overlap.csv`

O dashboard deve:
- montar automaticamente o caminho do CSV baseado nos seletores acima
- checar existência do arquivo e, se não existir, mostrar erro amigável no gráfico

## 4) Plot
Plotar em Plotly usando:
- eixo X = `CAN1`
- eixo Y = `CAN2` (se existir; senão cair para 1D ou mostrar mensagem “apenas 1 dimensão canônica disponível”)
- eixo Z = `CAN3` se o usuário selecionar 3D e se CAN3 existir

O color_by deve continuar igual ao dashboard atual (grupo/complexidade/overlap), mas:
- como o CSV já contém os labels, usar essas colunas direto.
- para Prot C, não permitir colorir por grupo.

## 5) Não mudar o resto
Manter:
- Comparison Mode (dois gráficos independentes)
- Tema claro/escuro
- Layout geral

# Requisitos técnicos
- Implementar uma função `list_matlab_cda_files(protocol)` que lista CSVs disponíveis na pasta e valida.
- Implementar uma função `load_matlab_cda_csv(path)` que retorna DataFrame pronto para plot.
- Ajustar callbacks para, quando o método for “CDA (MATLAB precomputed)”, ignorar o pipeline de embeddings e apenas usar `CAN*`.

# UX
- Se o CSV tiver apenas CAN1 (ex.: fator com 2 níveis), não quebrar: mostrar gráfico 1D ou avisar que só existe CAN1.
- Se o usuário escolher 3D mas não houver CAN3, reduzir automaticamente para 2D ou mostrar aviso.
