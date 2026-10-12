# Registro Histórico e Manifesto de Conversas (conversations.md)

---

## 📜 Manifesto do Arquivo

### 1. Finalidade e Intenção
Este documento é a **memória viva e persistente entre sessões de IA** para o repositório `Projeto-Estimulos-Vibrot-teis`.  
Ele resolve o problema de perda de contexto entre diferentes threads, garantindo que novos agentes:
1. **Não gastem tempo e tokens investigando históricos brutos de logs** (`transcript.jsonl`) para reconstruir o que aconteceu.
2. **Entendam instantaneamente o estado atual do projeto**, as decisões técnicas pregressas e os motivos pelos quais o código foi escrito de determinada maneira.
3. **Mantenham a continuidade lógica**, respeitando as decisões científicas já tomadas e evitando retrabalho ou desconfiguração de arquivos legados.

### 2. Protocolo Obrigatório para o Agente em Novas Sessões
Ao iniciar qualquer interação neste repositório:
* **Passo 1 (Leitura Obrigatória):** Leia este arquivo (`conversations.md`) e o [`Agents.md`](file:///c:/Users/diego/OneDrive/Documents/GitHub/Projeto-Estimulos-Vibrot-teis/Agent%20Conversations/Agents.md).
* **Passo 2 (Identificação do Contexto):** Localize a última entrada do registro cronológico para saber exatamente em que pé o trabalho parou.
* **Passo 3 (Atualização ao Encerrar):** Ao final da sessão (ou após implementar mudanças significativas), **adicione uma nova seção no histórico abaixo**, seguindo o template padrão.

### 3. Template Padrão de Registro de Sessão
```markdown
### [Data da Sessão] — Título / Tema Central da Sessão
* **ID da Conversa / Referência:** (se aplicável)
* **Demanda do Usuário:** Descrição sucinta do que foi solicitado.
* **Diagnóstico e Decisões Técnicas:** Explicação do problema encontrado e decisões tomadas.
* **Arquivos Modificados / Criados:** Lista com links no padrão GitHub markdown.
* **Estado do Projeto e Próximos Passos:** Em que estado o código ficou e quais são as próximas ações planejadas.
```

---

## 🗂️ Histórico Consolidado de Conversas e Decisões

---

### Sessão 1 (Maio – Julho de 2026) — Fundação do Dashboard e Decisões Iniciais
* **Tema Central:** Estruturação da suíte analítica em Dash/Plotly, integração MNE e normalização espectral.
* **Contexto e Demandas:**
  * O projeto precisava de uma interface para visualizar a Densidade de Potência Espectral (PSD) e testar a hipótese de decodificação sensório-motora.
  * Implementação da projeção por **Discriminante Canônica (CDA)** espelhando a função `manova1` do MATLAB.
* **Decisões Técnicas e Arquiteturais:**
  * **Resolução Espectral de 2 em 2 Hz:** Para a CDA, a PSD completa continha multicolinearidade severa e sobreajuste (*overfitting*). Decidiu-se condensar o espectro de 0 a 50 Hz em bins de 2 Hz (`psd_2em2_norm`, 25 features por canal).
  * **Correção FDR nos Topoplots:** Inicialmente, os topoplots marcavam canais apenas com $p < 0.05$ não corrigido. Para publicação científica, implementou-se a correção de **False Discovery Rate (Benjamini-Hochberg, `fdr_bh`)** em cima dos 32 canais por banda para mitigar falsos positivos espaciais.
* **Arquivos-Chave:** `eeg-psd-dashboard/app.py`, `manova1_matlab.py`, `topoplot_engine.py`.

---

### Sessão 2 (Julho – Início de Agosto de 2026) — Desenvolvimento da Aba Performance Evaluation
* **Tema Central:** Implementação do motor estatístico fatorial e gráficos comportamentais avançados.
* **Contexto e Demandas:**
  * O usuário precisava analisar o desempenho psicométrico dos voluntários (`Desempenho`, `Acuracia`, `Similaridade`, `Especificidade`).
* **Decisões Técnicas e Arquiteturais:**
  * **ANOVA Fatorial Tipo III:** Utilização do `statsmodels` com termos de interação total (`Complexidade * grupo * Overlap`).
  * **Inovação no Post-Hoc:** Desenvolveu-se a função `tukey_hsd_from_model` em `performance_engine.py`. A decisão crítica foi **reaproveitar o $MSE_{residual}$ e os graus de liberdade da tabela Tipo III**, evitando a perda de poder estatístico de um Tukey one-way ingênuo.
  * **Gráfico Híbrido:** Criação de visualização combinando boxplot/strip (pontos individuais com *jitter*) e linhas de interação de médias com intervalo de confiança de 95% ($CI_{95\%}$).
  * **Heatmaps Matriciais de Significância:** Criação de matrizes $k \times k$ em escala monocromática azul mostrando $-\log_{10}(p_{adj})$ para todas as combinações de fatores.
* **Arquivos-Chave:** `performance_engine.py`, `RELATORIO_PERFORMANCE.md`.

---

### Sessão 3 (20 de Agosto de 2026 — Commit `a9ad374`) — Variância em CDA, Overlays na PSD e Proporções Espaciais
* **Tema Central:** Refinamento matemático da CDA, sobreposição de curvas na PSD e novas variáveis espaciais.
* **Contexto e Demandas:**
  * Exibir a porcentagem real de variância explicada nas dimensões da CDA ($\text{CD}_1$, $\text{CD}_2$, $\text{CD}_3$).
  * Permitir sobrepor curvas de PSD de diferentes grupos no mesmo gráfico por canal.
  * Incluir análise das variáveis `Proporção espacial x` e `Proporção espacial y`.
* **Decisões Técnicas e Arquiteturais:**
  * **Variância CDA:** Corrigido o cálculo para utilizar a proporção dos autovalores da decomposição espectral das matrizes entre/dentro de grupos ($B$ e $W$).
  * **Proporções Espaciais:** Criada a função `run_spatial_proportions_analysis` em `performance_engine.py`, aplicando o **Teste $t$ de Welch** entre os grupos (CV vs SV e CF vs SF).
  * **Colchetes Coloridos:** Adicionada a opção visual `color_sig` no dotplot, diferenciando comparações inter-grupo (vermelho) de intra-grupo (verde).
* **Arquivos-Chave:** `analysis_engine.py`, `performance_engine.py`, `psd_visualizer.py`, `app.py`.

---

### Sessão 4 (31 de Agosto – 02 de Setembro de 2026 — ID `ca54a267`) — Padronização do Protocolo C e Correção de Gabaritos
* **Tema Central:** Integração completa do Protocolo C e correção na codificação de movimentos das trajetórias.
* **Contexto e Demandas:**
  * Erros ao carregar o Protocolo C no dashboard devido a diferenças na estrutura de dados (Protocolo C não possui variável `grupo` dividida nem `Overlap`).
  * Inconsistência nos gabaritos: os códigos estavam salvos com base na numeração de pinos dos motores físicos, e não na codificação direcional de movimento (1=direita, 2=baixo, 3=esquerda, 4=cima).
* **Decisões Técnicas e Arquiteturais:**
  * **Conversão do Gabarito do Protocolo C:** Criado `gabarito_protocolo_C.csv` traduzindo a matriz física de motores para os códigos de direção.
  * **Ajuste na UI do Dashboard:** O dashboard passou a ocultar dinamicamente os seletores de Fase e Interaction Plot para o Protocolo C, e a forçar internamente a regra de negócio `Fase == 'Fase Execucao'`.
  * **Geração de Figuras Científicas:** Criação do script para renderizar os gabaritos das trajetórias em alta resolução divididas em Fácil, Médio e Difícil para o artigo.
* **Arquivos-Chave:** `Codigos python/ler_protocolo_C.py`, `Codigos python/gabarito_protocolo_C.csv`, `eeg-psd-dashboard/app.py`.

---

### Sessão 5 (03 – 04 de Setembro de 2026 — ID `52977ee0`) — Auditoria da Figura 8 e Diagnóstico da ANOVA
* **Tema Central:** Investigação da perda de significância da interação após a correção da trajetória da Figura 8.
* **Contexto e Demandas:**
  * O usuário corrigiu o gabarito da Figura 8 (trajetória de nível Difícil / 8 curvas nos Protocolos A e B), substituindo um passo diagonal incorreto ("7") por passos ortogonais de 14 movimentos.
  * Ao rodar a ANOVA no dashboard com os novos dados, a interação tripla em `Desempenho` perdeu significância ($p = 0.0135 \to 0.1155$), gerando preocupação no pesquisador.
* **Diagnóstico e Conclusões Científicas:**
  * **Eliminação de um Falso Positivo:** O gabarito antigo penalizava assimetricamente o grupo SV devido à discrepância de passos, inflando falsamente a taxa de erros e gerando um falso cruzamento de curvas.
  * **Efeito de Cancelamento na Complexidade 8:** A Figura 8 corrigida teve desempenho onde SV superou CV, o que anulou os efeitos das Figuras 7 e 9 (onde CV superou SV). Ao agregar na categoria `Complexidade = 8`, as médias empataram e a variância aumentou, reduzindo o $F$.
  * **Manutenção da Evidência Real:** Ficou comprovado que a hipótese do estudo segue sólida: **Acurácia** ($p = 0.0052$) e **Similaridade** ($p = 0.0001$) mantiveram interações altamente significativas no Protocolo A.
* **Arquivos-Chave:** `Codigos python/df_protA.csv`, `eeg-psd-dashboard/data/df_protA_performance.csv`, `gen_performance_data.py`.

---

### Sessão 6 (16 – 20 de Setembro de 2026 — Atual) — Auditoria de Testes Post-Hoc, Mapeamento Amostral ($N$) e Hub de Continuidade
* **Tema Central:** Explicação detalhada de todos os testes post-hoc da suíte, levantamento preciso do $N$ amostral por módulo/protocolo e refinamento da redação do artigo científico.
* **Contexto e Demandas:**
  * O usuário solicitou explicação de todos os testes *post-hoc* feitos no dashboard (CDA, Topoplots, Performance), suas premissas matemáticas e o porquê de cada escolha.
  * Solicitou o levantamento do $N$ amostral exato (voluntários e trials) por protocolo e grupo, tanto na Performance quanto no EEG.
  * Pediu revisão de um parágrafo metodológico para o artigo acadêmico sobre os topoplots.
  * Solicitou a criação deste diretório [`Agent Conversations/`](file:///c:/Users/diego/OneDrive/Documents/GitHub/Projeto-Estimulos-Vibrot-teis/Agent%20Conversations) para servir de hub definitivo de contexto para as próximas threads.
* **Entregas e Decisões:**
  1. **Consolidação Estatística:**
     * **CDA:** ANOVA One-Way + **Tukey HSD** clássico nos scores latentes.
     * **Topoplots:** **Teste $t$ de Welch** (2 condições) ou **ANOVA One-Way F-test** (3 condições) canal por canal, com correção espacial de **FDR Benjamini-Hochberg (`fdr_bh`)** nos 32 canais. Esclarecido que *não há teste post-hoc posterior no topoplot*; os canais com $p_{adj} < 0.05$ já são o produto final marcado com "x".
     * **Performance:** ANOVA Fatorial Tipo III + **Tukey-Kramer baseado no $MSE_{residual}$ e $df_{residual}$ do modelo completo**.
  2. **Mapeamento do $N$ Amostral:**
     * **Prot A:** Performance = 16 voluntários ($N=1296$ trials: 648 CV / 648 SV). EEG = 13 voluntários válidos ($N=1053$ trials: 486 CV / 567 SV).
     * **Prot B:** Performance = 18 voluntários ($N=486$ trials: 243 CF / 243 SF). EEG = 10 voluntários processados no PSD ($N=270$ trials).
     * **Prot C:** 9 voluntários ($N=324$ trials: 162 Exploração / 162 Execução). Na aba de Performance usa-se $N=162$ (Execução).
  3. **Revisão Metodológica do Artigo:** Ajustado o texto substituindo "topológico" por "topográfico", "cenários" por "condições experimentais", e removendo a menção equivocada a teste post-hoc após o FDR no topoplot.
  4. **Criação do Hub do Agente:** Criados [`Agents.md`](file:///c:/Users/diego/OneDrive/Documents/GitHub/Projeto-Estimulos-Vibrot-teis/Agent%20Conversations/Agents.md) e [`conversations.md`](file:///c:/Users/diego/OneDrive/Documents/GitHub/Projeto-Estimulos-Vibrot-teis/Agent%20Conversations/conversations.md).
* **Arquivos Criados:**
  * [`Agent Conversations/Agents.md`](file:///c:/Users/diego/OneDrive/Documents/GitHub/Projeto-Estimulos-Vibrot-teis/Agent%20Conversations/Agents.md)
  * [`Agent Conversations/conversations.md`](file:///c:/Users/diego/OneDrive/Documents/GitHub/Projeto-Estimulos-Vibrot-teis/Agent%20Conversations/conversations.md)

---

### Sessão 7 (23 de Setembro de 2026) — Padronização Visual de Gráficos (Scatter/ANOVA) e Rótulos do Topoplot em Comparação
* **Tema Central:** Padronização visual completa dos gráficos de dispersão/ANOVA (labels em inglês, cores por grupo, barras de significância) e dinamização dos rótulos e layout no modo de comparação de Topoplots.
* **Contexto e Demandas:**
  1. **Padronização dos Gráficos de Dispersão / ANOVA:**
     * **1.1.1 Complexidades em Inglês:** Conversão de "Fácil", "Médio", "Difícil" para "Easy", "Medium", "Hard" em todo o dashboard (eixos, tooltips, legendas, ordenação semântica).
     * **1.1.2 Dispersão de Dados:** Pontos individuais de jitter removidos por padrão (`show_jitter=False`, checkbox desmarcado por padrão na Performance e removido da ANOVA Multivariada).
     * **1.1.3 Barras de Significância:** ANOVA da Multivariate Analysis padronizada com a mesma paleta da Performance: Vermelho (`#dc3545`) para comparações inter-grupo (CV/CF ↔ SV/SF), Verde (`#198754`) para comparações intra-grupo (CV ↔ CV / SV ↔ SV) e Cinza Escuro (`#333333`) para demais contrastes, com legendas explícitas. Na Performance, a opção vem marcada por padrão (`value=['yes']`).
     * **1.1.4 Colorir Markers por Grupo:** Adicionada a opção "Colorir markers por grupo" (ativa por padrão, oculta para Protocolo C) com tons padronizados de azul: Azul Claro (`#5dade2`) para CV/CF e Azul Escuro (`#0d47a1`) para SV/SF. Aplicado por padrão na ANOVA da Multivariate Analysis e configurável na Performance.
  2. **Modo Comparação de Topoplots:**
     * No grid invertido de comparação, substituição de "Painel 1", "Painel 2", etc. pelos nomes descritivos reais formatados (ex: `Prot. A Norm\n(Estimulação - CV)`).
     * Ajuste de layout (aumento da figura para `3.8 x 3.8` por painel, `pad=16` nos títulos e `subplots_adjust`) para prevenir sobreposição ou cortes.
     * Atualização dos títulos dos cards de comparação par-a-par (ex: `Comparação: Protocolo A Normalizado (Estimulação - Grupo CV) vs Protocolo B Normalizado (Estimulação - Grupo CF)`).
     * Atualização dos botões de download e dos nomes de arquivo salvos para refletir os rótulos corretos (ex: `Salvar Comp. A(CV) vs B(CF)` -> `topoplot_comp_ACV_vs_BCF.png`).
* **Arquivos Modificados:**
  * [`eeg-psd-dashboard/anova_engine.py`](file:///c:/Users/diego/OneDrive/Documents/GitHub/Projeto-Estimulos-Vibrot-teis/eeg-psd-dashboard/anova_engine.py)
  * [`eeg-psd-dashboard/performance_engine.py`](file:///c:/Users/diego/OneDrive/Documents/GitHub/Projeto-Estimulos-Vibrot-teis/eeg-psd-dashboard/performance_engine.py)
  * [`eeg-psd-dashboard/topoplot_engine.py`](file:///c:/Users/diego/OneDrive/Documents/GitHub/Projeto-Estimulos-Vibrot-teis/eeg-psd-dashboard/topoplot_engine.py)
  * [`eeg-psd-dashboard/psd_visualizer.py`](file:///c:/Users/diego/OneDrive/Documents/GitHub/Projeto-Estimulos-Vibrot-teis/eeg-psd-dashboard/psd_visualizer.py)
  * [`eeg-psd-dashboard/app.py`](file:///c:/Users/diego/OneDrive/Documents/GitHub/Projeto-Estimulos-Vibrot-teis/eeg-psd-dashboard/app.py)

---

### Sessão 8 (24 de Setembro de 2026) — Correção da Amostragem da Baseline (ICA), Unificação de Bandas e Topoplots
* **Tema Central:** Integração das novas baselines reprocessadas pela IC, remoção do downsample prejudicial (500 Hz -> 1000 Hz real), unificação das bandas de potência (clássicas + 2 em 2 Hz) e ajuste na exportação de Topoplots.
* **Contexto e Demandas:**
  * A aluna de IC reprocessou o ICA das baselines após ter inserido taxa de amostragem incorreta na versão anterior.
  * O usuário precisou remapear o script [`Codigos python/analise_psd_mat.py`](file:///c:/Users/diego/OneDrive/Documents/GitHub/Projeto-Estimulos-Vibrot-teis/Codigos%20python/analise_psd_mat.py) para os novos caminhos no HDD externo (`D:\Dados Bruna\TCC\Refazendo IC\filtrado - geral\arq_mat_OA` e `arq_mat_OF`).
  * Investigação e esclarecimento de que os arquivos `.set`/`.fdt` eram os formatos nativos intermediários do EEGLAB, enquanto os `.mat` com `eeg_data` já estavam presentes nas subpastas `arq_mat_OA` e `arq_mat_OF`.
* **Diagnósticos Críticos e Decisões Técnicas:**
  1. **Remoção do Downsample (`resample_poly`)**:
     * No código legado, havia `signal.resample_poly(eeg, up=1, down=2)` porque a baseline antiga marcava 2000 Hz por engano.
     * Os novos arquivos já vieram corretamente em **1000 Hz** (121.856 pontos = ~122 s). Se o `down=2` fosse mantido, a baseline cairia para 500 Hz e distorceria todo o Welch. A subamostragem foi desativada.
  2. **Unificação de Bandas de Potência (`bandas_totais`)**:
     * Em maio de 2026, a chamada de `add_bandpowers_per_channel` havia sido restringida para `bands=bandas_jean`, excluindo as bandas clássicas do EEG (`delta`, `theta`, `alfa`, `beta`, `gamma`).
     * Isso fez com que a rotina de Topoplots gerasse CSVs com 1664 linhas (bins de 2 em 2 Hz) em vez das 6 bandas esperadas pelo motor do MNE, causando `ValueError: No channels match the selection`.
     * **Solução:** Unificamos `bandas_totais = {**bandas_classicas, **bandas_jean}` para alimentar simultaneamente o histograma de 2 em 2 Hz (CDA) e as 6 bandas clássicas (Topoplots), com filtro estrito de `bandas_topo_nomes` na exportação de mapas topográficos.
* **Arquivos Modificados:**
  * [`Codigos python/analise_psd_mat.py`](file:///c:/Users/diego/OneDrive/Documents/GitHub/Projeto-Estimulos-Vibrot-teis/Codigos%20python/analise_psd_mat.py)
  * [`Agent Conversations/conversations.md`](file:///c:/Users/diego/OneDrive/Documents/GitHub/Projeto-Estimulos-Vibrot-teis/Agent%20Conversations/conversations.md)

---

### Sessão 9 (09 de Outubro de 2026) — Padronização Visual na Multivariate Analysis (Marcadores, Cores e Agrupamento da ANOVA)
* **Tema Central:** Padronização visual completa das projeções latentes (CDA/PCA/LDA/PLS) e dos testes ANOVA por dimensão latente.
* **Contexto e Demandas:**
  1. No `Color By (Visual)` e `Supervision (CDA/LDA Math)`, configurar `Group + Complexity` (`group_comp`) como padrão para Protocolos A e B, e reativo para `Complexity` (`complexity`) no Protocolo C.
  2. Nos Protocolos A e B, grupos iniciados com C (CV e CF) devem ter marcador **"X"**, e grupos iniciados com S (SV e SF) devem ter marcador de **bolinha ("Circle")**.
  3. Cores de **Easy**, **Medium** e **Hard** estritamente padronizadas e idênticas em todos os gráficos e protocolos (A, B e C): Easy = Verde (`#2ecc71`), Medium = Laranja (`#f39c12`), Hard = Vermelho (`#e74c3c`).
  4. Nos testes de ANOVA das dimensões canônicas (CDA), quando houver Grupo e Complexidade, ordenar no eixo X **primeiro por Grupo e depois por Complexidade** (ex: `CV_Easy`, `CV_Medium`, `CV_Hard` seguidos por `SV_Easy`, `SV_Medium`, `SV_Hard`), facilitando a inspeção intra-grupo e a comparação inter-grupo.
* **Decisões Técnicas e Implementação:**
  * **Callback Reativo (`update_color_options`):** Adicionado callback em `app.py` que ajusta opções e valores de `color-dropdown` dinamicamente conforme o protocolo ativo.
  * **Mapeamento de Marcadores e Cores:**
    * Em `app.py`, `symbol_map` padronizou `CV` e `CF` para `'x'` e `SV` e `SF` para `'circle'`. Traços do Plotly foram estilizados para que marcadores `'x'` mantenham espessura de 2px e coloração fiel, sem sobreposição de borda branca.
    * `COMPLEXITY_COLOR_MAP` (`Easy: #2ecc71`, `Medium: #f39c12`, `Hard: #e74c3c`) integrado em todas as projeções espectrais e no gráfico ANOVA.
  * **Ordenação e Estilização da ANOVA (`anova_engine.py`):**
    * `sort_key` reestruturado para ordenar por tupla `(group_rank, comp_rank, gs)`.
    * Cada condição plotada na ANOVA recebe a cor da sua complexidade e o símbolo do seu grupo (CV/CF com 'X' e SV/SF com bolinha).
    * Legendas dummy dinamizadas e blindadas contra erros de encoding `cp1252` no ambiente Windows.
* **Arquivos Modificados:**
  * [`eeg-psd-dashboard/app.py`](file:///c:/Users/diego/OneDrive/Documents/GitHub/Projeto-Estimulos-Vibrot-teis/eeg-psd-dashboard/app.py)
  * [`eeg-psd-dashboard/anova_engine.py`](file:///c:/Users/diego/OneDrive/Documents/GitHub/Projeto-Estimulos-Vibrot-teis/eeg-psd-dashboard/anova_engine.py)
  * [`Agent Conversations/conversations.md`](file:///c:/Users/diego/OneDrive/Documents/GitHub/Projeto-Estimulos-Vibrot-teis/Agent%20Conversations/conversations.md)

### Sessão 10 (09 de Outubro de 2026) — Tradução Completa dos Gráficos, Eixos, Legendas e UI para Inglês
* **Tema Central:** Internacionalização (Inglês) de toda a camada de visualização de dados (títulos, nomes de eixos, legendas, hover templates, opções de dropdown e cartões de resultados) em todos os gráficos do dashboard.
* **Contexto e Demandas:**
  * Solicitação do usuário para passar **tudo** o que é label, nome de eixo, legenda e elementos visuais dos gráficos para inglês:
    * Variáveis psicométricas: `Desempenho` → `Performance`, `Acurácia` → `Accuracy`, `Similaridade` → `Similarity`, `Especificidade` → `Specificity`, `Proporção espacial x/y` → `Spatial Proportion X/Y`.
    * Grupos e condições: `CV` → `V+`, `CF` → `F+`, `SV` → `V-`, `SF` → `F-`, `Ambos` → `Both`, `Grupo` → `Group`.
    * Fases e contextos: `Estimulação` → `Stimulation`, `Execução` → `Execution`, `Exploração` → `Exploration`, `Normalizado` → `Normalized`.
  * **Regra Fundamental Respeitada:** Nenhuma lógica de banco de dados, cabeçalhos de arquivos CSV ou nomes internos de colunas de DataFrames (`df['Desempenho']`, `df['grupo']`, `'CV'`, `'SV'`) foi alterada. A tradução ocorreu estritamente na camada de exibição (Plotly layouts, dropdown `label`, hover templates, anotações e tabelas de exibição).
* **Decisões Técnicas e Implementação:**
  * **`performance_engine.py`:**
    * Inclusão dos dicionários `VAR_NAME_EN` e `GRP_MAP_EN`.
    * Atualização de `format_factor_display` e `format_cat_display` mapeando `CV`/`CF` para `V+`/`F+` e `SV`/`SF` para `V-`/`F-`.
    * Eixos X e Y de Dotplots, Linhas de Interação, Gráfico Híbrido e Heatmaps atualizados para nomes em inglês (`Condition`, `Mean`, `CI`, `Group / Condition`, etc.).
    * Legendas de colchetes de significância: `Between-Group Sig. (V+/F+ <-> V-/F-)` e `Intra-Group Sig. (V+ <-> V+ / V- <-> V-)`.
    * Suporte flexível para colunas de ANOVA tanto em inglês (`Sum of Squares`, `Term`, `p-value`) quanto em português.
  * **`anova_engine.py`:**
    * Eixos e títulos das projeções CDA formatados em inglês (`val_disp by grp_disp_name`).
    * Brackets de significância com nomes e cores padronizados (`Between-Group Sig.`, `Intra-Group Sig.`).
    * Blindagem ASCII segura contra falhas de terminal Windows cp1252.
  * **`psd_visualizer.py`:**
    * Tradução dos títulos de estratificação e overlays para grupos `V+`, `V-`, `F+`, `F-`.
    * Traço de média renomeado de `PSD média` para `Mean PSD`.
    * Eixos atualizados para `Frequency (Hz)` e `Power`.
  * **`topoplot_engine.py`:**
    * Painéis nomeados em inglês (`Protocol {prot} Normalized ({fase} - Group {group})`).
    * Bandas traduzidas nos títulos (`Alpha` em vez de `alfa`).
  * **`app.py`:**
    * `Y_VARIABLES` atualizado com labels em inglês (`label: Performance, value: Desempenho`, etc.).
    * Dropdowns de Fase (`Exploration`, `Execution`, `Stimulation`), Grupos (`V+`, `V-`, `Both`), Modos de Escala de Topoplot e opções de ANOVA/Post-Hoc em inglês.
    * Títulos e modais de download de Topoplot (`📥 Save Comparison`, `📥 Save Topoplot`, `📥 Save ANOVA`).
    * Cartões de resultados estatísticos (Teste de Normalidade de Shapiro/KS, Tabela da ANOVA Tipo III e Teste $t$ de Welch) totalmente traduzidos para inglês.
* **Arquivos Modificados:**
  * [`eeg-psd-dashboard/performance_engine.py`](file:///c:/Users/diego/OneDrive/Documents/GitHub/Projeto-Estimulos-Vibrot-teis/eeg-psd-dashboard/performance_engine.py)
  * [`eeg-psd-dashboard/anova_engine.py`](file:///c:/Users/diego/OneDrive/Documents/GitHub/Projeto-Estimulos-Vibrot-teis/eeg-psd-dashboard/anova_engine.py)
  * [`eeg-psd-dashboard/psd_visualizer.py`](file:///c:/Users/diego/OneDrive/Documents/GitHub/Projeto-Estimulos-Vibrot-teis/eeg-psd-dashboard/psd_visualizer.py)
  * [`eeg-psd-dashboard/topoplot_engine.py`](file:///c:/Users/diego/OneDrive/Documents/GitHub/Projeto-Estimulos-Vibrot-teis/eeg-psd-dashboard/topoplot_engine.py)
  * [`eeg-psd-dashboard/app.py`](file:///c:/Users/diego/OneDrive/Documents/GitHub/Projeto-Estimulos-Vibrot-teis/eeg-psd-dashboard/app.py)
  * [`Agent Conversations/conversations.md`](file:///c:/Users/diego/OneDrive/Documents/GitHub/Projeto-Estimulos-Vibrot-teis/Agent%20Conversations/conversations.md)

---

## 📅 Sessão 10 - Refinamento Visual ANOVA, Redução de Pontos CDA e Correção de Overflow

* **Data/Hora:** 09/10/2026 - 15:30 (Horário de Brasília)
* **Status:** Concluído com Sucesso ✅

### 🎯 Objetivos da Sessão:
1. **Padronização Visual da ANOVA (Multivariate Analysis):**
   * Alinhar 100% o estilo dos gráficos de ANOVA das dimensões canônicas (CD1, CD2, CD3) com o padrão consagrado da aba **Performance Evaluation**.
   * Reverter a coloração por complexidade nos pontos de ANOVA, aplicando tons de azul por grupo: azul claro (`#5dade2`) para V+/F+ (`CV`/`CF`) e azul escuro (`#0d47a1`) para V-/F- (`SV`/`SF`).
   * Marcadores correspondentes aos grupos: `x` para V+/F+ e bolinha (`circle`) para V-/F-.
   * Barras de erro (IC 95%) desenhadas na cor do respectivo grupo.
   * Colchetes de significância padronizados: vermelho (`#dc3545`) para diferenças entre grupos (V+/F+ vs V-/F-), verde (`#198754`) para intra-grupo e cinza (`#333333`) neutro.
   * Legenda limpa, exibindo apenas os grupos e categorias de colchetes de significância.
2. **Dimensionamento dos Pontos no Scatter CDA/PCA/LDA:**
   * Reduzir o tamanho dos marcadores de dispersão (estavam excessivamente grandes).
   * 2D: marcador `x` reduzido de 11 para **7** (espessura 1.5) e círculo reduzido de 10 para **6** (borda 0.8).
   * 3D: marcador `x` reduzido de 8 para **5** e círculo de 6 para **4**.
3. **Resolução de Overflow do Card e Colisão de Título com Colchetes:**
   * **Causa do Overflow:** O componente `dcc.Graph` possuía `style={'height': '500px'}`, limitando a div externa enquanto o Plotly desenhava o SVG com `height = 618px` devido à altura dinâmica dos colchetes. Isso fazia o gráfico vazar para fora do card, corrigindo-se apenas quando um resize era acionado via zoom no browser.
   * **Solução:** Substituído para `minHeight: '500px'` (e `minHeight: '400px'` na visão de comparação) com `responsive=True` e `width: '100%'`, permitindo que o card abrace a altura total do gráfico no primeiro carregamento.
   * **Causa da Colisão do Título:** No `app.py`, o método `fig_a.update_layout(title=f"ANOVA: {d_name}")` sobrescrevia o dicionário de título do `anova_engine.py`, redefinindo o título para `yref='paper'` e `y ~ 1.10`, colidindo diretamente com os colchetes superiores.
   * **Solução:** Atualizado no `app.py` para manter o posicionamento `title=dict(text=..., y=0.98, yref='container', x=0.5, xanchor='center', yanchor='top')`, alocando o título no topo absoluto do canvas, acima de qualquer nível de colchetes.

### 🛠️ Modificações Realizadas:
* [`eeg-psd-dashboard/anova_engine.py`](file:///c:/Users/diego/OneDrive/Documents/GitHub/Projeto-Estimulos-Vibrot-teis/eeg-psd-dashboard/anova_engine.py):
  * Cores de grupo azuis (`#5dade2` e `#0d47a1`), marcadores por grupo (`x` e `circle`), barras IC95% na cor do grupo e legendas consolidadas.
  * Título com `yref='container', y=0.98`, margem inferior `b=80` (prevenindo corte de rótulos rotacionados a 45°).
  * **Correção de Hotfix:** Corrigido erro de interpolação no `hovertemplate` (onde `%{y:.3f}` em f-string disparava `NameError: name 'y' is not defined`), substituído pelo valor formatado da média `{m:.3f}`.
* [`eeg-psd-dashboard/app.py`](file:///c:/Users/diego/OneDrive/Documents/GitHub/Projeto-Estimulos-Vibrot-teis/eeg-psd-dashboard/app.py):
  * `run_dim_anova`: Título configurado com dicionário e `yref='container'`.
  * `update_multivariate_plot`: Marcadores 2D e 3D redimensionados para tamanhos discretos e legíveis.
  * `dcc.Graph` de ANOVA em Single View e Comparison View atualizados com `responsive=True, style={'minHeight': '500px', 'width': '100%'}, config={'responsive': True}`.
  * **Correção de Hotfix (Performance Evaluation ANOVA):** Importado `VAR_NAME_EN` do `performance_engine` no [`app.py`](file:///c:/Users/diego/OneDrive/Documents/GitHub/Projeto-Estimulos-Vibrot-teis/eeg-psd-dashboard/app.py), eliminando o `NameError: name 'VAR_NAME_EN' is not defined` que interrompia a atualização do gráfico `perf-anova-dotplot` no callback `update_posthoc_plot`.
* [`Agent Conversations/conversations.md`](file:///c:/Users/diego/OneDrive/Documents/GitHub/Projeto-Estimulos-Vibrot-teis/Agent%20Conversations/conversations.md):
  * Atualização do log da sessão.

---

## 📅 Sessão 11 (11 – 12 de Outubro de 2026) — Hotfix de Callbacks na Performance, Mapeamento Amostral Exaustivo (N e Exclusões) e Gerenciamento de Processos

* **Data/Hora:** 11/10/2026 a 12/10/2026
* **Status:** Concluído com Sucesso ✅

### 🎯 Contexto e Demandas do Usuário:
1. **Erro na ANOVA da Performance Evaluation:** O usuário reportou erro no callback ao tentar gerar o dotplot com os testes Post-Hoc na aba de Performance (`Callback error updating ..perf-anova-dotplot.figure..`).
2. **Confirmação Amostral Exaustiva ($N$ e Exclusões):** O usuário solicitou a confirmação exata do $N$ total para cada etapa (comportamento vs. EEG/CDA/Topoplot) e **quais indivíduos entram e quais são excluídos**, trial a trial e por grupo.
3. **Gerenciamento do Servidor:** Solicitação explícita para encerrar todos os processos do `eeg-psd-dashboard` e liberar a porta `localhost:8050`.

### 🔬 Diagnósticos e Decisões Técnicas:
1. **Hotfix de Importação no `app.py`:**
   * **Causa do Erro:** Na internacionalização (Sessão 10), adicionou-se a tradução `target_disp = VAR_NAME_EN.get(target_var, target_var)` no callback `update_posthoc_plot`, mas o dicionário `VAR_NAME_EN` não havia sido importado no cabeçalho do [`app.py`](file:///c:/Users/diego/OneDrive/Documents/GitHub/Projeto-Estimulos-Vibrot-teis/eeg-psd-dashboard/app.py). Isso gerava `NameError: name 'VAR_NAME_EN' is not defined`.
   * **Correção:** Importado `VAR_NAME_EN` de [`performance_engine.py`](file:///c:/Users/diego/OneDrive/Documents/GitHub/Projeto-Estimulos-Vibrot-teis/eeg-psd-dashboard/performance_engine.py) na linha 31 do [`app.py`](file:///c:/Users/diego/OneDrive/Documents/GitHub/Projeto-Estimulos-Vibrot-teis/eeg-psd-dashboard/app.py). Testado e validado com 100% de sucesso.
2. **Auditoria Amostral e Mapeamento de Exclusões:**
   * **Protocolo A:**
     * *Comportamento (Performance):* $N = 16$ voluntários ($1.296$ trials, $81$ por sujeito).
       * Grupo CV ($N = 8$): `ID_07`, `ID_10`, `ID_17`, `ID_21`, `ID_24`, `ID_31`, `ID_36`, `ID_39`.
       * Grupo SV ($N = 8$): `ID_06`, `ID_09`, `ID_16`, `ID_19`, `ID_23`, `ID_28` *(registrado como ID_27 na planilha de comportamento original)*, `ID_34`, `ID_37`.
     * *Exclusões no EEG ($N = 3$ sujeitos):* `ID_06` (SV), `ID_36` (CV) e `ID_39` (CV) foram excluídos devido a falhas de corte/sincronização temporal no sinal (`_trecho_info != 'ok'`).
     * *EEG (CDA/Topoplot):* $N = 13$ voluntários válidos ($1.053$ trials, $81$ por sujeito) em 32 canais. Grupo CV ($N = 6$): `ID_07`, `ID_10`, `ID_17`, `ID_21`, `ID_24`, `ID_31`. Grupo SV ($N = 7$): `ID_09`, `ID_16`, `ID_19`, `ID_23`, `ID_28`, `ID_34`, `ID_37`.
   * **Protocolo B:**
     * *Comportamento (Performance):* $N = 18$ voluntários ($486$ trials, $27$ por sujeito).
       * Grupo CF ($N = 9$): `ID_04`, `ID_05`, `ID_12`, `ID_18`, `ID_25`, `ID_26`, `ID_33`, `ID_38`, `ID_42`.
       * Grupo SF ($N = 9$): `ID_02`, `ID_03`, `ID_13`, `ID_15`, `ID_28`, `ID_29`, `ID_32`, `ID_40`, `ID_43`.
     * *Exclusões no EEG ($N = 8$ sujeitos):*
       * CF ($N = 4$ excluídos): `ID_04`, `ID_26`, `ID_33`, `ID_42`.
       * SF ($N = 4$ excluídos): `ID_13`, `ID_29`, `ID_40`, `ID_43`.
     * *EEG (CDA/Topoplot):* $N = 10$ voluntários balanceados ($270$ trials, $27$ por sujeito) em 32 canais. Grupo CF ($N = 5$): `ID_05`, `ID_12`, `ID_18`, `ID_25`, `ID_38`. Grupo SF ($N = 5$): `ID_02`, `ID_03`, `ID_15`, `ID_28`, `ID_32`.
   * **Protocolo C:**
     * *Comportamento (Performance):* $N = 9$ voluntários (`ID_08`, `ID_11`, `ID_14`, `ID_20`, `ID_22`, `ID_30`, `ID_35`, `ID_41`, `ID_44`). Filtro estrito de Fase Execução $\to 162$ trials ($18$ por sujeito).
     * *Exclusões no EEG:* $0$ exclusões (todos os 9 voluntários mantidos).
     * *EEG (CDA/Topoplot/Baseline):* $N = 9$ voluntários ($144$ trials na Execução, $108$ na Exploração; 30 canais no Topoplot sem CP2/CP4).
3. **Liberação de Portas e Processos:**
   * Tarefa de background `task-565` terminada.
   * Porta TCP `8050` confirmada livre e conexões recusadas com sucesso.

### 🛠️ Arquivos Modificados:
* [`eeg-psd-dashboard/app.py`](file:///c:/Users/diego/OneDrive/Documents/GitHub/Projeto-Estimulos-Vibrot-teis/eeg-psd-dashboard/app.py): Adicionada a importação de `VAR_NAME_EN` de `performance_engine`.
* [`Agent Conversations/conversations.md`](file:///c:/Users/diego/OneDrive/Documents/GitHub/Projeto-Estimulos-Vibrot-teis/Agent%20Conversations/conversations.md): Registro e consolidação da Sessão 11.

---

## 🚀 Estado Atual e Recomendações para a Próxima Sessão
1. **Dashboard 100% Funcional e Estável:**
   * ANOVA das CDAs formatada no padrão azul com colchetes coloridos e sem colisões.
   * ANOVA da Performance operando sem erros com suporte a interações simples, duplas e triplas.
2. **Porta `8050` Desocupada:**
   * Pronto para inicialização limpa quando o usuário desejar rodar novamente (`python eeg-psd-dashboard/app.py`).
3. **Documentação e Memória do Agente Sincronizada:**
   * Tanto [`Agents.md`](file:///c:/Users/diego/OneDrive/Documents/GitHub/Projeto-Estimulos-Vibrot-teis/Agent%20Conversations/Agents.md) quanto [`conversations.md`](file:///c:/Users/diego/OneDrive/Documents/GitHub/Projeto-Estimulos-Vibrot-teis/Agent%20Conversations/conversations.md) estão integralmente atualizados com todos os dados técnicos, amostrais e operacionais.



