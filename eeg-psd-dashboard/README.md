# EEG PSD Dashboard v2

**Interactive Scientific Dashboard for Multivariate Analysis of EEG Power Spectral Density (PSD) and Behavioral Metrics.**

Este projeto oferece uma suíte computacional rigorosa para a exploração multidimensional de dados neurofisiológicos oriundos de experimentos de estimulação vibrotátil e complexos labirínticos (Protocolos A, B e C).

---

## Estrutura dos Dados (Matrizes de Entrada)

A modelagem estática do painel atua sob o pressuposto de divisão matricial estrita (X e Y) baseada nas interações empíricas do voluntário:

*   **Matriz $X \in \mathbb{R}^{T \times F}$ (Features Neurofisiológicas):** Representa o domínio do sinal de Eletroencefalografia (EEG). 
    *   $T$ é o número de Trials/Épocas analisadas.
    *   $F$ é o número de *features* intruduzidas. Dependendo do recorte estabelecido no painel, pode representar escalares singulares absolutos ou relativos extratificados por bandas espectrais (Delta, Theta, Alfa, Beta, Gama) ou representar a integridade em série-temporal cruzada (Trecho Completo) contendo centenas de pontos flutuantes adjacentes aos canais (ex: CZ, C3, C4).
*   **Matriz $Y \in \mathbb{R}^{T \times C}$ (Features Comportamentais):** Representa o domínio de respostas de cognição ativa ligadas linearmente ao i-ésimo *trial*. A barreira $C$ consiste nas variáveis independentes captadas: Desempenho, Acurácia, Similaridade Espacial, etc.

## Algoritmos de Extração de Componentes (Engines)

### 1. PCA (Principal Component Analysis)
Método **não supervisionado**. O PCA atua encontrando projeções ortogonais onde a variância intrínseca global dos dados empíricos é matematicamente maximizada, diagnosticando a dispersão inata entre as features sem conhecimento prévio da classificação do participante (ex: Grupo ou Complexidade).

### 2. LDA (Linear Discriminant Analysis)
Método **supervisionado**. O foco do modelo Linear Discriminante passa da "dispersão global" (do PCA) para a maximização deliberada da variância ***entre*** classes em relação à variância ***intra*** classes. Utilizando o preenchedor de rótulos do campo "Color By", modela um hiperplano no qual as coortes são artificialmente distanciadas, propício à validação visual de agrupamentos comportamentais.

### 3. CDA (Canonical Discriminant Analysis via Identidade MATLAB)
Transformação canônica supervisionada que busca as melhores combinações lineares de uma matriz quantitativa. A formulação rigorosa foi transcrita da função teórica `manova1` do *MATLAB*.
*   O logaritmo deriva a soma de matrizes dos quadrados *Within-group* ($W$) e a matriz *Total* ($T$). 
*   Em seguida, gera uma decomposição em autovalores via Decomposição em Valores Singulares (SVD) sobre o traço $W^{-1}B$ (onde $B$ corresponde à dispersão de blocos) para encontrar auto-vetores discriminantes independentes que sumarizam variações globais entre as classes em representações ortogonais reduzidas.

### 4. PLS (Partial Least Squares Regression)
Um modelo preditivo de natureza **supervisionada projetada**. O PLS procura encontrar variáveis latentes essenciais modelando iterativamente uma matriz de covariância relacional $X^T Y$. Diferente de um PCA em X, o PLS restringe a sua explotação apenas a recortes do espaço frequencial (EEG) que efetivamente expliquem/difiram os vetores resultantes no mapeamento dos fatores do escore em $Y$.

---

## Visualização Multivariável e "Mixed Axes" (Domínios)

A área de "Data Domain" que substituiu a topologia legada possui capacidade de intercepção entre as matrizes.

*   **X Only / Y Only:** Projeção visual restrita aos componentes primários do modelo eleito extraídos estritamente em um dos ambientes isolados estatisticamente ($X$ para cérebro; $Y$ para cognição).
*   **Both (Mixed Axes) 🌐:** Libera o entrecruzamento das formulações onde o usuário seleciona livremente as ancoragens. Um Scatter plot pode ser fundado usando a Componente 1 Comportamental (PC1_Y) no eixo das abscissas cruzada matematicamente contra Componente 2 Eletrofisiológica (PC2_X) no eixo das ordenadas para uma correlação inter-domínio.

### Simbologia Paramétrica (Color By)
Se a escolha hierárquica atrelar chaves conjuntas no controle "Color By" (ex: `Complexity + Overlap`), o grafo Scatter particiona suas restrições:
1.  **Diferenciação Cromática (Cores):** Designada obrigatoriamente à superclasse (Complexidade). Componentes azuis englobam C4, vermelhas em C6, etc.
2.  **Diferenciação Geométrica (Símbolos):** Designada obrigatoriamente à subclasse iterativa. Overlap $0$ pode apresentar formato `circle` vazio, limitando fisicamente um conjunto específico no sistema sub-gráfico. O cruzamento crua automaticamente labels dinâmicas multivariáveis sem saturação.

---

## Relatório Estatístico Rigoroso (Teste de Hipóteses Oneway ANOVA)

Abaixo das análises de espaço transformado, o painel provê estatísticas cruas comportamentais parametrizadas avalizadas empiricamente:

### 1. Analysis of Variance (ANOVA 1-way)
Sempre atrelada dinamicamente à variável quantitativa de interesse (coluna $Y$) observada estritamente sobre as partições ativas de categoria selecionadas previamente. O sistema calcula a razão da variância ($F-statistic$), que divide os resíduos intragrupo da média marginal pelo resíduo global total da amostra.
*   **Condicional Lógica:** Estritamente caso $P-Value \le \alpha$ e $\alpha < 0.05$, a hipótese nula inicial (que afirma proveniência global das variáveis em populações equivalentes em média) é matematicamente rejeitada, denotando distinção formal sub-relevante e habilitando a camada preditiva analítica de *Post-Hoc*.

### 2. Post-Hoc Estendido (Tukey's HSD)
Na incidência positiva do salto estatístico anterior da ANOVA, ativa-se sistematicamente a computação do teste da "Diferença Honestamente Significativa" pareado de forma *tail-to-tail* via pacotes validados em *Statsmodels*.
*   O pipeline examina matrizes de Intervalo de Confiança 95% exclusivas entre duas parcelas amostrais pareadas (ex: Nível CV vs Subnível F ou C4x0 vs C6x0.5).
*   Se os limitantes de confiança inferior e superior **não interceptarem e cruzarem o Zero (0)**, conclui-se um salto empírico inegável na diferença pareada da subamostra.
*   Os resultados pareados conclusivos e restritos (rejeições reais da hipótese pareada independente) são mapeados em colchetes angulados interativos (Significance Brackets) e anexados à extremidade do gráfico de dispersão com anotações dinâmicas estrelares (\*\* = Diferença Estatística Detectada).

---

### Execução Local

O software opera sem depurações de retentiva pesada por `pandas` nos nós.

```bash
pip install -r requirements.txt
python app.py
```

 *(Acesso: `http://localhost:8050`)*
