# EEG PSD Dashboard v2 - Guia Rápido de Uso 🧠📊

Este guia rápido tem como objetivo orientar pesquisadores na utilização do Painel Interativo de Análise Multivariável de EEG, explicando o propósito de cada opção selecionável na interface e fundamentando as modelagens matemáticas subjacentes no processamento dos dados.

---

## 1. Configurações Globais

### Protocolo (Protocol)
Selecione a fase experimental do estudo.
*   **Protocol A / B:** Modos principais de análise onde existem categorias comportamentais bem definidas (Grupo) com ou sem sobreposição (Overlap).
*   **Protocol C:** Focado em trajetórias contínuas sem grupo restrito. O sistema adota automaticamente o rótulo "ALL" para processar extrações de forma íntegra.

### Dimensões Espaciais (Dimensions)
Define se as extrações matemáticas (ex: PCA1, PCA2, PCA3) serão renderizadas num plano Euclidiano padrão de **2D** (gráfico de dispersão clássico) ou se alocadas num ambiente tridimensional interativo **3D**.

### Modo de Comparação (Comparison Mode)
Habilita um *layout* *split-screen* inovador. Ele "clona" a barra lateral de configurações, permitindo que você instancie matrizes de recortes diferentes de forma conjunta. Por exemplo: você pode modelar o "Protocolo A" via PCA do lado esquerdo para debugar o perfil de onda delta, enquanto roda um PLS puramente relacional Comportamental no lado direito da tela.

---

## 2. Parâmetros da Construção Matricial

As análises multivariáveis no dashboard são puramente calcadas na geração em tempo real das matrizes base $X$ (Sinal Eletrofisiológico de Eletroencefalografica) e $Y$ (Métricas Comportamentais) sobre a dimensão transpassada dos $T$ Trials. Cada seletor define ativamente a engenharia da matriz antes de jogá-la para o teste.

### Categoria X (PSD Features)
Power Spectral Density (PSD) dos sinais do EEG registrados em $2.000 Hz$ e subamostrados em $1.000 Hz$. São três opções disponíveis de forma de visualizar as features ($F$). Em todos os cenários $X \in \mathbb{R}^{T \times F}$, em que a formação matricial individual se comporta como:

$$
X = \begin{bmatrix}
x_{11} & x_{12} & \dots & x_{1F} \\
x_{21} & x_{22} & \dots & x_{2F} \\
\vdots & \vdots & \ddots & \vdots \\
x_{T1} & x_{T2} & \dots & x_{TF}
\end{bmatrix}
$$

1.  **PSD trecho completo (psd_full):**
    Aqui as bandas são deixadas de lado. O vetor $X$ se apoia sobre o formato da onda contínua integral de CZ, C3 e C4 concatenada horizontalmente. A quantidade de Features $F$ engloba mais de $3.000$ pontos colineares. ($F = \sim3075$)
    $$
    X =
    \left[
    \begin{array}{ccc}
    \overbrace{x_{1,C_z,1} \;\cdots\; x_{1,C_z,F}}^{C_z} &
    \overbrace{x_{1,C_3,1} \;\cdots\; x_{1,C_3,F}}^{C_3} &
    \overbrace{x_{1,C_4,1} \;\cdots\; x_{1,C_4,F}}^{C_4}
    \\
    \vdots & \vdots & \vdots \\
    \overbrace{x_{T,C_z,1} \;\cdots\; x_{T,C_z,F}}^{C_z} &
    \overbrace{x_{T,C_3,1} \;\cdots\; x_{T,C_3,F}}^{C_3} &
    \overbrace{x_{T,C_4,1} \;\cdots\; x_{T,C_4,F}}^{C_4}
    \end{array}
    \right]
    \in \mathbb{R}^{T \times 3F}
    $$

    onde

    $$
    x_i =
    \left[
    \overbrace{x_{i,C_z,1} \;\cdots\; x_{i,C_z,F}}^{C_z} \;
    \overbrace{x_{i,C_3,1} \;\cdots\; x_{i,C_3,F}}^{C_3} \;
    \overbrace{x_{i,C_4,1} \;\cdots\; x_{i,C_4,F}}^{C_4}
    \right]
    \in \mathbb{R}^{1 \times 3F}
    $$


2.  **PSD estratificada (não-normalizada):**
    Seleciona apenas os valores unitários e unificados referentes às 5 faixas espectrais do EEG (Delta, Theta, Alfa, Beta e Gama) coletados estritamente sobre 3 canais cruciais (CZ, C3, C4). A matriz final é definida globalmente operando sob exatamente **15 Features** representativas quantificadas em escalas micro-volt globais. ($F = 15$)
    $${
    X = \begin{bmatrix}
    x_{11} & x_{12} & \dots & x_{1F} \\
    x_{21} & x_{22} & \dots & x_{2F} \\
    \vdots & \vdots & \ddots & \vdots \\
    x_{T1} & x_{T2} & \dots & x_{TF}
    \end{bmatrix}
    \in \mathbb{R}^{T \times F}
    }$$


3.  **PSD estratificada (normalizada) - *Padrão***
    Comporta a mesma estrutura base de 15 features correspondente as 5 faixas espectrais em cada um dos 3 principais canais (CZ, C3, C4), porém com os valores normalizados pelos sinais de _baseline_ daquele indivíduo($x_{norm} = x_{ij} / x_{i,baseline}$). ($F = 15$)

    $${
    X = \begin{bmatrix}
    x_{11} & x_{12} & \dots & x_{1F} \\
    x_{21} & x_{22} & \dots & x_{2F} \\
    \vdots & \vdots & \ddots & \vdots \\
    x_{T1} & x_{T2} & \dots & x_{TF}
    \end{bmatrix}
    \in \mathbb{R}^{T \times F}
    }$$



### Categoria Y (Behavioral Features)
Quadro de seleção múltipla em *checkboxes*. Vetor de comportamento. Representa a matriz alvo para calibração, englobando $C$ dimensões ($Y \in \mathbb{R}^{T \times C}$). 

$$
Y = \begin{bmatrix}
y_{11} & \dots & y_{1C} \\
y_{21} & \dots & y_{2C} \\
\vdots & \ddots & \vdots \\
y_{T1} & \dots & y_{TC}
\end{bmatrix}
$$
Selecione os escores específicos de calibração que serão confrontados ou cruzados. O modelo baseia-se fortemente na métrica de "Desempenho" ($\text{Desempenho}= \frac{\text{Acurácia} + \text{Similaridade} + \text{Especificidade}}{3}$), mas aceita métricas de Acurácia, Similaridade e Especificidade e de Proporção Espacial X e Y.

---

## 3. O Motor Analítico (Analysis Method)

Qual será o método utilizado para calcular o plano de visualização dos dados. Pode ser:
*   **PCA:** Método de redução padrão. Analisa de forma isolada *ou* $X$ *ou* $Y$. 
*   **LDA:** Linear Discriminante. 
*   **CDA *(Canônico)***: Canocial Discriminant Analysis imitando fielmente o módulo [`manova1`](https://www.mathworks.com/help/stats/manova1.html).
*   **PLS *(Projected Least Squares)***: Aplica o método [`PLSRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html).

---

## 4. Modelagem dos Gráficos Interativos

### Eixos Conjugados (Data Domain)
Determina visualmente qual grupo de Coordenadas povoarão o gráfico de Dispersão no Dashboard:
*   **X Only / Y Only:** Plota exclusivamente o comportamento reduzido matemático das matrizes únicas selecionadas e analisadas.
*   **Both (Mix Axes):** *Feature Exclusiva*. Ao selecionar "Both", o dashboard permite intercepção manual nos eixos. É possível fixar a linha do Eixo-X global do gráfico monitorando a Componente 2 Eletrofisiológica ($PC2\_X$) ao mesmo passo em que fixa a linha do Eixo-Y no monitoramento da Componente 1 Comportamental ($PC1\_Y$), forçando o usuário a ler visualmente uma fusão estatística direta.

### Mapeamento Multivariável Paramétrico (Color By)
No gráfico *Scatter*, os pontos individuais carregam *Simbologia Dupla Inteligente*. Quando se mapeia um label dimensional simples, a **cor** muda. Porém, mapeamentos paralelos aplicam regras separadas cruciais para a inspeção de alta-dimensionalidade:
1.  **Diferenciação Cromática (Cores):** Designada obrigatoriamente à superclasse (Complexidade / Group). Componentes azuis englobam (C4 ou CV) enquanto componentes vermelhos sinalizam classes conflitantes primais.
2.  **Diferenciação Geométrica (Símbolos):** Designada obrigatoriamente à subclasse iterativa (Overlap). Assim, C4 em Overlap 0 pode apresentar formato `circle` vazio (O), mas C4 em Overlap 0.5 transforma aquela cor azul num Losango ou Quadrado, retificando o cruzamento com maestria estatística visual instantânea.

---

## 5. Módulo Estatístico 

No painel interior, o painel provê a análise estatística:

### Analysis of Variance (ANOVA 1-way)
Sempre atrelada dinamicamente ao "Color By" na sua base, e à sua "Variável de Interesse" ($Y$) definida na barra lateral. O código calcula de forma assíncrona o valor da Distribuição-$F$. Caso encontre p-Valor estatisticamente contundente ($P \le \alpha$ onde $\alpha = 0.05$), ativa gatilhos interativos secundários para a quebra de classes.

### Post-Hoc Estendido (Tukey's HSD)
Em incidências críticas de rejeição estatística positiva detectada pela ANOVA, faz-se um post-Hoc.
*   A matemática pareia isoladamente todos os cruzamentos visuais existentes no gráfico (ex: CVxSV ou C4,O0 vs C6,O0).
*   Limites de Interceptação da Confiança estrita de 95% ($IC95$) determinam se a diferença de distribuição falha em cruzar o Zero (0 absoluto).
*   Geração visual autônoma embutida de *Brackets de Significância*, onde os limites marginais com diferenças validadas são apontados graficamente com estrelas de correlação significante ao longo da margem visual no painel do Dashboard.
