> # Explicação dos arquivos desta pasta 

# `protA`- Arquivos referentes ao protocolo A
- `protA_labels.csv`: São os possíveis labels para o protocolo A. Possui as colunas:
    1. **Complexidade**: Valores da _Complexidade_ da Curva executada naquele trial. Assume 4, 6 ou 8.
    2. **Overlap**: Valores do _Overlap_ dos motores naquele trial. Assume 0.0, 0.25 ou 0.5.
    3. **grupo**: valores do _Grupo_ daquele trial. Assume 'CV' ou 'SV'.
- `protA_X_psd_norm.csv`: matriz com os valores da power spectral density dos canais CZ, C3 e C4 normalizada pela baseline. Ou seja, tem-se uma matriz da seguinte forma:

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

    onde $T$ é o númeor de trials e F é a quantidade de pontos da psd. No caso, temos $F=150$.

- `protA_Y`: matriz com os pos´siveis valores que a variável de desempenho Y pode receber. A matriz está estruturada como:
    $$
    Y =
    \left[
    \begin{array}{ccc}
    \overbrace{x_{1,Desempenho}}^{Score \ Desempenho} &
    \overbrace{x_{1,Acuracia}}^{Acuracia} &
    \overbrace{x_{1,Similaridade}}^{Similaridade} &
    \overbrace{x_{1,Especificidade}}^{Especificidade} &
    \overbrace{x_{1,Proporcão \ espacial \ x}}^{Proporcão espacial \ x} &
    \overbrace{x_{1,Proporcão \ espacial \ y}}^{Proporcão espacial \ y} 
    \\
    \vdots & \vdots & \vdots & \vdots & \vdots & \vdots \\
    \overbrace{x_{T,Desempenho}}^{Score \ Desempenho} &
    \overbrace{x_{T,Acuracia}}^{Acuracia} &
    \overbrace{x_{T,Similaridade}}^{Similaridade} &
    \overbrace{x_{T,Especificidade}}^{Especificidade} &
    \overbrace{x_{T,Proporcão \ espacial \  x}}^{Proporcão \ espacial \ x} &
    \overbrace{x_{T,Proporcão \ espacial \ y}}^{Proporcão  \ espacial \ y} 
    \end{array}
    \right]
    \in \mathbb{R}^{T \times 3F}
    $$

--- 

# `protB`- Arquivos referentes ao protocolo B
- `protB_labels.csv`: São os possíveis labels para o protocolo B. Possui as colunas:
    1. **Complexidade**: Valores da _Complexidade_ da Curva executada naquele trial. Assume 4, 6 ou 8.
    2. **grupo**: valores do _Grupo_ daquele trial. Assume 'CF' ou 'SF'.
- `protB_X_psd_norm.csv`: matriz com os valores da power spectral density dos canais CZ, C3 e C4 normalizada pela baseline. Ou seja, tem-se uma matriz da seguinte forma:

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

    onde $T$ é o númeor de trials e F é a quantidade de pontos da psd. No caso, temos $F=150$.

- `protB_Y`: matriz com os pos´siveis valores que a variável de desempenho Y pode receber. A matriz está estruturada como:
    $$
    Y =
    \left[
    \begin{array}{ccc}
    \overbrace{x_{1,Desempenho}}^{Score \ Desempenho} &
    \overbrace{x_{1,Acuracia}}^{Acuracia} &
    \overbrace{x_{1,Similaridade}}^{Similaridade} &
    \overbrace{x_{1,Especificidade}}^{Especificidade} &
    \overbrace{x_{1,Proporcão \ espacial \ x}}^{Proporcão espacial \ x} &
    \overbrace{x_{1,Proporcão \ espacial \ y}}^{Proporcão espacial \ y} 
    \\
    \vdots & \vdots & \vdots & \vdots & \vdots & \vdots \\
    \overbrace{x_{T,Desempenho}}^{Score \ Desempenho} &
    \overbrace{x_{T,Acuracia}}^{Acuracia} &
    \overbrace{x_{T,Similaridade}}^{Similaridade} &
    \overbrace{x_{T,Especificidade}}^{Especificidade} &
    \overbrace{x_{T,Proporcão \ espacial \  x}}^{Proporcão \ espacial \ x} &
    \overbrace{x_{T,Proporcão \ espacial \ y}}^{Proporcão  \ espacial \ y} 
    \end{array}
    \right]
    \in \mathbb{R}^{T \times 3F}
    $$

--- 

# `protC`- Arquivos referentes ao protocolo C
- `protC_labels.csv`: São os possíveis labels para o protocolo C. Possui as colunas:
    1. **Complexidade**: Valores da _Complexidade_ da Curva executada naquele trial. Assume 4, 6 ou 8.
- `protC_X_psd_norm.csv`: matriz com os valores da power spectral density dos canais CZ, C3 e C4 normalizada pela baseline. Ou seja, tem-se uma matriz da seguinte forma:

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

    onde $T$ é o númeor de trials e F é a quantidade de pontos da psd. No caso, temos $F=150$.

- `protC_Y`: matriz com os pos´siveis valores que a variável de desempenho Y pode receber. A matriz está estruturada como:
    $$
    Y =
    \left[
    \begin{array}{ccc}
    \overbrace{x_{1,Desempenho}}^{Score \ Desempenho} &
    \overbrace{x_{1,Acuracia}}^{Acuracia} &
    \overbrace{x_{1,Similaridade}}^{Similaridade} &
    \overbrace{x_{1,Especificidade}}^{Especificidade} &
    \overbrace{x_{1,Proporcão \ espacial \ x}}^{Proporcão espacial \ x} &
    \overbrace{x_{1,Proporcão \ espacial \ y}}^{Proporcão espacial \ y} 
    \\
    \vdots & \vdots & \vdots & \vdots & \vdots & \vdots \\
    \overbrace{x_{T,Desempenho}}^{Score \ Desempenho} &
    \overbrace{x_{T,Acuracia}}^{Acuracia} &
    \overbrace{x_{T,Similaridade}}^{Similaridade} &
    \overbrace{x_{T,Especificidade}}^{Especificidade} &
    \overbrace{x_{T,Proporcão \ espacial \  x}}^{Proporcão \ espacial \ x} &
    \overbrace{x_{T,Proporcão \ espacial \ y}}^{Proporcão  \ espacial \ y} 
    \end{array}
    \right]
    \in \mathbb{R}^{T \times 3F}
    $$