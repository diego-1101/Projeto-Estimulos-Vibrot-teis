# 🗺️ Guia Rápido: Topoplots e Mapeamento Espacial

Este módulo permite visualizar a distribuição espacial da potência espectral (PSD) sobre o escalpo, facilitando a identificação de focos de ativação cerebral em diferentes bandas de frequência.

---

## 🛠️ Configurações das Opções

### 📈 Escala (Linear vs dB)
*   **Linear (psd_mean):** Exibe os valores brutos da potência média ($\mu \text{V}^2/\text{Hz}$). Útil para ver a magnitude real da energia.
*   **dB (psd_db_mean):** Converte os valores para escala logarítmica ($10 \cdot \log_{10}(\text{PSD})$). É a escala recomendada para EEG, pois compensa a queda natural de potência em frequências mais altas, tornando visíveis as variações em bandas como Beta e Gamma.

---

## ⚖️ Escala do Mapa (Normalização da Barra de Cores)

A escolha da escala é fundamental para uma comparação justa. O sistema adapta as opções disponíveis dependendo se você está visualizando um único plot ou comparando vários.

### Quando visualizando 1 Plot:
| Modo de Escala | Descrição |
| :--- | :--- |
| **Global** | Mínimo e máximo absoluto em **todos** os protocolos. |
| **Por Protocolo** | Considera todos os dados **apenas do protocolo atual**. |
| **Por Prot. e Fase** | Unifica as 6 bandas apenas para o momento atual (ex: Prot A - Execução). |
| **Por Prot., Fase e Grupo** | Filtra os dados apenas para o **grupo selecionado** (ex: apenas CV). |
| **Independente** | Cada banda calcula seu próprio limite local (foco na morfologia). |

### Quando visualizando 2 ou 3 Plots (Comparação):
Para garantir que as cores sejam comparáveis entre os painéis, oferecemos dois modos globais:
*   **Escala por banda (Global por Banda):** Sincroniza a escala de cada frequência (ex: Delta) entre todos os painéis ativos. O "Delta" do Painel 1 terá a mesma escala que o "Delta" do Painel 2 e 3. Útil para comparar a evolução de uma banda específica em diferentes contextos.
*   **Escala comum (Global Absoluto):** Cria uma única escala mestre para **todas as bandas** e **todos os painéis**. Isso permite ver, por exemplo, se a potência Alfa em um protocolo é genuinamente maior que a potência Beta em outro.

---

## 🤝 Opção "Ambos" e Médias Ponderadas

Ao selecionar a opção **"Ambos"** para o Grupo ou Fase, o sistema não faz uma média simples, mas sim uma **média ponderada pelo número de trials ($n$)** de cada condição.

$$ \text{Potência Combinada} = \frac{(\text{Média}_A \cdot n_A) + (\text{Média}_B \cdot n_B)}{n_A + n_B} $$

Isso garante que condições com mais dados (ex: uma fase de execução mais longa ou um grupo com mais participantes) tenham o peso correto no resultado final, evitando distorções estatísticas.

### 💡 Dicas de Escala para o modo "Ambos":
*   **Se usar "Ambos Grupo"**: Use a **Escala por Protocolo**. Isso permite comparar a ativação média do protocolo com estados de baseline ou outros protocolos de forma justa.
*   **Se usar "Ambos Fase"**: Use a **Escala por Protocolo** ou **Global**. Como você está olhando para a "sessão total", escalas mais amplas ajudam a contextualizar essa potência média dentro do experimento completo.
*   **Em Comparações (2-3 plots)**: Se um dos painéis for "Ambos" e o outro for um grupo específico, utilize obrigatoriamente a **Escala por Banda**. Isso evita que a magnitude naturalmente menor de uma média (Ambos) seja visualmente "engolida" por um grupo de alta ativação.

---

## 🔍 Comparação de Múltiplos Painéis

O dashboard permite exibir até **3 painéis** simultâneos para análise comparativa profunda.

1.  **Pairwise Comparison (Todos contra Todos):** Ao selecionar 2 ou 3 plots, o sistema gera automaticamente mapas de diferença estatística para todas as combinações possíveis (1 vs 2, 1 vs 3, 2 vs 3).
2.  **Diferença Estatística:** 
    *   **Mapa de Calor:** Tons de vermelho indicam potência maior no primeiro painel da dupla; tons de azul indicam potência maior no segundo.
    *   **Marcadores (X):** Canais onde a diferença é estatisticamente significativa (**$p < 0.05$** via *Independent T-Test*) são destacados.
    *   **Painel de Detalhes:** Abaixo de cada comparação, você pode expandir os "Detalhes Estatísticos" para ver os p-values exatos de cada canal significativo por banda.

---

## 📍 Referência de Canais
O sistema utiliza **32 canais** no padrão internacional **10-20**. Você pode consultar a posição exata de cada sensor (como CZ, F3, O1) abrindo o card "Channel Reference Map" na barra lateral.
