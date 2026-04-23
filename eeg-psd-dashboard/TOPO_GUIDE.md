# Guia Rápido - Módulo de Topoplots

Este guia explica como utilizar e interpretar as visualizações de mapas de escalpo (topoplots) no Dashboard EEG.

---

## 📊 Modos de Escala (Map Scaling)

A escala do mapa determina como as cores são distribuídas. Escolher a escala correta é fundamental para não tirar conclusões erradas:

1.  **Independente (Independent):** Cada topoplot tem sua própria escala (mínimo e máximo). 
    *   *Uso:* Bom para ver a topografia de um único estado, mas **péssimo para comparar** magnitudes entre grupos.
2.  **Por Banda (Per Band):** Todos os painéis de uma mesma banda (ex: todos os Alfas) compartilham a mesma escala.
    *   *Uso:* **Recomendado para comparações**. Permite ver se o Alfa do Grupo A é maior que o do Grupo B.
3.  **Por Protocolo (Per Protocol):** Cria uma escala única para todos os topoplots de um mesmo protocolo.
4.  **Global:** Cria uma escala mestre para todos os painéis e todas as bandas.

---

## 🤝 Opção "Ambos" e Médias Ponderadas

Ao selecionar a opção **"Ambos"** para o Grupo ou Fase, o sistema não faz uma média simples, mas sim uma **média ponderada pelo número de trials ($n$)** de cada condição.

$\text{Potência Combinada} = \frac{(\text{Média}_A \cdot n_A) + (\text{Média}_B \cdot n_B)}{n_A + n_B}$

Isso garante que condições com mais dados tenham o peso correto no resultado final.

### 💡 Dicas de Escala para o modo "Ambos":
*   **Se usar "Ambos Grupo"**: Use a **Escala por Protocolo**.
*   **Se usar "Ambos Fase"**: Use a **Escala por Protocolo** ou **Global**.
*   **Em Comparações (2-3 plots)**: Utilize obrigatoriamente a **Escala por Banda**.

---

## 🧪 Estatística Avançada

Para garantir o rigor científico, o sistema implementa métodos robustos de comparação:

1.  **Pairwise T-Test:** Compara dois painéis canal a canal.
    *   **Mapa de Calor:** Tons de vermelho indicam potência maior no primeiro painel; tons de azul indicam potência maior no segundo.
    *   **Marcadores (X):** Canais onde a diferença é estatisticamente significativa ($p < 0.05$) são destacados.
2.  **One-Way ANOVA (3 Painéis):** Gera um **Mapa ANOVA**. Ele exibe a probabilidade ($1-p$) de haver diferença significativa entre *qualquer* um dos três painéis. Áreas em vermelho intenso indicam alta probabilidade de efeito.

---

## 📍 Referência de Canais
O sistema utiliza **32 canais** no padrão internacional **10-20**. Você pode consultar a posição exata de cada sensor abrindo o card "Channel Reference Map" na barra lateral.
