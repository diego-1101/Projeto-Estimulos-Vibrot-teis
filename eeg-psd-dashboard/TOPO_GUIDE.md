# 🗺️ Guia Rápido: Topoplots e Mapeamento Espacial

Este módulo permite visualizar a distribuição espacial da potência espectral (PSD) sobre o escalpo, facilitando a identificação de focos de ativação cerebral em diferentes bandas de frequência.

---

## 🛠️ Configurações das Opções

### 📈 Escala (Linear vs dB)
*   **Linear (psd_mean):** Exibe os valores brutos da potência média ($\mu \text{V}^2/\text{Hz}$). Útil para ver a magnitude real da energia.
*   **dB (psd_db_mean):** Converte os valores para escala logarítmica ($10 \cdot \log_{10}(\text{PSD})$). É a escala recomendada para EEG, pois compensa a queda natural de potência em frequências mais altas, tornando visíveis as variações em bandas como Beta e Gamma.

---

## ⚖️ Escala do Mapa (Normalização da Barra de Cores)

A escolha da escala é fundamental para uma comparação justa entre diferentes estados. Sem padronização, uma banda Delta "azul" em um gráfico pode ter mais potência que uma banda Alfa "vermelha" em outro.

| Modo de Escala | Como é calculado | Quando usar |
| :--- | :--- | :--- |
| **Global** | Busca o mínimo e máximo absoluto em **todos** os protocolos e arquivos. | Para comparar a intensidade absoluta entre experimentos diferentes. |
| **Por Protocolo** | Considera todos os dados (estimulação, execução, grupos) **apenas do protocolo atual**. | Para comparar a ativação entre fases (Estimulação vs Execução) do mesmo estudo. |
| **Por Prot. e Fase** | Considera apenas o arquivo sendo exibido (ex: Prot A - Estimulação), unificando as 6 bandas. | Para comparar a ativação relativa entre Delta, Alfa, Gamma, etc. em um único momento. |
| **Por Prot., Fase e Grupo** | Filtra os dados apenas para o **grupo selecionado** (ex: apenas CV) naquele momento. | Para análise específica de um grupo sem interferência de outros. |
| **Independente** | Cada banda (Total, Delta, etc.) calcula seu próprio limite local. | Apenas para ver a **morfologia** (formato) da ativação em bandas muito fracas. |

> [!TIP]
> **Folga de Segurança (Buffer):** Em todos os modos padronizados, o sistema adiciona uma margem de **5%** nos limites. Isso evita que os valores fiquem saturados (totalmente vermelho/azul) nas bordas e permite ver melhor os gradientes.

---

## 🔍 Enable Comparison (Modo de Comparação)

Ao ativar o **Enable Comparison**, um segundo painel de controle é habilitado.

1.  **Sincronização Automática:** Se você escolher qualquer modo de escala padronizado (Global, Protocolo, etc.), o sistema calculará um limite comum entre os dois painéis. Isso garante que o "vermelho" no Painel 1 signifique exatamente o mesmo que no Painel 2.
2.  **Statistical Difference (Diferença Estatística):** Uma terceira linha de mapas será gerada automaticamente mostrando **(Painel 1 - Painel 2)**.
    *   **Mapa de Calor:** Tons de vermelho indicam potência maior no Painel 1; tons de azul indicam potência maior no Painel 2.
    *   **Marcadores (X):** Canais onde a diferença é estatisticamente significativa (**$p < 0.05$** via *Independent T-Test*) são destacados com um **X** preto.
    *   **Escala Única:** A linha de diferença também segue a padronização das 6 bandas para que você veja onde a mudança estatística foi mais intensa.

---

## 📍 Referência de Canais
O sistema utiliza **32 canais** no padrão internacional **10-20**. Você pode consultar a posição exata de cada sensor (como CZ, F3, O1) abrindo o card "Channel Reference Map" na barra lateral.
