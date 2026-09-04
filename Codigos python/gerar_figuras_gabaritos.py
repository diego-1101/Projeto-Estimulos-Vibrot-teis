import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
import pandas as pd
import numpy as np
import ast
import os

def decode_traj_to_coords(seq):
    """
    Decodifica uma sequência de códigos de movimento (1 a 8)
    partindo da origem (0, 0) em coordenadas discretas da grade.
    """
    x = [0]
    y = [0]
    
    # Mapeamento de direções
    # 1: Direita, 2: Esquerda, 3: Cima, 4: Baixo
    # 5: Diag Esq->Dir Cima, 6: Diag Dir->Esq Baixo, 7: Diag Dir->Esq Cima, 8: Diag Esq->Dir Baixo
    movs = {
        1: (1, 0),
        2: (-1, 0),
        3: (0, 1),
        4: (0, -1),
        5: (1, 1),
        6: (-1, -1),
        7: (-1, 1),
        8: (1, -1),
        9: (0, 0),
        0: (0, 0)
    }
    
    for num in seq:
        dx, dy = movs.get(int(num), (0, 0))
        x.append(x[-1] + dx)
        y.append(y[-1] + dy)
        
    return np.array(x), np.array(y)

def plot_single_subplot(ax, x, y):
    # Configurações de limites e estilo
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.5, 3.5)
    ax.set_aspect('equal')
    
    # Grade tracejada azul clarinha
    for grid_x in range(4):
        ax.axvline(grid_x, color='#d8e6f3', linestyle='--', linewidth=0.9, zorder=1)
    for grid_y in range(4):
        ax.axhline(grid_y, color='#d8e6f3', linestyle='--', linewidth=0.9, zorder=1)
        
    # Moldura do subplot
    for spine in ax.spines.values():
        spine.set_color('#b8c9db')
        spine.set_linewidth(1.3)
        
    ax.set_xticks([])
    ax.set_yticks([])

    # Desenhar segmentos e setas direcionais
    line_color = '#537895'
    node_color = '#3867d6'
    
    # Traçar linhas contínuas
    ax.plot(x, y, color=line_color, linewidth=2.2, zorder=2)
    
    # Adicionar setas no meio de cada segmento
    for i in range(len(x) - 1):
        x_start, y_start = x[i], y[i]
        x_end, y_end = x[i+1], y[i+1]
        
        # Ponto médio para a seta
        dx = x_end - x_start
        dy = y_end - y_start
        if dx == 0 and dy == 0:
            continue
            
        arrow = FancyArrowPatch(
            (x_start, y_start), (x_end, y_end),
            arrowstyle='-|>',
            mutation_scale=14,
            color=line_color,
            linewidth=2.2,
            zorder=3
        )
        ax.add_patch(arrow)

    # Nós intermediários
    ax.scatter(x[1:-1], y[1:-1], color=node_color, s=25, zorder=4)

    # Início (Verde)
    ax.scatter(x[0], y[0], color='#20bf6b', s=130, edgecolors='#107e43', linewidth=1.5, zorder=5)
    ax.text(x[0], y[0] - 0.28, 'Início', color='#107e43', fontsize=9.5, fontweight='bold', ha='center', va='top', zorder=6)

    # Fim (Vermelho)
    ax.scatter(x[-1], y[-1], color='#eb3b5a', s=130, edgecolors='#b71540', linewidth=1.5, zorder=5)
    ax.text(x[-1], y[-1] + 0.28, 'Fim', color='#b71540', fontsize=9.5, fontweight='bold', ha='center', va='bottom', zorder=6)

    # Numeração sequencial dos nós
    used_positions = {}
    for idx, (xi, yi) in enumerate(zip(x, y), start=1):
        pos = (xi, yi)
        count = used_positions.get(pos, 0)
        used_positions[pos] = count + 1
        
        # Pequeno offset para o texto do número não colidir
        offset_x = 0.12 if count == 0 else 0.12 + count * 0.14
        offset_y = 0.12 if count == 0 else 0.12 + count * 0.14
        
        # Se for o nó inicial ou final, ajustar posição do texto do número
        if idx == 1:
            ax.text(xi + 0.14, yi + 0.1, str(idx), fontsize=8.5, fontweight='bold', color='#2c3e50', zorder=7)
        elif idx == len(x):
            ax.text(xi + 0.14, yi + 0.1, str(idx), fontsize=8.5, fontweight='bold', color='#2c3e50', zorder=7)
        else:
            ax.text(xi + offset_x, yi + offset_y, str(idx), fontsize=8.5, fontweight='bold', color='#2c3e50', zorder=7)

def gerar_figura_grade_gabarito(csv_path, ordem_linhas, output_filename, titulo=None):
    """
    Gera a figura 3x3 dos gabaritos organizada por níveis de complexidade.
    ordem_linhas: lista com 3 listas de índices (1-based) para cada linha [Fácil, Médio, Difícil].
    """
    df = pd.read_csv(csv_path)
    # Parse das listas
    gabaritos = {}
    for col in df.columns:
        val = df[col].iloc[0]
        if isinstance(val, str):
            gabaritos[int(col)] = ast.literal_eval(val)
        else:
            gabaritos[int(col)] = list(val)
            
    fig, axes = plt.subplots(3, 3, figsize=(10, 10.5))
    fig.patch.set_facecolor('white')
    
    nomes_niveis = ['Easy', 'Medium', 'Hard']
    
    for row_idx, (nivel, indices) in enumerate(zip(nomes_niveis, ordem_linhas)):
        for col_idx, traj_num in enumerate(indices):
            ax = axes[row_idx, col_idx]
            seq = gabaritos[traj_num]
            x, y = decode_traj_to_coords(seq)
            plot_single_subplot(ax, x, y)
            
            # Adicionar o header de linha no primeiro subplot de cada linha
            if col_idx == 0:
                ax.set_ylabel(nivel, fontsize=16, fontweight='bold', color='#1e3799', labelpad=16)

    plt.tight_layout(pad=2.0)
    plt.subplots_adjust(left=0.10, right=0.96, top=0.96, bottom=0.04, hspace=0.20, wspace=0.15)
    
    fig.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Figura salva com sucesso em: {output_filename}")
    plt.close(fig)

if __name__ == '__main__':
    base_dir = os.path.dirname(__file__)
    
    # 1. Protocolo A e B (gab_seq_completa_converted.csv)
    # Linha 1: Fácil (4) -> 1, 2, 3
    # Linha 2: Médio (6) -> 4, 5, 6
    # Linha 3: Difícil (8) -> 7, 8, 9
    csv_ab = os.path.join(base_dir, 'Gabaritos', 'gab_seq_completa_converted.csv')
    ordem_ab = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    out_ab = os.path.join(base_dir, 'gabarito_integrado_protA_B.png')
    gerar_figura_grade_gabarito(csv_ab, ordem_ab, out_ab)
    
    # 2. Protocolo C (gabarito_protocolo_C.csv)
    # Linha 1: Fácil (1) -> 2, 6, 8
    # Linha 2: Médio (2) -> 3, 5, 7
    # Linha 3: Difícil (3) -> 1, 4, 9
    csv_c = os.path.join(base_dir, 'Gabaritos', 'gabarito_protocolo_C.csv')
    ordem_c = [[2, 6, 8], [3, 5, 7], [1, 4, 9]]
    out_c = os.path.join(base_dir, 'gabarito_integrado_protC.png')
    gerar_figura_grade_gabarito(csv_c, ordem_c, out_c)
