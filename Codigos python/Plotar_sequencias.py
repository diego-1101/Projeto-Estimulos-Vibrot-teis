#%% Função para calcular o (x,y) e dif(x) e dif(y) para pdoer plotar os gráficos das trajetórias
def traj_to_points(seq):
    import numpy as np
    import matplotlib.pyplot as plt

    # Gerar (x,y) para plotar o desenho
    x = [0]
    y = [0]
    sequencia = []
    for i, num in enumerate(seq):
        match num:
            case 2:
                # Movimento para Direita
                y.append(y[-1])
                x.append(x[-1]+10)
                sequencia.append('⮕')
                #print('⮕')
                pass
            case 1:
                #Movimento para Esquerda
                y.append(y[-1])
                x.append(x[-1]-10)
                sequencia.append('⬅')
                #print('⬅')
                pass
            case 3:
                # Movimento para Cima 
                y.append(y[-1]+10)
                x.append(x[-1])
                sequencia.append('⬆')
                #print('⬆')
                pass
            case 4:
                #Movimento para Baixo
                y.append(y[-1]-10)
                x.append(x[-1])
                sequencia.append('⬇')
                #print('⬇')
                pass
            case 5:
                # Movimento Diagonal esq->dir para cima
                y.append(y[-1]+10)
                x.append(x[-1]+10)
                sequencia.append('⬈')
                #print('⬈')
                pass
            case 6:
                #Movimento Diagonal dir->esq para baixo
                y.append(y[-1]-10)
                x.append(x[-1]-10)
                sequencia.append('⬋')
                #print('⬋')
                pass
            case 7:
                # Movimento Diagonal dir->esq para cima
                y.append(y[-1]+10)
                x.append(x[-1]-10)
                sequencia.append('⬉')
                #print('⬉')
                pass
            case 8:
                #Movimento Diagonal esq->dir para baixo
                y.append(y[-1]-10)
                x.append(x[-1]+10)
                sequencia.append('⬊')
                #print('⬊')
                pass
            case 9:
                #Parado
                y.append(y[-1])
                x.append(x[-1])
                sequencia.append('Parado')
                #print('Parado')
                pass

    # Calculate vector components (differences between points)
    u = np.diff(x)  # Differences in x-coordinates
    v = np.diff(y)  # Differences in y-coordinates

    return x,y,u,v,sequencia

#%%Função para plotar a trajetória feita
def plotar_trajetoria(seq = [3, 3, 3, 1, 4, 1, 4, 1, 4, 3, 3, 3], individuo = '1'):   
    import numpy as np
    import matplotlib.pyplot as plt

    # Gerar (x,y) para plotar o desenho
    x = [0]
    y = [0]
    sequencia = []
    for i, num in enumerate(seq):
        match num:
            case 1:
                # Movimento para Direita
                y.append(y[-1])
                x.append(x[-1]+10)
                sequencia.append('⮕')
                #print('⮕')
                pass
            case 2:
                #Movimento para Esquerda
                y.append(y[-1])
                x.append(x[-1]-10)
                sequencia.append('⬅')
                #print('⬅')
                pass
            case 3:
                # Movimento para Cima 
                y.append(y[-1]+10)
                x.append(x[-1])
                sequencia.append('⬆')
                #print('⬆')
                pass
            case 4:
                #Movimento para Baixo
                y.append(y[-1]-10)
                x.append(x[-1])
                sequencia.append('⬇')
                #print('⬇')
                pass
            case 5:
                # Movimento Diagonal esq->dir para cima
                y.append(y[-1]+10)
                x.append(x[-1]+10)
                sequencia.append('⬈')
                #print('⬈')
                pass
            case 6:
                #Movimento Diagonal dir->esq para baixo
                y.append(y[-1]-10)
                x.append(x[-1]-10)
                sequencia.append('⬋')
                #print('⬋')
                pass
            case 7:
                # Movimento Diagonal dir->esq para cima
                y.append(y[-1]+10)
                x.append(x[-1]-10)
                sequencia.append('⬉')
                #print('⬉')
                pass
            case 8:
                #Movimento Diagonal esq->dir para baixo
                y.append(y[-1]-10)
                x.append(x[-1]+10)
                sequencia.append('⬊')
                #print('⬊')
                pass
            case 9:
                #Parado
                y.append(y[-1])
                x.append(x[-1])
                sequencia.append('Parado')
                #print('Parado')
                pass

    # Calculate vector components (differences between points)
    u = np.diff(x)  # Differences in x-coordinates
    v = np.diff(y)  # Differences in y-coordinates

    # Criando o gráfico
    plt.figure(figsize=(6, 6))
    plt.plot(x, y, 'o', label="Points")  # Pontos conectados
    plt.quiver(x[:-1], y[:-1], u, v, angles="xy", scale_units="xy", scale=1, color="r", label="Vectors",alpha=0.5)  # Vetores

    # Evitando sobreposição de números
    used_positions = set()  # Armazena posições já utilizadas
    for i, (xi, yi) in enumerate(zip(x, y), start=1):
        pos = (xi, yi)
        if pos in used_positions:
            # Se a posição já foi usada, desloca o número levemente
            plt.text(xi + 3, yi, str(i), fontsize=12, ha='left', va='bottom', color='blue')
        else:
            # Adiciona a posição ao conjunto e coloca o número no ponto original
            plt.text(xi, yi, str(i), fontsize=12, ha='left', va='bottom', color='blue')
            used_positions.add(pos)

    # Configurações do gráfico
    #plt.xlabel("X")
    #plt.ylabel("Y")
    plt.xticks(np.arange(min(x)-10,max(x)+11,10))
    plt.yticks(np.arange(min(y)-10,max(y)+11,10))
    plt.axhline(0, color='gray', linewidth=0.5)
    plt.axvline(0, color='gray', linewidth=0.5)
    plt.grid()
    #plt.legend()
    plt.title(f"Trajetória de {individuo}")
    plt.show()

    #printando a sequencia
    print(sequencia)

# %% Função de plotar Gabarito x Trajetória feita
def plot_comparacao(gabarito = [], seq = [], score_parcial = 0, score_total = 0):
    import matplotlib.pyplot as plt
    import numpy as np

    x1,y1,u1,v1,sequencia1 = traj_to_points(gabarito)
    x2,y2,u2,v2,sequencia2 = traj_to_points(seq)
    
    # Criando dois gráficos lado a lado
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # 1 linha, 2 colunas
    fig.text(0.5, 1.02, "Comparação Gabarito x Trajetória Executada", fontsize=16, ha='center', fontweight='bold')
    fig.text(0.5, .98, f"Score total: {score_total*100:.2f}, Score parcial: {score_parcial*100:.2f}", fontsize=14, ha='center')
    # Plotando a Trajetória do Gabarito (Eixo Esquerdo)
    axes[0].plot(x1, y1, 'o', label=sequencia1, color='green')
    axes[0].quiver(x1[:-1], y1[:-1], u1, v1, angles="xy", scale_units="xy", scale=1, color="g", alpha=0.5)
    axes[0].set_title("Trajetória - Gabarito", fontsize = 14)
    axes[0].legend()
    axes[0].grid()

    # Evitando sobreposição de números
    used_positions = set()
    for i, (xi, yi) in enumerate(zip(x1, y1), start=1):
        pos = (xi, yi)
        if pos in used_positions:
            axes[0].text(xi + 3, yi, str(i), fontsize=12, ha='left', va='bottom', color='green')
        else:
            axes[0].text(xi, yi, str(i), fontsize=12, ha='left', va='bottom', color='green')
            used_positions.add(pos)

    # Plotando a Trajetória Executada (Eixo Direito)
    axes[1].plot(x2, y2, 'o', label=sequencia2, color='blue')
    axes[1].quiver(x2[:-1], y2[:-1], u2, v2, angles="xy", scale_units="xy", scale=1, color="b", alpha=0.5)
    axes[1].set_title("Trajetória - Executado",fontsize=14)
    axes[1].legend()
    axes[1].grid()

    # Evitando sobreposição de números
    used_positions = set()
    for i, (xi, yi) in enumerate(zip(x2, y2), start=1):
        pos = (xi, yi)
        if pos in used_positions:
            axes[1].text(xi + 3, yi, str(i), fontsize=12, ha='left', va='bottom', color='blue')
        else:
            axes[1].text(xi, yi, str(i), fontsize=12, ha='left', va='bottom', color='blue')
            used_positions.add(pos)

    # Ajustando escala dos eixos para ambos os gráficos
    for ax in axes:
        ax.set_xticks(np.arange(min(min(x1), min(x2)) - 10, max(max(x1), max(x2)) + 11, 10))
        ax.set_yticks(np.arange(min(min(y1), min(y2)) - 10, max(max(y1), max(y2)) + 11, 10))
        #ax.axhline(0, color='gray', linewidth=0.5)
        #ax.axvline(0, color='gray', linewidth=0.5)

    plt.tight_layout()  # Ajusta o layout para não sobrepor elementos
    plt.show()

#%%
plot_comparacao(gabarito = [3, 3, 3, 4, 4, 4, 1, 1, 1, 3, 3, 3, 4, 4, 4],seq= [3, 3, 3, 4, 4, 4, 1, 1, 1, 3, 3, 3, 4, 4, 4])

# %% 
plotar_trajetoria([3, 3, 3, 4, 4, 4, 1, 1, 1, 3, 3, 3, 4, 4, 4])
# %%
