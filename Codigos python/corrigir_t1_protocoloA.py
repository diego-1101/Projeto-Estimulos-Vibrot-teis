
import pandas as pd
import scipy.io as sio
import numpy as np

INPUT_CSV = "/mnt/data/df_protocoloA_tempos.csv"
INPUT_MAT = "/mnt/data/trajetorias_e_formas.mat"
OUTPUT_CSV = "/mnt/data/df_protocoloA_tempos_com_t1_corrigido.csv"

def matlab_blocks_and_duration(traj_seq, gabarito, overlap, dur=0.5):
    """
    Reproduz a lógica do protocolo_expA_1.m para obter a duração
    da estimulação sensorial no Protocolo A.

    Observação importante:
    Mantive a lógica exata do código MATLAB enviado, inclusive o detalhe
    da fórmula final que usa overlap(vet_traj(index(w))) no último termo.
    Para o conjunto de trajetórias 1..9, isso faz o último termo virar 0,
    preservando o comportamento do experimento como foi programado.
    """
    xs = []
    ys = []

    for val in traj_seq:
        pos = np.argwhere(gabarito == val)
        if len(pos) == 0:
            raise ValueError(f"Ponto {val} não encontrado no gabarito.")
        x, y = pos[0] + 1  # MATLAB é 1-indexado
        xs.append(int(x))
        ys.append(int(y))

    n = len(xs)
    blocos_x = [None] * n
    blocos_y = [None] * n

    for k in range(1, n):
        if xs[k - 1] == xs[k] and ys[k - 1] != ys[k]:
            blocos_x[k - 1] = [xs[k - 1], ys[k - 1]]
            blocos_x[k] = [xs[k], ys[k]]
        elif ys[k - 1] == ys[k] and xs[k - 1] != xs[k]:
            blocos_y[k - 1] = [xs[k - 1], ys[k - 1]]
            blocos_y[k] = [xs[k], ys[k]]

    a = [bx is None for bx in blocos_x]
    b = [by is None for by in blocos_y]

    bloco = [i + 1 for i, (aa, bb) in enumerate(zip(a, b)) if aa == bb]  # 1-indexado
    blocos = [1] + bloco + [len(blocos_x)]

    # Fórmula exatamente como o MATLAB do protocolo A efetivamente usa
    tempo_total = ((1 - overlap) * dur) * (len(blocos) - 1) + 0 * dur

    # Fórmula "pretendida", mantida só para transparência
    tempo_total_intended = ((1 - overlap) * dur) * (len(blocos) - 1) + overlap * dur

    return {
        "blocos": blocos,
        "n_blocos": len(blocos) - 1,
        "tempo_total_bug": tempo_total,
        "tempo_total_intended": tempo_total_intended,
    }

def main():
    df = pd.read_csv(INPUT_CSV)

    for col in ["Tempo 1", "Tempo 2", "Tempo 3"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    mat = sio.loadmat(INPUT_MAT, squeeze_me=True, struct_as_record=False)
    dados = mat["dados"]
    gabarito = np.array(dados.gabarito)
    trajetorias = dados.trajetorias

    rows = []
    for traj_num, traj_seq in enumerate(trajetorias, start=1):
        traj_seq = np.array(traj_seq).tolist()
        for overlap in [0.0, 0.25, 0.5]:
            res = matlab_blocks_and_duration(traj_seq, gabarito, overlap, dur=0.5)
            rows.append({
                "Número da Trajetória": traj_num,
                "Overlap": float(overlap),
                "Sequência Trajetória": str(traj_seq),
                "N pontos trajetória": len(traj_seq),
                "Blocos MATLAB": str(res["blocos"]),
                "N blocos": res["n_blocos"],
                "Duração estimulação sensorial (s)": res["tempo_total_bug"],
                "Duração estimulação sensorial fórmula pretendida (s)": res["tempo_total_intended"],
            })

    info_df = pd.DataFrame(rows)

    final = df.merge(
        info_df,
        on=["Número da Trajetória", "Overlap"],
        how="left",
        validate="many_to_one"
    )

    final["Tempo 1 Corrigido"] = final["Tempo 2"] - pd.to_timedelta(
        final["Duração estimulação sensorial (s)"], unit="s"
    )

    ordered_cols = [
        "Unnamed: 0",
        "ID",
        "Número da Trajetória",
        "Overlap",
        "Sequência Trajetória",
        "N pontos trajetória",
        "Blocos MATLAB",
        "N blocos",
        "Duração estimulação sensorial (s)",
        "Duração estimulação sensorial fórmula pretendida (s)",
        "Tempo 1",
        "Tempo 2",
        "Tempo 1 Corrigido",
        "Tempo 3",
    ]

    final = final[ordered_cols]
    final.to_csv(OUTPUT_CSV, index=False)

if __name__ == "__main__":
    main()
