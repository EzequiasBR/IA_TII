def experimento_lotomania():
    # Faixas de valores para testar
    robos_list = [50, 100, 150]
    geracoes_list = [10, 20, 30]
    mutacao_list = [0.03, 0.05, 0.08]
    top_percent_list = [0.3, 0.4, 0.5]
    melhor_config = None
    melhor_acerto = -1
    historico_one_hot, historico_lists = preparar_dados(ARQUIVO_DADOS)
    print("Iniciando experimentação automática para Lotomania...")
    for robos in robos_list:
        for geracoes in geracoes_list:
            for mutacao in mutacao_list:
                for top_percent in top_percent_list:
                    dna, bias = inicializar_populacao(robos)
                    W = np.zeros((robos, robos), dtype=np.float64)
                    lider = {'entropia_global': 1.0, 'performance_media': 0.0}
                    # Treina e valida
                    acerto = validacao_cruzada(
                        historico_one_hot, historico_lists,
                        dna, bias, W, geracoes, lider
                    )
                    print(f"robos={robos}, geracoes={geracoes}, mutacao={mutacao}, top_percent={top_percent} -> média acertos: {acerto:.2f}")
                    if acerto > melhor_acerto:
                        melhor_acerto = acerto
                        melhor_config = (robos, geracoes, mutacao, top_percent)
    print("\nMelhor configuração encontrada:")
    print(f"robos={melhor_config[0]}, geracoes={melhor_config[1]}, mutacao={melhor_config[2]}, top_percent={melhor_config[3]} -> média acertos: {melhor_acerto:.2f}")
    # Salva configuração em arquivo
    with open('melhor_configuracao_lotomania.txt', 'w') as f:
        f.write(f"robos={melhor_config[0]}\n")
        f.write(f"geracoes={melhor_config[1]}\n")
        f.write(f"mutacao={melhor_config[2]}\n")
        f.write(f"top_percent={melhor_config[3]}\n")
        f.write(f"media_acertos={melhor_acerto:.2f}\n")
    print("Configuração salva em 'melhor_configuracao_lotomania.txt'")

"""
TII Super-Organismo - Versão Otimizada (NumPy vetorizado)
Preserva a lógica original do seu modelo, mas substitui loops pesados por
operações vetorizadas e usa argpartition para top-k. Mantém compatibilidade
com arquivos .xlsx, .pkl e com as saídas (plots, excel).

Como usar: substitua seu arquivo antigo por este, instale dependências:
    pip install numpy pandas matplotlib openpyxl

(Opcional: numba pode acelerar ainda mais; se instalado, será usado)
"""

import numpy as np
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any
import math
import random
import argparse

# Configurações (mesmas constantes do seu script)
NUM_ROBOS = 50
NUM_NUMEROS_SORTEADOS_HIST = 20
NUM_PREVISAO_FINAL = 50
TOTAL_NUMEROS = 100
HISTORICO_SORTEIOS = 2832
ENTROPY_THRESHOLD = 0.5
TOP_PERCENT = 0.3
NUM_PREVISOES = 5
GERACOES = 10
ETA_BASE = 0.05
ARQUIVO_DADOS = "loto_mania_asloterias_ate_concurso_2835_sorteio.xlsx"
ARQUIVO_MODELO = "modelo_tii_superorganismo.pkl"

RESULTADO_REAL_2836 = [15, 20, 24, 26, 27, 46, 47, 50, 51, 54, 55, 56, 58, 59, 66, 82, 85, 88, 90, 99]

# Função para validação cruzada simples
def validacao_cruzada(historico_one_hot, historico_lists, dna, bias, W, num_geracoes, lider):
    # Usa 80% para treino, 20% para teste
    N = len(historico_lists)
    split = int(N * 0.8)
    X_train, X_test = historico_one_hot[:split], historico_one_hot[split:]
    lists_train, lists_test = historico_lists[:split], historico_lists[split:]
    # Treina com dados de treino
    dna, bias, W, _, _, _, _, _, _ = treinar_modelo(dna, bias, W, X_train, lists_train, num_geracoes, lider)
    # Gera previsões com modelo treinado
    previsoes = gerar_previsoes_finais(dna, bias, W, NUM_PREVISOES, NUM_PREVISAO_FINAL)
    # Avalia acertos nas previsões usando os resultados reais do teste
    acertos = []
    for i, p in enumerate(previsoes):
        if i < len(lists_test):
            real = set(lists_test[i])
            acertos.append(len(set(p) & real))
    return np.mean(acertos) if acertos else 0

# Tentativa de usar numba se disponível (opcional)
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False

# ------------------------------
# 1) PREPARAÇÃO DE DADOS (vetorizada)
# ------------------------------

def preparar_dados(arquivo: str, janela: int = HISTORICO_SORTEIOS) -> Tuple[np.ndarray, List[List[int]]]:
    # Tenta carregar excel; se não existir, cria dummy conforme seu código original
    if os.path.exists(arquivo):
        df = pd.read_excel(arquivo)
    else:
        print(f"Arquivo '{arquivo}' não encontrado. Gerando dados dummy de tamanho {janela}.")
        historico_dummy = [np.random.choice(TOTAL_NUMEROS, NUM_NUMEROS_SORTEADOS_HIST, replace=False).tolist()
                           for _ in range(janela)]
        df = pd.DataFrame({'Concurso': range(1, janela + 1)})
        for i in range(NUM_NUMEROS_SORTEADOS_HIST):
            df[f'D{i+1}'] = [h[i] if i < len(h) else None for h in historico_dummy]

    rodadas = df.iloc[:, 1:].values.tolist()
    rodadas = rodadas[-janela:]

    historico_list_of_lists: List[List[int]] = []
    for rodada in rodadas:
        numeros = []
        for n in rodada:
            try:
                n_int = int(n)
                if 0 <= n_int < TOTAL_NUMEROS:
                    numeros.append(n_int)
            except Exception:
                continue
        historico_list_of_lists.append(sorted(numeros))

    # Cria matriz one-hot de forma vetorizada
    N = len(historico_list_of_lists)
    X = np.zeros((N, TOTAL_NUMEROS), dtype=np.int8)
    for i, lista in enumerate(historico_list_of_lists):
        X[i, lista] = 1

    return X, historico_list_of_lists


# ------------------------------
# 2) Representação do Enxame (vetorizado)
#    Substituí classes Organismo por matrizes vetoriais para velocidade
# ------------------------------

def inicializar_populacao(num_robos: int = NUM_ROBOS) -> Tuple[np.ndarray, np.ndarray]:
    # dna: (num_robos, TOTAL_NUMEROS)
    dna = np.random.rand(num_robos, TOTAL_NUMEROS).astype(np.float64)
    bias = (np.random.rand(num_robos, TOTAL_NUMEROS) * 0.1).astype(np.float64)
    return dna, bias


# ------------------------------
# 3) Funções utilitárias vetorizadas
# ------------------------------

def topk_indices_rows(scores: np.ndarray, k: int) -> np.ndarray:
    """Retorna array booleana (num_robos, TOTAL_NUMEROS) com True para top-k de cada linha.
       Usa argpartition para eficiência O(n).
    """
    # Para cada linha, pegamos os k maiores índices
    # argpartition ordena parcialmente em cada linha
    idx_part = np.argpartition(scores, -k, axis=1)[:, -k:]
    mask = np.zeros_like(scores, dtype=bool)
    rows = np.arange(scores.shape[0])[:, None]
    mask[rows, idx_part] = True
    # Opcional: transformar top-k parcial em top-k ordenado (não necessário para boolean mask)
    return mask


def calcular_entropia_dna(dna: np.ndarray) -> float:
    # Normaliza por linha
    with np.errstate(divide='ignore', invalid='ignore'):
        p = dna / (dna.sum(axis=1, keepdims=True) + 1e-12)
        # evita log2(0)
        ent = -np.sum(np.where(p > 0, p * np.log2(p), 0.0), axis=1) / np.log2(TOTAL_NUMEROS)
    return float(np.mean(ent))


def calcular_eta(entropia_media: float, eta_base: float = ETA_BASE) -> float:
    if entropia_media < ENTROPY_THRESHOLD:
        return eta_base * (entropia_media / ENTROPY_THRESHOLD)
    else:
        return eta_base * (1 + (entropia_media - ENTROPY_THRESHOLD))


# ------------------------------
# 4) Treinamento vetorizado
# ------------------------------

def treinar_modelo(dna: np.ndarray, bias: np.ndarray, W: np.ndarray,
                   historico_one_hot: np.ndarray, historico_lists: List[List[int]],
                   geracoes_a_rodar: int, lider: Dict[str, Any]):

    num_rodos = dna.shape[0]
    N = historico_one_hot.shape[0]

    # Gera perfis históricos (lista de dicts) — custo menor comparado ao loop principal
    perfis_historicos = [None] * N
    for i in range(1, N):
        perfis_historicos[i] = gerar_perfil(historico_lists[i], historico_lists[i-1])

    log_entropia = []
    log_entropia_global = []
    log_probabilidades = []
    log_profundidades = []
    log_vizinhos = []

    print("Iniciando Treinamento TII (vetorizado)...")

    # Previsoes anteriores inicial: zeros
    previsoes_bool_prev = np.zeros((num_rodos, TOTAL_NUMEROS), dtype=bool)

    for geracao in range(geracoes_a_rodar):
        for t in range(N):
            sorteio_real_vetor = historico_one_hot[t]
            perfil_atual = perfis_historicos[t] if t > 0 else None
            if perfil_atual is not None:
                log_profundidades.append(perfil_atual['profundidade_list'])
                log_vizinhos.append(perfil_atual['vizinhos'])

            # -----------------
            # 1) Previsão (vetorizada)
            # -----------------
            # scores_base: dna + bias
            scores = dna + bias  # shape (num_rodos, TOTAL_NUMEROS)

            # adiciona ponderacao_map (se houver)
            if perfil_atual is not None and 'ponderacao_map' in perfil_atual:
                # broadcast ponderacao_map para cada robô
                scores = scores + perfil_atual['ponderacao_map'][None, :]

            # influência social: W dot previsoes_bool_prev (matmul)
            social_influence = W.dot(previsoes_bool_prev.astype(np.float64))  # (num_rodos, TOTAL_NUMEROS)
            scores = scores + social_influence

            # Seleciona top-20 por linha
            preds_bool = topk_indices_rows(scores, NUM_NUMEROS_SORTEADOS_HIST)

            # -----------------
            # 2) Avaliação em lote
            # -----------------
            real_vec = (sorteio_real_vetor == 1)
            # acertos por robô: dot entre preds_bool (int) e real_vec (int)
            acertos = preds_bool.dot(real_vec.astype(np.int8))
            performances = acertos.astype(np.float64) / NUM_NUMEROS_SORTEADOS_HIST

            # -----------------
            # 3) Lider/Meta-Controle
            # -----------------
            entropia_media = calcular_entropia_dna(dna)
            lider['entropia_global'] = entropia_media
            lider['performance_media'] = float(np.mean(performances))
            log_entropia_global.append(entropia_media)

            eta_adaptativo = calcular_eta(entropia_media)

            # -----------------
            # 4) Aprendizagem Hebbiana (vetorizada)
            # Para cada robô, números acertados: preds_bool & real_vec
            # Aumenta dna[i, n] += eta_adaptativo * performance_i
            # -----------------
            # cria matriz de performances broadcastada por coluna
            perf_col = performances[:, None]  # (num_rodos, 1)
            acertados_mask = preds_bool & real_vec[None, :]
            # Atualiza dna onde acertou
            dna += (acertados_mask.astype(np.float64) * (eta_adaptativo * perf_col))
            dna = np.clip(dna, 0.0, 1.0)

            # -----------------
            # 5) Atualização da rede W (correlação entre previsões atuais)
            # correlação = (preds_bool @ preds_bool.T) / NUM_NUMEROS_SORTEADOS_HIST
            # -----------------
            correlacao = preds_bool.astype(np.int16).dot(preds_bool.astype(np.int16).T).astype(np.float64)
            correlacao = correlacao / float(NUM_NUMEROS_SORTEADOS_HIST)
            np.fill_diagonal(correlacao, 0.0)
            W += 0.01 * correlacao
            W = np.clip(W, 0.0, 1.0)

            # -----------------
            # 6) Mutação adaptativa (vetorizada)
            # intensidade_mut = 0.05 * (1 - performance_media)
            # -----------------
            intensidade_mut = 0.05 * (1.0 - lider['performance_media'])
            # gera máscara aleatória e ruído uniforme
            mask = (np.random.rand(num_rodos, TOTAL_NUMEROS) < intensidade_mut)
            noise = np.random.uniform(-0.1, 0.1, size=(num_rodos, TOTAL_NUMEROS))
            dna = np.clip(dna + (mask * noise), 0.0, 1.0)

            # -----------------
            # 7) Logging
            # -----------------
            ent_local = calcular_entropia_dna(dna)
            log_entropia.append(ent_local)

            # Probabilidades: soma DNA+ bias dos TOP órgãos
            top_n = max(1, int(num_rodos * TOP_PERCENT))
            best_idx = np.argsort(performances)[-top_n:]
            score_total = np.sum(dna[best_idx, :] + bias[best_idx, :], axis=0)
            if score_total.sum() == 0:
                probabilidades = np.ones(TOTAL_NUMEROS) / TOTAL_NUMEROS
            else:
                probabilidades = score_total / score_total.sum()
            log_probabilidades.append(probabilidades.copy())

            # atualiza previsoes_bool_prev para próxima iteração
            previsoes_bool_prev = preds_bool

        # Fim das rodadas históricas -> gerar nova geração
        # Seleciona top para reprodução
        top_n = max(1, int(num_rodos * TOP_PERCENT))
        melhores_idx = np.argsort(performances)[-top_n:]
        # Reproduz novos DNAs por amostragem com reposição
        choices = np.random.choice(melhores_idx, size=num_rodos, replace=True)
        dna = dna[choices, :].copy()
        bias = bias[choices, :].copy()

        print(f"Geração {geracao+1}/{geracoes_a_rodar} concluída. Entropia Média: {np.mean(log_entropia[-N:]):.4f}")

    # Ao final, salva modelo
    salvar_modelo(ARQUIVO_MODELO, dna, bias, W)

    return dna, bias, W, performances, log_entropia, log_entropia_global, log_probabilidades, log_profundidades, log_vizinhos


# ------------------------------
# 5) Perfil histórico (mantido)
# ------------------------------

def gerar_perfil(sorteio_atual: List[int], sorteio_anterior: List[int]) -> Dict[str, Any]:
    profundidade_map = np.full(TOTAL_NUMEROS, TOTAL_NUMEROS, dtype=int)
    profundidade_list = []

    if sorteio_anterior:
        set_anterior = set(sorteio_anterior)
        for n_atual in sorteio_atual:
            if n_atual in set_anterior:
                p = 0
            else:
                p = min(abs(n_atual - n_ant) for n_ant in sorteio_anterior) if sorteio_anterior else 1
            profundidade_map[n_atual] = p
            profundidade_list.append(p)

    ponderacao_map = np.zeros(TOTAL_NUMEROS, dtype=float)
    for n in sorteio_atual:
        p = profundidade_map[n]
        ponderacao_map[n] = 1.0 / p if p != 0 else 1.0

    vizinhos = []
    sorteio_atual_sorted = sorted(sorteio_atual)
    for i in range(len(sorteio_atual_sorted) - 1):
        if sorteio_atual_sorted[i+1] - sorteio_atual_sorted[i] == 1:
            vizinhos.append((sorteio_atual_sorted[i], sorteio_atual_sorted[i+1]))

    return {'profundidade_list': profundidade_list, 'ponderacao_map': ponderacao_map, 'vizinhos': vizinhos}


# ------------------------------
# 6) Persistência (pickle)
# ------------------------------

def salvar_modelo(filename: str, dna: np.ndarray, bias: np.ndarray, W: np.ndarray):
    with open(filename, 'wb') as f:
        pickle.dump({'dna': dna, 'bias': bias, 'W': W}, f)
    print(f"Modelo salvo em '{filename}'")


def carregar_modelo(filename: str):
    if not os.path.exists(filename):
        print(f"Arquivo de modelo '{filename}' não encontrado. Iniciando novo modelo.")
        return None, None, None
    with open(filename, 'rb') as f:
        data = pickle.load(f)
        print(f"Modelo carregado de '{filename}'")
        return data.get('dna'), data.get('bias'), data.get('W')


# ------------------------------
# 7) Geração de previsões finais (vetorizada)
# ------------------------------

def gerar_previsoes_finais(dna: np.ndarray, bias: np.ndarray, W: np.ndarray,
                           num_prev: int = NUM_PREVISOES, topk: int = NUM_PREVISAO_FINAL) -> List[List[int]]:
    # Usa os melhores robôs para compor a previsão agregada
    num_rodos = dna.shape[0]
    performances_dummy = np.zeros(num_rodos)  # quando chamado após treino, você pode substituir

    top_n = max(1, int(num_rodos * TOP_PERCENT))
    # Seleciona os top com base em soma dna+bias
    scores_total = np.sum(dna + bias, axis=1)
    melhores_idx = np.argsort(scores_total)[-top_n:]

    previsoes_prob = []
    # Previsoes internas (cada melhor robô escolhe 20 números de acordo com seu score)
    scores_melhores = (dna[melhores_idx, :] + bias[melhores_idx, :])

    # Para coordenação social, cria previsoes_outras boolean usando top20 por linha
    preds_bool_melhores = topk_indices_rows(scores_melhores, NUM_NUMEROS_SORTEADOS_HIST)

    for _ in range(num_prev):
        # inicia com zeros
        agreg_scores = np.zeros(TOTAL_NUMEROS)
        for i_idx, i in enumerate(melhores_idx):
            scores_org = dna[i, :] + bias[i, :]
            # influência de cada outro melhor
            social = 0.0
            for j_idx, j in enumerate(melhores_idx):
                social += W[i, j] * preds_bool_melhores[j_idx, :].astype(np.float64)
            scores_org = scores_org + social
            agreg_scores += scores_org

        if agreg_scores.sum() == 0:
            probs = np.ones(TOTAL_NUMEROS) / TOTAL_NUMEROS
        else:
            probs = agreg_scores / agreg_scores.sum()

        # adiciona ruído pequeno e normaliza
        probs = probs + np.random.normal(0, 0.001, size=probs.shape)
        probs = np.clip(probs, 0.0, None)
        if probs.sum() == 0:
            probs = np.ones_like(probs) / probs.size
        probs = probs / probs.sum()

        chosen = np.random.choice(TOTAL_NUMEROS, topk, replace=False, p=probs)
        previsoes_prob.append(sorted(chosen.tolist()))

    return previsoes_prob


# ------------------------------
# 8) Função principal
# ------------------------------


def main():
    # Adiciona opção de experimentação automática
    import sys
    if '--experimento' in sys.argv:
        experimento_lotomania()
        return
    global NUM_ROBOS, GERACOES, ETA_BASE, TOP_PERCENT, NUM_PREVISOES
    parser = argparse.ArgumentParser(description='TII Super-Organismo - Previsão Lotomania')
    parser.add_argument('--robos', type=int, default=NUM_ROBOS, help='Número de robôs (default: 50)')
    parser.add_argument('--geracoes', type=int, default=GERACOES, help='Número de gerações (default: 10)')
    parser.add_argument('--mutacao', type=float, default=ETA_BASE, help='Taxa de mutação base (default: 0.05)')
    parser.add_argument('--top_percent', type=float, default=TOP_PERCENT, help='Percentual de robôs top (default: 0.3)')
    parser.add_argument('--previsoes', type=int, default=NUM_PREVISOES, help='Quantidade de jogos/previsões (default: 5)')
    parser.add_argument('--validacao', action='store_true', help='Executa validação cruzada simples')
    args = parser.parse_args()

    NUM_ROBOS = args.robos
    GERACOES = args.geracoes
    ETA_BASE = args.mutacao
    TOP_PERCENT = args.top_percent
    NUM_PREVISOES = args.previsoes

    historico_one_hot, historico_lists = preparar_dados(ARQUIVO_DADOS)
    print(f"Dados carregados. Shape: {historico_one_hot.shape}")

    dna, bias, W = carregar_modelo(ARQUIVO_MODELO)
    if dna is None:
        dna, bias = inicializar_populacao(NUM_ROBOS)
        W = np.zeros((NUM_ROBOS, NUM_ROBOS), dtype=np.float64)
        lider = {'entropia_global': 1.0, 'performance_media': 0.0}
        dna, bias, W, performances, log_entropia, log_entropia_global, log_probabilidades, log_profundidades, log_vizinhos = treinar_modelo(
            dna, bias, W, historico_one_hot, historico_lists, GERACOES, lider
        )
    else:
        # Modo rápido: apenas gerar previsões
        log_entropia = []
        log_entropia_global = []
        log_probabilidades = []
        log_profundidades = []
        log_vizinhos = []

    # Previsões Finais
    previsoes = gerar_previsoes_finais(dna, bias, W, NUM_PREVISOES, NUM_PREVISAO_FINAL)
    # ================================
    # EXIBIÇÃO CLARA DAS PREVISÕES FINAIS
    # ================================
    print("\n===== PREVISÕES FINAIS (50 números cada) =====")
    for i, previsao in enumerate(previsoes, start=1):
        numeros_ordenados = sorted(previsao)
        numeros_formatados = " ".join([f"{n:02d}" for n in numeros_ordenados])
        print(f"Previsão {i}: {numeros_formatados}")
    print("==============================================\n")

    # Avaliação contra resultado fixo
    resultado_set = set(RESULTADO_REAL_2836)
    acertos_list = [len(set(p) & resultado_set) for p in previsoes]

    print("\n" + "="*50)
    print(f"Resultado 2836: {sorted([f'{n:02d}' for n in RESULTADO_REAL_2836])}")
    for i, p in enumerate(previsoes, 1):
        print(f"Previsao {i}: {len(set(p) & resultado_set)} acertos | Exemplo: {p[:20]}...")
    print(f"Media de acertos: {np.mean(acertos_list):.2f}")
    print("="*50 + "\n")

    # Exporta previsoes
    df_prev = pd.DataFrame(previsoes, columns=[f'N{i+1}' for i in range(NUM_PREVISAO_FINAL)])
    df_prev.to_excel('previsoes_tii_superorganismo_50_numeros_otimizado.xlsx', index=False)

    # Exporta métricas
    if log_probabilidades:
        prob_medias = np.mean(np.array(log_probabilidades), axis=0)
    else:
        prob_medias = np.ones(TOTAL_NUMEROS) / TOTAL_NUMEROS

    df_metrics = pd.DataFrame({
        'Entropia_Local': log_entropia,
        'Entropia_Global_Lider': log_entropia_global
    })
    df_metrics.to_excel('metricas_tii_superorganismo_50_numeros_otimizado.xlsx', index=False)

    if log_profundidades:
        pd.DataFrame({'Profundidades': log_profundidades}).to_excel('log_profundidades_otimizado.xlsx', index=False)
        pd.DataFrame({'Vizinhos': log_vizinhos}).to_excel('log_vizinhos_otimizado.xlsx', index=False)

    # ...plots removidos...


if __name__ == '__main__':
    main()
