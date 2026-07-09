# ==============================================================================
# 現代アテンション・エコノミーの高度な課題検証シミュレーション
# Task 1: アルゴリズムによる選択的接触 (Algorithmic Amplification)
# Task 2: 進化ゲーム動学による戦略適応 (Evolutionary Dynamics)
# Task 3: 制度的介入と遅延効果 (Community Notes & Maximum Tolerable Delay)
# ==============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import shutil
import time
from google.colab import files

def setup_directories():
    out_dir = "advanced_simulation_results"
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

# ------------------------------------------------------------------------------
# Task 1: アルゴリズムによる選択的接触
# ------------------------------------------------------------------------------
def run_task1_algorithmic_amplification(out_dir):
    print("Task 1: アルゴリズムの選択的接触をシミュレート中...")
    T = 1000; M = 2000; GRID = 50
    sigma_L_sq_vals = np.linspace(0.0, 5.0, GRID)
    ratio_L = 0.5
    sigma_H = np.sqrt(0.05)
    
    # テンソル化 (sigma_L を次元0、エージェントを次元1)
    sigma_L_tensor = np.sqrt(sigma_L_sq_vals)[:, np.newaxis]
    
    def simulate_task1(use_algorithm):
        F = np.zeros((GRID, M))
        F_bar = np.random.normal(2.0, 0.5, (GRID, M)).clip(0.5, None)
        state_Plat = np.ones((GRID, M), dtype=bool)
        
        for t in range(T):
            if use_algorithm:
                # センセーショナルな投稿（ノイズ分散が大きい）ほどアルゴリズムがブーストする
                kappa = 1.0 # ブースト係数
                weight_L = ratio_L * (1 + kappa * sigma_L_sq_vals)
                weight_H = (1 - ratio_L) * (1 + kappa * 0.05)
                prob_L = weight_L / (weight_L + weight_H)
            else:
                prob_L = np.full(GRID, ratio_L)
                
            prob_L_tensor = prob_L[:, np.newaxis]
            picked_L = np.random.rand(GRID, M) < prob_L_tensor
            
            err_H = np.random.randn(GRID, M) * sigma_H
            err_L = np.random.randn(GRID, M) * sigma_L_tensor
            current_err = np.where(picked_L, err_L, err_H)
            
            F = np.where(state_Plat, 0.95 * F + 1.0 * np.abs(current_err), F)
            state_Plat = state_Plat & (F <= F_bar)
            
        return np.mean(state_Plat, axis=1)

    surv_base = simulate_task1(use_algorithm=False)
    surv_algo = simulate_task1(use_algorithm=True)
    
    df = pd.DataFrame({'Sigma_L_sq': sigma_L_sq_vals, 'Baseline_Random': surv_base, 'Algorithmic_Boost': surv_algo})
    df.to_csv(f"{out_dir}/task1_algorithmic_amplification.csv", index=False)
    
    plt.figure(figsize=(8, 5))
    plt.plot(sigma_L_sq_vals, surv_base, label="Baseline (Random Sampling)", color='blue', linewidth=2)
    plt.plot(sigma_L_sq_vals, surv_algo, label="Algorithmic Amplification", color='red', linestyle='--', linewidth=2)
    plt.title("Task 1: Impact of Algorithmic Amplification on Platform Survival", fontweight='bold')
    plt.xlabel(r"Noise Variance of Type-L ($\sigma_{\epsilon,L}^2$)")
    plt.ylabel("Platform Final Survival Rate")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{out_dir}/task1_plot.png", dpi=300)
    plt.close()

# ------------------------------------------------------------------------------
# Task 2: 送信者の進化ゲーム動学 (Replicator Dynamics)
# ------------------------------------------------------------------------------
def run_task2_evolutionary_dynamics(out_dir):
    print("Task 2: 進化ゲーム動学による戦略適応をシミュレート中...")
    T = 1500; M = 5000
    ratio_L = 0.05 # 初期は質の低い送信者が5%のみ
    sigma_H = np.sqrt(0.05); sigma_L = np.sqrt(3.0)
    
    F = np.zeros(M)
    F_bar = np.random.normal(2.0, 0.5, M).clip(0.5, None)
    state_Plat = np.ones(M, dtype=bool)
    
    history_ratio_L = []; history_survival = []
    
    kappa = 1.0 # アルゴリズムのエンゲージメント評価係数
    learning_rate = 0.05 # 送信者の戦略更新スピード
    
    for t in range(T):
        history_ratio_L.append(ratio_L)
        history_survival.append(np.mean(state_Plat))
        
        # 1. 受信者の情報処理
        weight_L = ratio_L * (1 + kappa * (sigma_L**2))
        weight_H = (1 - ratio_L) * (1 + kappa * (sigma_H**2))
        prob_L = weight_L / (weight_L + weight_H) if (weight_L + weight_H) > 0 else ratio_L
        
        picked_L = np.random.rand(M) < prob_L
        err_H = np.random.randn(M) * sigma_H
        err_L = np.random.randn(M) * sigma_L
        current_err = np.where(picked_L, err_L, err_H)
        
        F = np.where(state_Plat, 0.95 * F + 1.0 * np.abs(current_err), F)
        state_Plat = state_Plat & (F <= F_bar)
        
        # 2. レプリケーター方程式による戦略適応（Type-HがType-Lへと寝返る）
        payoff_L = 1 + kappa * (sigma_L**2)
        payoff_H = 1 + kappa * (sigma_H**2)
        avg_payoff = ratio_L * payoff_L + (1 - ratio_L) * payoff_H
        
        # 送信者がプラットフォームに残っている聴衆を奪い合う進化動学
        ratio_L = ratio_L + learning_rate * ratio_L * (payoff_L - avg_payoff)
        ratio_L = np.clip(ratio_L, 0.01, 0.99)
        
    df = pd.DataFrame({'Step': range(T), 'Ratio_Type_L': history_ratio_L, 'Platform_Survival': history_survival})
    df.to_csv(f"{out_dir}/task2_evolutionary_dynamics.csv", index=False)
    
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(range(T), history_ratio_L, label="Proportion of Type-L", color='red', linewidth=2)
    ax1.set_xlabel("Time (Steps)")
    ax1.set_ylabel("Proportion of Type-L Senders", color='red')
    ax1.tick_params(axis='y', labelcolor='red')
    
    ax2 = ax1.twinx()
    ax2.plot(range(T), history_survival, label="Platform Survival", color='blue', linestyle='--', linewidth=2)
    ax2.set_ylabel("Platform Survival Rate", color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    
    plt.title("Task 2: Gresham's Law via Replicator Dynamics", fontweight='bold')
    fig.tight_layout()
    plt.grid(True)
    plt.savefig(f"{out_dir}/task2_plot.png", dpi=300)
    plt.close()

# ------------------------------------------------------------------------------
# Task 3: 制度的介入（コミュニティノート）の遅延効果
# ------------------------------------------------------------------------------
def run_task3_community_notes(out_dir):
    print("Task 3: コミュニティノートの遅延効果をシミュレート中...")
    T = 1000; M = 1000
    delta_vals = np.arange(0, 51, 2) # 遅延ステップ数 (0〜50)
    sigma_L_sq_vals = np.linspace(0.0, 5.0, 26)
    ratio_L = 0.5; sigma_H = np.sqrt(0.05)
    
    survival_matrix = np.zeros((len(delta_vals), len(sigma_L_sq_vals)))
    
    for i, delta in enumerate(delta_vals):
        sigma_L_tensor = np.sqrt(sigma_L_sq_vals)[:, np.newaxis]
        F = np.zeros((len(sigma_L_sq_vals), M))
        F_bar = np.random.normal(2.0, 0.5, (len(sigma_L_sq_vals), M)).clip(0.5, None)
        state_Plat = np.ones((len(sigma_L_sq_vals), M), dtype=bool)
        
        # 遅延回復のためのリングバッファ
        buffer_L_err = []
        
        for t in range(T):
            picked_L = np.random.rand(len(sigma_L_sq_vals), M) < ratio_L
            err_H = np.random.randn(len(sigma_L_sq_vals), M) * sigma_H
            err_L = np.random.randn(len(sigma_L_sq_vals), M) * sigma_L_tensor
            current_err = np.where(picked_L, err_L, err_H)
            
            # Type-Lのノイズ量を記録
            buffer_L_err.append(np.where(picked_L, np.abs(err_L), 0.0))
            
            # 疲労の蓄積
            F = np.where(state_Plat, 0.95 * F + 1.0 * np.abs(current_err), F)
            
            # コミュニティノートによる事後回復（deltaステップ前のダメージを無効化）
            if len(buffer_L_err) > delta:
                recovery = buffer_L_err.pop(0)
                # 注：すでに閾値を超えて離脱した者は回復できない（致命的遅延）
                F = np.maximum(0, F - 1.0 * recovery)
                
            # 限界疲労を超えたら即座にオプトアウト（手遅れになる）
            state_Plat = state_Plat & (F <= F_bar)
            
        survival_matrix[i, :] = np.mean(state_Plat, axis=1)
        
    df = pd.DataFrame(survival_matrix, index=delta_vals, columns=sigma_L_sq_vals)
    df.to_csv(f"{out_dir}/task3_community_notes_delay.csv")
    
    plt.figure(figsize=(8, 6))
    X, Y = np.meshgrid(sigma_L_sq_vals, delta_vals)
    c = plt.pcolormesh(X, Y, survival_matrix, cmap='plasma', shading='auto', vmin=0, vmax=1)
    plt.colorbar(c, label="Platform Survival Rate")
    plt.title("Task 3: Maximum Tolerable Delay of Community Notes", fontweight='bold')
    plt.xlabel(r"Noise Variance of Type-L ($\sigma_{\epsilon,L}^2$)")
    plt.ylabel(r"Fact-check Delay ($\delta$ steps)")
    plt.savefig(f"{out_dir}/task3_plot.png", dpi=300)
    plt.close()

# ------------------------------------------------------------------------------
# 実行とZIP化
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    out_dir = setup_directories()
    
    # 各タスクの実行
    run_task1_algorithmic_amplification(out_dir)
    run_task2_evolutionary_dynamics(out_dir)
    run_task3_community_notes(out_dir)
    
    # ZIP化とダウンロード
    print("すべての計算が完了しました。ZIPファイルを作成します...")
    shutil.make_archive(out_dir, 'zip', out_dir)
    time.sleep(3)
    
    try:
        files.download(f"{out_dir}.zip")
        print("ダウンロード命令を送信しました。")
    except Exception as e:
        print(f"自動ダウンロードに失敗しました。手動で '{out_dir}.zip' をダウンロードしてください。エラー詳細: {e}")
