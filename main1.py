# ==============================================================================
# 動的チープトークと情報仲介モデルの大規模マルチエージェント・シミュレーション
# Dynamic Cheap Talk and Information Intermediation Model
# ==============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import shutil
from google.colab import files

def run_simulation():
    # --------------------------------------------------------------------------
    # 1. パラメータ設定 (Parameter Settings)
    # --------------------------------------------------------------------------
    T = 1000                 # シミュレーションのステップ数
    M = 1000                 # 各グリッドにおける受信者（一般層）のエージェント数
    GRID_SIZE = 50           # グリッドサーチの解像度 (50x50)
    
    # X軸：チープトークの蔓延度合（ノイズ分散 sigma_eps^2）
    # 0.0（ノイズなし）から 5.0（極めてノイズが多い）まで
    sigma_eps_sq_array = np.linspace(0.0, 5.0, GRID_SIZE)
    
    # Y軸：ファクトチェック・コスト (c_F)
    # 0.1（容易に検証可能）から 3.0（検証が極めて困難）まで
    c_F_array = np.linspace(0.1, 3.0, GRID_SIZE)
    
    # モデルの定数
    rho = 0.95               # 疲労の減衰率（忘却係数）
    alpha = 1.0              # 情報の歪みに対する疲労感受性
    K = 1.5                  # 情報回避（オプトアウト）による機会損失ペナルティ
    
    # 疲労閾値（エージェントごとに異質性を持たせるため正規分布から抽出）
    mu_F = 2.0               # 閾値の平均
    sigma_F = 0.5            # 閾値の標準偏差
    
    # --------------------------------------------------------------------------
    # 2. テンソル空間の初期化 (Tensor Initialization)
    # 形状: (len(c_F), len(sigma_eps_sq), M) = (50, 50, 1000)
    # --------------------------------------------------------------------------
    # Y軸(c_F)をaxis=0、X軸(sigma_eps_sq)をaxis=1にマッピングしてブロードキャスト
    c_F_tensor = c_F_array[:, np.newaxis, np.newaxis] 
    sigma_tensor = np.sqrt(sigma_eps_sq_array)[np.newaxis, :, np.newaxis]
    
    # エージェントの状態変数
    F = np.zeros((GRID_SIZE, GRID_SIZE, M))  # 情報的疲労 F_j(t)
    F_bar = np.random.normal(mu_F, sigma_F, (GRID_SIZE, GRID_SIZE, M)) # 許容閾値
    F_bar = np.clip(F_bar, 0.5, None)        # 閾値が負にならないようクリップ
    
    # エージェントの戦略状態（初期状態は全員「プラットフォーム依存」）
    state_Plat = np.ones((GRID_SIZE, GRID_SIZE, M), dtype=bool)
    state_Fact = np.zeros((GRID_SIZE, GRID_SIZE, M), dtype=bool)
    state_Avoid = np.zeros((GRID_SIZE, GRID_SIZE, M), dtype=bool)
    
    # プラットフォーム全体の総経済的利得を記録する配列
    total_payoff = np.zeros((GRID_SIZE, GRID_SIZE))

    print("シミュレーションを開始します... (テンソル演算により高速化されています)")
    
    # --------------------------------------------------------------------------
    # 3. 動的シミュレーション・ループ (Dynamic Simulation Loop)
    # --------------------------------------------------------------------------
    for t in range(T):
        # 毎期、標準正規分布に従うチープトークの誤差成分(epsilon)が発生
        # 誤差の絶対値 |s_it - omega_t| は |epsilon| に等しい
        epsilon = np.random.randn(GRID_SIZE, GRID_SIZE, M) * sigma_tensor
        abs_error = np.abs(epsilon)
        
        # プラットフォームに依存しているエージェントのみ疲労が蓄積・更新される
        # F_j(t+1) = rho * F_j(t) + alpha * |s_it - omega_t|
        F = np.where(state_Plat, rho * F + alpha * abs_error, F)
        
        # --- 戦略の移行判定 ---
        # 限界疲労閾値を超えた、またはプラットフォームの認知コスト(F)が代替手段(c_F, K)を上回った場合、離脱する
        leave_condition = (F > F_bar) | (F > c_F_tensor) | (F > K)
        
        # 今回新たにプラットフォームから離脱するエージェントを特定
        just_left = state_Plat & leave_condition
        
        if np.any(just_left):
            # プラットフォーム状態をFalseに更新
            state_Plat[just_left] = False
            
            # 離脱者のうち、FactとAvoidのどちらを選ぶかを利得（コスト）に基づいて決定
            # Factのコスト(c_F)がAvoidのペナルティ(K)より小さければFactを選択
            choose_fact_condition = (c_F_tensor < K)
            
            # ブロードキャストを用いて条件に合致するエージェントを割り振り
            # （just_leftのマスク内で条件判定）
            state_Fact = np.where(just_left & choose_fact_condition, True, state_Fact)
            state_Avoid = np.where(just_left & ~choose_fact_condition, True, state_Avoid)
            
        # 送信者の総アテンション（利得）を毎期加算
        total_payoff += np.sum(state_Plat, axis=2)
        
        if (t + 1) % 200 == 0:
            print(f"Step {t + 1}/{T} 完了")
            
    # --------------------------------------------------------------------------
    # 4. 結果の集計 (Aggregation)
    # --------------------------------------------------------------------------
    # 最終的な各戦略の割合を算出（エージェント軸=axis 2 で平均をとる）
    ratio_Plat = np.mean(state_Plat, axis=2)
    ratio_Fact = np.mean(state_Fact, axis=2)
    ratio_Avoid = np.mean(state_Avoid, axis=2)
    
    return ratio_Plat, ratio_Fact, ratio_Avoid, total_payoff, sigma_eps_sq_array, c_F_array

def save_and_plot_results(ratio_Plat, ratio_Fact, ratio_Avoid, total_payoff, X_vals, Y_vals):
    # 保存用ディレクトリの作成
    out_dir = "simulation_results"
    os.makedirs(out_dir, exist_ok=True)
    
    # --- CSVのエクスポート ---
    df_plat = pd.DataFrame(ratio_Plat, index=Y_vals, columns=X_vals)
    df_fact = pd.DataFrame(ratio_Fact, index=Y_vals, columns=X_vals)
    df_avoid = pd.DataFrame(ratio_Avoid, index=Y_vals, columns=X_vals)
    df_payoff = pd.DataFrame(total_payoff, index=Y_vals, columns=X_vals)
    
    df_plat.to_csv(f"{out_dir}/ratio_platform.csv")
    df_fact.to_csv(f"{out_dir}/ratio_fact_check.csv")
    df_avoid.to_csv(f"{out_dir}/ratio_avoidance.csv")
    df_payoff.to_csv(f"{out_dir}/total_payoff.csv")
    
    # --- 相図（Phase Diagram）のプロット ---
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    X, Y = np.meshgrid(X_vals, Y_vals)
    
    plots_info = [
        (axs[0, 0], ratio_Plat, "Ratio of Platform Dependence (State: Plat)", "viridis", [0, 1]),
        (axs[0, 1], ratio_Fact, "Ratio of Autonomous Fact-checking (State: Fact)", "plasma", [0, 1]),
        (axs[1, 0], ratio_Avoid, "Ratio of Information Avoidance (State: Avoid)", "magma", [0, 1]),
        (axs[1, 1], total_payoff, "Total Sender Payoff (Attention Accumulation)", "cividis", None)
    ]
    
    for ax, data, title, cmap, v_range in plots_info:
        if v_range:
            c = ax.pcolormesh(X, Y, data, cmap=cmap, shading='auto', vmin=v_range[0], vmax=v_range[1])
        else:
            c = ax.pcolormesh(X, Y, data, cmap=cmap, shading='auto')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(r"Spread of Cheap Talk / Noise Variance ($\sigma_\epsilon^2$)", fontsize=12)
        ax.set_ylabel(r"Fact-checking Cost ($c_F$)", fontsize=12)
        fig.colorbar(c, ax=ax)
        
    plt.tight_layout()
    plt.savefig(f"{out_dir}/phase_diagrams.png", dpi=300)
    plt.show()
    
    # --- ZIPファイル化とダウンロード ---
    shutil.make_archive(out_dir, 'zip', out_dir)
    print("解析が完了しました。CSVファイルとグラフをZIP形式でダウンロードします。")
    files.download(f"{out_dir}.zip")

# 実行ブロック
if __name__ == "__main__":
    res_plat, res_fact, res_avoid, res_payoff, x_ax, y_ax = run_simulation()
    save_and_plot_results(res_plat, res_fact, res_avoid, res_payoff, x_ax, y_ax)
