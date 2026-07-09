# ==============================================================================
# 送信者の異質性モデル (Heterogeneous Senders Model)
# 動的レモン市場の証明 (Proof of the Dynamic Market for Lemons)
# ==============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import shutil
import time
from google.colab import files

def run_heterogeneous_simulation():
    # --------------------------------------------------------------------------
    # 1. パラメータ設定 (Parameter Settings)
    # --------------------------------------------------------------------------
    T = 1000                 # シミュレーションのステップ数
    M = 1000                 # 各グリッドにおける受信者エージェント数
    GRID_SIZE = 50           # グリッドサーチの解像度 (50x50)
    
    # X軸：質の低い送信者(Type-L)が付加するノイズ分散 sigma_L^2 (0.0 〜 5.0)
    sigma_L_sq_array = np.linspace(0.0, 5.0, GRID_SIZE)
    
    # Y軸：プラットフォームにおける質の低い送信者(Type-L)の割合 (1% 〜 99%)
    # ※ 0%と100%はゼロ除算を防ぐため除外
    ratio_L_array = np.linspace(0.01, 0.99, GRID_SIZE)
    
    # モデルの定数
    sigma_H_sq = 0.05        # 質の高い送信者(Type-H)のノイズ分散（極めて低い固定値）
    sigma_H = np.sqrt(sigma_H_sq)
    rho = 0.95               # 疲労の減衰率
    alpha = 1.0              # 情報の歪みに対する疲労感受性
    
    # --------------------------------------------------------------------------
    # 2. テンソル空間の初期化 (Tensor Initialization)
    # --------------------------------------------------------------------------
    # ブロードキャスト用のテンソル変形
    ratio_L_tensor = ratio_L_array[:, np.newaxis, np.newaxis]
    sigma_L_tensor = np.sqrt(sigma_L_sq_array)[np.newaxis, :, np.newaxis]
    
    # エージェントの状態変数
    F = np.zeros((GRID_SIZE, GRID_SIZE, M))  # 情報的疲労 F_j(t)
    F_bar = np.random.normal(2.0, 0.5, (GRID_SIZE, GRID_SIZE, M)) # 許容閾値
    F_bar = np.clip(F_bar, 0.5, None)
    
    state_Plat = np.ones((GRID_SIZE, GRID_SIZE, M), dtype=bool) # プラットフォーム滞在フラグ
    
    # 各タイプの送信者が獲得した総アテンション
    payoff_H = np.zeros((GRID_SIZE, GRID_SIZE))
    payoff_L = np.zeros((GRID_SIZE, GRID_SIZE))

    print("送信者の異質性シミュレーションを開始します...")
    
    # --------------------------------------------------------------------------
    # 3. 動的シミュレーション・ループ (Dynamic Simulation Loop)
    # --------------------------------------------------------------------------
    for t in range(T):
        # 毎期、受信者は送信者をランダムにサンプリングする
        # U < ratio_L の場合は Type-L を、それ以外は Type-H を引く
        U = np.random.rand(GRID_SIZE, GRID_SIZE, M)
        picked_L = U < ratio_L_tensor
        
        # それぞれのタイプのノイズを発生させる
        err_H = np.random.randn(GRID_SIZE, GRID_SIZE, M) * sigma_H
        err_L = np.random.randn(GRID_SIZE, GRID_SIZE, M) * sigma_L_tensor
        
        # 受信者が引いた送信者に応じた誤差（チープトーク）を観測
        current_err = np.where(picked_L, err_L, err_H)
        
        # 疲労の蓄積（プラットフォーム滞在者のみ）
        F = np.where(state_Plat, rho * F + alpha * np.abs(current_err), F)
        
        # 限界疲労を超えたエージェントはプラットフォームからオプトアウト
        state_Plat = state_Plat & (F <= F_bar)
        
        # アテンション（利得）の集計
        # プラットフォームに残っており、かつそのタイプを引いたエージェントの数を加算
        payoff_H += np.sum(state_Plat & ~picked_L, axis=2)
        payoff_L += np.sum(state_Plat & picked_L, axis=2)
        
        if (t + 1) % 200 == 0:
            print(f"Step {t + 1}/{T} 完了")

    # --------------------------------------------------------------------------
    # 4. 指標の正規化 (Normalization)
    # --------------------------------------------------------------------------
    # 単純な総和ではなく、「送信者1人あたり、1ステップあたり」が
    # 最大ポテンシャルに対して何％のアテンションを獲得できたか（正規化利得）を算出
    max_possible_H = M * T * (1.0 - ratio_L_array[:, np.newaxis])
    max_possible_L = M * T * ratio_L_array[:, np.newaxis]
    
    norm_payoff_H = payoff_H / max_possible_H
    norm_payoff_L = payoff_L / max_possible_L
    survival_rate = np.mean(state_Plat, axis=2)

    return norm_payoff_H, norm_payoff_L, survival_rate, sigma_L_sq_array, ratio_L_array

def save_and_plot_results(norm_payoff_H, norm_payoff_L, survival_rate, X_vals, Y_vals):
    out_dir = "heterogeneous_results"
    os.makedirs(out_dir, exist_ok=True)
    
    # --- CSVエクスポート ---
    df_H = pd.DataFrame(norm_payoff_H, index=Y_vals, columns=X_vals)
    df_L = pd.DataFrame(norm_payoff_L, index=Y_vals, columns=X_vals)
    df_surv = pd.DataFrame(survival_rate, index=Y_vals, columns=X_vals)
    
    df_H.to_csv(f"{out_dir}/normalized_payoff_Type_H.csv")
    df_L.to_csv(f"{out_dir}/normalized_payoff_Type_L.csv")
    df_surv.to_csv(f"{out_dir}/platform_survival_rate.csv")
    
    # --- グラフ描画 ---
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    X, Y = np.meshgrid(X_vals, Y_vals)
    
    # (1) Type-Hの正規化利得（ヒートマップ）
    c1 = axs[0, 0].pcolormesh(X, Y, norm_payoff_H, cmap='viridis', shading='auto', vmin=0, vmax=1)
    axs[0, 0].set_title("Normalized Attention for Type-H (Honest)", fontsize=13, fontweight='bold')
    axs[0, 0].set_xlabel(r"Noise Variance of Type-L ($\sigma_{\epsilon,L}^2$)")
    axs[0, 0].set_ylabel("Proportion of Type-L Senders (ratio_L)")
    fig.colorbar(c1, ax=axs[0, 0])
    
    # (2) プラットフォームの最終生存率（ヒートマップ）
    c2 = axs[0, 1].pcolormesh(X, Y, survival_rate, cmap='plasma', shading='auto', vmin=0, vmax=1)
    axs[0, 1].set_title("Platform Final Survival Rate", fontsize=13, fontweight='bold')
    axs[0, 1].set_xlabel(r"Noise Variance of Type-L ($\sigma_{\epsilon,L}^2$)")
    axs[0, 1].set_ylabel("Proportion of Type-L Senders (ratio_L)")
    fig.colorbar(c2, ax=axs[0, 1])
    
    # (3) Type-Lが50%混在時の利得崩壊プロセス（断面図グラフ）
    mid_idx = len(Y_vals) // 2  # ratio_L ≒ 0.5 のインデックス
    axs[1, 0].plot(X_vals, norm_payoff_H[mid_idx, :], label="Type-H (Honest)", color='blue', linewidth=2)
    axs[1, 0].plot(X_vals, norm_payoff_L[mid_idx, :], label="Type-L (Noise-makers)", color='red', linestyle='--', linewidth=2)
    axs[1, 0].set_title(f"Attention Collapse (Fixed Type-L Ratio = {Y_vals[mid_idx]:.2f})", fontsize=13, fontweight='bold')
    axs[1, 0].set_xlabel(r"Noise Variance of Type-L ($\sigma_{\epsilon,L}^2$)")
    axs[1, 0].set_ylabel("Normalized Attention (per sender)")
    axs[1, 0].legend()
    axs[1, 0].grid(True)
    
    # (4) Type-Lのノイズ分散を3.0に固定した際の、割合に対する崩壊（断面図グラフ）
    high_noise_idx = np.abs(X_vals - 3.0).argmin()
    axs[1, 1].plot(Y_vals, norm_payoff_H[:, high_noise_idx], label="Type-H (Honest)", color='blue', linewidth=2)
    # raw文字列 (r"...") を使用してLaTeXエラーを回避
    axs[1, 1].set_title(r"Impact of Type-L Population Size (Fixed $\sigma_{\epsilon, L}^2 \approx 3.0$)", fontsize=13, fontweight='bold')
    axs[1, 1].set_xlabel("Proportion of Type-L Senders")
    axs[1, 1].set_ylabel("Normalized Attention for Type-H")
    axs[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{out_dir}/heterogeneous_dynamics.png", dpi=300)
    plt.show()
    
    # --- ZIP化とダウンロードの安定化 ---
    shutil.make_archive(out_dir, 'zip', out_dir)
    print("ZIPファイルの作成が完了しました。ファイルシステムの同期を待機しています...")
    time.sleep(3) # ダウンロード失敗を防ぐための待機時間
    
    try:
        files.download(f"{out_dir}.zip")
        print("ダウンロード命令を送信しました。ブラウザの通知を確認してください。")
    except Exception as e:
        print(f"自動ダウンロードに失敗しました。画面左側のフォルダアイコンから手動でダウンロードしてください。エラー詳細: {e}")

# ==============================================================================
# 実行ブロック
# ==============================================================================
if __name__ == "__main__":
    res_H, res_L, res_surv, x_ax, y_ax = run_heterogeneous_simulation()
    save_and_plot_results(res_H, res_L, res_surv, x_ax, y_ax)
