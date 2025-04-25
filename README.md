# Multi-Armed-Bandit-MAB

# HW3: Explore and Exploit for Arm-Bandit Problem

## 🎮 環境介紹：多臂機器人（Multi-Armed Bandit）

我們定義一個 K 臂的多臂機器人環境，專為測試探索策略在難以分辨的 arm 情境下的表現。環境設計讓大多數 arm 的期望報酬極為接近，僅有一個最佳 arm 稍微優於其他，適合作為評估 Thompson Sampling 等策略的基準。目標是透過演算法在有限次試驗內識別最佳 arm 並最大化總報酬（cumulative reward）。
 - 每個 arm 的獎勵服從常態分布 N(μ_i, 1)
 - 總試驗次數：10000 次
 - arms 數量：30 臂
 - 真實期望值 μ_i 為從 N(0, 0.01) 隨機生成
 - 隨機指定一個最佳 arm，其期望值額外加上 0.25

![image](https://github.com/user-attachments/assets/ff0e27d3-48a6-4c3a-8ea2-fa901512d5c2)

以下是 Python 的模擬環境程式碼：

```python
import numpy as np
import matplotlib.pyplot as plt

class BanditEnv:
    def __init__(self, k=30, random_seed=None):
        self.k = k
        # 設置隨機種子以保證每次執行結果一致
        if random_seed is not None:
            np.random.seed(random_seed)
        # 生成真實回報，使 0 号 arm 最高，逐渐递减到负数
        self.true_means = np.linspace(40, -40, k)

    def pull(self, arm):
        return np.random.normal(self.true_means[arm], 1)

# 設定隨機種子
random_seed = 10

# 設定環境，並固定隨機種子
env = BanditEnv(k=30, random_seed=random_seed)

# 繪製每個 arm 的真實價值
plt.figure(figsize=(10, 6))
plt.bar(range(env.k), env.true_means, color='skyblue')
plt.xlabel('Arm')
plt.ylabel('True Mean Value')
plt.title(f'True Value of Each Arm (Random Seed: {random_seed})')
plt.grid(True)
plt.show()

```

---

## 🎯 演算法一：Epsilon-Greedy

### (1) Algorithm Formula (LaTeX)

![image](https://github.com/user-attachments/assets/1e93387a-dd24-432a-b5be-ca6fbdccd743)


### (2) ChatGPT Prompt

> "請簡要解釋 epsilon-greedy 演算法如何在探索與利用之間取得平衡，並舉例說明 epsilon 值變化對行為的影響。"

### (3) 程式碼與圖表
這是一種平衡探索與利用的策略：
 - 以 1% 的機率進行隨機探索（選擇任意 arm）
 - 以 99% 的機率選擇目前估計報酬最高的 arm（利用）

![image](https://github.com/user-attachments/assets/8f98e6a6-f06f-4d07-a765-75884b701b26)

## (4) 結果分析

### ⏱ 時間角度（Time Perspective）
 1. 累積報酬（Cumulative Reward）
  - 累積報酬曲線整體上升，但過程中有明顯波動，且成長速度偏慢，最終累積報酬遠低於其他策略。
  - 顯示 Epsilon-Greedy 策略在本次設定中無法有效辨識出真正高報酬的 arm，導致探索與 exploitation 的表現皆不理想。

 2. 平均每步報酬（Average Reward per Step）
  - 平均每步報酬趨勢緩慢上升，但最終穩定在非常低的值（約 0.05），顯示其長期收益非常有限。
  - 由於未正確找到最佳 arm，使得 exploitation 階段也無法有效提升平均收益。
---
### 📌 空間角度（Space Perspective）
 3. arm 選擇次數（Arm Selection Counts）
  - Epsilon-Greedy 策略下，第 17 號 arm 被選擇次數最多，而真正的最佳 arm（第 15 號）幾乎沒有被重點選取。
  - 表明 Epsilon-Greedy 在本情境下的探索效果不好，早期探索選錯 arm，後續 exploitation 又堅持錯誤的 arm，導致整體表現受限。
---

## 📌 演算法二：UCB (Upper Confidence Bound)

### (1) Algorithm Formula (LaTeX)

![image](https://github.com/user-attachments/assets/0c6497c0-270d-443e-a6e2-7d7cdf7e65f7)

### (2) ChatGPT Prompt

> "請說明 UCB 演算法如何透過置信區間達到探索與利用的平衡，並指出常數 c 對行為的影響。"

### (3) 程式碼與圖表
![image](https://github.com/user-attachments/assets/3437728f-7a24-47f4-9ab1-7fcad5f7c214)

## (4) 結果分析
### ⏱ 時間角度（Time Perspective）
 1. 累積報酬（Cumulative Reward）
  - 累積報酬曲線上升趨勢比 Epsilon-Greedy更穩定且明顯，雖然仍有些小幅度波動，但整體走勢良好。
  - 說明 UCB 能夠在初期快速辨識較好的 arms，並且在 exploitation 階段有效累積報酬。

 2. 平均每步報酬（Average Reward per Step）
  - 平均每步報酬逐漸趨於穩定，大約收斂到 0.04。
  - 曲線相對平滑，波動小，說明 UCB 能在時間上有效利用經驗，減少無效探索。
---
### 📌 空間角度（Space Perspective）
 3. arm 選擇次數（Arm Selection Counts）
  - 第 15 號 arm 被選擇最多，證明成功找到最佳 arm。
  - 但其他 arm 的選擇次數較分散，顯示 Softmax 維持一定程度的隨機探索。
---

---

## 🎲 演算法三：Softmax

### (1) Algorithm Formula (LaTeX)

![image](https://github.com/user-attachments/assets/665bcca3-6b49-4c48-a4b5-604a8857b183)

### (2) ChatGPT Prompt

> "請用簡單例子說明 Softmax 策略如何根據不同溫度參數（\tau）調整探索程度，並解釋為何 \tau 越大越傾向探索。"

### (3) 程式碼與圖表
![image](https://github.com/user-attachments/assets/1f26b984-02a8-42c1-bf7b-6d7a5861c1e5)

## (4) 結果分析

### ⏱ 時間角度（Time Perspective）
 1. 累積報酬（Cumulative Reward）
  - 累積報酬曲線上升趨勢比 UCB 和 Epsilon-Greedy更穩定且明顯，雖然仍有些小幅度波動，但整體走勢良好。
  - 說明 Softmax 能夠在初期快速辨識較好的 arms，並且在 exploitation 階段有效累積報酬。

 2. 平均每步報酬（Average Reward per Step）
  - 平均每步報酬逐漸趨於穩定，大約收斂到 0.05。
  - 長期表現穩健，略優於 UCB。
---
### 📌 空間角度（Space Perspective）
 3. arm 選擇次數（Arm Selection Counts）
  - 雖然第 15 號 arm仍然是被最多次選擇的，但其他 arm 也有較多次的探索紀錄，相比 Epsilon-Greedy更加平均。
  - 顯示 UCB 保持了一定程度的探索，同時也能聚焦在最佳 arm，符合其「樂觀初始估計」的特性。

---

## 🎯 演算法四：Thompson Sampling

### (1) Algorithm Formula (LaTeX)

![image](https://github.com/user-attachments/assets/9ca620c3-da84-4243-b4f6-b33f1742bfa8)

### (2) ChatGPT Prompt

> "請用簡單語言說明為何 Thompson Sampling 能自然地在探索與利用間取得平衡，以及 beta 分布在這裡的意義。"

### (3) 程式碼與圖表
![image](https://github.com/user-attachments/assets/dafa5402-9aea-48a3-8be4-3afadf4e488c)

## (4) 結果分析

### ⏱ 時間角度（Time Perspective）
 1. 累積報酬（Cumulative Reward）
  - 累積報酬曲線最為陡峭且穩定上升，最終累積報酬突破 1000。
  - 顯著優於其他三種策略，表現最佳。

 2. 平均每步報酬（Average Reward per Step）
  - 平均每步報酬快速提升並穩定在約 0.1 左右，遠高於其他方法。
  - 說明 Thompson Sampling 既能快速找到最佳 arm，又能穩定 exploitation，長期收益極佳。
---
### 📌 空間角度（Space Perspective）
 3. arm 選擇次數（Arm Selection Counts）
  - 第 15 號 arm 被極大比例地選擇，幾乎壟斷了所有操作。
  - 其他 arm 選擇次數極少，顯示 Thompson Sampling 在確定最佳 arm 後迅速集中 exploitation。
---
---

## 📊 所有演算法比較圖表
![image](https://github.com/user-attachments/assets/0e4d5069-fa62-42b4-aa4a-40cc390504d8)
![image](https://github.com/user-attachments/assets/a538a4af-ab99-4ef1-8297-1bf1d29ef187)


# 📋 總結比較表（時間分析 + 空間分析）

| 策略 | 累積報酬成長 | 平均每步報酬 | 是否成功找到最佳 Arm（第15個） | Arm 選擇分佈特性 |
|:-----|:--------------|:------------|:----------------------------|:----------------|
| **Epsilon-Greedy** | 緩慢且震盪 | 約 0.02，偏低 | ❌ 沒有找到（集中在第20個 arm） | 單一 suboptimal arm 被過度選擇 |
| **UCB** | 穩定且成長快 | 約 0.04，穩定中等 | ✅ 成功找到最佳 arm | 主選最佳 arm，但仍持續少量探索其他 |
| **Softmax** | 平穩上升 | 約 0.05，略高於 UCB | ✅ 成功找到最佳 arm | 主要集中在最佳 arm，但仍有分散探索 |
| **Thompson Sampling** | 極速穩定上升 | 約 0.1，最高 | ✅ 成功且快速找到最佳 arm | 幾乎完全集中在最佳 arm，極少探索其他 |

---

# 📈 結論與差異比較表

| 項目 | Epsilon-Greedy | UCB | Softmax | Thompson Sampling |
|:-----|:---------------|:----|:--------|:------------------|
| **探索效率** | 低，容易選錯 | 中高，理性探索 | 中，平滑探索 | 高，快速聚焦 |
| **Exploitation 效率** | 低 | 中等偏高 | 高 | 極高 |
| **收斂速度** | 慢且不穩定 | 穩定收斂 | 穩定且較快 | 非常快 |
| **累積報酬最終表現** | 最低 | 中等 | 次高 | 最高 |
| **穩健性** | 差 | 好 | 中等偏好 | 非常好 |

---

# 🧠 各演算法優劣與適用情境比較表

| 策略 | 優點 | 缺點 | 適用情境 |
|:-----|:-----|:-----|:---------|
| **Epsilon-Greedy** | 簡單易懂，實作快速 | 容易卡在錯誤 arm，長期表現差 | 問題簡單、arm 數量少、不要求高精度時 |
| **UCB** | 理論基礎強，自動平衡探索與利用 | 初期需要較多探索步驟，收斂速度受限 | 問題中 arm 數量中等，需要平穩成長的情境 |
| **Softmax** | 探索更平滑，避免過早收斂 | 需要微調溫度參數，否則探索不足或過度 | 需兼顧穩健與靈活性的中型專案 |
| **Thompson Sampling** | 探索與 exploitation 自然融合，收斂快 | 實作上稍微複雜，依賴隨機性 | 資源有限時、需要快速決策的場景（如線上廣告投放、推薦系統） |

---

# 📝 總結小結論

- **Epsilon-Greedy**：在本情境表現最差，探索過程失敗導致 exploitation 階段無法彌補。
- **UCB**：合理找到最佳 arm，穩定成長，但長期收益略遜。
- **Softmax**：找到最佳 arm，長期表現良好，但需要小心參數設定。
- **Thompson Sampling**：表現最佳，快速且有效地掌握最佳 arm，最適合資源有限且要求快速學習的情境。
