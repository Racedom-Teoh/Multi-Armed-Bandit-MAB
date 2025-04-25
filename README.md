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

```latex
\documentclass{article}
\usepackage{amsmath}
\usepackage[margin=1in]{geometry}  
\usepackage{tcolorbox}  

\begin{document}

\begin{center}
    \LARGE \textbf{$\varepsilon$-Greedy Algorithm}
\end{center}

\vspace{0.5em}
\hrule
\vspace{1em}

\begin{tcolorbox}[colback=gray!5, colframe=black!40, title=Action Selection Rule]
At each time step $t$, the action $A_t$ is selected as:

\[
A_t =
\begin{cases}
\arg\max\limits_a Q_t(a) & \text{with probability } 1 - \varepsilon \\
\text{random arm} & \text{with probability } \varepsilon
\end{cases}
\]

Where:
\begin{itemize}
    \item $Q_t(a)$ is the estimated value of the action $a$ at time $t$.
    \item $\varepsilon$ is a small value (e.g. 0.1) controlling exploration.
\end{itemize}
\end{tcolorbox}

\vspace{1em}
\hrule
\vspace{1em}

\begin{tcolorbox}[colback=gray!5, colframe=black!40, title=Value Update Rule]
After choosing action $a$ and receiving reward $R_t$, the value estimate is updated as:

\[
Q_{t+1}(a) = Q_t(a) + \alpha \left( R_t - Q_t(a) \right)
\]

Here, $\alpha \in (0,1]$ is the learning rate that controls how quickly the estimates adapt to new rewards.
\end{tcolorbox}

\end{document}


```

### (2) ChatGPT Prompt

> "請簡要解釋 epsilon-greedy 演算法如何在探索與利用之間取得平衡，並舉例說明 epsilon 值變化對行為的影響。"

### (3) 程式碼與圖表
這是一種平衡探索與利用的策略：
 - 以 1% 的機率進行隨機探索（選擇任意 arm）
 - 以 99% 的機率選擇目前估計報酬最高的 arm（利用）

![image](https://github.com/user-attachments/assets/8f98e6a6-f06f-4d07-a765-75884b701b26)
```python
# epsilon-greedy 策略
def epsilon_greedy(env, epsilon=0.01, steps=10000):
    k = env.k
    Q = np.zeros(k)  # 初始化每个 arm 的估计值
    N = np.zeros(k)  # 初始化每个 arm 被选择的次数
    rewards = []
    cumulative = 0

    for t in range(steps):
        if np.random.rand() < epsilon:
            action = np.random.choice(k)  # 探索
        else:
            action = np.argmax(Q)  # 利用

        reward = env.pull(action)
        N[action] += 1
        Q[action] += (reward - Q[action]) / N[action]  # 更新估计
        cumulative += reward
        rewards.append(cumulative)

    return rewards, N

# 执行改进的 epsilon-greedy 策略
eps_rewards, eps_N = epsilon_greedy(env, epsilon=0.01)

# 计算平均每步报酬
avg_rewards = [r / (i + 1) for i, r in enumerate(eps_rewards)]

# 绘制图表
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. 累积报酬
axes[0].plot(eps_rewards)
axes[0].set_title('Cumulative Reward')
axes[0].set_xlabel('Steps')
axes[0].set_ylabel('Total Reward')
axes[0].grid(True)

# 2. 平均每步报酬
axes[1].plot(avg_rewards)
axes[1].set_title('Average Reward per Step')
axes[1].set_xlabel('Steps')
axes[1].set_ylabel('Average Reward')
axes[1].grid(True)

# 3. 每个 arm 的选择次数
axes[2].bar(np.arange(env.k), eps_N)
axes[2].set_title('Arm Selection Counts')
axes[2].set_xlabel('Arm')
axes[2].set_ylabel('Times Selected')
axes[2].grid(True, axis='y')

plt.tight_layout()
plt.suptitle('Epsilon-Greedy Strategy Summary', fontsize=16, y=1.05)
plt.show()
```

### (4) 結果分析

## ⏱ 時間角度（Time Perspective）
# 1. 累積報酬（Cumulative Reward）
- 圖中顯示累積報酬隨時間穩定成長。
- 初期因為策略還在探索（尤其是 ε = 0.01 時偶爾會隨機選擇），所以報酬成長較慢。
- 隨著時間增加，演算法逐漸學會最佳 arm，報酬成長曲線變得更陡峭。

# 2. 平均每步報酬（Average Reward per Step）
- 一開始報酬震盪大，表示演算法還在嘗試與學習。
- 隨著步數增加，平均報酬逐漸穩定上升並趨近於最佳 arm 的期望值（接近 0.25）。
- 這表明 epsilon-greedy 在時間推移中學會了接近最優策略。
---
## 📌 空間角度（Space Perspective）
# 3. arm 選擇次數（Arm Selection Counts）
- 最佳 arm（被標記為金色的那一個）被選擇得最多，顯示策略成功辨識出它。
- 其餘 arm 的選擇次數非常少，只在早期探索階段或偶爾隨機選擇中出現。
- 這種選擇分佈符合 epsilon-greedy 的性質：絕大多數時間都選擇目前預估最好的選項，只有少部分時間進行隨機探索。
---

## 📌 演算法二：UCB (Upper Confidence Bound)

### (1) Algorithm Formula (LaTeX)

![image](https://github.com/user-attachments/assets/0c6497c0-270d-443e-a6e2-7d7cdf7e65f7)


```latex
\documentclass{article}
\usepackage{amsmath}
\usepackage[margin=1in]{geometry}  % 更好看一点的页面边距
\usepackage{tcolorbox}  % 用于漂亮的内容框

\begin{document}

\begin{center}
    \LARGE \textbf{UCB Algorithm for Multi-Armed Bandit}
\end{center}

\vspace{0.5em}
\hrule
\vspace{1em}

\begin{tcolorbox}[colback=gray!5, colframe=black!40, title=Action Selection Rule]
At each time step $t$, the action $A_t$ is selected as:

\[
A_t = \arg\max_a \left[ Q_t(a) + c \cdot \sqrt{\frac{\ln t}{N_t(a)}} \right]
\]

Where:
\begin{itemize}
    \item $Q_t(a)$ is the estimated value of action $a$ at time $t$.
    \item $N_t(a)$ is the number of times action $a$ has been selected up to time $t$.
    \item $c > 0$ is a tunable parameter that balances exploration and exploitation.
    \item $\ln t$ encourages exploration of less-frequently selected actions.
\end{itemize}
\end{tcolorbox}

\vspace{1em}
\hrule
\vspace{1em}

\begin{tcolorbox}[colback=gray!5, colframe=black!40, title=Value Update Rule]
After choosing action $a$ and receiving reward $R_t$, the value estimate is updated as:

\[
Q_{t+1}(a) = Q_t(a) + \alpha \left( R_t - Q_t(a) \right)
\]

Here, $\alpha \in (0,1]$ is the learning rate that controls how quickly the estimates adapt to new rewards. Alternatively, one can use a sample average (i.e., $\alpha = \frac{1}{N_t(a)}$) for non-stationary problems.
\end{tcolorbox}

\end{document}

```

### (2) ChatGPT Prompt

> "請說明 UCB 演算法如何透過置信區間達到探索與利用的平衡，並指出常數 c 對行為的影響。"

### (3) 程式碼與圖表

```python
def ucb(env, c=2, steps=1000):
    k = env.k
    Q = np.zeros(k)
    N = np.zeros(k)
    rewards = []
    cumulative = 0

    for t in range(1, steps + 1):
        ucb_values = np.where(N > 0, Q + c * np.sqrt(np.log(t) / N), float('inf'))
        action = np.argmax(ucb_values)
        reward = env.pull(action)

        N[action] += 1
        Q[action] += (reward - Q[action]) / N[action]
        cumulative += reward
        rewards.append(cumulative)
    
    return rewards

env = BanditEnv()
ucb_rewards = ucb(env)

plt.plot(ucb_rewards, label='UCB')
plt.xlabel('Steps')
plt.ylabel('Cumulative Reward')
plt.title('UCB Performance')
plt.legend()
plt.grid(True)
plt.show()

```
![image](https://github.com/user-attachments/assets/0085c65d-3f4a-4b98-8021-2fd81cd55f0f)

### (4) 結果分析

- **時間分析：**  根據圖表，UCB 曲線在前期上升速度比 Epsilon-Greedy 快，表示它在較短時間內就發現報酬較高的臂，並集中選擇。這是因為 UCB 會優先探索信心區間較大的 arm，在 early stage 就能更有效識別最優 arm。而到了中後期，曲線變得非常平滑且斜率高，表現穩定，收斂速度極快，是所有演算法中收斂表現最佳者之一。
- **空間分析：** 需要記錄所有臂的選擇次數與 Q 值，空間複雜度與 ε-greedy 相當，但計算成本較高（log 與 sqrt 計算）。

---

## 🎲 演算法三：Softmax

### (1) Algorithm Formula (LaTeX)

![image](https://github.com/user-attachments/assets/665bcca3-6b49-4c48-a4b5-604a8857b183)


```latex
\documentclass{article}
\usepackage{amsmath}
\usepackage[margin=1in]{geometry}  % 更好看一点的页面边距
\usepackage{tcolorbox}  % 用于漂亮的内容框

\begin{document}

\begin{center}
    \LARGE \textbf{Softmax Algorithm for Multi-Armed Bandit}
\end{center}

\vspace{0.5em}
\hrule
\vspace{1em}

\begin{tcolorbox}[colback=gray!5, colframe=black!40, title=Action Selection Rule]
At each time step $t$, the probability of selecting action $a$ is computed using the softmax function:

\[
P_t(a) = \frac{e^{Q_t(a)/\tau}}{\sum\limits_b e^{Q_t(b)/\tau}}
\]

Then, action $A_t$ is sampled according to the distribution $P_t(a)$.

Where:
\begin{itemize}
    \item $Q_t(a)$ is the estimated value of action $a$ at time $t$.
    \item $\tau > 0$ is the \textbf{temperature} parameter that controls the randomness of action selection:
    \begin{itemize}
        \item High $\tau$: more exploration (actions have similar probabilities)
        \item Low $\tau$: more exploitation (focuses on high-value actions)
    \end{itemize}
\end{itemize}
\end{tcolorbox}

\vspace{1em}
\hrule
\vspace{1em}

\begin{tcolorbox}[colback=gray!5, colframe=black!40, title=Value Update Rule]
After choosing action $a$ and receiving reward $R_t$, the value estimate is updated as:

\[
Q_{t+1}(a) = Q_t(a) + \alpha \left( R_t - Q_t(a) \right)
\]

Here, $\alpha \in (0,1]$ is the learning rate that determines how quickly the estimate adapts to new information. A decaying or sample-average approach to $\alpha$ is also commonly used.
\end{tcolorbox}

\end{document}

```

### (2) ChatGPT Prompt

> "請用簡單例子說明 Softmax 策略如何根據不同溫度參數（\tau）調整探索程度，並解釋為何 \tau 越大越傾向探索。"

### (3) 程式碼與圖表

```python
def softmax(env, tau=0.1, steps=1000):
    k = env.k
    Q = np.zeros(k)
    N = np.zeros(k)
    rewards = []
    cumulative = 0

    for t in range(steps):
        exp_Q = np.exp(Q / tau)
        probs = exp_Q / np.sum(exp_Q)
        action = np.random.choice(k, p=probs)
        
        reward = env.pull(action)
        N[action] += 1
        Q[action] += (reward - Q[action]) / N[action]
        cumulative += reward
        rewards.append(cumulative)
    
    return rewards

env = BanditEnv()
softmax_rewards = softmax(env)

plt.plot(softmax_rewards, label='Softmax')
plt.xlabel('Steps')
plt.ylabel('Cumulative Reward')
plt.title('Softmax Performance')
plt.legend()
plt.grid(True)
plt.show()
```
![image](https://github.com/user-attachments/assets/e9aa5448-1206-4d6a-990b-bcec5ba811d9)

### (4) 結果分析

- **時間分析：** 從圖中可見 Softmax 的表現整體穩定但偏慢。初期成長率與 Epsilon-Greedy 接近，甚至略為落後。這是因為 Softmax 採用機率選擇，可能會頻繁選擇次佳臂。然而隨著時間推移，它的累積報酬曲線逐漸趨於穩定。整體來說收斂速度較慢，尤其是當 τ（溫度參數）設定不當時，會顯著拖慢學習進度。
- **空間分析：** 與前述方法相同，主要成本在於每步需計算 softmax 機率分布。

---

## 🎯 演算法四：Thompson Sampling

### (1) Algorithm Formula (LaTeX)

![image](https://github.com/user-attachments/assets/9ca620c3-da84-4243-b4f6-b33f1742bfa8)


```latex
\documentclass{article}
\usepackage{amsmath}
\usepackage[margin=1in]{geometry}  % 更好看一点的页面边距
\usepackage{tcolorbox}  % 用于漂亮的内容框

\begin{document}

\begin{center}
    \LARGE \textbf{Thompson Sampling for Multi-Armed Bandit}
\end{center}

\vspace{0.5em}
\hrule
\vspace{1em}

\begin{tcolorbox}[colback=gray!5, colframe=black!40, title=Action Selection Rule]
At each time step $t$, the agent maintains a posterior distribution over the reward of each action. The action $A_t$ is selected as:

\[
A_t = \arg\max_a \theta_a
\]

Where:
\begin{itemize}
    \item For each action $a$, sample $\theta_a$ from its posterior distribution $P(\theta_a \mid \text{data})$.
    \item Choose the action $a$ with the highest sampled $\theta_a$.
\end{itemize}

\textbf{Typical case: Bernoulli rewards}

\begin{itemize}
    \item Use Beta distribution as the conjugate prior.
    \item Maintain parameters $(\alpha_a, \beta_a)$ for each action $a$:
    \[
    \theta_a \sim \text{Beta}(\alpha_a, \beta_a)
    \]
\end{itemize}
\end{tcolorbox}

\vspace{1em}
\hrule
\vspace{1em}

\begin{tcolorbox}[colback=gray!5, colframe=black!40, title=Posterior Update Rule]
After choosing action $a$ and observing reward $R_t \in \{0,1\}$ (e.g., Bernoulli reward), update the Beta distribution parameters:

\[
\alpha_a \leftarrow \alpha_a + R_t, \quad \beta_a \leftarrow \beta_a + (1 - R_t)
\]

This keeps the posterior up to date:
\[
\theta_a \sim \text{Beta}(\alpha_a, \beta_a)
\]
\end{tcolorbox}

\end{document}

```

### (2) ChatGPT Prompt

> "請用簡單語言說明為何 Thompson Sampling 能自然地在探索與利用間取得平衡，以及 beta 分布在這裡的意義。"

### (3) 程式碼與圖表

```python
def thompson_sampling(env, steps=1000):
    k = env.k
    alpha = np.ones(k)
    beta = np.ones(k)
    rewards = []
    cumulative = 0

    for t in range(steps):
        theta = np.random.beta(alpha, beta)
        action = np.argmax(theta)
        reward = env.pull(action)
        reward_bin = 1 if reward > 0 else 0  # 把常態分布轉為伯努力

        alpha[action] += reward_bin
        beta[action] += 1 - reward_bin
        cumulative += reward
        rewards.append(cumulative)

    return rewards

env = BanditEnv()
ts_rewards = thompson_sampling(env)

plt.plot(ts_rewards, label='Thompson Sampling')
plt.xlabel('Steps')
plt.ylabel('Cumulative Reward')
plt.title('Thompson Sampling Performance')
plt.legend()
plt.grid(True)
plt.show()
```
![image](https://github.com/user-attachments/assets/5a58a159-e615-4ce8-8bff-f626fc7e7567)

### (4) 結果分析

- **時間分析：**  Thompson Sampling 的曲線一開始即呈現平穩上升，表示即使在初期，該方法也能有效選出高報酬 arm。從圖中可以清楚看到它早期就超越其他演算法，代表其探索效率高。中期與後期，TS 的累積報酬持續保持最高，且震盪最小。這顯示它能快速收斂，且不易被誤導至次優 arm，整體來說表現極為穩定。
- **空間分析：**須追蹤每個 arm 的 alpha/beta，對空間需求略高；儘管計算略複雜，但在大多數 Python 環境中效率仍可接受。空間為 O(2k)，取樣操作是其唯一潛在瓶頸。但以換取超強穩定與收斂速度來看，極具性價比。

---

## 📊 所有演算法比較圖表

```python
env = BanditEnv()
eps = epsilon_greedy(env)
env = BanditEnv()
ucb_ = ucb(env)
env = BanditEnv()
soft = softmax(env)
env = BanditEnv()
ts = thompson_sampling(env)

plt.figure(figsize=(10,6))
plt.plot(eps, label='Epsilon-Greedy')
plt.plot(ucb_, label='UCB')
plt.plot(soft, label='Softmax')
plt.plot(ts, label='Thompson Sampling')
plt.xlabel('Steps')
plt.ylabel('Cumulative Reward')
plt.title('Algorithm Comparison: Cumulative Reward')
plt.legend()
plt.grid(True)
plt.show()
```
![image](https://github.com/user-attachments/assets/50c8a93b-eb9a-4e35-8510-f37ffe6da88f)

## 🧾 總結表格

| 演算法             | 初期表現   | 收斂速度 | 空間需求   | 運算成本                 | 圖表收斂曲線特性             |
|--------------------|------------|----------|------------|--------------------------|------------------------------|
| Epsilon-Greedy     | 波動大     | 中等     | 低 `O(k)`  | 低                        | 後期漸穩，略落後             |
| UCB                | 穩定快速   | 快       | 低 `O(k)`  | 中（需 `log` 與 `sqrt`） | 斜率穩定，快速上升           |
| Softmax            | 穩定但慢   | 慢       | 低 `O(k)`  | 高（需 `exp` 與正規化）  | 緩慢收斂，波動小             |
| Thompson Sampling  | 穩定且高   | 快速     | 中 `O(2k)` | 中（需 `beta` sampling） | 一路領先，最穩定             |

---

# 🔍 結論與差異比較

| 演算法             | 優點                                                                 | 缺點                                                                 |
|--------------------|----------------------------------------------------------------------|----------------------------------------------------------------------|
| **Epsilon-Greedy** | - 實作簡單、邏輯直觀。<br>- 在大多數情境中表現穩定，具備基本探索與利用能力。 | - 探索機率 ε 為固定值，無法根據學習進度自調整。<br>- 容易在初期探索不足、後期探索過多。 |
| **UCB**            | - 理論有保證（後悔界 upper bound）。<br>- 初期強制探索，有效避免陷入局部最優。 | - 每次需計算置信上界，運算較複雜。<br>- 對高變異臂可能高估其價值，導致過度探索。     |
| **Softmax**        | - 機率式選擇方式，每臂皆有被選中的機會。<br>- 溫度參數具彈性，可控制探索程度。 | - 對溫度參數敏感，難以調整最適值。<br>- 實作時需進行指數與歸一化運算，計算成本較高。 |
| **Thompson Sampling** | - 自然平衡探索與利用，依據後驗分布選臂。<br>- 收斂快，效果良好。             | - 需對獎勵有先驗假設（如伯努利分布）。<br>- 每次更新需採樣，運算邏輯較複雜。        |

---

## ✅ 空間分析

- 所有演算法的空間需求皆為 $O(k)$，其中 $k$ 為拉霸臂數量。
- Epsilon-Greedy、UCB、Softmax 僅需儲存平均獎勵與次數。
- **Thompson Sampling** 額外儲存每個臂的 Beta 分布參數（α 與 β），空間雖仍為 $O(k)$，但實際儲存量稍多。

---

## ✅ 時間分析

| 演算法             | 每次選擇臂的時間成本                                                                 |
|--------------------|----------------------------------------------------------------------------------------|
| **Epsilon-Greedy** | 最快。僅需比較平均獎勵與產生一個隨機值，時間複雜度為 $O(k)$。                           |
| **UCB**            | 需計算每個臂的 $\text{平均} + \text{信賴區間}$，包含對數與開根號，複雜度為 $O(k)$。         |
| **Softmax**        | 需計算每個臂的指數值與歸一化機率，包含浮點數與指數運算，時間複雜度為 $O(k)$。             |
| **Thompson Sampling** | 每次需從每個臂的 Beta 分布採樣，雖然仍為 $O(k)$，但實際運算成本視實作與採樣效率而定。     |


### 🎯 各演算法優劣與適用情境比較表

| 演算法              | 優勢                                                                 | 限制                                                                 | 適用情境說明                                                                 |
|----------------------|----------------------------------------------------------------------|----------------------------------------------------------------------|------------------------------------------------------------------------------|
| **Epsilon-Greedy**   | - 實作簡單<br>- 低計算成本                                             | - 探索率 ε 固定可能導致收斂慢或過度探索                                 | - 適合臂的數量少、任務時間長（可慢慢學習）                                   |
| **UCB**              | - 理論有保證<br>- 可自動調整探索程度                                     | - 對估計不準時表現不穩定<br>- 需知道總時間步數                         | - 適合臂之間差距明顯、需快速收斂的情境                                       |
| **Softmax**          | - 採用機率方式平衡探索與利用<br>- 可調整溫度參數來控制選擇行為              | - 對參數（溫度）敏感<br>- 可能長期維持在次佳選擇                         | - 適合臂之間差異不大、希望保有一定隨機性以避免陷入局部最優                   |
| **Thompson Sampling**| - 自然整合探索與利用（Bayesian 方法）<br>- 表現穩定、收斂速度快            | - 實作與計算較複雜<br>- 需建構每個臂的後驗分布                          | - 適合臂數量不多但需要高效率決策的場景<br>- 表現最佳於不確定性高或報酬差距小的任務 |
