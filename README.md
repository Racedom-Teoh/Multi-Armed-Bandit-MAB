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

```latex
\documentclass{article}
\usepackage{amsmath}
\usepackage[margin=1in]{geometry} 
\usepackage{tcolorbox}  

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
![image](https://github.com/user-attachments/assets/3437728f-7a24-47f4-9ab1-7fcad5f7c214)
```python
# UCB 策略
def ucb(env, c=3, steps=10000):
    k = env.k
    Q = np.zeros(k)  # 每個 arm 的估算期望報酬
    N = np.zeros(k)  # 每個 arm 被選擇的次數
    rewards = []
    cumulative = 0

    for t in range(1, steps + 1):
        ucb_values = Q + c * np.sqrt(np.log(t) / (N + 1e-6))  # 計算 UCB 值，避免除以零
        # 隨機選擇具有最大 UCB 值的 arm
        max_ucb_value = np.max(ucb_values)
        best_arms = np.where(ucb_values == max_ucb_value)[0]
        action = np.random.choice(best_arms)  # 隨機選擇其中一個最佳 arm
        
        reward = env.pull(action)

        N[action] += 1
        Q[action] += (reward - Q[action]) / N[action]  # 更新 Q 值
        cumulative += reward
        rewards.append(cumulative)

    return rewards, N

# 执行 UCB 策略
ucb_rewards, ucb_N = ucb(env)

# 计算平均每步报酬
avg_rewards = [r / (i + 1) for i, r in enumerate(ucb_rewards)]

# 绘制图表
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. 累积报酬
axes[0].plot(ucb_rewards)
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
axes[2].bar(np.arange(env.k), ucb_N)
axes[2].set_title('Arm Selection Counts')
axes[2].set_xlabel('Arm')
axes[2].set_ylabel('Times Selected')
axes[2].grid(True, axis='y')

plt.tight_layout()
plt.suptitle('UCB Strategy Summary', fontsize=16, y=1.05)
plt.show()
```
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
![image](https://github.com/user-attachments/assets/1f26b984-02a8-42c1-bf7b-6d7a5861c1e5)
```python
# Softmax 策略
def softmax(env, tau=0.2, steps=10000):
    k = env.k
    Q = np.zeros(k)  # 每個 arm 的估算期望報酬
    N = np.zeros(k)  # 每個 arm 被選擇的次數
    rewards = []
    cumulative = 0

    for t in range(steps):
        exp_Q = np.exp(Q / tau)  # 計算 Q 值的 Softmax 機率
        probs = exp_Q / np.sum(exp_Q)  # 機率正規化
        action = np.random.choice(k, p=probs)  # 根據機率選擇 arm

        reward = env.pull(action)
        N[action] += 1
        Q[action] += (reward - Q[action]) / N[action]
        cumulative += reward
        rewards.append(cumulative)

    return rewards, N

# 执行 Softmax 策略
softmax_rewards, softmax_N = softmax(env)

# 计算平均每步报酬
avg_rewards = [r / (i + 1) for i, r in enumerate(softmax_rewards)]

# 绘制图表
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. 累积报酬
axes[0].plot(softmax_rewards)
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
axes[2].bar(np.arange(env.k), softmax_N)
axes[2].set_title('Arm Selection Counts')
axes[2].set_xlabel('Arm')
axes[2].set_ylabel('Times Selected')
axes[2].grid(True, axis='y')

plt.tight_layout()
plt.suptitle('Softmax Strategy Summary', fontsize=16, y=1.05)
plt.show()
```
![image](https://github.com/user-attachments/assets/e9aa5448-1206-4d6a-990b-bcec5ba811d9)

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
![image](https://github.com/user-attachments/assets/dafa5402-9aea-48a3-8be4-3afadf4e488c)

```python
def thompson_sampling(env, steps=10000):
    k = env.k
    alpha = np.ones(k)
    beta = np.ones(k)
    rewards = []
    cumulative = 0
    N = np.zeros(k)  # 記錄每個 arm 被選擇的次數

    for t in range(steps):
        theta = np.random.beta(alpha, beta)  # 根據 beta 分佈進行抽樣
        action = np.argmax(theta)  # 選擇期望報酬最大的 arm
        reward = env.pull(action)

        # 將常態分佈的報酬轉換為二元回報
        reward_bin = 1 if reward > 0 else 0  

        # 更新 Beta 分佈參數
        alpha[action] += reward_bin
        beta[action] += 1 - reward_bin
        cumulative += reward
        rewards.append(cumulative)
        N[action] += 1

    return rewards, N

# 执行 Thompson Sampling 策略
ts_rewards, ts_N = thompson_sampling(env)

# 计算平均每步报酬
avg_rewards = [r / (i + 1) for i, r in enumerate(ts_rewards)]

# 绘制图表
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. 累积报酬
axes[0].plot(ts_rewards)
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
axes[2].bar(np.arange(env.k), ts_N)
axes[2].set_title('Arm Selection Counts')
axes[2].set_xlabel('Arm')
axes[2].set_ylabel('Times Selected')
axes[2].grid(True, axis='y')

plt.tight_layout()
plt.suptitle('Thompson Sampling Strategy Summary', fontsize=16, y=1.05)
plt.show()
```
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
