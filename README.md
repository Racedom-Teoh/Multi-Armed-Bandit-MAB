# Multi-Armed-Bandit-MAB

# HW3: Explore and Exploit for Arm-Bandit Problem

## 🎮 環境介紹：多臂機器人（Multi-Armed Bandit）

本作業中，我們模擬一個包含 k 臂的拉霸機，每個臂在每次拉下時會根據固定的分布給出不同的報酬。這是一個探索（exploration）與利用（exploitation）兼具的經典問題，需找出在有限試驗次數下最大化累積獎勵的策略。

以下是 Python 的模擬環境程式碼：

```python
import numpy as np

class BanditEnv:
    def __init__(self, k=10):
        self.k = k
        self.true_means = np.random.normal(0, 1, k)
    
    def pull(self, arm):
        return np.random.normal(self.true_means[arm], 1)
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

```python
import matplotlib.pyplot as plt

def epsilon_greedy(env, epsilon=0.1, steps=1000):
    k = env.k
    Q = np.zeros(k)
    N = np.zeros(k)
    rewards = []
    cumulative = 0

    for t in range(steps):
        if np.random.rand() < epsilon:
            action = np.random.choice(k)
        else:
            action = np.argmax(Q)
        
        reward = env.pull(action)
        N[action] += 1
        Q[action] += (reward - Q[action]) / N[action]
        cumulative += reward
        rewards.append(cumulative)
    
    return rewards

env = BanditEnv()
eps_rewards = epsilon_greedy(env)

plt.plot(eps_rewards, label='Epsilon-Greedy')
plt.xlabel('Steps')
plt.ylabel('Cumulative Reward')
plt.title('Epsilon-Greedy Performance')
plt.legend()
plt.grid(True)
plt.show()
```
![image](https://github.com/user-attachments/assets/e5210a9e-e987-45ef-be65-263dc2005d0a)

### (4) 結果分析

- **時間分析：** 收斂速度依賴於 epsilon 值；較高 epsilon 導致長期學習更慢。
- **空間分析：** 僅需記錄每個 arm 的 Q 值與選擇次數，空間複雜度為 \( O(k) \)。

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

- **時間分析：** 初期探索效果佳，收斂速度快。
- **空間分析：** 與 epsilon-greedy 相同，計算成本增加。

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

- **時間分析：** 收斂速度取決於 \( \tau \) 值，過高會阻礙學習。
- **空間分析：** 需額外計算 softmax 分布。

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

- **時間分析：** 收斂速度快，穩定性佳。
- **空間分析：** 需記錄 \( \alpha, \beta \) 參數，略高於其他方法。

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
