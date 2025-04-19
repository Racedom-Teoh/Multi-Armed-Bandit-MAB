![image](https://github.com/user-attachments/assets/62e7f50c-9b5d-41a5-8344-97552307c9ac)# Multi-Armed-Bandit-MAB

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

\[
A_t = \begin{cases}
\arg\max_a Q_t(a) & \text{with probability } 1 - \epsilon \\
a \sim \text{Uniform}(0, k-1) & \text{with probability } \epsilon
\end{cases}
\]

```latex
A_t = \begin{cases}
\arg\max_a Q_t(a) & \text{with probability } 1 - \epsilon \\
a \sim \text{Uniform}(0, k-1) & \text{with probability } \epsilon
\end{cases}
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

\[
A_t = \arg\max_a \left[ Q_t(a) + c \cdot \sqrt{\frac{\ln t}{N_t(a)}} \right]
\]

```latex
A_t = \arg\max_a \left[ Q_t(a) + c \cdot \sqrt{\frac{\ln t}{N_t(a)}} \right]
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

\[
P(a) = \frac{\exp(Q_t(a) / \tau)}{\sum_{b=1}^{k} \exp(Q_t(b) / \tau)}
\]

```latex
P(a) = \frac{\exp(Q_t(a) / \tau)}{\sum_{b=1}^{k} \exp(Q_t(b) / \tau)}
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

\[
\theta_a \sim \text{Beta}(\alpha_a, \beta_a), \quad A_t = \arg\max_a \theta_a
\]

```latex
\theta_a \sim \text{Beta}(\alpha_a, \beta_a), \quad A_t = \arg\max_a \theta_a
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

## 🔍 結論與差異比較

| 演算法 | 優點 | 缺點 |
|--------|------|------|
| Epsilon-Greedy | 簡單實作、有效探索 | 探索固定不變、不夠自適應 |
| UCB | 理論有保證、強制早期探索 | 計算較繁、容易高估罕見臂 |
| Softmax | 機率式選擇、更平滑探索 | 對溫度參數敏感、難以調參 |
| Thompson Sampling | 自然平衡探索與利用、收斂快 | 需假設獎勵分布、更新較複雜 |

### ✅ 空間分析
所有演算法空間需求為 \( O(k) \)，Thompson Sampling 多維護兩組參數。

### ✅ 時間分析
UCB 與 Softmax 計算成本較高，Thompson Sampling 需取樣 beta 分布，Epsilon-Greedy 最快。

