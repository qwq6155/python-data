import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import os

# ==========================================
# 1. 代理设置 (Proxy Setup) - 根据你的情况修改
# ==========================================
# 如果你在国内，通常需要开启这一步。
# 请把 '7890' 改成你代理软件 (Clash/V2Ray) 的端口号
USE_PROXY = True  # 如果不走代理，改成 False
PROXY_PORT = "7897"

if USE_PROXY:
    proxy_url = f"http://127.0.0.1:{PROXY_PORT}"
    os.environ["http_proxy"] = proxy_url
    os.environ["https_proxy"] = proxy_url
    print(f"🌍 代理已配置: {proxy_url}")
else:
    print("🌍 未使用代理，直接连接...")

# ==========================================
# 2. 获取数据 (Data Fetching)
# ==========================================
try:
    print("⏳ 正在下载 AAPL 数据...")
    # 下载最近 1 年半的数据，这样能看到比较明显的趋势
    df = yf.download('AAPL', start='2023-01-01', end='2024-06-01', progress=False)

    # 检查数据是否为空
    if df.empty:
        raise ValueError("数据为空，请检查网络或代理设置。")
    print(f"✅ 下载成功！共获取 {len(df)} 个交易日数据。")

except Exception as e:
    print(f"❌ 下载失败: {e}")
    # 为了演示代码，如果下载失败，这里紧急生成一份模拟数据 (备用方案)
    print("⚠️ 启用备用方案：生成模拟数据继续运行...")
    dates = pd.date_range(start='2023-01-01', periods=300, freq='D')
    df = pd.DataFrame(index=dates)
    df['Close'] = 150 * (1 + np.random.normal(0.0005, 0.02, len(dates))).cumprod()

# ==========================================
# 3. 策略逻辑 (Strategy Logic)
# ==========================================
# 计算均线
df['MA5'] = df['Close'].rolling(window=5).mean()  # 快线
df['MA20'] = df['Close'].rolling(window=20).mean()  # 慢线

# 初始化信号列
df['Signal'] = 0
# 初始化仓位 (Position): 1代表持有，0代表空仓
df['Position'] = 0

# --- 核心策略：寻找金叉与死叉 ---
# 金叉条件：今天 MA5 > MA20  且  昨天 MA5 < MA20
condition_buy = (df['MA5'] > df['MA20']) & (df['MA5'].shift(1) < df['MA20'].shift(1))
# 死叉条件：今天 MA5 < MA20  且  昨天 MA5 > MA20
condition_sell = (df['MA5'] < df['MA20']) & (df['MA5'].shift(1) > df['MA20'].shift(1))

# 标记信号 (1: 买入, -1: 卖出)
df.loc[condition_buy, 'Signal'] = 1
df.loc[condition_sell, 'Signal'] = -1

# 生成仓位：
# 如果发出了买入信号，之后的日子我们就一直持有 (Position=1)，直到发出卖出信号变成 0
df['Position'] = df['Signal'].replace(to_replace=0, method='ffill')
# 把最开始的 NaN 填为 0
df['Position'] = df['Position'].fillna(0)

# 如果 Position 只有 -1 和 1，把 -1 (卖出信号那一刻) 变成 0 (空仓)
df['Position'] = df['Position'].replace(-1, 0)

# ==========================================
# 4. 回测收益 (Backtesting) - 我们赚了多少？
# ==========================================
# 计算每天的股价变化率 (今天比昨天涨了百分之多少)
df['Market_Return'] = df['Close'].pct_change()

# 计算我们的策略收益
# 逻辑：如果我们昨天持有 (Position=1)，那我们要吃今天的涨跌幅
# shift(1) 是为了避免"未来函数" (即只能用昨天收盘的信号决定今天开盘的操作)
df['Strategy_Return'] = df['Position'].shift(1) * df['Market_Return']

# 计算累计收益 (Cumulative Return) - 也就是资金曲线
# (1 + 收益率).cumprod() 模拟复利增长
df['Cumulative_Market_Returns'] = (1 + df['Market_Return']).cumprod()
df['Cumulative_Strategy_Returns'] = (1 + df['Strategy_Return']).cumprod()

# 输出最终结果
final_market_return = df['Cumulative_Market_Returns'].iloc[-1] - 1
final_strategy_return = df['Cumulative_Strategy_Returns'].iloc[-1] - 1

print("\n" + "=" * 30)
print(f"💰 回测结果报告")
print(f"1. 傻瓜式持有收益率: {final_market_return:.2%}")
print(f"2. 均线策略交易收益率: {final_strategy_return:.2%}")
if final_strategy_return > final_market_return:
    print("🏆 恭喜！你的策略战胜了市场！")
else:
    print("📉 遗憾，这一波操作猛如虎，不如原地不动。")
print("=" * 30 + "\n")

# ==========================================
# 5. 可视化 (Visualization)
# ==========================================
plt.figure(figsize=(14, 8))

# --- 子图 1: 股价与买卖点 ---
ax1 = plt.subplot(2, 1, 1)  # 2行1列，第1张图
ax1.plot(df.index, df['Close'], label='Close Price', alpha=0.5, color='gray')
ax1.plot(df.index, df['MA5'], label='MA5', alpha=0.8, color='orange', linestyle='--')
ax1.plot(df.index, df['MA20'], label='MA20', alpha=0.8, color='blue')

# 画出买入信号 (红三角)
ax1.plot(df[df['Signal'] == 1].index,
         df['MA5'][df['Signal'] == 1],
         '^', markersize=10, color='red', label='Buy Signal')

# 画出卖出信号 (绿三角)
ax1.plot(df[df['Signal'] == -1].index,
         df['MA5'][df['Signal'] == -1],
         'v', markersize=10, color='green', label='Sell Signal')

ax1.set_title('AAPL Trading Strategy (Golden Cross)')
ax1.set_ylabel('Price ($)')
ax1.legend(loc='upper left')
ax1.grid(True)

# --- 子图 2: 资金曲线对比 (最刺激的部分) ---
ax2 = plt.subplot(2, 1, 2)  # 2行1列，第2张图
ax2.plot(df.index, df['Cumulative_Market_Returns'], label='Buy & Hold (Market)', color='gray')
ax2.plot(df.index, df['Cumulative_Strategy_Returns'], label='MA Strategy (You)', color='red', linewidth=2)

ax2.set_title('Cumulative Returns Comparison')
ax2.set_ylabel('Cumulative Return (1.0 = Initial Capital)')
ax2.legend(loc='upper left')
ax2.grid(True)

plt.tight_layout()
plt.show()