# =============================================================================
# EDA — Predicting Daily Stock Price Movement for S&P 500 Companies
# =============================================================================
import matplotlib
matplotlib.use('Agg') 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'figure.dpi': 120,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# =============================================================================
# 1. Load and dataset statistics
# =============================================================================
# Load without date parsing for speed 
df_raw = pd.read_csv('sp500_stocks.csv')

print("=" * 60)
print("DATASET OVERVIEW (full)")
print("=" * 60)
print(f"Shape: {df_raw.shape[0]:,} rows * {df_raw.shape[1]} columns")
print(f"Unique companies: {df_raw['Symbol'].nunique()}")
print(f"\nMissing values:\n{df_raw.isnull().sum()}")

# Sample 100 stocks early to avoid processing the full dataset
np.random.seed(42)
sampled_symbols = np.random.choice(df_raw['Symbol'].unique(), size=100, replace=False)
df = df_raw[df_raw['Symbol'].isin(sampled_symbols)].copy()
del df_raw  # free memory

df['Date'] = pd.to_datetime(df['Date'])
df.sort_values(['Symbol', 'Date'], inplace=True)
df.reset_index(drop=True, inplace=True)

print(f"\nSampled {len(sampled_symbols)} stocks, {len(df):,} rows")
print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")


# =============================================================================
# 2. Missing data — drop rows where price/volume are all NaN
# =============================================================================
# The dataset contains placeholder rows for tickers that weren't yet listed
# These carry no information and must be removed before feature engineering.

price_cols = ['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
before = len(df)
df = df.dropna(subset=price_cols).copy()
after = len(df)

print(f"\nDropped {before - after:,} rows with missing price/volumn data")
print(f"Remaining rows: {after:,}")
print(f"\nMissing values:\n{df[price_cols].isnull().sum()}")
print(f"\nBasic statistics:\n{df.describe().round(3)}")

# =============================================================================
# 3. Feature Engineering
# =============================================================================

Features = [
    'return_1d', 'return_5d', 'return_10d',
    'ma_5', 'ma_20', 'ma_50',
    'price_to_ma20', 'price_to_ma50',
    'vol_10', 'vol_20', 'rsi_14', 'price_range', 'vol_ratio'
]

print("Starting feature engineering...")

# Group once and use last 252 trading days per stock for plotting
df = df.groupby('Symbol').tail(252).copy()
grouped = df.groupby('Symbol')  # fix: define grouped

# Returns (vectorized)
df['return_1d'] = grouped['Close'].pct_change(1)
df['return_5d'] = grouped['Close'].pct_change(5)
df['return_10d'] = grouped['Close'].pct_change(10)

# Moving averages
df['ma_5'] = grouped['Close'].rolling(5).mean().reset_index(level=0, drop=True)
df['ma_20'] = grouped['Close'].rolling(20).mean().reset_index(level=0, drop=True)
df['ma_50'] = grouped['Close'].rolling(50).mean().reset_index(level=0, drop=True)

# Price relative to MAs
df['price_to_ma20'] = df['Close'] / df['ma_20'] - 1
df['price_to_ma50'] = df['Close'] / df['ma_50'] - 1

# Volatility (rolling std of returns)
df['vol_10'] = grouped['return_1d'].rolling(10).std().reset_index(level=0, drop=True)
df['vol_20'] = grouped['return_1d'].rolling(20).std().reset_index(level=0, drop=True)

# Price range
df['price_range'] = (df['High'] - df['Low']) / df['Close']

# Volume ratio
df['vol_ratio'] = df['Volume'] / grouped['Volume'].rolling(20).mean().reset_index(level=0, drop=True)

# Target (next day movement)
df['target'] = (grouped['Close'].shift(-1) > df['Close']).astype(int)

# RSI (vectorized per group)
def calc_rsi_vectorized(series):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

df['rsi_14'] = grouped['Close'].apply(calc_rsi_vectorized).reset_index(level=0, drop=True)

# Drop NaNs from rolling calculations
df = df.dropna(subset=Features + ['target']).copy()

print("Finished feature engineering")
print(f"Usable rows: {len(df):,}")
print(f"Features: {len(Features)}  |  Target: binary (1=up, 0=down)")

feat_info = {
    'return_1d':     'Lagged return    — 1-day % price change',
    'return_5d':     'Lagged return    — 5-day % price change',
    'return_10d':    'Lagged return    — 10-day % price change',
    'ma_5':          'Moving average   — 5-day SMA',
    'ma_20':         'Moving average   — 20-day SMA',
    'ma_50':         'Moving average   — 50-day SMA',
    'price_to_ma20': 'MA signal        — Close / MA20 − 1',
    'price_to_ma50': 'MA signal        — Close / MA50 − 1',
    'vol_10':        'Volatility       — 10-day rolling std of returns',
    'vol_20':        'Volatility       — 20-day rolling std of returns',
    'rsi_14':        'Momentum osc.    — 14-day RSI (>70 overbought, <30 oversold)',
    'price_range':   'Intraday range   — (High − Low) / Close',
    'vol_ratio':     'Volume signal    — Volume / 20d average volume',
}
print("\n--- Feature Definitions ---")
for k, v in feat_info.items():
    print(f"  {k:<16} {v}")

# =============================================================================
# 4. Target Class Distributions
# =============================================================================
plot_feats = ['return_1d', 'return_5d', 'rsi_14', 'vol_20', 'price_range',
                'vol_ratio', 'price_to_ma20', 'price_to_ma50']

fig, axes = plt.subplots(2, 4, figsize=(15, 6))
fig.suptitle('Feature Distributions', fontweight='bold')

for ax, feat in zip(axes.flatten(), plot_feats):
    data = df[feat].clip(df[feat].quantile(0.01), df[feat].quantile(0.99))
    ax.hist(data, bins=60, color ='#378ADD', alpha=0.75, edgecolor='none')
    ax.set_title(feat, fontsize=9)
    ax.text(0.97, 0.93, f'skew={df[feat].skew():.2f}\nkurt={df[feat].kurt():.2f}',
            transform=ax.transAxes, ha='right', va='top', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.6))
plt.tight_layout()
plt.savefig('eda_feature_distributions.png', bbox_inches='tight')
plt.close()

# Distribution split by class
fig, axes = plt.subplots(2, 3, figsize=(14, 7))
fig.suptitle('Feature Distribution: Up VS Down', fontweight='bold')

for ax, feat in zip(axes.flatten(), ['return_1d', 'return_5d', 'return_10d', 'rsi_14', 'vol_20', 'vol_ratio']):
    lo, hi = df[feat].quantile(0.01), df[feat].quantile(0.99)
    for cls, col, lbl in [(0, '#E24B4A', 'Down'), (1, '#378ADD', 'Up')]:
        ax.hist(df[df['target'] == cls][feat].clip(lo, hi), bins=60, alpha=0.5, density=True, color=col, label=lbl, edgecolor='none')
    ax.set_title(feat, fontsize=9)
    ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig('eda_up_vs_down_distributions.png', bbox_inches='tight')
plt.close()

# =============================================================================
# 6. Correlation Analysis
# =============================================================================
target_corr = df[Features + ['target']].corr()['target'].drop('target').sort_values()

print("\n--- Feature-Target Correlation ---")
print(target_corr.round(4).to_string())

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Correlation Analysis', fontweight='bold')

# Left: Feature-Target correlations
axes[0].barh(target_corr.index, target_corr.values, color=['#E24B4A' if v < 0 else '#378ADD' for v in target_corr.values], edgecolor='none')
axes[0].axvline(0, color='gray', linewidth=0.8)
axes[0].set_xlabel('Pearson r with target')
axes[0].set_title('Feature-Target correlation')

# Right: mean feature vaues for Up Vs down days, z-score scaled for comparability
feat_scaled = pd.DataFrame(StandardScaler().fit_transform(df[Features]), columns=Features, index=df.index)
feat_scaled['target'] = df['target'].values
means = feat_scaled.groupby('target')[Features].mean()

x = np.arange(len(Features))
width = 0.38
axes[1].bar(x - width / 2, means.loc[0], width, label = 'Down(0)', 
            color='#E24B4A', alpha=0.85, edgecolor='none')
axes[1].bar(x + width / 2, means.loc[1], width, label='Up(1)',
            color='#378ADD', alpha=0.85, edgecolor='none')
axes[1].axhline(0, color='gray', linewidth=0.8)
axes[1].set_xticks(x)
axes[1].set_xticklabels(Features, rotation=45, ha='right', fontsize=7)
axes[1].set_ylabel('Mean (z-score scaled)')
axes[1].set_title('Feature Means: Up vs Down')
axes[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig('eda_correlation_analysis.png', bbox_inches='tight')
plt.close()

# =============================================================================
# 7. Sample Stock: Price, RSI, Volume
# =============================================================================
sym = 'AAPL'
samp = df[df['Symbol'] == sym].tail(252)

fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
fig.suptitle(f'{sym} - Price, RSI, Volumn (last 1 year)', fontweight='bold')

axes[0].plot(samp['Date'], samp['Close'], color='#444', linewidth=1.2, label='Close')
axes[0].plot(samp['Date'], samp['ma_20'], color='#378ADD', linewidth=1, label='MA 20', alpha=0.8)
axes[0].plot(samp['Date'], samp['ma_50'], color='#E24B4A', linewidth=1, label='MA 50', alpha=0.8)
axes[0].set_ylabel('Price ($)')
axes[0].legend(fontsize=9)

axes[1].plot(samp['Date'], samp['rsi_14'], color='#3B6D11', linewidth=1)
axes[1].axhline(70, color='#E24B4A', linestyle='--', linewidth=0.8)
axes[1].axhline(30, color='#378ADD', linestyle='--', linewidth=0.8)
axes[1].set_ylabel('RSI')
axes[1].set_ylim(0, 100)

axes[2].bar(samp['Date'], samp['vol_ratio'], color='#888', alpha=0.6, width=1)
axes[2].axhline(1, color='gray', linestyle='--', linewidth=0.8)
axes[2].set_ylabel('Volume ratio')
axes[2].set_xlabel('Date')

plt.tight_layout()
plt.savefig('eda_aapl_price_rsi_volume.png', bbox_inches='tight')
plt.close()

# =============================================================================
# 8. Cross-Stock Volatility trend
# =============================================================================
vol_trend = df.groupby(df['Date'].dt.to_period('M'))['vol_20'].mean()

fig, ax = plt.subplots(figsize=(13, 3.5))
ax.plot(vol_trend.index.to_timestamp(), vol_trend.values, color='#E24B4A', linewidth=1)
ax.fill_between(vol_trend.index.to_timestamp(), vol_trend.values, alpha=0.15, color='#E24B4A')
ax.set_title('Average 20 day volatility across all stocks (2010 - 2014)', fontweight='bold')
ax.set_ylabel('Avg volatility')

plt.tight_layout()
plt.savefig('eda_cross_stock_volatility.png', bbox_inches='tight')
plt.close()

# =============================================================================
# 9. Return Distribution
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax_sub, feat, color in zip(axes, ['return_5d', 'return_10d'], ['#378ADD', '#E24B4A']):
    vals = df[feat].dropna()
    vals = vals.clip(vals.quantile(0.01), vals.quantile(0.99))
    ax_sub.hist(vals, bins=80, density=True, color=color, alpha=0.75, edgecolor='none')
    ax_sub.axvline(0, color='gray', linestyle='--', linewidth=0.8)
    ax_sub.set_xlabel(feat)
    ax_sub.set_ylabel('Density')
    ax_sub.set_title(f'{feat} distribution\nskew={df[feat].skew():.2f}, kurt={df[feat].kurt():.2f}')
fig.suptitle('5 day vs 10 day Return Distributions', fontweight='bold')
plt.tight_layout()
plt.savefig('eda_return_distributions.png', bbox_inches='tight')
plt.close()

# =============================================================================
# 10. Boxplots by class
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle('Feature Boxplots: Up vs Down', fontweight='bold')

for ax, feat in zip(axes, ['rsi_14', 'return_1d']):
    lo, hi = df[feat].quantile(0.01), df[feat].quantile(0.99)
    d0 = df[df['target'] == 0][feat].clip(lo, hi)
    d1 = df[df['target'] == 1][feat].clip(lo, hi)
    bp = ax.boxplot([d0, d1], labels=['Down(0)', 'Up(1)'], patch_artist=True, medianprops=dict(color='white', linewidth=2))
    bp['boxes'][0].set_facecolor('#E24B4A')
    bp['boxes'][1].set_facecolor('#378ADD')
    ax.set_title(feat)
    t, p = stats.ttest_ind(d0, d1)
    ax.text(0.97, 0.97, f'p={p:.2e}', transform=ax.transAxes, ha='right', va='top', fontsize=8)

plt.tight_layout()
plt.savefig('eda_boxplots_by_class.png', bbox_inches='tight')
plt.close()

# =============================================================================
# 11. RSI Bucket Analysis
# =============================================================================
# Bins RSI into four ranges and shows the mean next-day return per bucket.
# This directly motivates RSI as a feature by showing directional signal
# before any model training errors

df['rsi_bucket'] = pd.cut(df['rsi_14'], bins=[0, 30, 50, 70, 100], 
                        labels=['<30\n(Oversold)', '30-50', '50-70', '>70\n(Overbought)'])
rsi_stats = df.groupby('rsi_bucket', observed=True)['return_1d'].agg(['mean', 'sem'])

fig, ax = plt.subplots(figsize=(7, 4))
colors = ['#378ADD' if v > 0 else '#E24B4A' for v in rsi_stats['mean']]
ax.bar(rsi_stats.index, rsi_stats['mean'] * 100, yerr=rsi_stats['sem'] * 100, capsize=5, color=colors, edgecolor='none', alpha=0.85, width=0.5)
ax.axhline(0, color='gray', linewidth=0.8)
ax.set_ylabel('Mean next day return (%)')
ax.set_xlabel('RSI bucket')
ax.set_title('Mean next day return by RSI bucket\n(+/- 1 std err)', fontweight='bold')

plt.tight_layout()
plt.savefig('eda_rsi_bucket_analysis.png', bbox_inches='tight')
plt.close()

print("\nDone. All EDA plots saved to current directory.")