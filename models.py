# =============================================================================
# Train & Evaluate on a Random Sample of S&P 500 Stocks
# =============================================================================
import matplotlib
matplotlib.use('Agg')
 
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve, confusion_matrix)
from torch.utils.data import DataLoader, TensorDataset
 
# Use MPS (Apple Silicon), CUDA or CPU
if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
elif torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
print(f"PyTorch device: {DEVICE}")
 
 
# =============================================================================
# 0. LOAD DATA
# =============================================================================
df = pd.read_csv('sp500_stocks.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values(['Symbol', 'Date'], inplace=True)
df.reset_index(drop=True, inplace=True)
 
price_cols = ['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
df = df.dropna(subset=price_cols).copy()
n_stocks = df['Symbol'].nunique()
print(f"Full dataset: {n_stocks} stocks, {len(df):,} rows")
 
 
# =============================================================================
# 1. NOVEL CONTRIBUTION: MARKET REGIME FEATURE
# =============================================================================
# For each trading date, compute a cross-sectional median close price across
# all S&P 500 stocks in the dataset as a proxy for the broad market index.
# Then label each day as BULL (1) if the market proxy is above its 200-day MA,
# or BEAR (0) if below. This captures whether the broad market environment
# favors upward moves, which individual stock models typically ignore.
#
# Why this is novel:
#   - Standard pipelines use only per-stock technical indicators
#   - This adds macro context without requiring a separate index dataset
#   - The regime label can interact with per-stock features (see interaction terms below)
 
print("\nComputing market regime feature...")
market_daily = df.groupby('Date')['Close'].median().rename('market_proxy').reset_index()
market_daily = market_daily.sort_values('Date').reset_index(drop=True)
market_daily['market_ma200'] = market_daily['market_proxy'].rolling(200).mean()
market_daily['bull_regime'] = (market_daily['market_proxy'] > market_daily['market_ma200']).astype(np.float32)
 
# Merge regime back into main df
df = df.merge(market_daily[['Date', 'bull_regime']], on='Date', how='left')
print(f"Bull regime days: {df['bull_regime'].mean()*100:.1f}% of rows")
 
 
# =============================================================================
# 2. FEATURE ENGINEERING (original + interaction features)
# =============================================================================
BASE_FEATURES = [
    'return_1d', 'return_5d', 'return_10d',
    'ma_5', 'ma_20', 'ma_50',
    'price_to_ma20', 'price_to_ma50',
    'vol_10', 'vol_20', 'rsi_14', 'price_range', 'vol_ratio'
]
 
# Novel: interaction features that encode regime-conditional momentum
# return_x_regime: does the stock move WITH the market regime?
# rsi_x_regime: does RSI signal persist differently in bull vs bear markets?
INTERACTION_FEATURES = [
    'return1_x_regime',   # return_1d * bull_regime
    'rsi_x_regime',       # rsi_14 * bull_regime
    'ptma20_x_regime',    # price_to_ma20 * bull_regime
]
 
Features = BASE_FEATURES + ['bull_regime'] + INTERACTION_FEATURES
 
grouped = df.groupby('Symbol')
 
# Original features
df['return_1d']  = grouped['Close'].pct_change(1)
df['return_5d']  = grouped['Close'].pct_change(5)
df['return_10d'] = grouped['Close'].pct_change(10)
df['ma_5']  = grouped['Close'].rolling(5).mean().reset_index(level=0, drop=True)
df['ma_20'] = grouped['Close'].rolling(20).mean().reset_index(level=0, drop=True)
df['ma_50'] = grouped['Close'].rolling(50).mean().reset_index(level=0, drop=True)
df['price_to_ma20'] = df['Close'] / df['ma_20'] - 1
df['price_to_ma50'] = df['Close'] / df['ma_50'] - 1
df['vol_10'] = grouped['return_1d'].rolling(10).std().reset_index(level=0, drop=True)
df['vol_20'] = grouped['return_1d'].rolling(20).std().reset_index(level=0, drop=True)
df['price_range'] = (df['High'] - df['Low']) / df['Close']
df['vol_ratio'] = df['Volume'] / grouped['Volume'].rolling(20).mean().reset_index(level=0, drop=True)
df['target'] = (grouped['Close'].shift(-1) > df['Close']).astype(int)
 
def calc_rsi_vectorized(series):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss
    return 100 - (100 / (1 + rs))
 
df['rsi_14'] = grouped['Close'].apply(calc_rsi_vectorized).reset_index(level=0, drop=True)
 
# Drop NaNs before computing interaction terms (need base features first)
df = df.dropna(subset=BASE_FEATURES + ['target', 'bull_regime']).copy()
 
# Compute interaction features
df['return1_x_regime'] = df['return_1d'] * df['bull_regime']
df['rsi_x_regime']     = df['rsi_14']    * df['bull_regime']
df['ptma20_x_regime']  = df['price_to_ma20'] * df['bull_regime']
 
df = df.dropna(subset=Features).copy()
df[Features] = df[Features].astype(np.float32)
 
print(f"\nUsable rows after feature engineering: {len(df):,}")
print(f"Total features: {len(Features)} (13 original + 1 regime + 3 interaction)")
 
class_counts = pd.Series(df['target'].values).value_counts().sort_index()
print(f"\nClass distribution:")
print(f"  0 (Down): {class_counts.get(0,0):,}  ({class_counts.get(0,0)/len(df)*100:.1f}%)")
print(f"  1 (Up)  : {class_counts.get(1,0):,}  ({class_counts.get(1,0)/len(df)*100:.1f}%)")
 
X = df[Features].values
y = df['target'].values
 
 
# =============================================================================
# 3. REGIME-STRATIFIED ANALYSIS
# =============================================================================
# Validate that bull_regime has predictive value by checking class distributions
# in bull vs bear regimes — if Up% differs meaningfully, the feature is informative
 
bull_up = df[df['bull_regime'] == 1]['target'].mean()
bear_up = df[df['bull_regime'] == 0]['target'].mean()
print(f"\nRegime validation:")
print(f"  Bull regime — Up day rate: {bull_up*100:.1f}%")
print(f"  Bear regime — Up day rate: {bear_up*100:.1f}%")
print(f"  Difference: {(bull_up - bear_up)*100:.2f} percentage points")
 
 
# =============================================================================
# 4. TIMESERIES SPLIT EVALUATION
# =============================================================================
N_SPLITS = 5
tscv = TimeSeriesSplit(n_splits=N_SPLITS)
 
COLORS = {
    'LR (balanced)':   '#378ADD',
    'RidgeClassifier': '#E24B4A',
    'LDA':             '#2EAD6D',
    'kNN':             '#F5A623',
    'LSTM':            '#8B5CF6',
}
 
def run_cv(model_name, make_model_fn, X, y, tscv):
    fold_results = []
    print(f"\n{'─'*55}")
    print(f"  {model_name}")
    print(f"{'─'*55}")
 
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_tr_raw, X_val_raw = X[train_idx], X[val_idx]
        y_tr, y_val         = y[train_idx], y[val_idx]
 
        scaler = StandardScaler()
        X_tr   = scaler.fit_transform(X_tr_raw)
        X_val  = scaler.transform(X_val_raw)
 
        model = make_model_fn()
        model.fit(X_tr, y_tr)
 
        if hasattr(model, 'predict_proba'):
            y_score = model.predict_proba(X_val)[:, 1]
        else:
            y_score = model.decision_function(X_val)
 
        y_pred = model.predict(X_val)
 
        metrics = {
            'accuracy':  accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred, zero_division=0),
            'recall':    recall_score(y_val, y_pred, zero_division=0),
            'f1':        f1_score(y_val, y_pred, zero_division=0),
            'auc':       roc_auc_score(y_val, y_score),
            'y_pred':    y_pred,
            'y_score':   y_score,
            'y_true':    y_val,
        }
        fold_results.append(metrics)
        print(f"  Fold {fold}: acc={metrics['accuracy']:.4f}  "
              f"f1={metrics['f1']:.4f}  auc={metrics['auc']:.4f}")
 
    print()
    for key in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        vals = [r[key] for r in fold_results]
        print(f"  Mean {key:9s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
 
    return fold_results
 
 
# =============================================================================
# 5a. SKLEARN MODELS
# =============================================================================
MODELS = {
    'LR (balanced)':   lambda: LogisticRegression(C=1.0, solver='lbfgs', max_iter=300, random_state=42, class_weight='balanced'),
    'RidgeClassifier': lambda: RidgeClassifier(alpha=1.0),
    'LDA':             lambda: LinearDiscriminantAnalysis(solver='svd'),
    'kNN':             lambda: KNeighborsClassifier(n_neighbors=31, weights='distance', metric='minkowski', n_jobs=-1),
}
 
all_results = {}
for name, factory in MODELS.items():
    all_results[name] = run_cv(name, factory, X, y, tscv)
 
 
# =============================================================================
# 5b. LSTM
# =============================================================================
SEQ_LEN = 10
BATCH   = 256
EPOCHS  = 30
LR_RATE = 1e-3
HIDDEN  = 64
N_LAYERS = 2
DROPOUT  = 0.2
 
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
 
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        return self.fc(self.dropout(last_hidden)).squeeze(-1)
 
def build_sequences(X_flat, y_flat, seq_len):
    X_seq, y_seq = [], []
    for i in range(seq_len, len(X_flat)):
        X_seq.append(X_flat[i - seq_len: i])
        y_seq.append(y_flat[i])
    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)
 
def train_lstm_fold(X_tr_seq, y_tr_seq, X_val_seq, y_val_seq):
    train_ds = TensorDataset(torch.tensor(X_tr_seq), torch.tensor(y_tr_seq))
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=False)
    model = LSTMClassifier(X_tr_seq.shape[2], HIDDEN, N_LAYERS, DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    criterion = nn.BCEWithLogitsLoss()
    model.train()
    for epoch in range(EPOCHS):
        epoch_loss, n_samples = 0.0, 0
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
            n_samples += len(xb)
        scheduler.step(epoch_loss / n_samples)
    model.eval()
    with torch.no_grad():
        xv = torch.tensor(X_val_seq).to(DEVICE)
        probs = torch.sigmoid(model(xv)).cpu().numpy()
        preds = (probs >= 0.5).astype(int)
    return preds, probs
 
print(f"\n{'-'*55}")
print(f"  LSTM (seq_len={SEQ_LEN}, hidden={HIDDEN}, layers={N_LAYERS})")
print(f"{'-'*55}")
 
lstm_fold_results = []
for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
    X_tr_raw, X_val_raw = X[train_idx], X[val_idx]
    y_tr_raw, y_val_raw = y[train_idx], y[val_idx]
 
    scaler = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_tr_raw)
    X_val_sc = scaler.transform(X_val_raw)
 
    X_tr_seq, y_tr_seq   = build_sequences(X_tr_sc, y_tr_raw, SEQ_LEN)
    X_bridge = np.concatenate([X_tr_sc[-SEQ_LEN:], X_val_sc], axis=0)
    y_bridge = np.concatenate([y_tr_raw[-SEQ_LEN:], y_val_raw], axis=0)
    X_val_seq, y_val_seq = build_sequences(X_bridge, y_bridge, SEQ_LEN)
 
    preds, probs = train_lstm_fold(X_tr_seq, y_tr_seq, X_val_seq, y_val_seq)
    metrics = {
        'accuracy':  accuracy_score(y_val_seq, preds),
        'precision': precision_score(y_val_seq, preds, zero_division=0),
        'recall':    recall_score(y_val_seq, preds, zero_division=0),
        'f1':        f1_score(y_val_seq, preds, zero_division=0),
        'auc':       roc_auc_score(y_val_seq, probs),
        'y_pred': preds, 'y_score': probs, 'y_true': y_val_seq,
    }
    lstm_fold_results.append(metrics)
    print(f"  Fold {fold}: acc={metrics['accuracy']:.4f}  f1={metrics['f1']:.4f}  auc={metrics['auc']:.4f}")
 
print()
for key in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
    vals = [r[key] for r in lstm_fold_results]
    print(f"  Mean {key:9s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
 
all_results['LSTM'] = lstm_fold_results
 
 
# =============================================================================
# 6. SUMMARY TABLE
# =============================================================================
METRICS = ['accuracy', 'precision', 'recall', 'f1', 'auc']
rows = []
for name, folds in all_results.items():
    row = {'Model': name}
    for m in METRICS:
        vals = [f[m] for f in folds]
        row[m.capitalize()] = f"{np.mean(vals):.4f} ± {np.std(vals):.4f}"
    rows.append(row)
 
summary_df = pd.DataFrame(rows).set_index('Model')
print("\n\n=== CROSS-VALIDATED RESULTS (mean ± std, 5 folds) ===")
print(summary_df.to_string())
 
 
# =============================================================================
# 7. PLOTS
# =============================================================================
COLORS_FINAL = {
    'LR (balanced)':   '#378ADD',
    'RidgeClassifier': '#E24B4A',
    'LDA':             '#2EAD6D',
    'kNN':             '#F5A623',
    'LSTM':            '#8B5CF6',
}
 
# 7a. Metric comparison bar chart
fig, axes = plt.subplots(1, len(METRICS), figsize=(20, 5))
fig.suptitle('Model Comparison — Mean CV Metrics (+/- 1 std)', fontweight='bold')
names = list(all_results.keys())
x = np.arange(len(names))
for ax, metric in zip(axes, METRICS):
    means = [np.mean([f[metric] for f in all_results[n]]) for n in names]
    stds  = [np.std( [f[metric] for f in all_results[n]]) for n in names]
    ax.bar(x, means, yerr=stds, capsize=5, width=0.55,
           color=[COLORS_FINAL[n] for n in names], edgecolor='none', alpha=0.85)
    ax.axhline(0.50, color='gray', linestyle='--', linewidth=0.9, label='50% baseline')
    ax.set_title(metric.upper(), fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(' ', '\n') for n in names], fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))
    ax.set_ylim(bottom=max(0.40, min(means) - 0.02))
axes[0].legend(fontsize=8)
plt.tight_layout()
plt.savefig('results_metric_comparison.png', bbox_inches='tight')
plt.close()
 
# 7b. ROC curves (last fold)
fig, ax = plt.subplots(figsize=(7, 5.5))
ax.plot([0, 1], [0, 1], 'k--', linewidth=0.9, label='Random (AUC = 0.50)')
for name, folds in all_results.items():
    last = folds[-1]
    fpr, tpr, _ = roc_curve(last['y_true'], last['y_score'])
    auc_val = roc_auc_score(last['y_true'], last['y_score'])
    ax.plot(fpr, tpr, color=COLORS_FINAL[name], linewidth=1.8,
            label=f"{name} (AUC = {auc_val:.3f})")
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves — Final Fold', fontweight='bold')
ax.legend(fontsize=8, loc='lower right')
plt.tight_layout()
plt.savefig('results_roc_curves.png', bbox_inches='tight')
plt.close()
 
# 7c. Confusion matrices
n_models = len(all_results)
fig, axes = plt.subplots(1, n_models, figsize=(4.2 * n_models, 4))
fig.suptitle('Confusion Matrices — Final Fold (row-normalized)', fontweight='bold')
for ax, (name, folds) in zip(axes, all_results.items()):
    last = folds[-1]
    cm = confusion_matrix(last['y_true'], last['y_pred'])
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_pct, ax=ax, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'],
                cbar=False, linewidths=0.5)
    ax.set_title(name, fontsize=9, fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
plt.tight_layout()
plt.savefig('results_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
 
# 7d. LR feature coefficients
scaler_full = StandardScaler()
X_scaled = scaler_full.fit_transform(X)
lr_final = LogisticRegression(C=1.0, solver='lbfgs', max_iter=300, random_state=42,
                               class_weight='balanced')
lr_final.fit(X_scaled, y)
coefs  = pd.Series(lr_final.coef_[0], index=Features).sort_values()
colors_lr = ['#E24B4A' if v < 0 else '#378ADD' for v in coefs.values]
fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(coefs.index, coefs.values, color=colors_lr, edgecolor='none')
ax.axvline(0, color='gray', linewidth=0.8)
ax.set_title('Logistic Regression — Feature Coefficients (balanced, z-scaled)', fontweight='bold')
ax.set_xlabel('Coefficient value')
plt.tight_layout()
plt.savefig('results_lr_coefficients.png', bbox_inches='tight')
plt.close()
 
# 7e. LDA scalings
lda_final = LinearDiscriminantAnalysis(solver='svd')
lda_final.fit(X_scaled, y)
lda_coefs = pd.Series(lda_final.scalings_.ravel(), index=Features).sort_values()
colors_lda = ['#E24B4A' if v < 0 else '#2EAD6D' for v in lda_coefs.values]
fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(lda_coefs.index, lda_coefs.values, color=colors_lda, edgecolor='none')
ax.axvline(0, color='gray', linewidth=0.8)
ax.set_title('LDA — Discriminant Axis Scalings (z-scaled)', fontweight='bold')
ax.set_xlabel('Scaling value')
plt.tight_layout()
plt.savefig('results_lda_scalings.png', dpi=150, bbox_inches='tight')
plt.close()
 
# 7f. LSTM loss curve
print("\nFitting LSTM on full data for loss curve...")
scaler_lstm = StandardScaler()
X_lstm_full = scaler_lstm.fit_transform(X)
X_full_seq, y_full_seq = build_sequences(X_lstm_full, y, SEQ_LEN)
train_ds = TensorDataset(torch.tensor(X_full_seq), torch.tensor(y_full_seq))
train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=False)
lstm_full = LSTMClassifier(len(Features), HIDDEN, N_LAYERS, DROPOUT).to(DEVICE)
optimizer = torch.optim.Adam(lstm_full.parameters(), lr=LR_RATE)
criterion = nn.BCEWithLogitsLoss()
loss_history = []
lstm_full.train()
for epoch in range(EPOCHS):
    epoch_loss, n_samples = 0.0, 0
    for xb, yb in train_dl:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(lstm_full(xb), yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * len(xb)
        n_samples += len(xb)
    loss_history.append(epoch_loss / n_samples)
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(range(1, EPOCHS + 1), loss_history, color=COLORS_FINAL['LSTM'], linewidth=1.5)
ax.set_xlabel('Epoch')
ax.set_ylabel('Training Loss (BCE)')
ax.set_title(f'LSTM — Training Loss Curve ({N_LAYERS} layers, hidden={HIDDEN})', fontweight='bold')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('results_lstm_loss_curve.png', dpi=150, bbox_inches='tight')
plt.close()
 
# 7g. Per-fold accuracy over time
fig, ax = plt.subplots(figsize=(8, 4.5))
folds_x = np.arange(1, N_SPLITS + 1)
for name, folds in all_results.items():
    ax.plot(folds_x, [f['accuracy'] for f in folds], marker='o', linewidth=1.8,
            color=COLORS_FINAL[name], label=name)
ax.axhline(0.50, color='gray', linestyle='--', linewidth=0.9, label='50% baseline')
ax.set_xlabel('Fold (chronological)')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy per Fold — Temporal Stability', fontweight='bold')
ax.set_xticks(folds_x)
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig('results_fold_accuracy.png', bbox_inches='tight')
plt.close()
 
# 7h. Novel: Regime split accuracy — compare model accuracy in bull vs bear regimes
print("\nComputing regime-split accuracy...")
fig, ax = plt.subplots(figsize=(9, 4.5))
x_pos = np.arange(len(all_results))
width = 0.35
 
# Use final fold only for regime analysis
regime_bull_acc, regime_bear_acc = [], []
for name, folds in all_results.items():
    last = folds[-1]
    y_true, y_pred = last['y_true'], last['y_pred']
    # Align regime labels to final fold val indices
    val_idx = list(tscv.split(X))[-1][1]
    # For LSTM, val_idx is offset by SEQ_LEN
    if name == 'LSTM':
        regime_vals = df['bull_regime'].values[val_idx[SEQ_LEN:] if len(val_idx) > len(y_true) else val_idx[:len(y_true)]]
    else:
        regime_vals = df['bull_regime'].values[val_idx[:len(y_true)]]
    bull_mask = regime_vals == 1
    bear_mask = regime_vals == 0
    ba = accuracy_score(y_true[bull_mask], y_pred[bull_mask]) if bull_mask.sum() > 0 else np.nan
    bea = accuracy_score(y_true[bear_mask], y_pred[bear_mask]) if bear_mask.sum() > 0 else np.nan
    regime_bull_acc.append(ba)
    regime_bear_acc.append(bea)
 
ax.bar(x_pos - width/2, regime_bull_acc, width, label='Bull Regime', color='#378ADD', alpha=0.85, edgecolor='none')
ax.bar(x_pos + width/2, regime_bear_acc, width, label='Bear Regime', color='#E24B4A', alpha=0.85, edgecolor='none')
ax.axhline(0.50, color='gray', linestyle='--', linewidth=0.9)
ax.set_xticks(x_pos)
ax.set_xticklabels(list(all_results.keys()), fontsize=9)
ax.set_ylabel('Accuracy')
ax.set_title('Model Accuracy by Market Regime — Final Fold', fontweight='bold')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig('results_regime_accuracy.png', dpi=150, bbox_inches='tight')
plt.close()
 
print("\nDone. All result plots saved.")