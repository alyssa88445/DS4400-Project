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
# 0. LOAD DATA & SAMPLE STOCKS
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
# 1. FEATURE ENGINEERING
# =============================================================================
Features = [
    'return_1d', 'return_5d', 'return_10d',
    'ma_5', 'ma_20', 'ma_50',
    'price_to_ma20', 'price_to_ma50',
    'vol_10', 'vol_20', 'rsi_14', 'price_range', 'vol_ratio'
]

grouped = df.groupby('Symbol')

# Returns
df['return_1d']  = grouped['Close'].pct_change(1)
df['return_5d']  = grouped['Close'].pct_change(5)
df['return_10d'] = grouped['Close'].pct_change(10)

# Moving averages
df['ma_5']  = grouped['Close'].rolling(5).mean().reset_index(level=0, drop=True)
df['ma_20'] = grouped['Close'].rolling(20).mean().reset_index(level=0, drop=True)
df['ma_50'] = grouped['Close'].rolling(50).mean().reset_index(level=0, drop=True)

# Price relative to MAs
df['price_to_ma20'] = df['Close'] / df['ma_20'] - 1
df['price_to_ma50'] = df['Close'] / df['ma_50'] - 1

# Volatility
df['vol_10'] = grouped['return_1d'].rolling(10).std().reset_index(level=0, drop=True)
df['vol_20'] = grouped['return_1d'].rolling(20).std().reset_index(level=0, drop=True)

# Price range & volume ratio
df['price_range'] = (df['High'] - df['Low']) / df['Close']
df['vol_ratio']   = df['Volume'] / grouped['Volume'].rolling(20).mean().reset_index(level=0, drop=True)

# Target: 1 if next day close > current close, else 0
df['target'] = (grouped['Close'].shift(-1) > df['Close']).astype(int)

# RSI-14 (Wilder smoothing via 14-day rolling mean of gains/losses)
# Interpretation: RSI > 70 → overbought (bearish signal), RSI < 30 → oversold (bullish signal).
def calc_rsi_vectorized(series):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss
    return 100 - (100 / (1 + rs))

df['rsi_14'] = grouped['Close'].apply(calc_rsi_vectorized).reset_index(level=0, drop=True)

# Drop NaNs and convert to float32
df = df.dropna(subset=Features + ['target']).copy()
df[Features] = df[Features].astype(np.float32)

print(f"Usable rows after feature engineering: {len(df):,}")

# Class balance check
class_counts = pd.Series(df['target'].values).value_counts().sort_index()
print("\nClass distribution:")
print(f"  0 (Down): {class_counts.get(0, 0):,}  ({class_counts.get(0, 0)/len(df)*100:.1f}%)")
print(f"  1 (Up)  : {class_counts.get(1, 0):,}  ({class_counts.get(1, 0)/len(df)*100:.1f}%)")

X = df[Features].values
y = df['target'].values

# =============================================================================
# 2. TIMESERIES SPLIT EVALUATION
# =============================================================================
N_SPLITS = 5
tscv = TimeSeriesSplit(n_splits=N_SPLITS)

COLORS = {
    'Logistic Regression': '#378ADD',
    'RidgeClassifier':     '#E24B4A',
    'LDA':                 '#2EAD6D',
    'kNN':                 '#F5A623',
    'LSTM':                '#8B5CF6',
}

def run_cv(model_name, make_model_fn, X, y, tscv):
    fold_results = []
    print(f"\n{'─'*52}")
    print(f"  {model_name}")
    print(f"{'─'*52}")

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_tr_raw, X_val_raw = X[train_idx], X[val_idx]
        y_tr, y_val         = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_tr   = scaler.fit_transform(X_tr_raw)
        X_val  = scaler.transform(X_val_raw)

        model = make_model_fn()
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)

        if hasattr(model, 'predict_proba'):
            y_score = model.predict_proba(X_val)[:, 1]
        else:
            y_score = model.decision_function(X_val)

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
# 3a. TRAIN sklean MODELS (LR, Ridge, LDA, kNN)
# =============================================================================
MODELS = {
    'Logistic Regression': lambda: LogisticRegression(C=1.0, solver='lbfgs', max_iter=300, random_state=42),
    'RidgeClassifier':     lambda: RidgeClassifier(alpha=1.0),
    'LDA':                 lambda: LinearDiscriminantAnalysis(solver='svd'),
    'kNN':                 lambda: KNeighborsClassifier(n_neighbors=21, weights='distance', metric='minkowski', n_jobs=-1),

}

all_results = {}
for name, factory in MODELS.items():
    all_results[name] = run_cv(name, factory, X, y, tscv)

# =============================================================================
# 3b. LSTM with TimeSeriesSplit (PyTorch)
# =============================================================================
# The LSTM sees a sliding window of SEQ_LEN consecutive days of features
# and predicts whether the stock goes up on the day after the window. 
# Each fold uses the same TimeSeriesSplit indices as the sklearn models
# so the forward-chaining discipline is identicial

SEQ_LEN = 10 # look back window: 10 day trading days, predict day 11
BATCH = 256
EPOCHS = 30
LR = 1e-3
HIDDEN = 64
N_LAYERS = 2
DROPOUT = 0.2

class LSTMClassifier(nn.Module):
    """ LSTM for binary classification on time-series feature sequences """
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size = input_size, 
            hidden_size = hidden_size,
            num_layers = num_layers,
            batch_first = True,
            dropout = dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :] # take last time step
        out = self.dropout(last_hidden)
        return self.fc(out).squeeze(-1) # (batch) raw logits

def build_sequences(X_flat, y_flat, seq_len):
    """
    Slide a window of length 'seq_len' over the flat (row-ordered) feature
    matrix. The label for each window is the target of the last row in the window. 
    Returns (X_seq, y_seq) as numpy arrays
    """
    X_seq, y_seq = [], []
    for i in range(seq_len, len(X_flat)):
        X_seq.append(X_flat[i - seq_len : i])
        y_seq.append(y_flat[i])
    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)

def train_lstm_fold(X_tr_seq, y_tr_seq, X_val_seq, y_val_seq):
    """ Train one LSTM fold and return predictions and probability scores """
    train_ds = TensorDataset(torch.tensor(X_tr_seq), torch.tensor(y_tr_seq))
    train_d1 = DataLoader(train_ds, batch_size=BATCH, shuffle=False)

    model = LSTMClassifier(
        input_size = X_tr_seq.shape[2], hidden_size = HIDDEN,
        num_layers = N_LAYERS, dropout = DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    # Training
    model.train()
    for epoch in range(EPOCHS):
        for xb, yb in train_d1:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
    
    # Validation
    model.eval()
    with torch.no_grad():
        xv = torch.tensor(X_val_seq).to(DEVICE)
        logits = model(xv)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs >= 0.5).astype(int)
    return preds, probs

print(f"\n{'-'*52}")
print(f" LSTM (seq_len={SEQ_LEN}, hidden={HIDDEN}, layers={N_LAYERS})")
print(f"\n{'-'*52}")

lstm_fold_results = []

for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
    X_tr_raw, X_val_raw = X[train_idx], X[val_idx]
    y_tr_raw, y_val_raw = y[train_idx], y[val_idx]

    # Scale using training data only 
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr_raw)
    X_val_sc = scaler.transform(X_val_raw)

    # Build sequences from the scaled data
    # Training sequences: built entirely within training set
    X_tr_seq, y_tr_seq = build_sequences(X_tr_sc, y_tr_raw, SEQ_LEN)

    # Validation sequences: bridge last SEQ_LEN training rows as context 
    # so the first validation prediction has a full look back window
    # The bridged training rows are already scaled
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
        'y_pred':    preds,
        'y_score':   probs,
        'y_true':    y_val_seq,
    }
    lstm_fold_results.append(metrics)
    print(f" Fold {fold}: acc={metrics['accuracy']:.4f}  f1={metrics['f1']:.4f}  auc={metrics['auc']:.4f}")

print()
for key in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
    vals = [r[key] for r in lstm_fold_results]
    print(f" Mean {key:9s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

all_results['LSTM'] = lstm_fold_results

# =============================================================================
# 4. SUMMARY TABLE
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
# 5. PLOTS
# =============================================================================

# 5a. Metric comparison bar chart
fig, axes = plt.subplots(1, len(METRICS), figsize=(20, 5))
fig.suptitle('Model Comparison — Mean CV Metrics (+/- 1 std)', fontweight='bold')

names = list(all_results.keys())
x = np.arange(len(names))

for ax, metric in zip(axes, METRICS):
    means = [np.mean([f[metric] for f in all_results[n]]) for n in names]
    stds  = [np.std( [f[metric] for f in all_results[n]]) for n in names]
    ax.bar(x, means, yerr=stds, capsize=5, width=0.55,
           color=[COLORS[n] for n in names], edgecolor='none', alpha=0.85)
    ax.axhline(0.50, color='gray', linestyle='--', linewidth=0.9, label='50% baseline')
    ax.set_title(metric.upper(), fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(' ', '\n') for n in names], fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))
    bottom = min(means) - 0.02
    ax.set_ylim(bottom=max(0.40, bottom))

axes[0].legend(fontsize=8)
plt.tight_layout()
plt.savefig('results_metric_comparison.png', bbox_inches='tight')
plt.close()

# 5b. ROC curves (last fold)
fig, ax = plt.subplots(figsize=(7, 5.5))
ax.plot([0, 1], [0, 1], 'k--', linewidth=0.9, label='Random (AUC = 0.50)')

for name, folds in all_results.items():
    last = folds[-1]
    fpr, tpr, _ = roc_curve(last['y_true'], last['y_score'])
    auc_val = roc_auc_score(last['y_true'], last['y_score'])
    ax.plot(fpr, tpr, color=COLORS[name], linewidth=1.8,
            label=f"{name} (AUC = {auc_val:.3f})")

ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves — Final Fold', fontweight='bold')
ax.legend(fontsize=8, loc='lower right')
plt.tight_layout()
plt.savefig('results_roc_curves.png', bbox_inches='tight')
plt.close()

# 5c. Confusion matrices
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

# 5d. Logistic Regression feature coefficients
scaler_full = StandardScaler()
X_scaled = scaler_full.fit_transform(X)
lr_final = LogisticRegression(C=1.0, solver='lbfgs', max_iter=300, random_state=42)
lr_final.fit(X_scaled, y)

coefs  = pd.Series(lr_final.coef_[0], index=Features).sort_values()
colors = ['#E24B4A' if v < 0 else '#378ADD' for v in coefs.values]

fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(coefs.index, coefs.values, color=colors, edgecolor='none')
ax.axvline(0, color='gray', linewidth=0.8)
ax.set_title('Logistic Regression — Feature Coefficients\n(full dataset, z-score scaled)',
             fontweight='bold')
ax.set_xlabel('Coefficient value')
plt.tight_layout()
plt.savefig('results_lr_coefficients.png', bbox_inches='tight')
plt.close()

# 5e. LDA feature weights (scalings on the single discriminant axis)
lda_final = LinearDiscriminantAnalysis(solver='svd')
lda_final.fit(X_scaled, y)

lda_coefs = pd.Series(lda_final.scalings_.ravel(), index=Features).sort_values()
colors_lda = ['#E24B4A' if v < 0 else '#2EAD6D' for v in lda_coefs.values]

fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(lda_coefs.index, lda_coefs.values, color=colors_lda, edgecolor='none')
ax.axvline(0, color='gray', linewidth=0.8)
ax.set_title('LDA - Discriminant Axis Scalings\n(sampled dataset, z-scaled)', fontweight='bold')
ax.set_xlabel('Scaling value')
plt.tight_layout()
plt.savefig('results_lda_scalings.png', dpi=150, bbox_inches='tight')
plt.close()

# 5f. LSTM training loss curve (full data refit for visualization)
print("\nFitting LSTM on full data for loss curve plot...")
scaler_lstm = StandardScaler()
X_lstm_full = scaler_lstm.fit_transform(X)
X_full_seq, y_full_seq = build_sequences(X_lstm_full, y, SEQ_LEN)

train_ds = TensorDataset(torch.tensor(X_full_seq), torch.tensor(y_full_seq))
train_d1 = DataLoader(train_ds, batch_size=BATCH, shuffle=False)

lstm_full = LSTMClassifier(input_size = len(Features), hidden_size= HIDDEN, num_layers=N_LAYERS, dropout=DROPOUT).to(DEVICE)
optimizer = torch.optim.Adam(lstm_full.parameters(), lr = LR)
criterion = nn.BCEWithLogitsLoss()

loss_history = []
lstm_full.train()
for epoch in range(EPOCHS):
    epoch_loss = 0.0
    n_samples = 0
    for xb, yb in train_d1:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        logits = lstm_full(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * len(xb)
        n_samples += len(xb)
    loss_history.append(epoch_loss / n_samples)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(range(1, EPOCHS + 1), loss_history, color=COLORS['LSTM'], linewidth=1.5)
ax.set_xlabel('Epoch')
ax.set_ylabel('Training Loss (BCE)')
ax.set_title(f'LSTM - Training Loss Curve\n{N_LAYERS} layers, hidden={HIDDEN}, seq_len={SEQ_LEN}', fontweight='bold')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('results_lstm_loss_curve.png', dpi=150, bbox_inches='tight')
plt.close()

# 5g. Per-fold accuracy over time
fig, ax = plt.subplots(figsize=(8, 4.5))
folds_x = np.arange(1, N_SPLITS + 1)

for name, folds in all_results.items():
    accs = [f['accuracy'] for f in folds]
    ax.plot(folds_x, accs, marker='o', linewidth=1.8,
            color=COLORS[name], label=name)

ax.axhline(0.50, color='gray', linestyle='--', linewidth=0.9, label='50% baseline')
ax.set_xlabel('Fold (chronological, earlier → later)')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy per Fold — Temporal Stability', fontweight='bold')
ax.set_xticks(folds_x)
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig('results_fold_accuracy.png', bbox_inches='tight')
plt.close()

# RSI sanity check
print("\n\n=== RSI CHECK ===")
print(f"LR coefficient for rsi_14: {lr_final.coef_[0][Features.index('rsi_14')]:.6f}")
print(f"LDA scaling for rsi_14: {lda_final.scalings_[Features.index('rsi_14')][0]:.6f}")
print("Interpretation: a positive coefficient means higher RSI which predicts UP")
print("This is counterintuitive (RSI > 70 = overbought which is typically bearish).")
print("If positive, this may reflect momentum persistence in the sample per period")
print("rather than a mean-reversion signal. Verify by checking class conditional RSI means:")

up_rsi = df.loc[df['target'] == 1, 'rsi_14'].mean()
down_rsi = df.loc[df['target'] == 0, 'rsi_14'].mean()

print(f" Mean RSI when target = 1 (Up): {up_rsi:.2f}")
print(f" Mean RSI when target = 0 (Down): {down_rsi:.2f}")

print("\nDone. All result plots saved.")