# emg_id_cnn.py
import os, math, glob, json, random
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# Utilities
# -----------------------------
def bandpass_notch(x, fs=200, bp=(20,450), notch=None):
    # x: (T, C). Implement with scipy if available; here we keep it as a no-op placeholder.
    # Replace with proper filters in your pipeline (scipy.signal butter + filtfilt, + iirnotch).
    return x

def zscore_per_channel(x, mean, std, eps=1e-8):
    return (x - mean) / (std + eps)

def compute_channel_stats(arrays: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    # arrays: list of (T, C)
    concat = np.concatenate(arrays, axis=0)  # (sumT, C)
    mean = concat.mean(axis=0)
    std  = concat.std(axis=0)
    return mean, std

def load_emg_array(path: str) -> np.ndarray:
    # Expect .npy with shape (T, 8). Modify if you store CSV.
    # If CSV: use np.loadtxt(path, delimiter=',') or pandas.read_csv(...).values
    return np.load(path)

# -----------------------------
# Windowing Dataset
# -----------------------------
class EMGWindows(Dataset):
    def __init__(self, meta: pd.DataFrame, label_encoder: LabelEncoder,
                 sessions_for_stats: List[int], window_ms=200, fs=200, overlap=0.5,
                 apply_filter=False, notch=None, channel_stats: Dict[str, np.ndarray]=None,
                 augment=False):
        """
        meta: DataFrame with [path, session, subject, gesture]
        sessions_for_stats: which sessions are used to fit normalization stats
        channel_stats: if provided, dict with 'mean' (8,), 'std' (8,)
        """
        self.meta = meta.reset_index(drop=True)
        self.le = label_encoder
        self.fs = fs
        self.win = int(round(window_ms * fs / 1000))
        self.hop = max(1, int(round(self.win * (1 - overlap))))
        self.apply_filter = apply_filter
        self.notch = notch
        self.augment = augment

        # Load all arrays once and pre-window
        self.samples = []  # list of (C, T), label_id
        self.labels = []
        raw_for_stats = []

        # Preload arrays and (optionally) accumulate for stats
        cache = {}
        for i, row in self.meta.iterrows():
            p = row['path']
            if p not in cache:
                arr = load_emg_array(p)  # (T, 8)
                if self.apply_filter:
                    arr = bandpass_notch(arr, fs=self.fs, bp=(20,450), notch=self.notch)
                cache[p] = arr
            arr = cache[p]
            # Collect for stats only from specified sessions
            if int(row['session']) in sessions_for_stats:
                raw_for_stats.append(arr)

        # Fit or use provided channel stats
        if channel_stats is None:
            mean, std = compute_channel_stats(raw_for_stats if raw_for_stats else [np.zeros((1,8))])
            self.stats = {'mean': mean.astype(np.float32), 'std': std.astype(np.float32)}
        else:
            self.stats = channel_stats

        # Now window all recordings
        for i, row in self.meta.iterrows():
            p, sess, subj = row['path'], int(row['session']), str(row['subject'])
            y = self.le.transform([subj])[0]
            arr = cache[p]  # (T, 8)

            # Normalize with train stats
            arr = zscore_per_channel(arr, self.stats['mean'], self.stats['std'])

            T = arr.shape[0]
            for start in range(0, max(0, T - self.win + 1), self.hop):
                w = arr[start:start+self.win]                 # (win, 8)
                w = w.T.astype(np.float32, copy=False)       # (C=8, T)
                self.samples.append(w)
                self.labels.append(y)

        self.samples = np.stack(self.samples)  # (N, C, T)
        self.labels  = np.array(self.labels, dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def channel_dropout(self, x, p=0.1):
        # x: (C,T)
        if random.random() < p:
            c = random.randrange(x.shape[0])
            x[c, :] = 0.0
        return x

    def gaussian_noise(self, x, std=0.02):
        return x + np.random.randn(*x.shape).astype(np.float32)*std

    def time_jitter(self, x, max_shift=2):
        # shift along time axis by -max_shift..+max_shift samples
        s = random.randint(-max_shift, max_shift)
        if s == 0: return x
        if s > 0:
            return np.concatenate([x[:, s:], np.zeros((x.shape[0], s), dtype=x.dtype)], axis=1)
        else:
            s = abs(s)
            return np.concatenate([np.zeros((x.shape[0], s), dtype=x.dtype), x[:, :-s]], axis=1)

    def __getitem__(self, idx):
        x = self.samples[idx].copy()
        y = self.labels[idx]
        if self.augment:
            if random.random() < 0.5: x = self.gaussian_noise(x, std=0.01)
            if random.random() < 0.3: x = self.channel_dropout(x, p=1.0)  # drop exactly one channel sometimes
            if random.random() < 0.3: x = self.time_jitter(x, max_shift=2)
        return torch.from_numpy(x), torch.tensor(y)

# -----------------------------
# 2D CNN over (Channels x Time)
# -----------------------------
class EMG2DCNN(nn.Module):
    def __init__(self, n_classes: int, in_ch=1):
        super().__init__()
        # Treat (C=8, T) as an "image": add a fake 1 channel dim -> (N, 1, 8, T)
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 16, kernel_size=(3,5), padding=(1,2)),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,2)),     # -> (16, 4, T/2)

            nn.Conv2d(16, 32, kernel_size=(3,5), padding=(1,2)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,2)),     # -> (32, 2, T/4)

            nn.Conv2d(32, 64, kernel_size=(2,3), padding=(0,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d((1, 8)),        # -> (64, 1, 8)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                         # 64*1*8 = 512
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        # x: (N, C=8, T) -> add channel dim
        x = x.unsqueeze(1)           # (N,1,8,T)
        x = self.features(x)
        x = self.classifier(x)
        return x

# -----------------------------
# Train / Eval
# -----------------------------
def train_epoch(model, loader, optim, loss_fn, device):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optim.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        optim.step()
        loss_sum += loss.item() * yb.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)
    return loss_sum/total, correct/total

@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    all_y, all_p = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()
        all_p.extend(preds)
        all_y.extend(yb.numpy().tolist())
    acc = accuracy_score(all_y, all_p)
    f1  = f1_score(all_y, all_p, average='macro')
    return acc, f1

def make_label_encoder(subject_list: List[str]) -> LabelEncoder:
    le = LabelEncoder()
    le.fit(subject_list)
    return le

# -----------------------------
# Entry: provide your metadata CSV
# -----------------------------
def main(meta_csv: str, out_dir="runs_emg_id",
         window_ms=200, fs=200, overlap=0.5,
         batch_size=256, epochs=30, lr=1e-3, weight_decay=1e-4,
         train_sessions=(1,2), test_session=3, augment=True):

    os.makedirs(out_dir, exist_ok=True)
    meta = pd.read_csv(meta_csv)  # columns: path, session, subject, gesture
    meta['session'] = meta['session'].astype(int)
    subjects = sorted(meta['subject'].unique().tolist())
    le = make_label_encoder(subjects)
    n_classes = len(subjects)

    train_meta = meta[meta.session.isin(train_sessions)].copy()
    test_meta  = meta[meta.session == test_session].copy()

    # Build train dataset to estimate normalization stats from train sessions only
    tmp_ds = EMGWindows(train_meta, le, sessions_for_stats=list(train_sessions),
                        window_ms=window_ms, fs=fs, overlap=overlap,
                        apply_filter=False, notch=None, augment=False)
    stats = tmp_ds.stats  # {'mean':(8,), 'std':(8,)}

    # Rebuild with augmentation + fixed stats
    train_ds = EMGWindows(train_meta, le, sessions_for_stats=list(train_sessions),
                          window_ms=window_ms, fs=fs, overlap=overlap,
                          apply_filter=False, notch=None, augment=augment,
                          channel_stats=stats)
    test_ds  = EMGWindows(test_meta,  le, sessions_for_stats=list(train_sessions),
                          window_ms=window_ms, fs=fs, overlap=overlap,
                          apply_filter=False, notch=None, augment=False,
                          channel_stats=stats)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EMG2DCNN(n_classes=n_classes).to(device)

    # Optional: class weights if subjects are imbalanced by window count
    class_counts = np.bincount(train_ds.labels, minlength=n_classes)
    weights = (class_counts.sum() / (class_counts + 1e-8))
    weights = weights / weights.mean()
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='max', factor=0.5, patience=3, verbose=False)

    best = {'acc': 0.0, 'f1': 0.0, 'epoch': -1}
    for ep in range(1, epochs+1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optim, loss_fn, device)
        te_acc, te_f1   = eval_epoch(model, test_loader, device)
        scheduler.step(te_acc)

        print(f"Epoch {ep:02d} | train loss {tr_loss:.4f} acc {tr_acc:.3f} | test acc {te_acc:.3f} f1 {te_f1:.3f}")

        if te_acc > best['acc']:
            best = {'acc': te_acc, 'f1': te_f1, 'epoch': ep}
            torch.save({
                'model': model.state_dict(),
                'label_encoder_classes': subjects,
                'stats': stats,
                'cfg': {
                    'window_ms': window_ms, 'fs': fs, 'overlap': overlap
                }
            }, os.path.join(out_dir, f'best_ep{ep:02d}_acc{te_acc:.3f}.pt'))

    print("Best:", best)

if __name__ == "__main__":
    # Example:
    # python emg_id_cnn.py meta.csv
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("meta_csv", type=str, help="CSV with columns: path,session,subject,gesture")
    ap.add_argument("--out_dir", type=str, default="runs_emg_id")
    ap.add_argument("--window_ms", type=int, default=200)
    ap.add_argument("--fs", type=int, default=200)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--train_sessions", type=int, nargs="+", default=[1,2])
    ap.add_argument("--test_session", type=int, default=3)
    ap.add_argument("--augment", action="store_true")
    args = ap.parse_args()
    main(**vars(args))

