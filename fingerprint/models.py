"""FingerprintCNN (1D CNN on I/Q) and FingerprintRNN (BiLSTM on the I/Q sequence).

Both take x of shape (B, 2, L) (real/imag channels) and optionally a vector of handcrafted
features (B, F) that is fused before the classifier head. forward() returns class logits;
embed() returns the penultimate embedding (useful for open-set matching of unseen aircraft).
"""
import torch
from torch import nn


def _conv_block(cin, cout, k, stride=1):
    return nn.Sequential(nn.Conv1d(cin, cout, k, stride=stride, padding=k // 2, bias=False),
                         nn.BatchNorm1d(cout), nn.ReLU(inplace=True))


class _Head(nn.Module):
    def __init__(self, emb_in, n_classes, feat_dim, emb=128, dropout=0.3):
        super().__init__()
        self.feat = None
        if feat_dim:
            self.feat = nn.Sequential(nn.Linear(feat_dim, 64), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.emb = nn.Sequential(nn.Linear(emb_in + (64 if feat_dim else 0), emb),
                                 nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.out = nn.Linear(emb, n_classes)

    def embed(self, h, feats):
        if self.feat is not None:
            h = torch.cat([h, self.feat(feats)], dim=1)
        return self.emb(h)

    def forward(self, h, feats):
        return self.out(self.embed(h, feats))


class FingerprintCNN(nn.Module):
    def __init__(self, n_classes, feat_dim=0, base=32, dropout=0.3):
        super().__init__()
        c = [base, base * 2, base * 4, base * 4]
        self.body = nn.Sequential(
            _conv_block(2, c[0], 7), _conv_block(c[0], c[0], 5),
            _conv_block(c[0], c[1], 5, stride=2), _conv_block(c[1], c[1], 3),
            _conv_block(c[1], c[2], 3, stride=2), _conv_block(c[2], c[2], 3),
            _conv_block(c[2], c[3], 3, stride=2))
        self.head = _Head(c[3] * 2, n_classes, feat_dim, dropout=dropout)

    def _pool(self, x):
        h = self.body(x)
        return torch.cat([h.mean(-1), h.amax(-1)], dim=1)

    def embed(self, x, feats=None):
        return self.head.embed(self._pool(x), feats)

    def forward(self, x, feats=None):
        return self.head(self._pool(x), feats)


class FingerprintRNN(nn.Module):
    def __init__(self, n_classes, feat_dim=0, hidden=128, layers=2, dropout=0.3):
        super().__init__()
        self.front = nn.Sequential(_conv_block(2, 32, 5), _conv_block(32, 64, 3, stride=2))
        self.lstm = nn.LSTM(64, hidden, num_layers=layers, batch_first=True, bidirectional=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.head = _Head(hidden * 4, n_classes, feat_dim, dropout=dropout)

    def _pool(self, x):
        h = self.front(x).transpose(1, 2)           # (B, L/2, 64)
        o, _ = self.lstm(h)
        return torch.cat([o.mean(1), o[:, -1]], dim=1)

    def embed(self, x, feats=None):
        return self.head.embed(self._pool(x), feats)

    def forward(self, x, feats=None):
        return self.head(self._pool(x), feats)


def build(name, n_classes, feat_dim=0):
    return {"cnn": FingerprintCNN, "rnn": FingerprintRNN}[name](n_classes, feat_dim)
