"""
Run this script on YOUR machine to fetch real 2-year stock data
and regenerate tsne.csv for HW3.

Usage:
    pip install yfinance torch scikit-learn pandas
    python generate_hw3_data.py
"""

import yfinance as yf
import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.manifold import TSNE

# ── Config ───────────────────────────────────────────────────────────────────

TICKERS = [
    'XOM', 'CVX', 'HAL',                        # Energy
    'MMM', 'CAT', 'DAL',                         # Industrials
    'MCD', 'NKE', 'KO',                          # Consumer
    'JNJ', 'PFE', 'UNH',                         # Healthcare
    'JPM', 'GS', 'BAC',                          # Financials
    'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META'     # Technology
]

SECTORS = {
    'XOM': 'Energy',    'CVX': 'Energy',    'HAL': 'Energy',
    'MMM': 'Industrials','CAT': 'Industrials','DAL': 'Industrials',
    'MCD': 'Consumer',  'NKE': 'Consumer',  'KO':  'Consumer',
    'JNJ': 'Healthcare','PFE': 'Healthcare','UNH': 'Healthcare',
    'JPM': 'Financials','GS':  'Financials','BAC': 'Financials',
    'AAPL':'Technology','MSFT':'Technology','NVDA':'Technology',
    'GOOGL':'Technology','META':'Technology'
}

# ── Step 1: Fetch stock data ─────────────────────────────────────────────────

def fetch_and_save(tickers, period='2y', folder='stockdata'):
    os.makedirs(folder, exist_ok=True)
    for ticker in tickers:
        df = yf.Ticker(ticker).history(period=period)
        df = df.reset_index()
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        df['Date'] = df['Date'].dt.tz_localize(None)
        df.to_csv(os.path.join(folder, f'{ticker}.csv'), index=False)
        print(f'Saved {ticker}.csv — {len(df)} rows')

# ── Step 2: LSTM Autoencoder ─────────────────────────────────────────────────

class StockDataset(Dataset):
    def __init__(self, folder_path='stockdata',
                 feature_cols=['Open', 'High', 'Low', 'Close', 'Volume']):
        self.sequences = []
        self.tickers = []
        for fname in sorted(os.listdir(folder_path)):
            if not fname.endswith('.csv'):
                continue
            df = pd.read_csv(os.path.join(folder_path, fname))
            df = df[feature_cols].dropna()
            df = (df - df.min()) / (df.max() - df.min() + 1e-8)
            self.sequences.append(torch.tensor(df.values, dtype=torch.float32))
            self.tickers.append(fname.replace('.csv', ''))

    def __len__(self): return len(self.sequences)
    def __getitem__(self, idx): return self.sequences[idx]


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, latent_dim=16, seq_len=500):
        super().__init__()
        self.seq_len = seq_len
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.encoder_fc   = nn.Linear(hidden_dim, latent_dim)
        self.decoder_fc   = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output_fc    = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        _, (h_n, _) = self.encoder_lstm(x)
        z = self.encoder_fc(h_n.squeeze(0))
        dec = self.decoder_fc(z).unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _ = self.decoder_lstm(dec)
        return self.output_fc(out), z


def train_and_get_latents(folder='stockdata', num_epochs=50):
    dataset = StockDataset(folder)
    seq_len = dataset[0].shape[0]
    dataloader = DataLoader(dataset, batch_size=1)

    model = LSTMAutoencoder(seq_len=seq_len)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(num_epochs):
        total = 0
        for batch in dataloader:
            optimizer.zero_grad()
            recon, _ = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            total += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}  loss={total/len(dataloader):.6f}')

    model.eval()
    latents = []
    with torch.no_grad():
        for batch in dataloader:
            _, z = model(batch)
            latents.append(z)

    return torch.cat(latents, dim=0).numpy(), dataset.tickers


# ── Step 3: t-SNE → tsne.csv ─────────────────────────────────────────────────

def generate_tsne(Z, tickers, sectors):
    tsne = TSNE(n_components=2, random_state=42, perplexity=5)
    Z2 = tsne.fit_transform(Z)
    df = pd.DataFrame({
        'ticker': tickers,
        'x': Z2[:, 0],
        'y': Z2[:, 1],
        'sector': [sectors[t] for t in tickers]
    })
    df.to_csv('tsne.csv', index=False)
    print('Saved tsne.csv')
    print(df)


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('=== Step 1: Fetching stock data ===')
    fetch_and_save(TICKERS, period='2y')

    print('\n=== Step 2: Training autoencoder ===')
    Z, tickers = train_and_get_latents()

    print('\n=== Step 3: Generating tsne.csv ===')
    generate_tsne(Z, tickers, SECTORS)

    print('\nDone! Copy stockdata/ and tsne.csv into your HW3 data/ folder.')