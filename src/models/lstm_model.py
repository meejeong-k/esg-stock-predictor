# src/models/lstm_model.py
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
import os
import warnings
warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"사용 디바이스: {DEVICE}")

ESG_RISK = {
    '005930': 38.12, '000660': 47.45, '005380': 42.84,
    '373220': 44.06, '207940': 43.36, '005490': 34.15,
    '105560': 36.17, '055550': 36.20, '051910': 30.92,
    '035720': 38.80,
}
DEFAULT_ESG = 39.0  # 나머지 종목 기본값


def load_and_merge_data():
    print("\n[1/5] 데이터 로드 및 통합 중...")

    stock_df = pd.read_csv('data/parquet/stocks/data.csv',
                           encoding='utf-8-sig', low_memory=False)
    stock_df['date']   = pd.to_datetime(stock_df['date'])
    stock_df['ticker'] = stock_df['ticker'].astype(str).str.zfill(6)

    weather_df = pd.read_csv('data/parquet/weather/data.csv',
                             encoding='utf-8-sig')
    weather_df['date'] = pd.to_datetime(weather_df['date'])

    # 기후 데이터 병합
    df = stock_df.merge(
        weather_df[['date', 'national_avg_temp', 'national_climate_risk']],
        on='date', how='left'
    )
    df['national_avg_temp']     = df.groupby('ticker')['national_avg_temp'].ffill()
    df['national_climate_risk'] = df['national_climate_risk'].fillna(0)

    # ESG 리스크 점수 추가
    df['esg_risk_score'] = df['ticker'].map(ESG_RISK).fillna(DEFAULT_ESG)

    # 레이블: 다음날 종가 상승 여부 (종목별)
    df = df.sort_values(['ticker', 'date'])
    df['next_close'] = df.groupby('ticker')['close'].shift(-1)
    df['label']      = (df['next_close'] > df['close']).astype(int)
    df = df.dropna(subset=['label', 'close', 'ma5', 'ma20'])

    print(f"  전체 데이터: {len(df):,}행")
    print(f"  종목 수: {df['ticker'].nunique()}")
    print(f"  기간: {df['date'].min().date()} ~ {df['date'].max().date()}")
    print(f"  상승: {df['label'].sum():,}일 / 하락: {(df['label']==0).sum():,}일")
    return df


def make_sequences(df, seq_len=30):
    print(f"\n[2/5] 시퀀스 생성 중 (윈도우={seq_len}일)...")

    features = ['open', 'high', 'low', 'close', 'volume',
                'ma5', 'ma20', 'ma60', 'daily_return',
                'national_avg_temp', 'national_climate_risk',
                'esg_risk_score']

    df[features] = df[features].fillna(0)

    # 종목별 정규화 + 시퀀스 생성
    all_X, all_y = [], []
    tickers = df['ticker'].unique()

    for ticker in tickers:
        t_df = df[df['ticker'] == ticker].copy().reset_index(drop=True)
        if len(t_df) < seq_len + 1:
            continue

        scaler = StandardScaler()
        t_df[features] = scaler.fit_transform(t_df[features])

        for i in range(seq_len, len(t_df)):
            all_X.append(t_df[features].iloc[i-seq_len:i].values)
            all_y.append(t_df['label'].iloc[i])

    X = np.array(all_X, dtype=np.float32)
    y = np.array(all_y, dtype=np.int64)

    print(f"  총 시퀀스 수: {len(y):,}")
    print(f"  입력 shape: {X.shape}")
    return X, y


class StockLSTM(nn.Module):
    def __init__(self, input_dim=12, hidden_size=128,
                 num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.classifier(h[-1])


def train_model(X, y, epochs=50, batch_size=64, lr=0.001):
    print("\n[3/5] 모델 학습 중...")

    # 시계열 순서 유지하며 분할
    split = int(len(y) * 0.8)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    print(f"  학습: {len(y_tr):,}개 / 테스트: {len(y_te):,}개")

    Xtr = torch.FloatTensor(X_tr).to(DEVICE)
    ytr = torch.LongTensor(y_tr).to(DEVICE)
    Xte = torch.FloatTensor(X_te).to(DEVICE)

    model     = StockLSTM().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc   = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        n_batches  = 0

        # 셔플
        idx = torch.randperm(len(y_tr))
        Xtr_s = Xtr[idx]
        ytr_s = ytr[idx]

        for i in range(0, len(y_tr), batch_size):
            xb = Xtr_s[i:i+batch_size]
            yb = ytr_s[i:i+batch_size]
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1

        scheduler.step()

        model.eval()
        with torch.no_grad():
            # 메모리 절약을 위해 배치로 평가
            preds = []
            for i in range(0, len(y_te), 512):
                preds.extend(model(Xte[i:i+512]).argmax(dim=1).cpu().numpy())
            acc = accuracy_score(y_te, preds)
            f1  = f1_score(y_te, preds, average='macro')

        if acc > best_acc:
            best_acc   = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}/{epochs} | "
                  f"Loss: {total_loss/n_batches:.4f} | "
                  f"Val Acc: {acc:.4f} | F1: {f1:.4f}")

    model.load_state_dict(best_state)
    print(f"\n  최고 검증 정확도: {best_acc:.4f}")
    return model, Xte, y_te


def evaluate_model(model, Xte, y_te):
    print("\n[4/5] 모델 평가 중...")
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(y_te), 512):
            preds.extend(model(Xte[i:i+512]).argmax(dim=1).cpu().numpy())

    acc = accuracy_score(y_te, preds)
    f1  = f1_score(y_te, preds, average='macro')

    print(f"\n=== 최종 평가 결과 ===")
    print(f"  정확도 (Accuracy): {acc:.4f} ({acc*100:.1f}%)")
    print(f"  F1 Score (macro):  {f1:.4f}")
    print(f"\n{classification_report(y_te, preds, target_names=['하락', '상승'])}")
    return acc, f1


def save_model(model, save_path='data/processed'):
    os.makedirs(save_path, exist_ok=True)
    path = os.path.join(save_path, 'multimodal_lstm.pt')
    torch.save(model.state_dict(), path)
    print(f"\n[5/5] 모델 저장 완료: {path}")


def run_lstm_pipeline():
    print("=" * 50)
    print("ESG-LENS LSTM 학습 시작 (코스피 200 전체)")
    print("=" * 50)

    df       = load_and_merge_data()
    X, y     = make_sequences(df, seq_len=30)
    model, Xte, y_te = train_model(X, y, epochs=50)
    acc, f1  = evaluate_model(model, Xte, y_te)
    save_model(model)

    print("\n" + "=" * 50)
    print("LSTM 학습 완료!")
    print("=" * 50)
    return model, acc, f1


if __name__ == '__main__':
    run_lstm_pipeline()