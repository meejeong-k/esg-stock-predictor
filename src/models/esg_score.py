# src/models/esg_score.py
import pandas as pd
import numpy as np
from datetime import datetime
import os

COMPANY_TICKER = {
    '삼성전자':         '005930',
    'SK하이닉스':       '000660',
    '현대차':           '005380',
    'LG에너지솔루션':   '373220',
    '삼성바이오로직스': '207940',
    'POSCO홀딩스':      '005490',
    'KB금융':           '105560',
    '신한지주':         '055550',
    'LG화학':           '051910',
    '카카오':           '035720',
}


def load_data():
    """전처리된 데이터 로드"""
    print("데이터 로드 중...")

    sentiment_df = pd.read_csv(
        'data/processed/news_sentiment_20260331.csv',
        encoding='utf-8-sig'
    )
    sentiment_df['news_date'] = pd.to_datetime(
        sentiment_df['news_date'], errors='coerce'
    )

    weather_df = pd.read_csv(
        'data/parquet/weather/data.csv',
        encoding='utf-8-sig'
    )
    weather_df['date'] = pd.to_datetime(weather_df['date'])

    stock_df = pd.read_csv(
        'data/parquet/stocks/data.csv',
        encoding='utf-8-sig',
        low_memory=False
    )
    stock_df['date']   = pd.to_datetime(stock_df['date'])
    stock_df['ticker'] = stock_df['ticker'].astype(str).str.zfill(6)

    print(f"  감성 데이터: {len(sentiment_df):,}건")
    print(f"  뉴스 날짜 범위: {sentiment_df['news_date'].min()} ~ {sentiment_df['news_date'].max()}")
    print(f"  기후 데이터: {len(weather_df):,}건")
    print(f"  주가 데이터: {len(stock_df):,}건")

    return sentiment_df, weather_df, stock_df


def calc_sentiment_score(sentiment_df: pd.DataFrame) -> pd.DataFrame:
    """기업별 감성 점수 집계 — 날짜 없으면 crawled_at 사용"""
    print("\n[1/3] 감성 점수 집계 중...")

    df = sentiment_df.copy()

    # news_date가 NaT이면 crawled_at 사용
    df['crawled_at'] = pd.to_datetime(df['crawled_at'], errors='coerce')
    df['date'] = df['news_date'].fillna(df['crawled_at'])

    # 날짜가 여전히 없으면 오늘 날짜
    df['date'] = df['date'].fillna(pd.Timestamp.today().normalize())

    # 티커 추가
    df['ticker'] = df['company'].map(COMPANY_TICKER)

    print(f"  날짜 분포:\n{df['date'].dt.date.value_counts().head()}")

    # 기업별 전체 평균 감성 점수 (날짜 무관)
    company_sentiment = df.groupby('company').agg(
        avg_sentiment=('sentiment_score', 'mean'),
        news_count=('sentiment_score', 'count'),
        positive_count=('sentiment_label', lambda x: (x == '긍정').sum()),
        negative_count=('sentiment_label', lambda x: (x == '부정').sum()),
        neutral_count=('sentiment_label',  lambda x: (x == '중립').sum()),
    ).reset_index()

    company_sentiment['ticker'] = company_sentiment['company'].map(COMPANY_TICKER)

    print(f"\n=== 기업별 평균 감성 점수 ===")
    print(company_sentiment[['company', 'avg_sentiment', 'news_count']].to_string(index=False))

    return company_sentiment


def calc_climate_risk(weather_df: pd.DataFrame) -> pd.DataFrame:
    """기후 리스크 점수 정규화"""
    print("\n[2/3] 기후 리스크 점수 정규화 중...")

    df = weather_df.copy()
    max_risk = df['national_climate_risk'].max()
    min_risk = df['national_climate_risk'].min()

    if max_risk > min_risk:
        df['climate_risk_normalized'] = (
            (df['national_climate_risk'] - min_risk) / (max_risk - min_risk)
        )
    else:
        df['climate_risk_normalized'] = 0.0

    # 전체 평균 기후 리스크
    avg_climate_risk = df['climate_risk_normalized'].mean()
    print(f"  평균 기후 리스크: {avg_climate_risk:.4f}")
    print(f"  폭염 일수: {int(df['heat_wave_count'].sum())}일")
    print(f"  혹한 일수: {int(df['cold_wave_count'].sum())}일")
    print(f"  폭우 일수: {int(df['heavy_rain_count'].sum())}일")

    return df, avg_climate_risk


def calc_esg_score(company_sentiment: pd.DataFrame,
                   avg_climate_risk: float,
                   stock_df: pd.DataFrame) -> pd.DataFrame:
    """ESG 리스크 점수 계산 후 주가 데이터와 결합"""
    print("\n[3/3] ESG 리스크 점수 계산 중...")

    df = company_sentiment.copy()

    # 감성 리스크 (0~1) — 부정적일수록 높음
    df['sentiment_risk'] = (-df['avg_sentiment'] + 1) / 2

    # ESG 리스크 점수 (0~100)
    df['esg_risk_score'] = (
        df['sentiment_risk'] * 0.6 +
        avg_climate_risk * 0.4
    ) * 100
    df['esg_risk_score'] = df['esg_risk_score'].round(2)

    print(f"\n=== 기업별 ESG 리스크 점수 ===")
    result = df[['company', 'ticker', 'avg_sentiment',
                 'sentiment_risk', 'esg_risk_score',
                 'news_count', 'positive_count',
                 'negative_count', 'neutral_count']].sort_values(
        'esg_risk_score', ascending=False
    )
    print(result[['company', 'avg_sentiment', 'esg_risk_score']].to_string(index=False))

    # 주가 데이터와 결합 (티커별 최신 주가)
    latest_stock = stock_df.sort_values('date').groupby('ticker').last().reset_index()
    latest_stock = latest_stock[['ticker', 'date', 'close', 'daily_return', 'ma5', 'ma20']]

    final_df = result.merge(latest_stock, on='ticker', how='left')

    return final_df


def run_esg_pipeline(save_path: str = 'data/processed'):
    """전체 ESG 점수 계산 파이프라인"""
    print("=" * 50)
    print("ESG 리스크 점수 계산 시작")
    print("=" * 50)

    os.makedirs(save_path, exist_ok=True)

    sentiment_df, weather_df, stock_df = load_data()
    company_sentiment             = calc_sentiment_score(sentiment_df)
    weather_daily, avg_climate    = calc_climate_risk(weather_df)
    esg_df                        = calc_esg_score(company_sentiment, avg_climate, stock_df)

    today = datetime.today().strftime('%Y%m%d')
    save_file = f'{save_path}/esg_risk_score_{today}.csv'
    esg_df.to_csv(save_file, index=False, encoding='utf-8-sig')

    print(f"\n저장 완료: {save_file}")
    print(f"총 {len(esg_df):,}개 기업 ESG 점수 산출")
    print("\n" + "=" * 50)
    print("ESG 리스크 점수 계산 완료!")
    print("=" * 50)

    return esg_df


if __name__ == '__main__':
    df = run_esg_pipeline()
    print("\n최종 결과:")
    print(df[['company', 'avg_sentiment', 'esg_risk_score', 'close']].to_string(index=False))