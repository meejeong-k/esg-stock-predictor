# src/crawling/stock_api.py
import FinanceDataReader as fdr
import pandas as pd
from datetime import datetime, timedelta
import os
import time

def fetch_stock_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    FinanceDataReader로 주가 OHLCV 데이터 수집
    ticker: 종목코드 (예: '005930' = 삼성전자)
    start/end: 'YYYY-MM-DD' 형식
    """
    df = fdr.DataReader(ticker, start, end)
    df.index.name = 'date'
    df.reset_index(inplace=True)
    df['ticker'] = ticker
    return df

def fetch_kospi_tickers() -> pd.DataFrame:
    """코스피 전체 종목 리스트 반환"""
    df = fdr.StockListing('KOSPI')
    return df

def fetch_all_stocks(save_path: str = 'data/raw/stocks'):
    os.makedirs(save_path, exist_ok=True)

    end   = datetime.today().strftime('%Y-%m-%d')
    start = (datetime.today() - timedelta(days=365*5)).strftime('%Y-%m-%d')

    print(f"수집 기간: {start} ~ {end}")

    listing = fetch_kospi_tickers()
    tickers = listing['Code'].tolist()[:200]  # 상위 200개
    print(f"총 {len(tickers)}개 종목 수집 시작...")

    all_data = []
    for i, ticker in enumerate(tickers):
        try:
            df = fetch_stock_data(ticker, start, end)
            if len(df) > 0:
                all_data.append(df)
            if (i+1) % 10 == 0:
                print(f"  {i+1}/{len(tickers)} 완료")
            time.sleep(0.3)
        except Exception as e:
            print(f"  {ticker} 실패: {e}")

    result = pd.concat(all_data, ignore_index=True)
    result.to_csv(f'{save_path}/kospi200_5y.csv',
                  index=False, encoding='utf-8-sig')

    print(f"\n저장 완료!")
    print(f"파일 위치: {save_path}/kospi200_5y.csv")
    print(f"총 데이터: {len(result):,}행")
    return result

if __name__ == '__main__':
    fetch_all_stocks()