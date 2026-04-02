# test_stock.py
from src.crawling.stock_api import fetch_stock_data

df = fetch_stock_data(
    ticker='005930',  # 삼성전자
    start='20230101',
    end='20231231'
)

print(df.head(10))
print(f"\n총 {len(df)}행 수집 완료!")