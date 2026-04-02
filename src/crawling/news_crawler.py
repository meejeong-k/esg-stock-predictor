# src/crawling/news_crawler.py
import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import os
from dotenv import load_dotenv

load_dotenv()

CLIENT_ID     = os.getenv('NAVER_CLIENT_ID')
CLIENT_SECRET = os.getenv('NAVER_CLIENT_SECRET')

def get_news_api(company_name: str, display: int = 100,
                 start: int = 1) -> list:
    headers = {
        'X-Naver-Client-Id':     CLIENT_ID,
        'X-Naver-Client-Secret': CLIENT_SECRET
    }
    params = {
        'query':   company_name,
        'display': display,
        'start':   start,
        'sort':    'date'
    }
    url = "https://openapi.naver.com/v1/search/news.json"
    response = requests.get(url, headers=headers, params=params, timeout=10)

    if response.status_code != 200:
        print(f"  API 오류: {response.status_code}")
        return []

    items = response.json().get('items', [])
    result = []
    for item in items:
        result.append({
            'company':     company_name,
            'title':       item.get('title', '').replace('<b>', '').replace('</b>', ''),
            'description': item.get('description', '').replace('<b>', '').replace('</b>', ''),
            'press':       item.get('originallink', '').split('/')[2] if item.get('originallink') else '',
            'date':        item.get('pubDate', ''),
            'link':        item.get('originallink', ''),
            'naver_link':  item.get('link', ''),
            'crawled_at':  datetime.today().strftime('%Y-%m-%d')
        })
    return result


def crawl_company_news(company_name: str, max_articles: int = 1000) -> pd.DataFrame:
    """기업명으로 최대 1000건 수집 (API 최대 한도)"""
    all_news = []
    start = 1

    while start <= min(max_articles, 1000):
        display = min(100, max_articles - len(all_news))
        news = get_news_api(company_name, display=display, start=start)

        if not news:
            break

        all_news.extend(news)

        if len(news) < display:
            break

        start += display
        time.sleep(0.3)

    print(f"  {company_name}: {len(all_news)}건 수집")
    return pd.DataFrame(all_news)


def crawl_all_companies(
    company_list: list,
    save_path: str = 'data/raw/news',
    max_articles: int = 1000
):
    os.makedirs(save_path, exist_ok=True)
    all_data = []

    for i, company in enumerate(company_list):
        print(f"\n[{i+1}/{len(company_list)}] {company} 수집 중...")
        df = crawl_company_news(company, max_articles=max_articles)
        all_data.append(df)
        time.sleep(0.5)

    result = pd.concat(all_data, ignore_index=True)

    today = datetime.today().strftime('%Y%m%d')
    save_file = f'{save_path}/naver_news_{today}_v2.csv'
    result.to_csv(save_file, index=False, encoding='utf-8-sig')

    print(f"\n저장 완료: {save_file}")
    print(f"총 {len(result):,}건 수집")
    return result


if __name__ == '__main__':
    companies = [
        '삼성전자', 'SK하이닉스', '현대차', 'LG에너지솔루션',
        '삼성바이오로직스', 'POSCO홀딩스', 'KB금융', '신한지주',
        'LG화학', '카카오'
    ]
    crawl_all_companies(companies, max_articles=1000)