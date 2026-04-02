# test_forum.py
import requests
from bs4 import BeautifulSoup
import time

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                  'AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/120.0.0.0 Safari/537.36',
    'Referer': 'https://finance.naver.com'
}

def crawl_forum_page(code: str, page: int, date: str = None) -> list:
    url = "https://finance.naver.com/item/board.naver"
    params = {'code': code, 'page': page}
    if date:
        params['date'] = date  # YYYYMMDD 형식

    response = requests.get(url, headers=headers, params=params, timeout=10)
    soup = BeautifulSoup(response.text, 'html.parser')

    results = []
    table = soup.select_one('table.type2')
    if not table:
        return results

    for row in table.select('tr'):
        cols = row.select('td')
        if len(cols) < 4:
            continue
        date_val  = cols[0].get_text(strip=True)
        title_tag = cols[1].select_one('a')
        title     = title_tag.get_text(strip=True) if title_tag else ''
        if not date_val or not title or '날짜' in date_val:
            continue
        results.append({'date': date_val, 'title': title})

    return results

# 과거 날짜로 테스트
test_dates = ['20240101', '20230101', '20220101', '20210101']

print("=== 날짜 파라미터 테스트 ===")
for d in test_dates:
    posts = crawl_forum_page('005930', page=1, date=d)
    if posts:
        dates = [p['date'] for p in posts]
        print(f"날짜={d} → {min(dates)} ~ {max(dates)} ({len(posts)}건)")
    else:
        print(f"날짜={d} → 데이터 없음")
    time.sleep(0.5)