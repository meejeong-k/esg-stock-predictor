# test_news.py
from src.crawling.news_crawler import get_news_api

# 삼성전자 뉴스 10건만 테스트
news = get_news_api('삼성전자', display=10)

print(f"수집된 기사 수: {len(news)}")
if news:
    for i, n in enumerate(news[:3]):
        print(f"\n--- 기사 {i+1} ---")
        print(f"제목:   {n['title']}")
        print(f"언론사: {n['press']}")
        print(f"날짜:   {n['date']}")
        print(f"요약:   {n['description'][:60]}")
else:
    print("수집 실패! .env 파일의 API 키를 확인해주세요.")