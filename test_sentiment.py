# test_sentiment.py
from src.models.kobert_sentiment import KoBERTSentimentAnalyzer

analyzer = KoBERTSentimentAnalyzer()

# 테스트 문장 3개
test_texts = [
    "삼성전자 실적 호조로 주가 급등, 투자자들 기대감 높아져",
    "SK하이닉스, 반도체 업황 악화로 영업이익 급감 우려",
    "현대차 3분기 실적 시장 예상치 부합, 전망 중립적"
]

results = analyzer.predict_batch(test_texts)
for text, result in zip(test_texts, results):
    print(f"\n텍스트: {text[:30]}...")
    print(f"감성:   {result['sentiment_label']} ({result['sentiment_score']:+.4f})")