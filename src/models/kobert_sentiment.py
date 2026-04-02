# src/models/kobert_sentiment.py
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import os

# GPU 없으면 CPU 사용
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"사용 디바이스: {DEVICE}")


class NewsDataset(Dataset):
    """뉴스 텍스트 데이터셋"""
    def __init__(self, texts, tokenizer, max_len=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids':      encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }


class KoBERTSentimentAnalyzer:
    """KoBERT 기반 한국어 감성 분석기"""

    def __init__(self):
        print("KoBERT 모델 로딩 중...")
        # snunlp/KR-FinBert-SC: 금융 특화 한국어 BERT
        model_name = "snunlp/KR-FinBert-SC"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(DEVICE)
        self.model.eval()

        # 레이블: 0=부정, 1=중립, 2=긍정
        self.labels = {0: '부정', 1: '중립', 2: '긍정'}
        print("모델 로딩 완료!")

    def predict_batch(self, texts: list, batch_size: int = 32) -> list:
        """배치 단위 감성 예측"""
        dataset = NewsDataset(texts, self.tokenizer)
        loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        all_probs  = []
        all_labels = []

        with torch.no_grad():
            for batch in loader:
                input_ids      = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)

                all_probs.extend(probs.tolist())
                all_labels.extend(preds.tolist())

        results = []
        for i, (prob, label) in enumerate(zip(all_probs, all_labels)):
            # 감성 점수: -1(부정) ~ +1(긍정)
            score = prob[2] - prob[0]
            results.append({
                'sentiment_label': self.labels[label],
                'sentiment_score': round(score, 4),
                'prob_negative':   round(prob[0], 4),
                'prob_neutral':    round(prob[1], 4),
                'prob_positive':   round(prob[2], 4)
            })

        return results

    def analyze_news_file(self, save_path: str = 'data/processed'):
        """뉴스 파일 전체 감성 분석"""
        os.makedirs(save_path, exist_ok=True)

        # 수정 코드 — 새로 수집한 10,000건 파일 사용
        news_df = pd.read_csv(
            'data/raw/news/naver_news_20260331_v2.csv',
            encoding='utf-8-sig'
        )
        print(f"\n분석 대상: {len(news_df):,}건")

        # 감성 분석 실행
        # 수정 코드 — text 컬럼 없으면 title + description 합치기
        if 'text' not in news_df.columns:
            news_df['text'] = (
                    news_df['title'].fillna('') + ' ' +
                    news_df['description'].fillna('')
            )
        texts = news_df['text'].fillna('').tolist()
        print("감성 분석 중... (시간이 걸릴 수 있어요)")

        results = self.predict_batch(texts, batch_size=16)
        results_df = pd.DataFrame(results)

        # 원본 데이터와 결합
        final_df = pd.concat([
            news_df.reset_index(drop=True),
            results_df
        ], axis=1)

        # 저장
        today = datetime.today().strftime('%Y%m%d')
        save_file = f'{save_path}/news_sentiment_{today}.csv'
        final_df.to_csv(save_file, index=False, encoding='utf-8-sig')

        print(f"\n저장 완료: {save_file}")
        print(f"총 {len(final_df):,}건 분석 완료")

        # 감성 분포 출력
        print("\n=== 감성 분포 ===")
        print(final_df['sentiment_label'].value_counts())
        print(f"\n평균 감성 점수: {final_df['sentiment_score'].mean():.4f}")

        return final_df


if __name__ == '__main__':
    analyzer = KoBERTSentimentAnalyzer()
    df = analyzer.analyze_news_file()
    print("\n샘플 결과:")
    print(df[['company', 'title', 'sentiment_label', 'sentiment_score']].head(5))