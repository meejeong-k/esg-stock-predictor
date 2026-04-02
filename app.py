# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="ESG-LENS",
    page_icon="🔍",
    layout="wide"
)

@st.cache_data
def load_stock_data():
    df = pd.read_csv('data/parquet/stocks/data.csv',
                     encoding='utf-8-sig', low_memory=False)
    df['date']   = pd.to_datetime(df['date'])
    df['ticker'] = df['ticker'].astype(str).str.zfill(6)
    return df

@st.cache_data
def load_esg_data():
    return pd.read_csv('data/processed/esg_risk_score_20260331.csv',
                       encoding='utf-8-sig')

@st.cache_data
def load_sentiment_data():
    return pd.read_csv('data/processed/news_sentiment_20260331.csv',
                       encoding='utf-8-sig')

@st.cache_data
def load_weather_data():
    df = pd.read_csv('data/parquet/weather/data.csv', encoding='utf-8-sig')
    df['date'] = pd.to_datetime(df['date'])
    return df

COMPANY_TICKER = {
    '삼성전자': '005930', 'SK하이닉스': '000660',
    '현대차': '005380', 'LG에너지솔루션': '373220',
    '삼성바이오로직스': '207940', 'POSCO홀딩스': '005490',
    'KB금융': '105560', '신한지주': '055550',
    'LG화학': '051910', '카카오': '035720',
}
TICKER_COMPANY = {v: k for k, v in COMPANY_TICKER.items()}

# ── 사이드바 ──────────────────────────────────────────────────────
st.sidebar.title("🔍 ESG-LENS")
st.sidebar.markdown("ESG 리스크를 통해 주식 시장을 바라보는 렌즈")
st.sidebar.divider()

page = st.sidebar.radio(
    "메뉴",
    ["📊 ESG 리스크 대시보드",
     "📈 주가 분석",
     "📰 뉴스 감성 분석",
     "🌡️ 기후 리스크",
     "🤖 주가 예측"]
)

# ── 1. ESG 리스크 대시보드 ────────────────────────────────────────
if page == "📊 ESG 리스크 대시보드":
    st.title("📊 ESG 리스크 대시보드")
    st.markdown("코스피 200 주요 기업의 ESG 리스크 점수를 분석합니다.")

    esg_df = load_esg_data()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("분석 기업 수", f"{len(esg_df)}개")
    with col2:
        st.metric("평균 ESG 리스크", f"{esg_df['esg_risk_score'].mean():.1f}점")
    with col3:
        max_row = esg_df.loc[esg_df['esg_risk_score'].idxmax()]
        st.metric("최고 리스크", max_row['company'],
                  delta=f"{max_row['esg_risk_score']:.1f}점", delta_color="inverse")
    with col4:
        min_row = esg_df.loc[esg_df['esg_risk_score'].idxmin()]
        st.metric("최저 리스크", min_row['company'],
                  delta=f"{min_row['esg_risk_score']:.1f}점", delta_color="normal")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🏆 기업별 ESG 리스크 점수")
        fig = px.bar(
            esg_df.sort_values('esg_risk_score', ascending=True),
            x='esg_risk_score', y='company', orientation='h',
            color='esg_risk_score', color_continuous_scale='RdYlGn_r',
            labels={'esg_risk_score': 'ESG 리스크 점수', 'company': '기업'},
            title="높을수록 위험"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📉 ESG 리스크 vs 현재 주가")
        fig = px.scatter(
            esg_df.dropna(subset=['close']),
            x='esg_risk_score', y='close', text='company',
            color='esg_risk_score', color_continuous_scale='RdYlGn_r',
            labels={'esg_risk_score': 'ESG 리스크 점수', 'close': '현재 주가(원)'},
            title="ESG 리스크 vs 주가"
        )
        fig.update_traces(textposition='top center')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("📋 ESG 리스크 상세 데이터")
    display_df = esg_df[['company', 'avg_sentiment', 'esg_risk_score',
                          'positive_count', 'negative_count', 'close']].copy()
    display_df.columns = ['기업', '평균감성점수', 'ESG리스크점수', '긍정뉴스수', '부정뉴스수', '현재주가']
    display_df = display_df.sort_values('ESG리스크점수', ascending=False)
    st.dataframe(
        display_df.style.background_gradient(subset=['ESG리스크점수'], cmap='RdYlGn_r'),
        use_container_width=True
    )

# ── 2. 주가 분석 ──────────────────────────────────────────────────
elif page == "📈 주가 분석":
    st.title("📈 주가 분석")

    stock_df = load_stock_data()
    esg_df   = load_esg_data()

    col1, col2 = st.columns([1, 3])
    with col1:
        company = st.selectbox("기업 선택", list(COMPANY_TICKER.keys()))
        ticker  = COMPANY_TICKER[company]
        period  = st.selectbox("기간", ["1개월", "3개월", "6개월", "1년", "전체"])

    period_days = {"1개월": 30, "3개월": 90, "6개월": 180, "1년": 365, "전체": 9999}
    days = period_days[period]

    df = stock_df[stock_df['ticker'] == ticker].copy().sort_values('date')
    if days < 9999:
        df = df[df['date'] >= df['date'].max() - pd.Timedelta(days=days)]

    with col2:
        esg_row = esg_df[esg_df['ticker'] == ticker]
        if not esg_row.empty:
            score     = esg_row['esg_risk_score'].values[0]
            sentiment = esg_row['avg_sentiment'].values[0]
            ca, cb, cc = st.columns(3)
            ca.metric("ESG 리스크 점수", f"{score:.1f}점")
            cb.metric("평균 감성 점수", f"{sentiment:.4f}")
            cc.metric("현재 주가",
                      f"{df['close'].iloc[-1]:,.0f}원" if not df.empty else "N/A")

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.7, 0.3],
                        subplot_titles=['주가 & 이동평균', '거래량'])
    fig.add_trace(go.Candlestick(
        x=df['date'], open=df['open'], high=df['high'],
        low=df['low'], close=df['close'], name='주가'
    ), row=1, col=1)
    for ma, color in [('ma5', 'blue'), ('ma20', 'orange'), ('ma60', 'red')]:
        if ma in df.columns:
            fig.add_trace(go.Scatter(
                x=df['date'], y=df[ma], name=ma.upper(),
                line=dict(color=color, width=1)
            ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=df['date'], y=df['volume'], name='거래량', marker_color='lightblue'
    ), row=2, col=1)
    fig.update_layout(height=600, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

# ── 3. 뉴스 감성 분석 ────────────────────────────────────────────
elif page == "📰 뉴스 감성 분석":
    st.title("📰 뉴스 감성 분석")
    st.markdown("KoBERT (snunlp/KR-FinBert-SC) 모델로 금융 뉴스 10,000건을 분석했습니다.")

    sent_df = load_sentiment_data()

    # KPI
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("분석 뉴스 수", f"{len(sent_df):,}건")
    col2.metric("긍정", f"{(sent_df['sentiment_label']=='긍정').sum():,}건",
                delta=f"{(sent_df['sentiment_label']=='긍정').mean()*100:.1f}%")
    col3.metric("중립", f"{(sent_df['sentiment_label']=='중립').sum():,}건",
                delta=f"{(sent_df['sentiment_label']=='중립').mean()*100:.1f}%")
    col4.metric("부정", f"{(sent_df['sentiment_label']=='부정').sum():,}건",
                delta=f"{(sent_df['sentiment_label']=='부정').mean()*100:.1f}%", delta_color="inverse")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("전체 감성 분포")
        counts = sent_df['sentiment_label'].value_counts()
        fig = px.pie(
            values=counts.values, names=counts.index,
            color=counts.index,
            color_discrete_map={'긍정': '#22C55E', '중립': '#94A3B8', '부정': '#EF4444'},
            title=f"전체 {len(sent_df):,}건 감성 분포"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("기업별 평균 감성 점수")
        company_sent = sent_df.groupby('company')['sentiment_score'].mean().reset_index()
        company_sent = company_sent.sort_values('sentiment_score')
        fig = px.bar(
            company_sent, x='sentiment_score', y='company', orientation='h',
            color='sentiment_score', color_continuous_scale='RdYlGn',
            title="-1(부정) ~ +1(긍정)"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # 한계점 안내
    st.warning("""
    ⚠️ **감성 분석 한계점**

    KoBERT로 10,000건의 뉴스 감성 분석을 완료했지만,
    **네이버 뉴스 API는 최신 뉴스(최근 1주일)만 제공**하여
    5년치 주가 데이터와 날짜 매칭이 되지 않았습니다.

    이로 인해 감성 점수는 **ESG 리스크 점수 산출(60% 가중치)에 반영**되었지만,
    LSTM 모델에 시계열로 직접 연결하지는 못했습니다.

    → 향후 유료 뉴스 데이터 확보 시 모델 성능 향상 가능
    """)

    st.subheader("기업별 뉴스 상세 보기")
    company  = st.selectbox("기업 선택", list(COMPANY_TICKER.keys()))
    labels   = st.multiselect("감성 필터", ['긍정', '중립', '부정'], default=['긍정', '부정'])

    filtered = sent_df[
        (sent_df['company'] == company) &
        (sent_df['sentiment_label'].isin(labels))
    ][['title', 'sentiment_label', 'sentiment_score', 'crawled_at']].head(20)
    filtered.columns = ['제목', '감성', '점수', '날짜']

    def color_sentiment(val):
        if val == '긍정': return 'color: green'
        if val == '부정': return 'color: red'
        return 'color: gray'

    st.dataframe(
        filtered.style.applymap(color_sentiment, subset=['감성']),
        use_container_width=True
    )

# ── 4. 기후 리스크 ────────────────────────────────────────────────
elif page == "🌡️ 기후 리스크":
    st.title("🌡️ 기후 리스크 분석")
    st.markdown("기상청 ASOS API로 수집한 8개 도시 5년치 기후 데이터를 분석합니다.")

    weather_df = load_weather_data()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("수집 기간", "2022 ~ 2026년")
    col2.metric("폭염 일수",  f"{int(weather_df['heat_wave_count'].sum())}일",
                delta="최고기온 ≥ 30°C")
    col3.metric("혹한 일수",  f"{int(weather_df['cold_wave_count'].sum())}일",
                delta="최저기온 ≤ -10°C", delta_color="inverse")
    col4.metric("폭우 일수",  f"{int(weather_df['heavy_rain_count'].sum())}일",
                delta="강수량 ≥ 80mm", delta_color="inverse")

    st.divider()
    st.subheader("📅 일별 기후 리스크 추이")

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=['전국 평균기온 (°C)', '기후 리스크 점수'])
    fig.add_trace(go.Scatter(
        x=weather_df['date'], y=weather_df['national_avg_temp'],
        name='평균기온', line=dict(color='orange')
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=weather_df['date'], y=weather_df['national_climate_risk'],
        name='기후리스크', marker_color='red'
    ), row=2, col=1)
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 기후 리스크 점수 계산 방법")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("🌡️ **폭염** (+1점)\n\n최고기온 ≥ 30°C인 날")
    with col2:
        st.info("❄️ **혹한** (+1점)\n\n최저기온 ≤ -10°C인 날")
    with col3:
        st.info("🌧️ **폭우** (+1점)\n\n일 강수량 ≥ 80mm인 날")
    st.markdown("→ 전국 8개 도시 합산 후 정규화 (0~1) → **ESG 리스크 점수에 40% 반영**")

# ── 5. 주가 예측 ──────────────────────────────────────────────────
elif page == "🤖 주가 예측":
    st.title("🤖 LSTM 주가 예측 모델")
    st.markdown("주가 + 기후 + ESG 점수를 융합한 딥러닝 모델입니다.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("최종 정확도",   "53.6%", delta="+5.6%p vs 주가만")
    col2.metric("F1 Score",     "0.4542")
    col3.metric("학습 시퀀스",  "224,950개")
    col4.metric("대상 종목",    "코스피 200")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 모델 성능 비교")
        perf_df = pd.DataFrame({
            '모델': [
                '① 주가만 (단일종목)',
                '② 주가+기후+ESG (단일종목)',
                '③ 주가+기후+ESG (코스피200 전체)'
            ],
            '정확도': [48.0, 50.7, 53.6]
        })
        fig = px.bar(
            perf_df, x='모델', y='정확도',
            color='정확도', color_continuous_scale='Blues',
            title="모델별 정확도 비교 (%)",
            text='정확도'
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(height=400, yaxis_range=[44, 56])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🏗️ 모델 입력 피처 (12개)")
        feature_df = pd.DataFrame({
            '카테고리': ['주가']*9 + ['기후']*2 + ['ESG'],
            '피처': [
                '시가(Open)', '고가(High)', '저가(Low)', '종가(Close)', '거래량(Volume)',
                'MA5 (5일 이동평균)', 'MA20 (20일 이동평균)', 'MA60 (60일 이동평균)',
                '일별 수익률',
                '전국 평균기온', '기후 리스크 점수',
                'ESG 리스크 점수'
            ]
        })
        fig = px.sunburst(
            feature_df, path=['카테고리', '피처'],
            title="12개 입력 피처 구성"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("🏗️ LSTM 모델 아키텍처")
    st.code("""
입력층:  30일 시퀀스 × 12개 피처
          ↓
LSTM Layer 1  (hidden_size=128, dropout=0.3)
          ↓
LSTM Layer 2  (hidden_size=128, dropout=0.3)
          ↓
FC Layer      (128 → 64)  + ReLU + Dropout(0.3)
          ↓
출력층:  상승(1) / 하락(0)

학습: Epochs=50, lr=0.001, CosineAnnealingLR
데이터: 코스피 200 × 5년치 → 224,950개 시퀀스
    """)

    st.subheader("⚠️ 한계 및 향후 개선 방향")
    st.warning("""
    **현재 한계점**
    - KoBERT 감성 분석을 구현했지만, 과거 뉴스 데이터 확보 한계로 LSTM에 시계열 직접 연결 못 함
    - 상승 예측 정밀도(Precision 53%)가 아직 낮음
    """)
    st.info("""
    **향후 개선 방향**
    1. 유료 뉴스 데이터 확보 → 감성 피처 시계열 직접 연결
    2. LSTM + XGBoost 앙상블 모델로 성능 향상
    3. 실시간 자동 재학습 파이프라인 구축
    4. 탄소배출량 등 추가 ESG 지표 반영
    """)