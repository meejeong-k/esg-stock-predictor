# src/crawling/dart_api.py
import requests
import pandas as pd
from datetime import datetime
import time
import os
from dotenv import load_dotenv

load_dotenv()

DART_API_KEY = os.getenv('DART_API_KEY')

COMPANY_CODES = {
    '삼성전자':         '00126380',
    'SK하이닉스':       '00164779',
    '현대차':           '00164742',
    'LG에너지솔루션':   '01515323',  # 수정 ✅
    '삼성바이오로직스': '00877059',  # 수정 ✅
    'POSCO홀딩스':      '00155319',  # 수정 ✅
    'KB금융':           '00688996',  # 수정 ✅
    '신한지주':         '00382199',
    'LG화학':           '00356361',
    '카카오':           '00266961',
}

def get_company_reports(corp_code: str, company_name: str, years: int = 3) -> list:
    """사업보고서, 반기보고서, 분기보고서 수집"""
    results = []
    current_year = datetime.today().year

    for year in range(current_year - years, current_year + 1):
        for report_code in ['A001', 'A002', 'A003']:
            url = "https://opendart.fss.or.kr/api/list.json"
            params = {
                'crtfc_key': DART_API_KEY,
                'corp_code': corp_code,
                'bgn_de': f'{year}0101',
                'end_de': f'{year}1231',
                'pblntf_detail_ty': report_code,
                'page_count': 10
            }
            try:
                response = requests.get(url, params=params, timeout=10)
                data = response.json()
                if data.get('status') == '000' and data.get('list'):
                    for item in data['list']:
                        results.append({
                            'company': company_name,
                            'corp_code': corp_code,
                            'report_name': item.get('report_nm', ''),
                            'rcept_no': item.get('rcept_no', ''),
                            'rcept_dt': item.get('rcept_dt', ''),
                            'report_type': report_code,
                            'crawled_at': datetime.today().strftime('%Y-%m-%d')
                        })
            except Exception as e:
                print(f"  {company_name} {year}년 오류: {e}")
            time.sleep(0.2)

    return results


def get_esg_disclosure(corp_code: str, company_name: str) -> list:
    """ESG 관련 공시 수집 — pblntf_ty 전체 범위로 확장"""
    results = []
    today = datetime.today().strftime('%Y%m%d')

    # 공시 유형을 넓게 설정해서 ESG 키워드 필터링
    for pblntf_ty in ['A', 'B', 'C', 'D', 'E', 'F']:
        url = "https://opendart.fss.or.kr/api/list.json"
        params = {
            'crtfc_key': DART_API_KEY,
            'corp_code': corp_code,
            'bgn_de': '20210101',
            'end_de': today,
            'pblntf_ty': pblntf_ty,
            'page_count': 100
        }
        try:
            response = requests.get(url, params=params, timeout=10)
            data = response.json()

            if data.get('status') == '000' and data.get('list'):
                keywords = ['ESG', '환경', '지속가능', '탄소', '온실가스',
                           'CDP', '기후', '녹색', '친환경', 'RE100']
                for item in data['list']:
                    report_name = item.get('report_nm', '')
                    if any(kw in report_name for kw in keywords):
                        results.append({
                            'company': company_name,
                            'corp_code': corp_code,
                            'report_name': report_name,
                            'rcept_no': item.get('rcept_no', ''),
                            'rcept_dt': item.get('rcept_dt', ''),
                            'report_type': 'ESG',
                            'crawled_at': datetime.today().strftime('%Y-%m-%d')
                        })
        except Exception as e:
            print(f"  {company_name} ESG({pblntf_ty}) 오류: {e}")
        time.sleep(0.2)

    return results


def crawl_all_dart(save_path: str = 'data/raw/dart'):
    """전체 기업 DART 공시 데이터 수집"""
    os.makedirs(save_path, exist_ok=True)
    all_reports = []
    all_esg = []

    for i, (company, corp_code) in enumerate(COMPANY_CODES.items()):
        print(f"\n[{i+1}/{len(COMPANY_CODES)}] {company} 수집 중...")

        reports = get_company_reports(corp_code, company, years=3)
        all_reports.extend(reports)
        print(f"  보고서: {len(reports)}건")

        esg = get_esg_disclosure(corp_code, company)
        all_esg.extend(esg)
        print(f"  ESG 공시: {len(esg)}건")

        time.sleep(0.5)

    today = datetime.today().strftime('%Y%m%d')

    if all_reports:
        reports_df = pd.DataFrame(all_reports)
        reports_file = f'{save_path}/dart_reports_{today}.csv'
        reports_df.to_csv(reports_file, index=False, encoding='utf-8-sig')
        print(f"\n보고서 저장: {reports_file} ({len(all_reports)}건)")

    if all_esg:
        esg_df = pd.DataFrame(all_esg)
        esg_file = f'{save_path}/dart_esg_{today}.csv'
        esg_df.to_csv(esg_file, index=False, encoding='utf-8-sig')
        print(f"ESG 저장: {esg_file} ({len(all_esg)}건)")
    else:
        print("\nESG 공시: 0건 (키워드에 해당하는 공시 없음)")

    return pd.DataFrame(all_reports), pd.DataFrame(all_esg) if all_esg else pd.DataFrame()


if __name__ == '__main__':
    crawl_all_dart()