# test_dart.py
import requests
import zipfile
import xml.etree.ElementTree as ET
import os
import io
from dotenv import load_dotenv

load_dotenv()
DART_API_KEY = os.getenv('DART_API_KEY')

def download_corp_codes():
    """DART 전체 법인코드 ZIP 다운로드 및 파싱"""
    url = f"https://opendart.fss.or.kr/api/corpCode.xml?crtfc_key={DART_API_KEY}"
    print("법인코드 목록 다운로드 중...")
    response = requests.get(url, timeout=30)

    # ZIP 파일 압축 해제
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        with z.open('CORPCODE.xml') as f:
            tree = ET.parse(f)
            root = tree.getroot()

    # 법인 목록 파싱
    corps = {}
    for corp in root.findall('list'):
        corp_code  = corp.findtext('corp_code', '')
        corp_name  = corp.findtext('corp_name', '')
        stock_code = corp.findtext('stock_code', '')
        # 상장사만 (stock_code가 있는 기업)
        if stock_code and stock_code.strip():
            corps[corp_name] = {
                'corp_code': corp_code,
                'stock_code': stock_code.strip()
            }

    print(f"상장 법인 수: {len(corps):,}개")
    return corps

# 법인코드 다운로드
corps = download_corp_codes()

# 0건이었던 기업 검색
targets = ['LG에너지솔루션', '삼성바이오로직스', 'POSCO홀딩스', 'KB금융지주']

print("\n=== 법인코드 검색 결과 ===")
for target in targets:
    # 정확한 이름 먼저 검색
    if target in corps:
        info = corps[target]
        print(f"{target}: corp_code={info['corp_code']}, stock_code={info['stock_code']}")
    else:
        # 유사 이름 검색
        similar = [(name, info) for name, info in corps.items() if target[:4] in name]
        if similar:
            print(f"\n[{target}] 유사 검색 결과:")
            for name, info in similar[:5]:
                print(f"  → {name}: corp_code={info['corp_code']}, stock_code={info['stock_code']}")
        else:
            print(f"{target}: 검색 결과 없음")