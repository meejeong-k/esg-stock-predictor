# test_weather.py
import requests
import os
from dotenv import load_dotenv

load_dotenv()
WEATHER_API_KEY = os.getenv('WEATHER_API_KEY')

url = "https://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList"
params = {
    'serviceKey': WEATHER_API_KEY,
    'numOfRows': '999',
    'pageNo': '1',
    'dataType': 'JSON',
    'dataCd': 'ASOS',
    'dateCd': 'DAY',
    'startDt': '20210401',
    'endDt': '20260331',
    'stnIds': '108'
}

print("=== 5년치 서울 데이터 테스트 ===")
response = requests.get(url, params=params, timeout=15)
print(f"상태코드: {response.status_code}")
data = response.json()

body = data.get('response', {}).get('body', {})
total_count = body.get('totalCount', 0)
items = body.get('items', {})
item_list = items.get('item', []) if isinstance(items, dict) else []

print(f"totalCount: {total_count}")
print(f"수집된 item 수: {len(item_list)}")
if item_list:
    print(f"첫 번째: {item_list[0].get('tm')} | 평균기온: {item_list[0].get('avgTa')}")
    print(f"마지막: {item_list[-1].get('tm')} | 평균기온: {item_list[-1].get('avgTa')}")