# src/crawling/weather_api.py
import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import os
from dotenv import load_dotenv

load_dotenv()

WEATHER_API_KEY = os.getenv('WEATHER_API_KEY')

STATION_CODES = {
    '서울':  '108',
    '부산':  '159',
    '대구':  '143',
    '인천':  '112',
    '광주':  '156',
    '대전':  '133',
    '울산':  '152',
    '수원':  '119',
}

def get_weather_by_year(station_id: str, station_name: str, year: int) -> list:
    """연도별로 나눠서 기상 데이터 수집"""
    url = "https://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList"
    start_date = f"{year}0101"
    end_date   = f"{year}1231"

    params = {
        'serviceKey': WEATHER_API_KEY,
        'numOfRows': '999',
        'pageNo': '1',
        'dataType': 'JSON',
        'dataCd': 'ASOS',
        'dateCd': 'DAY',
        'startDt': start_date,
        'endDt': end_date,
        'stnIds': station_id
    }

    try:
        response = requests.get(url, params=params, timeout=15)
        data = response.json()

        body = data.get('response', {}).get('body', {})
        items = body.get('items', {})
        item_list = items.get('item', []) if isinstance(items, dict) else []

        results = []
        for item in item_list:
            results.append({
                'station_id':    station_id,
                'station_name':  station_name,
                'date':          item.get('tm', ''),
                'avg_temp':      item.get('avgTa', ''),
                'max_temp':      item.get('maxTa', ''),
                'min_temp':      item.get('minTa', ''),
                'precipitation': item.get('sumRn', ''),
                'avg_humidity':  item.get('avgRhm', ''),
                'avg_wind':      item.get('avgWs', ''),
                'sunshine':      item.get('sumSsHr', ''),
                'snow':          item.get('ddMes', ''),
                'crawled_at':    datetime.today().strftime('%Y-%m-%d')
            })
        return results

    except Exception as e:
        print(f"  {station_name} {year}년 오류: {e}")
        return []


def crawl_all_weather(save_path: str = 'data/raw/weather', years: int = 5):
    os.makedirs(save_path, exist_ok=True)

    current_year = datetime.today().year
    year_list = list(range(current_year - years + 1, current_year + 1))
    print(f"수집 연도: {year_list[0]} ~ {year_list[-1]}")

    all_data = []

    for i, (station_name, station_id) in enumerate(STATION_CODES.items()):
        print(f"\n[{i+1}/{len(STATION_CODES)}] {station_name} 수집 중...")
        station_total = 0

        for year in year_list:
            data = get_weather_by_year(station_id, station_name, year)
            all_data.extend(data)
            station_total += len(data)
            time.sleep(0.3)

        print(f"  {station_name}: 총 {station_total}건")
        time.sleep(0.3)

    if not all_data:
        print("\n수집된 데이터가 없어요!")
        return pd.DataFrame()

    result = pd.DataFrame(all_data)

    numeric_cols = ['avg_temp', 'max_temp', 'min_temp',
                    'precipitation', 'avg_humidity', 'avg_wind', 'sunshine', 'snow']
    for col in numeric_cols:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors='coerce')

    today = datetime.today().strftime('%Y%m%d')
    save_file = f'{save_path}/weather_{today}.csv'
    result.to_csv(save_file, index=False, encoding='utf-8-sig')

    print(f"\n저장 완료: {save_file}")
    print(f"총 {len(result):,}건 수집")
    return result


if __name__ == '__main__':
    crawl_all_weather()