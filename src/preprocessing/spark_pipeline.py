# src/preprocessing/spark_pipeline.py
import os
import glob
import warnings
warnings.filterwarnings('ignore')

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StringType
from pyspark.sql.window import Window


def create_spark_session():
    spark = SparkSession.builder \
        .appName("ESG-LENS Pipeline") \
        .config("spark.driver.memory", "2g") \
        .config("spark.sql.shuffle.partitions", "4") \
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def find_latest_file(pattern: str) -> str:
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"파일을 찾을 수 없어요: {pattern}")
    return max(files, key=os.path.getmtime)


def process_stock_data(spark: SparkSession):
    print("\n[1/4] 주가 데이터 처리 중...")
    path = find_latest_file("data/raw/stocks/*.csv")
    print(f"  파일: {path}")
    df = spark.read.csv(path, header=True, inferSchema=True)
    for col in df.columns:
        df = df.withColumnRenamed(col, col.lower())
    df = df.withColumn('ticker', F.lpad(F.col('ticker').cast(StringType()), 6, '0'))
    df = df.withColumn('date', F.to_date(F.col('date')))
    window_5  = Window.partitionBy('ticker').orderBy('date').rowsBetween(-4, 0)
    window_20 = Window.partitionBy('ticker').orderBy('date').rowsBetween(-19, 0)
    window_60 = Window.partitionBy('ticker').orderBy('date').rowsBetween(-59, 0)
    df = df.withColumn('ma5',  F.avg('close').over(window_5))
    df = df.withColumn('ma20', F.avg('close').over(window_20))
    df = df.withColumn('ma60', F.avg('close').over(window_60))
    window_lag = Window.partitionBy('ticker').orderBy('date')
    df = df.withColumn(
        'daily_return',
        (F.col('close') - F.lag('close', 1).over(window_lag)) /
        F.lag('close', 1).over(window_lag) * 100
    )
    df = df.dropna(subset=['close', 'ticker', 'date'])
    print(f"  행 수: {df.count():,}")
    print(f"  종목 수: {df.select('ticker').distinct().count()}")
    return df


def process_news_data(spark: SparkSession):
    print("\n[2/4] 뉴스 데이터 처리 중...")
    path = find_latest_file("data/raw/news/*.csv")
    print(f"  파일: {path}")
    df = spark.read.csv(path, header=True, inferSchema=True)

    # 날짜 파싱 — LEGACY 모드로 처리
    df = df.withColumn(
        'news_date',
        F.to_date(F.col('date'), "EEE, dd MMM yyyy HH:mm:ss Z")
    )
    # 파싱 실패 시 crawled_at 사용
    df = df.withColumn(
        'news_date',
        F.when(F.col('news_date').isNull(),
               F.to_date(F.col('crawled_at'), "yyyy-MM-dd"))
        .otherwise(F.col('news_date'))
    )

    df = df.withColumn('text', F.concat_ws(' ', F.col('title'), F.col('description')))
    df = df.filter(F.length(F.col('text')) > 10)
    df = df.select('company', 'title', 'description', 'text',
                   'press', 'news_date', 'link', 'crawled_at')
    print(f"  행 수: {df.count():,}")
    print(f"  기업 수: {df.select('company').distinct().count()}")
    return df


def process_weather_data(spark: SparkSession):
    print("\n[3/4] 기후 데이터 처리 중...")
    path = find_latest_file("data/raw/weather/*.csv")
    print(f"  파일: {path}")
    df = spark.read.csv(path, header=True, inferSchema=True)
    df = df.withColumn('date', F.to_date(F.col('date'), "yyyy-MM-dd"))
    numeric_cols = ['avg_temp', 'max_temp', 'min_temp',
                    'precipitation', 'avg_humidity', 'avg_wind', 'sunshine']
    for col in numeric_cols:
        df = df.withColumn(col, F.col(col).cast(DoubleType()))
    df = df.withColumn('heat_wave',  F.when(F.col('max_temp') >= 30, 1).otherwise(0))
    df = df.withColumn('cold_wave',  F.when(F.col('min_temp') <= -10, 1).otherwise(0))
    df = df.withColumn('heavy_rain', F.when(F.col('precipitation') >= 80, 1).otherwise(0))
    df = df.withColumn(
        'climate_risk_score',
        F.col('heat_wave') + F.col('cold_wave') + F.col('heavy_rain')
    )
    df_daily = df.groupBy('date').agg(
        F.avg('avg_temp').alias('national_avg_temp'),
        F.avg('climate_risk_score').alias('national_climate_risk'),
        F.sum('heat_wave').alias('heat_wave_count'),
        F.sum('cold_wave').alias('cold_wave_count'),
        F.sum('heavy_rain').alias('heavy_rain_count')
    )
    print(f"  행 수 (도시별): {df.count():,}")
    print(f"  행 수 (전국 일별): {df_daily.count():,}")
    return df_daily


def process_dart_data(spark: SparkSession):
    print("\n[4/4] DART 공시 데이터 처리 중...")
    path = find_latest_file("data/raw/dart/dart_reports*.csv")
    print(f"  파일: {path}")
    df = spark.read.csv(path, header=True, inferSchema=True)
    df = df.withColumn('report_date', F.to_date(F.col('rcept_dt'), "yyyyMMdd"))
    df = df.withColumn('year', F.year(F.col('report_date')))
    df_agg = df.groupBy('company', 'year').agg(
        F.count('rcept_no').alias('report_count'),
        F.countDistinct('report_type').alias('report_type_count')
    )
    print(f"  행 수: {df.count():,}")
    print(f"  기업 수: {df.select('company').distinct().count()}")
    return df_agg


def save_processed(df, path: str):
    """Pandas 경유 CSV 저장 (Windows 호환)"""
    os.makedirs(path, exist_ok=True)
    pandas_df = df.toPandas()
    save_file = os.path.join(path, "data.csv")
    pandas_df.to_csv(save_file, index=False, encoding='utf-8-sig')
    print(f"  저장 완료: {save_file} ({len(pandas_df):,}행)")


def run_pipeline():
    print("=" * 50)
    print("ESG-LENS PySpark 전처리 파이프라인 시작")
    print("=" * 50)

    spark = create_spark_session()

    stock_df   = process_stock_data(spark)
    news_df    = process_news_data(spark)
    weather_df = process_weather_data(spark)
    dart_df    = process_dart_data(spark)

    print("\n=== 전처리 데이터 저장 중 ===")
    save_processed(stock_df,   "data/parquet/stocks")
    save_processed(news_df,    "data/parquet/news")
    save_processed(weather_df, "data/parquet/weather")
    save_processed(dart_df,    "data/parquet/dart")

    print("\n" + "=" * 50)
    print("파이프라인 완료!")
    print("=" * 50)

    spark.stop()


if __name__ == '__main__':
    run_pipeline()