# test_env.py
import torch
import pandas as pd
import numpy as np
import transformers
import xgboost
import streamlit
import pykrx
from pyspark.sql import SparkSession

print("=== 환경 세팅 확인 ===")
print(f"PyTorch     : {torch.__version__}")
print(f"Pandas      : {pd.__version__}")
print(f"Numpy       : {np.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"XGBoost     : {xgboost.__version__}")

# PySpark 확인 (조금 시간이 걸려요)
print("PySpark 확인 중...")
spark = SparkSession.builder.appName("test").getOrCreate()
print(f"PySpark     : {spark.version}")
spark.stop()

print("\n모든 라이브러리 정상 설치 완료!")