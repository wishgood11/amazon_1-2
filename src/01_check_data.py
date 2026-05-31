# import pandas as pd

# amazon = pd.read_csv("data/raw/amazon_2009_2022"
# ".csv")

# print(amazon.head())
# print(amazon.info())
# print(amazon.columns)
import pandas as pd
from pathlib import Path

# 이 파일 기준으로 프로젝트 폴더 찾기
BASE_DIR = Path(__file__).resolve().parents[1]

# data/raw 폴더 찾기
RAW_DIR = BASE_DIR / "data" / "raw"

# raw 폴더 안의 csv 파일 자동 찾기
csv_files = list(RAW_DIR.glob("*.csv"))

print("프로젝트 폴더:", BASE_DIR)
print("raw 폴더:", RAW_DIR)
print("찾은 CSV 파일:", [file.name for file in csv_files])

if len(csv_files) == 0:
    raise FileNotFoundError("data/raw 폴더 안에 CSV 파일이 없습니다.")

# 첫 번째 CSV 파일 읽기
amazon = pd.read_csv(csv_files[0])

print(amazon.head())
print(amazon.info())
print(amazon.columns)