import pandas as pd
import numpy as np
import re
from pathlib import Path

# 프로젝트 폴더 기준 경로 설정
BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# processed 폴더가 없으면 자동 생성
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# raw 폴더 안의 csv 파일 자동 찾기
csv_files = list(RAW_DIR.glob("*.csv"))

if len(csv_files) == 0:
    raise FileNotFoundError("data/raw 폴더 안에 CSV 파일이 없습니다.")

# 첫 번째 csv 파일 읽기
amazon = pd.read_csv(csv_files[0])

print("원본 데이터 크기:", amazon.shape)
print("원본 컬럼:", amazon.columns.tolist())

# 컬럼명 정리
amazon = amazon.rename(columns={
    "Name": "title",
    "Author": "author",
    "User Rating": "amazon_rating",
    "Reviews": "amazon_reviews",
    "Price": "price",
    "Year": "year",
    "Genre": "genre"
})

# 문자열 정리 함수
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# 제목/저자 정리용 key 생성
amazon["title_key"] = amazon["title"].apply(clean_text)
amazon["author_key"] = amazon["author"].apply(clean_text)

# 자료형 변환
amazon["amazon_rating"] = pd.to_numeric(amazon["amazon_rating"], errors="coerce")
amazon["amazon_reviews"] = pd.to_numeric(amazon["amazon_reviews"], errors="coerce")
amazon["price"] = pd.to_numeric(amazon["price"], errors="coerce")
amazon["year"] = pd.to_numeric(amazon["year"], errors="coerce")

# 제목 기반 파생변수
amazon["title_length"] = amazon["title"].astype(str).str.len()
amazon["title_word_count"] = amazon["title"].astype(str).str.split().str.len()
amazon["has_colon"] = amazon["title"].astype(str).str.contains(":", regex=False).astype(int)
amazon["has_number"] = amazon["title"].astype(str).str.contains(r"\d", regex=True).astype(int)

# 리뷰 수 로그 변환
amazon["review_log"] = np.log1p(amazon["amazon_reviews"])

# 리뷰 수 구간화
amazon["review_level"] = pd.qcut(
    amazon["amazon_reviews"],
    q=3,
    labels=["low", "middle", "high"]
)

# 평점 구간화
amazon["rating_level"] = pd.cut(
    amazon["amazon_rating"],
    bins=[0, 4.5, 4.8, 5.0],
    labels=["normal", "high", "very_high"],
    include_lowest=True
)
# ==============================
# 같은 도서-저자-연도 기준 중복 통합
# ==============================

before_count = len(amazon)

amazon = amazon.groupby(["title_key", "author_key", "year"], as_index=False).agg(
    title=("title", "first"),
    author=("author", "first"),
    amazon_rating=("amazon_rating", "mean"),
    amazon_reviews=("amazon_reviews", "mean"),
    price=("price", "mean"),
    genre=("genre", "first"),
    title_length=("title_length", "first"),
    title_word_count=("title_word_count", "first"),
    has_colon=("has_colon", "first"),
    has_number=("has_number", "first"),
    review_log=("review_log", "mean"),
    review_level=("review_level", "first"),
    rating_level=("rating_level", "first")
)

after_count = len(amazon)

print("중복 통합 전 행 수:", before_count)
print("중복 통합 후 행 수:", after_count)
print("통합된 중복 행 수:", before_count - after_count)

# 저장
save_path = PROCESSED_DIR / "amazon_clean.csv"
amazon.to_csv(save_path, index=False, encoding="utf-8-sig")

# 저장
save_path = PROCESSED_DIR / "amazon_clean.csv"
amazon.to_csv(save_path, index=False, encoding="utf-8-sig")

print("전처리 완료!")
print("저장 위치:", save_path)
print("전처리 데이터 크기:", amazon.shape)
print(amazon.head())

