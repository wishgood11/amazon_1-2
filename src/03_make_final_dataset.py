import pandas as pd
from pathlib import Path

# 프로젝트 폴더 기준 경로 설정
BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# 전처리된 Amazon 데이터 읽기
amazon_path = PROCESSED_DIR / "amazon_clean.csv"
amazon = pd.read_csv(amazon_path)

print("전처리 데이터 크기:", amazon.shape)

# 책 단위로 베스트셀러 등장 정보 요약
book_summary = amazon.groupby(["title_key", "author_key"]).agg(
    title=("title", "first"),
    author=("author", "first"),
    genre=("genre", "first"),

    first_year=("year", "min"),
    last_year=("year", "max"),
    appear_count=("year", "nunique"),

    avg_rating=("amazon_rating", "mean"),
    avg_reviews=("amazon_reviews", "mean"),
    max_reviews=("amazon_reviews", "max"),
    avg_price=("price", "mean"),

    title_length=("title_length", "first"),
    title_word_count=("title_word_count", "first"),
    has_colon=("has_colon", "first"),
    has_number=("has_number", "first")
).reset_index()

# 지속 기간 계산
book_summary["duration"] = book_summary["last_year"] - book_summary["first_year"] + 1

# 2년 이상 등장 여부
book_summary["is_multi_year_bestseller"] = (book_summary["appear_count"] >= 2).astype(int)

# 3년 이상 등장 여부
book_summary["is_long_seller"] = (book_summary["appear_count"] >= 3).astype(int)

# 중간에 빠졌다가 다시 등장했는지 확인
def check_reentry(years):
    years = sorted(list(years))
    if len(years) <= 1:
        return 0
    
    full_range = list(range(min(years), max(years) + 1))
    
    # 실제 등장 연도와 전체 연도 범위가 다르면 중간에 빠진 적이 있다는 뜻
    if years != full_range:
        return 1
    else:
        return 0

reentry = amazon.groupby(["title_key", "author_key"])["year"].apply(check_reentry).reset_index()
reentry = reentry.rename(columns={"year": "has_reentry"})

book_summary = book_summary.merge(reentry, on=["title_key", "author_key"], how="left")

# 보기 좋게 정렬
book_summary = book_summary.sort_values(
    by=["appear_count", "duration", "avg_reviews"],
    ascending=[False, False, False]
)

# 저장
save_path = PROCESSED_DIR / "final_books_dataset.csv"
book_summary.to_csv(save_path, index=False, encoding="utf-8-sig")

print("최종 데이터셋 생성 완료!")
print("저장 위치:", save_path)
print("최종 데이터 크기:", book_summary.shape)
print(book_summary.head(10))