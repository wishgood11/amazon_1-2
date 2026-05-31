import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"

def read_csv_auto(path):
    encodings = ["utf-8-sig", "utf-8", "cp949", "euc-kr"]

    for enc in encodings:
        try:
            print(f"{path.name} 파일을 {enc} 인코딩으로 읽는 중...")
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            print(f"{enc} 실패")
    
    raise UnicodeDecodeError("인코딩 오류", b"", 0, 1, "지원하는 인코딩으로 읽을 수 없습니다.")

amazon_clean = read_csv_auto(PROCESSED_DIR / "amazon_clean.csv")
final_books = read_csv_auto(PROCESSED_DIR / "final_books_dataset.csv")

print("===== 1. 데이터 크기 확인 =====")
print("amazon_clean:", amazon_clean.shape)
print("final_books_dataset:", final_books.shape)

print("\n===== 2. amazon_clean 결측치 확인 =====")
print(amazon_clean.isnull().sum())

print("\n===== 3. final_books_dataset 결측치 확인 =====")
print(final_books.isnull().sum())

print("\n===== 4. 중복 확인 =====")
dup_clean = amazon_clean.duplicated(subset=["title_key", "author_key", "year"]).sum()
dup_final = final_books.duplicated(subset=["title_key", "author_key"]).sum()

print("amazon_clean 도서-저자-연도 중복:", dup_clean)
print("final_books 도서-저자 중복:", dup_final)

print("\n===== 5. 가격 0원 데이터 확인 =====")
print("amazon_clean 가격 0원 개수:", (amazon_clean["price"] == 0).sum())

print("\n===== 6. 장르 분포 =====")
print(amazon_clean["genre"].value_counts())

print("\n===== 7. 장기 베스트셀러 여부 분포 =====")
print("2년 이상 등장 여부")
print(final_books["is_multi_year_bestseller"].value_counts())

print("\n3년 이상 등장 여부")
print(final_books["is_long_seller"].value_counts())

print("\n===== 8. 상위 장기 베스트셀러 TOP 10 =====")
print(final_books[["title", "author", "genre", "appear_count", "duration", "avg_reviews"]].head(10))