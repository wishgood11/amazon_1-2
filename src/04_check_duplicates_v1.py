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
            print(f"{enc} 인코딩 실패")
        except Exception as e:
            print(f"{enc} 시도 중 오류 발생:", e)

    raise Exception("CSV 파일을 읽을 수 없습니다. 인코딩 또는 파일 형식을 확인하세요.")

amazon_path = PROCESSED_DIR / "amazon_clean.csv"

print("읽을 파일 경로:", amazon_path)

amazon = read_csv_auto(amazon_path)

duplicates = amazon[amazon.duplicated(
    subset=["title_key", "author_key", "year"],
    keep=False
)]

print("중복 데이터 개수:", len(duplicates))

if len(duplicates) > 0:
    print(
        duplicates[[
            "title",
            "author",
            "year",
            "genre",
            "amazon_rating",
            "amazon_reviews",
            "price"
        ]].sort_values(["title", "author", "year"])
    )
else:
    print("중복 데이터가 없습니다.")