import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ==============================
# 1. 경로 설정
# ==============================
BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "outputs" / "genre_analysis"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==============================
# 2. 데이터 불러오기
# ==============================

# amazon_clean: 연도별 베스트셀러 등장 데이터
amazon = pd.read_csv(PROCESSED_DIR / "amazon_clean.csv")

# final_books_dataset: 책 단위로 정리된 데이터
books = pd.read_csv(PROCESSED_DIR / "final_books_dataset.csv")

print("amazon_clean 데이터 크기:", amazon.shape)
print("final_books_dataset 데이터 크기:", books.shape)

# ==============================
# 3. Fiction / Non Fiction 전체 비율 비교
# ==============================
genre_count = amazon["genre"].value_counts()
genre_ratio = amazon["genre"].value_counts(normalize=True) * 100

genre_summary = pd.DataFrame({
    "count": genre_count,
    "ratio_percent": genre_ratio.round(2)
})

print("\n===== 장르별 전체 등장 비율 =====")
print(genre_summary)

genre_summary.to_csv(
    OUTPUT_DIR / "genre_overall_ratio.csv",
    encoding="utf-8-sig"
)

plt.figure(figsize=(6, 4))
genre_count.plot(kind="bar")
plt.title("Fiction과 Non Fiction 전체 등장 횟수")
plt.xlabel("Genre")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "genre_overall_count.png", dpi=300)
plt.close()

# ==============================
# 4. 연도별 장르 비율 변화
# ==============================
year_genre_count = pd.crosstab(amazon["year"], amazon["genre"])
year_genre_ratio = pd.crosstab(
    amazon["year"],
    amazon["genre"],
    normalize="index"
) * 100

print("\n===== 연도별 장르 비율 =====")
print(year_genre_ratio.round(2))

year_genre_ratio.round(2).to_csv(
    OUTPUT_DIR / "yearly_genre_ratio.csv",
    encoding="utf-8-sig"
)

plt.figure(figsize=(10, 5))
for genre in year_genre_ratio.columns:
    plt.plot(
        year_genre_ratio.index,
        year_genre_ratio[genre],
        marker="o",
        label=genre
    )

plt.title("연도별 Fiction / Non Fiction 비율 변화")
plt.xlabel("Year")
plt.ylabel("Ratio (%)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "yearly_genre_ratio.png", dpi=300)
plt.close()

# ==============================
# 5. 장르별 평균 평점 비교
# ==============================
genre_rating = amazon.groupby("genre")["amazon_rating"].mean().round(2)

print("\n===== 장르별 평균 평점 =====")
print(genre_rating)

genre_rating.to_csv(
    OUTPUT_DIR / "genre_avg_rating.csv",
    encoding="utf-8-sig"
)

plt.figure(figsize=(6, 4))
genre_rating.plot(kind="bar")
plt.title("장르별 평균 평점")
plt.xlabel("Genre")
plt.ylabel("Average Rating")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "genre_avg_rating.png", dpi=300)
plt.close()

# ==============================
# 6. 장르별 평균 리뷰 수 비교
# ==============================
genre_reviews = amazon.groupby("genre")["amazon_reviews"].mean().round(2)

print("\n===== 장르별 평균 리뷰 수 =====")
print(genre_reviews)

genre_reviews.to_csv(
    OUTPUT_DIR / "genre_avg_reviews.csv",
    encoding="utf-8-sig"
)

plt.figure(figsize=(6, 4))
genre_reviews.plot(kind="bar")
plt.title("장르별 평균 리뷰 수")
plt.xlabel("Genre")
plt.ylabel("Average Reviews")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "genre_avg_reviews.png", dpi=300)
plt.close()

# ==============================
# 7. 장르별 평균 가격 비교
# ==============================
genre_price = amazon.groupby("genre")["price"].mean().round(2)

print("\n===== 장르별 평균 가격 =====")
print(genre_price)

genre_price.to_csv(
    OUTPUT_DIR / "genre_avg_price.csv",
    encoding="utf-8-sig"
)

plt.figure(figsize=(6, 4))
genre_price.plot(kind="bar")
plt.title("장르별 평균 가격")
plt.xlabel("Genre")
plt.ylabel("Average Price")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "genre_avg_price.png", dpi=300)
plt.close()

# ==============================
# 8. 장르별 베스트셀러 지속 기간 비교
# ==============================

# appear_count: 베스트셀러 목록에 등장한 연도 수
# duration: 처음 등장 연도부터 마지막 등장 연도까지의 기간
genre_duration = books.groupby("genre").agg(
    avg_appear_count=("appear_count", "mean"),
    avg_duration=("duration", "mean"),
    long_seller_rate=("is_long_seller", "mean")
).round(3)

genre_duration["long_seller_rate_percent"] = (
    genre_duration["long_seller_rate"] * 100
).round(2)

print("\n===== 장르별 베스트셀러 지속 기간 =====")
print(genre_duration)

genre_duration.to_csv(
    OUTPUT_DIR / "genre_duration_summary.csv",
    encoding="utf-8-sig"
)

plt.figure(figsize=(6, 4))
genre_duration["avg_appear_count"].plot(kind="bar")
plt.title("장르별 평균 베스트셀러 등장 연도 수")
plt.xlabel("Genre")
plt.ylabel("Average Appear Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "genre_avg_appear_count.png", dpi=300)
plt.close()

plt.figure(figsize=(6, 4))
genre_duration["avg_duration"].plot(kind="bar")
plt.title("장르별 평균 베스트셀러 지속 기간")
plt.xlabel("Genre")
plt.ylabel("Average Duration")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "genre_avg_duration.png", dpi=300)
plt.close()

# ==============================
# 9. 최종 해석용 문장 출력
# ==============================
print("\n===== 분석 질문 답변용 요약 =====")

more_common_genre = genre_count.idxmax()
more_common_ratio = genre_ratio.loc[more_common_genre].round(2)

longer_genre = genre_duration["avg_appear_count"].idxmax()
longer_value = genre_duration.loc[longer_genre, "avg_appear_count"]

print(f"전체 베스트셀러 등장 횟수는 {more_common_genre} 장르가 가장 많음.")
print(f"{more_common_genre}의 전체 등장 비율은 약 {more_common_ratio}%임.")
print(f"평균 베스트셀러 등장 연도 수는 {longer_genre} 장르가 더 높음.")
print(f"{longer_genre}의 평균 등장 연도 수는 약 {longer_value:.2f}년임.")

print("결과 저장 위치:", OUTPUT_DIR)


import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ==============================
# 경로 설정
# ==============================
BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "outputs" / "genre_visualization"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)



# ==============================
# 데이터 불러오기
# ==============================
amazon = pd.read_csv(PROCESSED_DIR / "amazon_clean.csv")
books = pd.read_csv(PROCESSED_DIR / "final_books_dataset.csv")

# ==============================
# 막대그래프 숫자 표시 함수
# ==============================
def add_labels(ax, fmt="{:.2f}", suffix=""):
    for container in ax.containers:
        labels = [fmt.format(v) + suffix for v in container.datavalues]
        ax.bar_label(container, labels=labels, padding=3, fontsize=9)

# ==============================
# 1. 장르별 전체 등장 비율
# ==============================
genre_ratio = amazon["genre"].value_counts(normalize=True) * 100

plt.figure(figsize=(7, 5))
ax = genre_ratio.plot(kind="bar")
plt.title("Fiction과 Non Fiction 전체 베스트셀러 등장 비율")
plt.xlabel("장르")
plt.ylabel("비율 (%)")
plt.xticks(rotation=0)
add_labels(ax, fmt="{:.2f}", suffix="%")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_genre_overall_ratio.png", dpi=300)
plt.close()

# ==============================
# 2. 연도별 장르 비율 변화
# ==============================
year_genre_ratio = pd.crosstab(
    amazon["year"],
    amazon["genre"],
    normalize="index"
) * 100

year_genre_ratio = year_genre_ratio.sort_index()

plt.figure(figsize=(10, 5))
for genre in year_genre_ratio.columns:
    plt.plot(
        year_genre_ratio.index,
        year_genre_ratio[genre],
        marker="o",
        label=genre
    )

plt.title("연도별 Fiction / Non Fiction 비율 변화")
plt.xlabel("연도")
plt.ylabel("비율 (%)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_yearly_genre_ratio.png", dpi=300)
plt.close()

# ==============================
# 3. 장르별 평균 리뷰 수
# ==============================
genre_reviews = amazon.groupby("genre")["amazon_reviews"].mean().round(2)

plt.figure(figsize=(7, 5))
ax = genre_reviews.plot(kind="bar")
plt.title("장르별 평균 리뷰 수 비교")
plt.xlabel("장르")
plt.ylabel("평균 리뷰 수")
plt.xticks(rotation=0)
add_labels(ax, fmt="{:.0f}")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_genre_avg_reviews.png", dpi=300)
plt.close()

# ==============================
# 4. 장르별 평균 가격
# ==============================
genre_price = amazon.groupby("genre")["price"].mean().round(2)

plt.figure(figsize=(7, 5))
ax = genre_price.plot(kind="bar")
plt.title("장르별 평균 가격 비교")
plt.xlabel("장르")
plt.ylabel("평균 가격")
plt.xticks(rotation=0)
add_labels(ax, fmt="{:.2f}")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "04_genre_avg_price.png", dpi=300)
plt.close()

# ==============================
# 5. 장르별 평균 베스트셀러 등장 연도 수
# ==============================
genre_appear = books.groupby("genre")["appear_count"].mean().round(2)

plt.figure(figsize=(7, 5))
ax = genre_appear.plot(kind="bar")
plt.title("장르별 평균 베스트셀러 등장 연도 수")
plt.xlabel("장르")
plt.ylabel("평균 등장 연도 수")
plt.xticks(rotation=0)
add_labels(ax, fmt="{:.2f}", suffix="년")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "05_genre_avg_appear_count.png", dpi=300)
plt.close()