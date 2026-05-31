import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ==============================
# 1. 경로 설정
# ==============================
BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "outputs" / "review_duration_analysis"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==============================
# 2. 데이터 불러오기
# ==============================

# 책 단위 데이터 사용
# avg_reviews: 책별 평균 리뷰 수
# appear_count: 베스트셀러 목록에 등장한 연도 수
# duration: 처음 등장 연도부터 마지막 등장 연도까지의 기간
# is_long_seller: 3년 이상 등장 여부
books = pd.read_csv(PROCESSED_DIR / "final_books_dataset.csv")

print("final_books_dataset 데이터 크기:", books.shape)
print(books.head())

# ==============================
# 3. 리뷰 수 분포 확인
# ==============================
review_desc = books["avg_reviews"].describe().round(2)

print("\n===== 리뷰 수 기초통계 =====")
print(review_desc)

review_desc.to_csv(
    OUTPUT_DIR / "review_distribution_summary.csv",
    encoding="utf-8-sig"
)

plt.figure(figsize=(8, 5))
plt.hist(books["avg_reviews"], bins=30)
plt.title("리뷰 수 분포")
plt.xlabel("Average Reviews")
plt.ylabel("Book Count")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "review_distribution_hist.png", dpi=300)
plt.close()

# ==============================
# 4. 리뷰 수 로그 변환
# ==============================
books["review_log"] = np.log1p(books["avg_reviews"])

log_review_desc = books["review_log"].describe().round(2)

print("\n===== 로그 변환 리뷰 수 기초통계 =====")
print(log_review_desc)

log_review_desc.to_csv(
    OUTPUT_DIR / "review_log_distribution_summary.csv",
    encoding="utf-8-sig"
)

plt.figure(figsize=(8, 5))
plt.hist(books["review_log"], bins=30)
plt.title("로그 변환 후 리뷰 수 분포")
plt.xlabel("Log Average Reviews")
plt.ylabel("Book Count")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "review_log_distribution_hist.png", dpi=300)
plt.close()

# ==============================
# 5. 리뷰 수와 지속 기간의 관계 분석
# ==============================

# Pearson 상관계수: 선형 관계 확인
pearson_corr = books["review_log"].corr(books["appear_count"], method="pearson")

# Spearman 상관계수: 순위 기반 관계 확인
spearman_corr = books["review_log"].corr(books["appear_count"], method="spearman")

corr_summary = pd.DataFrame({
    "method": ["pearson", "spearman"],
    "correlation": [pearson_corr, spearman_corr]
})

corr_summary["correlation"] = corr_summary["correlation"].round(3)

print("\n===== 리뷰 수 로그값과 베스트셀러 등장 연도 수의 상관계수 =====")
print(corr_summary)

corr_summary.to_csv(
    OUTPUT_DIR / "review_duration_correlation.csv",
    index=False,
    encoding="utf-8-sig"
)

plt.figure(figsize=(7, 5))
plt.scatter(books["review_log"], books["appear_count"], alpha=0.6)
plt.title("리뷰 수와 베스트셀러 등장 연도 수의 관계")
plt.xlabel("Log Average Reviews")
plt.ylabel("Appear Count")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "review_log_vs_appear_count.png", dpi=300)
plt.close()

plt.figure(figsize=(7, 5))
plt.scatter(books["review_log"], books["duration"], alpha=0.6)
plt.title("리뷰 수와 베스트셀러 지속 기간의 관계")
plt.xlabel("Log Average Reviews")
plt.ylabel("Duration")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "review_log_vs_duration.png", dpi=300)
plt.close()

# ==============================
# 6. 리뷰 수 구간별 장기 베스트셀러 비율 비교
# ==============================

# 리뷰 수를 낮음 / 중간 / 높음 3개 구간으로 나누기
books["review_group"] = pd.qcut(
    books["avg_reviews"],
    q=3,
    labels=["low", "middle", "high"]
)

review_group_summary = books.groupby("review_group").agg(
    book_count=("title", "count"),
    avg_reviews=("avg_reviews", "mean"),
    avg_appear_count=("appear_count", "mean"),
    avg_duration=("duration", "mean"),
    long_seller_rate=("is_long_seller", "mean")
).round(3)

review_group_summary["long_seller_rate_percent"] = (
    review_group_summary["long_seller_rate"] * 100
).round(2)

print("\n===== 리뷰 수 구간별 장기 베스트셀러 비율 =====")
print(review_group_summary)

review_group_summary.to_csv(
    OUTPUT_DIR / "review_group_long_seller_summary.csv",
    encoding="utf-8-sig"
)

plt.figure(figsize=(7, 5))
review_group_summary["avg_appear_count"].plot(kind="bar")
plt.title("리뷰 수 구간별 평균 베스트셀러 등장 연도 수")
plt.xlabel("Review Group")
plt.ylabel("Average Appear Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "review_group_avg_appear_count.png", dpi=300)
plt.close()

plt.figure(figsize=(7, 5))
review_group_summary["long_seller_rate_percent"].plot(kind="bar")
plt.title("리뷰 수 구간별 장기 베스트셀러 비율")
plt.xlabel("Review Group")
plt.ylabel("Long Seller Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "review_group_long_seller_rate.png", dpi=300)
plt.close()

# ==============================
# 7. 장기 베스트셀러 여부별 리뷰 수 비교
# ==============================

long_seller_review_summary = books.groupby("is_long_seller").agg(
    book_count=("title", "count"),
    avg_reviews=("avg_reviews", "mean"),
    median_reviews=("avg_reviews", "median"),
    avg_review_log=("review_log", "mean"),
    avg_appear_count=("appear_count", "mean"),
    avg_duration=("duration", "mean")
).round(2)

print("\n===== 장기 베스트셀러 여부별 리뷰 수 비교 =====")
print(long_seller_review_summary)

long_seller_review_summary.to_csv(
    OUTPUT_DIR / "long_seller_review_summary.csv",
    encoding="utf-8-sig"
)

plt.figure(figsize=(6, 4))
long_seller_review_summary["avg_reviews"].plot(kind="bar")
plt.title("장기 베스트셀러 여부별 평균 리뷰 수")
plt.xlabel("Is Long Seller")
plt.ylabel("Average Reviews")
plt.xticks(
    ticks=[0, 1],
    labels=["Not Long Seller", "Long Seller"],
    rotation=0
)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "long_seller_avg_reviews.png", dpi=300)
plt.close()

# ==============================
# 8. 최종 해석용 문장 출력
# ==============================
print("\n===== 분석 질문 답변용 요약 =====")

low_rate = review_group_summary.loc["low", "long_seller_rate_percent"]
high_rate = review_group_summary.loc["high", "long_seller_rate_percent"]
low_appear = review_group_summary.loc["low", "avg_appear_count"]
high_appear = review_group_summary.loc["high", "avg_appear_count"]

print(f"리뷰 수 low 구간의 장기 베스트셀러 비율: {low_rate:.2f}%")
print(f"리뷰 수 high 구간의 장기 베스트셀러 비율: {high_rate:.2f}%")
print(f"리뷰 수 low 구간의 평균 등장 연도 수: {low_appear:.2f}년")
print(f"리뷰 수 high 구간의 평균 등장 연도 수: {high_appear:.2f}년")
print(f"Spearman 상관계수: {spearman_corr:.3f}")

if high_rate > low_rate:
    print("해석: 리뷰 수가 많은 도서일수록 장기 베스트셀러가 될 가능성이 더 높게 나타남.")
else:
    print("해석: 리뷰 수가 많다고 해서 장기 베스트셀러 비율이 더 높다고 보기 어려움.")

print("결과 저장 위치:", OUTPUT_DIR)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ==============================
# 경로 설정
# ==============================
BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "outputs" / "review_duration_visualization"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)



# ==============================
# 데이터 불러오기
# ==============================
books = pd.read_csv(PROCESSED_DIR / "final_books_dataset.csv")

# ==============================
# 리뷰 수 로그 변환
# ==============================
books["review_log"] = np.log1p(books["avg_reviews"])

# ==============================
# 리뷰 수 구간 나누기
# ==============================
books["review_group"] = pd.qcut(
    books["avg_reviews"],
    q=3,
    labels=["low", "middle", "high"]
)

# ==============================
# 막대그래프 숫자 표시 함수
# ==============================
def add_labels(ax, fmt="{:.2f}", suffix=""):
    for container in ax.containers:
        labels = [fmt.format(v) + suffix for v in container.datavalues]
        ax.bar_label(container, labels=labels, padding=3, fontsize=9)

# ==============================
# 1. 원본 리뷰 수 분포
# ==============================
plt.figure(figsize=(8, 5))
plt.hist(books["avg_reviews"], bins=30)
plt.title("리뷰 수 분포")
plt.xlabel("평균 리뷰 수")
plt.ylabel("도서 수")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_review_distribution.png", dpi=300)
plt.close()

# ==============================
# 2. 로그 변환 후 리뷰 수 분포
# ==============================
plt.figure(figsize=(8, 5))
plt.hist(books["review_log"], bins=30)
plt.title("로그 변환 후 리뷰 수 분포")
plt.xlabel("로그 변환 리뷰 수")
plt.ylabel("도서 수")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_review_log_distribution.png", dpi=300)
plt.close()

# ==============================
# 3. 리뷰 수와 등장 연도 수 산점도
# ==============================
plt.figure(figsize=(8, 5))
plt.scatter(
    books["review_log"],
    books["appear_count"],
    alpha=0.6
)
plt.title("리뷰 수와 베스트셀러 등장 연도 수의 관계")
plt.xlabel("로그 변환 리뷰 수")
plt.ylabel("베스트셀러 등장 연도 수")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_review_log_vs_appear_count.png", dpi=300)
plt.close()

# ==============================
# 4. 리뷰 수 구간별 평균 등장 연도 수
# ==============================
review_summary = books.groupby("review_group").agg(
    avg_appear_count=("appear_count", "mean"),
    long_seller_rate=("is_long_seller", "mean")
)

review_summary["long_seller_rate_percent"] = (
    review_summary["long_seller_rate"] * 100
)

review_summary = review_summary.round(2)

plt.figure(figsize=(7, 5))
ax = review_summary["avg_appear_count"].plot(kind="bar")
plt.title("리뷰 수 구간별 평균 베스트셀러 등장 연도 수")
plt.xlabel("리뷰 수 구간")
plt.ylabel("평균 등장 연도 수")
plt.xticks(rotation=0)
add_labels(ax, fmt="{:.2f}", suffix="년")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "04_review_group_avg_appear_count.png", dpi=300)
plt.close()

# ==============================
# 5. 리뷰 수 구간별 장기 베스트셀러 비율
# ==============================
plt.figure(figsize=(7, 5))
ax = review_summary["long_seller_rate_percent"].plot(kind="bar")
plt.title("리뷰 수 구간별 장기 베스트셀러 비율")
plt.xlabel("리뷰 수 구간")
plt.ylabel("장기 베스트셀러 비율 (%)")
plt.xticks(rotation=0)
add_labels(ax, fmt="{:.2f}", suffix="%")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "05_review_group_long_seller_rate.png", dpi=300)
plt.close()

