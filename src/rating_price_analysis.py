# -*- coding: utf-8 -*-
"""
비빔데이터분석가로 팀 프로젝트
주제 : 아마존 도서 데이터를 활용한 베스트셀러 성공 요인 분석 및 다년 베스트셀러 여부 분석

담당 파트(추천 분석 방향 3, 4번만 구현):
  3. 평점과 베스트셀러 지속 기간 분석
  4. 가격과 베스트셀러 지속 기간 분석

[분석의 한계 명시]
본 데이터셋에는 실제 판매 수량이나 판매량 변수가 포함되어 있지 않으므로, 본 분석의
목적은 판매 예측이 아니라 평점·가격과 베스트셀러 지속 기간의 관계를 확인하는 데 있습니다.

실행 방법:
    python src/rating_price_analysis.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# 0. 환경 설정 (한글 폰트 / 경로 / 출력 폴더)
# =============================================================================

def set_korean_font():
    from matplotlib import font_manager
    installed = {f.name for f in font_manager.fontManager.ttflist}
    candidates = ["AppleGothic", "Malgun Gothic", "NanumGothic",
                  "Noto Sans CJK KR", "Noto Sans KR"]
    for name in candidates:
        if name in installed:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False

set_korean_font()
sns.set_style("whitegrid")
matplotlib.rcParams["figure.dpi"] = 110

# --- 경로 설정 ---
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"

amazon_clean_path = DATA_DIR / "amazon_clean.csv"
final_books_path = DATA_DIR / "final_books_dataset.csv"

if not amazon_clean_path.exists() or not final_books_path.exists():
    print("[오류] amazon_clean.csv 와 final_books_dataset.csv 를 찾을 수 없습니다.")
    print(f"       확인 경로: {DATA_DIR}")
    sys.exit(1)

# --- 출력 폴더 생성 ---
FIG_DIR = BASE_DIR / "outputs" / "rating_price_visualization"
RES_DIR = BASE_DIR / "outputs" / "rating_price_analysis"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)

def savefig(fig, name):
    path = FIG_DIR / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  - 그래프 저장: {path.name}")

# 상관계수 결과를 모아 마지막에 한 번에 저장하기 위한 리스트
correlation_records = []

def add_corr(label, x, y, n):
    r = float(np.corrcoef(x, y)[0, 1])
    correlation_records.append({"분석": label, "표본수": int(n),
                                "피어슨_상관계수": round(r, 4)})
    return r

# =============================================================================
# 1. 데이터 불러오기 및 기본 구조 확인
# =============================================================================
print("=" * 70)
print("1. 데이터 불러오기 및 기본 구조 확인")
print("=" * 70)

amazon_clean = pd.read_csv(amazon_clean_path, encoding="utf-8-sig")
final_books = pd.read_csv(final_books_path, encoding="utf-8-sig")

print(f"\n[amazon_clean.csv]  (연도별 베스트셀러 기록 데이터)")
print(f"  크기: {amazon_clean.shape[0]}행 x {amazon_clean.shape[1]}열")
print(f"  결측치 합계: {int(amazon_clean.isna().sum().sum())}개")
print(f"  중복 행: {int(amazon_clean.duplicated().sum())}개")
print(f"  가격(price) 0인 기록: {int((amazon_clean['price'] == 0).sum())}건")

print(f"\n[final_books_dataset.csv]  (도서 단위 요약 데이터)")
print(f"  크기: {final_books.shape[0]}행 x {final_books.shape[1]}열")
print(f"  결측치 합계: {int(final_books.isna().sum().sum())}개")
print(f"  중복 행: {int(final_books.duplicated().sum())}개")
print(f"  평균 가격(avg_price) 0인 도서: {int((final_books['avg_price'] == 0).sum())}건")
print(f"  단년 베스트셀러(0) / 다년 베스트셀러(1): "
      f"{final_books['is_multi_year_bestseller'].value_counts().to_dict()}")
print(f"  비장기(0) / 장기 베스트셀러(1): "
      f"{final_books['is_long_seller'].value_counts().to_dict()}")

# =============================================================================
# 2. 평점 구간 / 가격 구간 변수 생성
# =============================================================================

RATING_LABELS = ["4.0 미만", "4.0~4.3", "4.4~4.6", "4.7 이상"]
def make_rating_group(s):
    return pd.cut(s, bins=[-np.inf, 4.0, 4.4, 4.7, np.inf],
                  right=False, labels=RATING_LABELS)

PRICE_LABELS = ["5달러 이하", "6~10달러", "11~15달러", "16달러 이상"]
def make_price_group(s):
    return pd.cut(s, bins=[-np.inf, 5, 10, 15, np.inf],
                  right=True, labels=PRICE_LABELS)

# =============================================================================
# 3. 평점과 베스트셀러 지속 기간 분석
# =============================================================================
print("\n" + "=" * 70)
print("3. 평점과 베스트셀러 지속 기간 분석")
print("=" * 70)

# ----- 3.1 평점 분포 확인 -----
print("\n[3.1] 평점 분포 확인")
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(amazon_clean["amazon_rating"], bins=20,
        color="#4C72B0", edgecolor="white")
ax.set_title("평점 분포", fontsize=14)
ax.set_xlabel("Amazon 사용자 평점")
ax.set_ylabel("도서 기록 수")
savefig(fig, "rating_distribution.png")
print(f"  평점 평균: {amazon_clean['amazon_rating'].mean():.3f}, "
      f"중앙값: {amazon_clean['amazon_rating'].median():.3f}, "
      f"최소~최대: {amazon_clean['amazon_rating'].min()}~{amazon_clean['amazon_rating'].max()}")

# ----- 3.2 장르별 평균 평점 비교 -----
print("\n[3.2] 장르별 평균 평점 비교")
genre_rating = amazon_clean.groupby("genre")["amazon_rating"].mean().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(genre_rating.index.astype(str), genre_rating.values,
       color=["#4C72B0", "#DD8452"])
ax.set_title("장르별 평균 평점 비교", fontsize=14)
ax.set_xlabel("장르")
ax.set_ylabel("평균 평점")
ax.set_ylim(genre_rating.min() - 0.1, genre_rating.max() + 0.1)
for i, v in enumerate(genre_rating.values):
    ax.text(i, v + 0.005, f"{v:.3f}", ha="center", va="bottom")
savefig(fig, "avg_rating_by_genre.png")
print("  장르별 평균 평점:", {k: round(v, 3) for k, v in genre_rating.items()})

# ----- 3.3 단년 vs 다년 베스트셀러 평균 평점 비교 -----
print("\n[3.3] 단년·다년 베스트셀러 평균 평점 비교")
label_map_my = {0: "단년 베스트셀러", 1: "다년 베스트셀러"}
fb_r = final_books.copy()
fb_r["status"] = fb_r["is_multi_year_bestseller"].map(label_map_my)

rating_my_summary = (fb_r.groupby("status")["avg_rating"]
                     .agg(도서수="count", 평균="mean", 중앙값="median", 표준편차="std")
                     .reindex(["단년 베스트셀러", "다년 베스트셀러"])
                     .round(4))
print(rating_my_summary)
rating_my_summary.to_csv(RES_DIR / "rating_multi_year_summary.csv", encoding="utf-8-sig")

fig, ax = plt.subplots(figsize=(7, 5))
order = ["단년 베스트셀러", "다년 베스트셀러"]
vals = [rating_my_summary.loc[o, "평균"] for o in order]
ax.bar(order, vals, color=["#999999", "#4C72B0"])
ax.set_title("단년·다년 베스트셀러 평균 평점 비교", fontsize=14)
ax.set_xlabel("베스트셀러 유형")
ax.set_ylabel("평균 평점(avg_rating)")
ax.set_ylim(min(vals) - 0.1, max(vals) + 0.1)
for i, v in enumerate(vals):
    ax.text(i, v + 0.005, f"{v:.3f}", ha="center", va="bottom")
savefig(fig, "avg_rating_by_multi_year_status.png")

# ----- 3.4 평점과 베스트셀러 등장 횟수의 관계 -----
print("\n[3.4] 평점과 베스트셀러 등장 횟수의 관계")
r_appear = add_corr("평점(avg_rating) vs 등장횟수(appear_count)",
                    final_books["avg_rating"], final_books["appear_count"],
                    len(final_books))
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(final_books["avg_rating"], final_books["appear_count"],
           alpha=0.4, color="#4C72B0", edgecolor="none")
ax.set_title("평점과 베스트셀러 등장 횟수의 관계", fontsize=14)
ax.set_xlabel("평균 평점(avg_rating)")
ax.set_ylabel("베스트셀러 등장 횟수(appear_count)")
ax.text(0.02, 0.95, f"피어슨 상관계수 = {r_appear:.3f}",
        transform=ax.transAxes, va="top",
        bbox=dict(boxstyle="round", fc="white", ec="#cccccc"))
savefig(fig, "rating_vs_appear_count.png")
print(f"  피어슨 상관계수: {r_appear:.4f}")

# ----- 3.5 평점과 베스트셀러 지속 기간의 관계 -----
print("\n[3.5] 평점과 베스트셀러 지속 기간의 관계")
r_dur = add_corr("평점(avg_rating) vs 지속기간(duration)",
                 final_books["avg_rating"], final_books["duration"],
                 len(final_books))
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(final_books["avg_rating"], final_books["duration"],
           alpha=0.4, color="#DD8452", edgecolor="none")
ax.set_title("평점과 베스트셀러 지속 기간의 관계", fontsize=14)
ax.set_xlabel("평균 평점(avg_rating)")
ax.set_ylabel("베스트셀러 지속 기간(duration)")
ax.text(0.02, 0.95, f"피어슨 상관계수 = {r_dur:.3f}",
        transform=ax.transAxes, va="top",
        bbox=dict(boxstyle="round", fc="white", ec="#cccccc"))
savefig(fig, "rating_vs_duration.png")
print(f"  피어슨 상관계수: {r_dur:.4f}")

# ----- 3.6 / 3.7 평점 구간별 다년 / 장기 베스트셀러 비율 -----
print("\n[3.6 / 3.7] 평점 구간별 다년·장기 베스트셀러 비율")
fb_rg = final_books.copy()
fb_rg["rating_group"] = make_rating_group(fb_rg["avg_rating"])

rating_group_summary = (fb_rg.groupby("rating_group", observed=False)
                        .agg(도서수=("avg_rating", "count"),
                             다년비율=("is_multi_year_bestseller", "mean"),
                             장기비율=("is_long_seller", "mean"))
                        .reindex(RATING_LABELS))
rating_group_summary["다년비율"] = (rating_group_summary["다년비율"] * 100).round(2)
rating_group_summary["장기비율"] = (rating_group_summary["장기비율"] * 100).round(2)
print(rating_group_summary)
rating_group_summary.to_csv(RES_DIR / "rating_group_summary.csv", encoding="utf-8-sig")

# 3.6 다년 베스트셀러 비율
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(RATING_LABELS, rating_group_summary["다년비율"].values, color="#4C72B0")
ax.set_title("평점 구간별 다년 베스트셀러 비율", fontsize=14)
ax.set_xlabel("평점 구간")
ax.set_ylabel("다년 베스트셀러 비율(%)")
for i, v in enumerate(rating_group_summary["다년비율"].values):
    if not np.isnan(v):
        ax.text(i, v + 0.5, f"{v:.1f}%", ha="center", va="bottom")
savefig(fig, "multi_year_ratio_by_rating_group.png")

# 3.7 장기 베스트셀러 비율
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(RATING_LABELS, rating_group_summary["장기비율"].values, color="#55A868")
ax.set_title("평점 구간별 장기 베스트셀러 비율", fontsize=14)
ax.set_xlabel("평점 구간")
ax.set_ylabel("장기 베스트셀러 비율(%)")
for i, v in enumerate(rating_group_summary["장기비율"].values):
    if not np.isnan(v):
        ax.text(i, v + 0.5, f"{v:.1f}%", ha="center", va="bottom")
savefig(fig, "long_seller_ratio_by_rating_group.png")

# ----- 3.8 평점 분석 결론 -----
print("\n[3.8] 평점 분석 결론")
print("  질문: 평점이 높은 도서가 반드시 장기 베스트셀러가 되는가?")
print("  결론: 평점이 높은 구간에서 다년 베스트셀러 비율이 높게 나타날 수 있지만,")
print("        평점만으로 장기 베스트셀러 여부를 단정하기는 어렵다. 특히 베스트셀러")
print("        도서 대부분이 이미 높은 평점을 가지고 있어 평점의 구분력이 제한될 수 있다.")

# =============================================================================
# 4. 가격과 베스트셀러 지속 기간 분석
# =============================================================================
print("\n" + "=" * 70)
print("4. 가격과 베스트셀러 지속 기간 분석")
print("=" * 70)

# ----- 4.1 가격 0인 처리 -----
n_zero_clean = int((amazon_clean["price"] == 0).sum())
n_zero_final = int((final_books["avg_price"] == 0).sum())
amazon_price = amazon_clean[amazon_clean["price"] > 0].copy()
final_price = final_books[final_books["avg_price"] > 0].copy()
print("\n[4.1] 가격 0인 처리")
print(f"  amazon_clean: price==0 {n_zero_clean}건 제외 → {len(amazon_price)}건 사용")
print(f"  final_books : avg_price==0 {n_zero_final}건 제외 → {len(final_price)}권 사용")

# ----- 4.2 가격 분포 확인 -----
print("\n[4.2] 가격 분포 확인")
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(amazon_price["price"], bins=30, color="#DD8452", edgecolor="white")
ax.set_title("가격 분포", fontsize=14)
ax.set_xlabel("가격(달러)")
ax.set_ylabel("도서 기록 수")
savefig(fig, "price_distribution.png")
print(f"  가격 평균: {amazon_price['price'].mean():.2f}, "
      f"중앙값: {amazon_price['price'].median():.2f}, "
      f"최소~최대: {amazon_price['price'].min()}~{amazon_price['price'].max()}")

# ----- 4.3 장르별 평균 가격 비교 -----
print("\n[4.3] 장르별 평균 가격 비교")
genre_price = amazon_price.groupby("genre")["price"].mean().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(genre_price.index.astype(str), genre_price.values, color=["#4C72B0", "#DD8452"])
ax.set_title("장르별 평균 가격 비교", fontsize=14)
ax.set_xlabel("장르")
ax.set_ylabel("평균 가격(달러)")
for i, v in enumerate(genre_price.values):
    ax.text(i, v + 0.1, f"{v:.2f}", ha="center", va="bottom")
savefig(fig, "avg_price_by_genre.png")
print("  장르별 평균 가격:", {k: round(v, 2) for k, v in genre_price.items()})

# ----- 4.4 가격과 리뷰 수의 관계 -----
print("\n[4.4] 가격과 리뷰 수의 관계")
review_log_for_analysis = np.log1p(amazon_price["amazon_reviews"])
r_price_rev = add_corr("가격(price) vs 리뷰수(amazon_reviews)",
                       amazon_price["price"], amazon_price["amazon_reviews"],
                       len(amazon_price))
r_price_revlog = add_corr("가격(price) vs 리뷰수 로그(review_log)",
                          amazon_price["price"], review_log_for_analysis,
                          len(amazon_price))
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(amazon_price["price"], review_log_for_analysis,
           alpha=0.4, color="#4C72B0", edgecolor="none")
ax.set_title("가격과 리뷰 수의 관계", fontsize=14)
ax.set_xlabel("가격(달러)")
ax.set_ylabel("리뷰 수(log1p 변환)")
ax.text(0.02, 0.95,
        f"상관계수(원자료) = {r_price_rev:.3f}\n상관계수(로그) = {r_price_revlog:.3f}",
        transform=ax.transAxes, va="top",
        bbox=dict(boxstyle="round", fc="white", ec="#cccccc"))
savefig(fig, "price_vs_reviews.png")
print(f"  가격 vs 리뷰수(원자료) 상관계수: {r_price_rev:.4f}")
print(f"  가격 vs 리뷰수(로그변환) 상관계수: {r_price_revlog:.4f}")

# ----- 4.5 가격과 베스트셀러 등장 횟수의 관계 -----
print("\n[4.5] 가격과 베스트셀러 등장 횟수의 관계")
r_price_appear = add_corr("평균가격(avg_price) vs 등장횟수(appear_count)",
                          final_price["avg_price"], final_price["appear_count"],
                          len(final_price))
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(final_price["avg_price"], final_price["appear_count"],
           alpha=0.4, color="#DD8452", edgecolor="none")
ax.set_title("가격과 베스트셀러 등장 횟수의 관계", fontsize=14)
ax.set_xlabel("평균 가격(avg_price, 달러)")
ax.set_ylabel("베스트셀러 등장 횟수(appear_count)")
ax.text(0.02, 0.95, f"피어슨 상관계수 = {r_price_appear:.3f}",
        transform=ax.transAxes, va="top",
        bbox=dict(boxstyle="round", fc="white", ec="#cccccc"))
savefig(fig, "price_vs_appear_count.png")
print(f"  피어슨 상관계수: {r_price_appear:.4f}")

# ----- 4.6 가격과 베스트셀러 지속 기간의 관계 -----
print("\n[4.6] 가격과 베스트셀러 지속 기간의 관계")
r_price_dur = add_corr("평균가격(avg_price) vs 지속기간(duration)",
                       final_price["avg_price"], final_price["duration"],
                       len(final_price))
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(final_price["avg_price"], final_price["duration"],
           alpha=0.4, color="#4C72B0", edgecolor="none")
ax.set_title("가격과 베스트셀러 지속 기간의 관계", fontsize=14)
ax.set_xlabel("평균 가격(avg_price, 달러)")
ax.set_ylabel("베스트셀러 지속 기간(duration)")
ax.text(0.02, 0.95, f"피어슨 상관계수 = {r_price_dur:.3f}",
        transform=ax.transAxes, va="top",
        bbox=dict(boxstyle="round", fc="white", ec="#cccccc"))
savefig(fig, "price_vs_duration.png")
print(f"  피어슨 상관계수: {r_price_dur:.4f}")

# ----- 4.7 단년 vs 다년 베스트셀러 평균 가격 비교 -----
print("\n[4.7] 단년·다년 베스트셀러 평균 가격 비교")
fp = final_price.copy()
fp["status"] = fp["is_multi_year_bestseller"].map(label_map_my)
price_my_summary = (fp.groupby("status")["avg_price"]
                    .agg(도서수="count", 평균="mean", 중앙값="median", 표준편차="std")
                    .reindex(["단년 베스트셀러", "다년 베스트셀러"])
                    .round(4))
print(price_my_summary)
price_my_summary.to_csv(RES_DIR / "price_multi_year_summary.csv", encoding="utf-8-sig")

fig, ax = plt.subplots(figsize=(7, 5))
order = ["단년 베스트셀러", "다년 베스트셀러"]
vals = [price_my_summary.loc[o, "평균"] for o in order]
ax.bar(order, vals, color=["#999999", "#DD8452"])
ax.set_title("단년·다년 베스트셀러 평균 가격 비교", fontsize=14)
ax.set_xlabel("베스트셀러 유형")
ax.set_ylabel("평균 가격(avg_price, 달러)")
for i, v in enumerate(vals):
    ax.text(i, v + 0.1, f"{v:.2f}", ha="center", va="bottom")
savefig(fig, "avg_price_by_multi_year_status.png")

# ----- 4.8 / 4.9 가격 구간별 다년 / 장기 베스트셀러 비율 -----
print("\n[4.8 / 4.9] 가격 구간별 다년·장기 베스트셀러 비율")
fp_g = final_price.copy()
fp_g["price_group"] = make_price_group(fp_g["avg_price"])
price_group_summary = (fp_g.groupby("price_group", observed=False)
                       .agg(도서수=("avg_price", "count"),
                            다년비율=("is_multi_year_bestseller", "mean"),
                            장기비율=("is_long_seller", "mean"))
                       .reindex(PRICE_LABELS))
price_group_summary["다년비율"] = (price_group_summary["다년비율"] * 100).round(2)
price_group_summary["장기비율"] = (price_group_summary["장기비율"] * 100).round(2)
print(price_group_summary)
price_group_summary.to_csv(RES_DIR / "price_group_summary.csv", encoding="utf-8-sig")

# 4.8 다년 베스트셀러 비율
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(PRICE_LABELS, price_group_summary["다년비율"].values, color="#DD8452")
ax.set_title("가격 구간별 다년 베스트셀러 비율", fontsize=14)
ax.set_xlabel("가격 구간")
ax.set_ylabel("다년 베스트셀러 비율(%)")
for i, v in enumerate(price_group_summary["다년비율"].values):
    if not np.isnan(v):
        ax.text(i, v + 0.5, f"{v:.1f}%", ha="center", va="bottom")
savefig(fig, "multi_year_ratio_by_price_group.png")

# 4.9 장기 베스트셀러 비율
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(PRICE_LABELS, price_group_summary["장기비율"].values, color="#C44E52")
ax.set_title("가격 구간별 장기 베스트셀러 비율", fontsize=14)
ax.set_xlabel("가격 구간")
ax.set_ylabel("장기 베스트셀러 비율(%)")
for i, v in enumerate(price_group_summary["장기비율"].values):
    if not np.isnan(v):
        ax.text(i, v + 0.5, f"{v:.1f}%", ha="center", va="bottom")
savefig(fig, "long_seller_ratio_by_price_group.png")

# ----- 4.10 가격 분석 결론 -----
print("\n[4.10] 가격 분석 결론")
print("  질문: 가격이 낮은 도서가 장기 베스트셀러가 될 가능성이 높은가?")
print("  결론: 일부 가격 구간에서 다년 베스트셀러 비율이 높게 나타날 수 있지만,")
print("        가격이 낮다고 해서 반드시 장기 베스트셀러가 된다고 보기는 어렵다.")
print("        가격은 장르, 독자층, 리뷰 수, 출판 시기 등 다른 요인과 함께 해석해야 한다.")

# =============================================================================
# 5. 상관계수 요약 저장
# =============================================================================
corr_df = pd.DataFrame(correlation_records)
corr_df.to_csv(RES_DIR / "correlation_summary.csv", index=False, encoding="utf-8-sig")

print("\n" + "=" * 70)
print("상관계수 요약 (correlation_summary.csv)")
print("=" * 70)
print(corr_df.to_string(index=False))

print("\n" + "=" * 70)
print("분석 완료. 산출물 위치")
print("=" * 70)
print(f"  그래프 : {FIG_DIR}")
print(f"  결과표 : {RES_DIR}")
print("\n[분석 한계] 본 데이터셋에는 실제 판매 수량이나 판매량 변수가 포함되어 있지 않으므로,")
print("            본 분석은 판매 예측이 아니라 평점·가격과 베스트셀러 지속 기간의 관계를")
print("            확인하는 데 목적을 둡니다.")
