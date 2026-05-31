import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# outputs 하위 폴더 경로 설정
output_dir = './outputs/title_features_analysis'
os.makedirs(output_dir, exist_ok=True)

# 데이터 로드 (final_books_dataset 사용)
df = pd.read_csv('./data/final_books_dataset.csv')


# ==========================================
# [1] 제목 길이 및 단어 수 분석
# ==========================================
# 결측치 제거 후 피어슨 상관계수 및 p-value 계산
df_clean_len = df[['title_length', 'duration']].dropna()
corr_len, p_len = stats.pearsonr(df_clean_len['title_length'], df_clean_len['duration'])

df_clean_word = df[['title_word_count', 'duration']].dropna()
corr_word, p_word = stats.pearsonr(df_clean_word['title_word_count'], df_clean_word['duration'])

# 그래프 그리기
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 제목 글자 수 vs 지속 기간
sns.regplot(data=df, x='title_length', y='duration', ax=axes[0], color='skyblue', scatter_kws={'alpha':0.6})
axes[0].set_title(f'제목 글자 수 vs 베스트셀러 지속 기간\n(r: {corr_len:.3f}, p-val: {p_len:.4f})', fontsize=12)

# 제목 단어 수 vs 지속 기간
sns.regplot(data=df, x='title_word_count', y='duration', ax=axes[1], color='salmon', scatter_kws={'alpha':0.6})
axes[1].set_title(f'제목 단어 수 vs 베스트셀러 지속 기간\n(r: {corr_word:.3f}, p-val: {p_word:.4f})', fontsize=12)

plt.tight_layout()
# ★ 결과 이미지 저장
plt.savefig(f'{output_dir}/title_length_word_vs_duration.png', dpi=300)
plt.close()


# ==========================================
# [2] 숫자 및 콜론 포함 여부 분석
# ==========================================
colon_cross = pd.crosstab(df['is_long_seller'], df['has_colon'], normalize='index') * 100
number_cross = pd.crosstab(df['is_long_seller'], df['has_number'], normalize='index') * 100

colon_cross.to_csv(f'{output_dir}/is_long_seller_vs_colon_ratio.csv')
number_cross.to_csv(f'{output_dir}/is_long_seller_vs_number_ratio.csv')

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

colon_cross.plot(kind='bar', stacked=True, ax=axes[0], color=['#d3d3d3', '#2b5c8f'])
axes[0].set_title('장기 베스트셀러 여부별 콜론(:) 포함 비율')
axes[0].set_xticklabels(['단기 (False)', '장기 (True)'], rotation=0)

number_cross.plot(kind='bar', stacked=True, ax=axes[1], color=['#e9ecef', '#e67e22'])
axes[1].set_title('장기 베스트셀러 여부별 숫자 포함 비율')
axes[1].set_xticklabels(['단기 (False)', '장기 (True)'], rotation=0)

plt.tight_layout()
plt.savefig(f'{output_dir}/title_attributes_vs_long_seller.png', dpi=300)
plt.close()


# ==========================================
# [3] 콜론 및 숫자 포함 여부에 따른 지속 기간 분포
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 콜론 포함 여부별 지속 기간 분포 (Violin Plot)
sns.violinplot(data=df, x='has_colon', y='duration', ax=axes[0], palette='Pastel1', inner='quartile')
axes[0].set_title('콜론(:) 포함 여부별 베스트셀러 지속 기간 분포')
axes[0].set_xticklabels(['미포함 (False)', '포함 (True)'])
axes[0].set_xlabel('콜론 포함 여부')
axes[0].set_ylabel('지속 기간 (주/월 등)')

# 숫자 포함 여부별 지속 기간 분포 (Violin Plot)
sns.violinplot(data=df, x='has_number', y='duration', ax=axes[1], palette='Pastel2', inner='quartile')
axes[1].set_title('숫자 포함 여부별 베스트셀러 지속 기간 분포')
axes[1].set_xticklabels(['미포함 (False)', '포함 (True)'])
axes[1].set_xlabel('숫자 포함 여부')
axes[1].set_ylabel('지속 기간 (주/월 등)')

plt.tight_layout()
# ★ 바이올린 플롯 결과 이미지 저장
plt.savefig(f'{output_dir}/title_attributes_vs_duration_violin.png', dpi=300)
plt.close()


print("5번 제목 특성 분석 완료! (피어슨 상관계수 및 바이올린 플롯 반영)")
print(f"결과물은 {output_dir} 폴더에서 확인하세요.")