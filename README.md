# Contest-Dacon-Smart-Maritime-Anomaly-Diagnosis

**스마트 해운물류 x AI 미션 챌린지 : 이상신호 감지 기반 비정상 작동 진단** (해양수산부 · DACON)

센서 데이터(`X_01 ~ X_52`)만으로 장비의 정상/비정상 작동 유형(21개 클래스)을 분류하는 대회입니다.

| 항목 | 내용 |
|---|---|
| 평가 지표 | Macro F1 |
| 최종 Private Score | **Macro F1 87.793** |
| 데이터 특성 | 시계열 정보 없음, 피처명 비식별화, 다중 클래스(21) |
| 팀 | SKKU_SME_TM |
| 발표 자료 | [docs/solution_SKKU_SME_TM.pdf](docs/solution_SKKU_SME_TM.pdf) |

## 솔루션 한눈에 보기

베이스라인 Confusion Matrix에서 클래스 간 난이도 차이가 크다는 점(클래스 0·9·15 극상, 3·16 상, 10·11·14 중)을 확인했고, 어려운 클래스로 예측된 샘플만 따로 다시 분류하는 **3단 Cascade**를 구성했습니다.

```
          ┌────────────── Feature Store ──────────────┐
 X_01~52 ─► Scaled raw | PCA | LDA | Log-Lik | Mahalanobis ─┐
          └───────────────────────────────────────────┘     │
                                                            ▼
 [Stage 1] 21개 전체 클래스 XGBoost (5-fold bagging)
      │ MID 클래스(0,3,9,10,11,14,15,16,19)로 예측된 샘플만
      ▼
 [Stage 2] MID 9개 클래스 전용 XGBoost 재분류
      │ HARD 클래스(0,3,9,15,16,19)로 예측된 샘플만
      ▼
 [Stage 3] HARD 6개 클래스 전용 XGBoost 정밀 재분류  ─► 최종 예측
```

## 사용한 기법 (적용 순서)

각 단계의 구현 위치와 함께 정리했습니다.

### 1. 전처리

| # | Method | 기능 | 효과 | 코드 |
|---|---|---|---|---|
| 1 | **StandardScaler** | 각 피처를 평균 0, 표준편차 1로 변환 | 센서마다 다른 스케일을 보정. PCA 등 선형변환 기반 기법의 전제 조건이며, 모델 수렴 속도 향상 | [src/preprocessing.py](src/preprocessing.py) `ScalePCA` |
| 2 | **PCA (52개 전체 성분 사용)** | 공분산 행렬의 고유벡터 방향으로 공간 회전 | 차원을 줄이지 않고 **회전 + 정규 직교화**만 수행. 다중공선성 제거, 중복 정보 감소, 공분산 역행렬 계산의 수치 안정성 증가(→ LDA·Gaussian 계열에 유리), Boosting 모델이 독립 피처에 더 잘 수렴. 피처 셀렉션으로 인한 정보 손실 방지 | [src/preprocessing.py](src/preprocessing.py) `ScalePCA` |

### 2. Feature Engineering (Feature Store)

PCA 공간에서 클래스 정보를 요약한 피처를 만들어 분류기에 함께 제공합니다. 시계열이 없고 피처가 비식별화되어 도메인 지식을 쓸 수 없으므로, **순수 통계 기반 판단 구조**를 피처로 만든 것입니다.

| # | Feature | 기능 | 효과 | 코드 |
|---|---|---|---|---|
| 1 | **Raw (scaled)** | 스케일링된 원본 피처 | 원본 정보 보존 | [src/features.py](src/features.py) `FeatureStore` |
| 2 | **PCA Feature** | 52개 주성분 점수 | 서로 직교하는 정규화된 독립 피처 제공 | 〃 |
| 3 | **LDA Feature** | 클래스 간 분산 / 클래스 내 분산 비를 최대화하는 판별축(최대 K-1개)에 투영 (`S_B w = λ S_W w`) | "샘플이 어느 클래스에 가까운가"를 담은 정제된 상대 위치 정보. 후속 분류기가 더 정확한 결정 경계를 학습 | 〃 |
| 4 | **Gaussian Log-Likelihood** | PCA 공간에서 클래스별 가우시안 `N(μc, Σc + εI)` 적합 후 `ℓc(x) = -½[(x-μc)ᵀΣc⁻¹(x-μc) + log\|Σc\|]` (K차원) | "x가 클래스 c의 분포에서 나왔을 법한가"를 직접 수치화하는 생성적 점수. 축 척도·상관 구조를 반영해 유클리드 거리보다 정교 | [src/features.py](src/features.py) `fit_class_gaussians`, `gaussian_loglik` |
| 5 | **Mahalanobis Distance** | 클래스 중심·공분산 기준 거리 `dc(x) = √((x-μc)ᵀΣc⁻¹(x-μc))` (K차원) | 로그우도와 단조 관계지만 수치 안정성·결정경계 보정에 실전적으로 도움. 어느 클래스 중심에 더 가까운지를 직접 거리로 제공 | [src/features.py](src/features.py) `mahalanobis` |

`FeatureStore`는 fold의 **학습 구간에서만** fit하므로 검증 구간 정보가 scaler / PCA / LDA / Gaussian 통계에 섞이지 않습니다.

### 3. 모델링

| # | Method | 기능 | 효과 | 코드 |
|---|---|---|---|---|
| 1 | **XGBoost** (`multi:softprob`) | 클래스별 확률을 출력하는 gradient boosting. 하이퍼파라미터는 [src/config.py](src/config.py) | 정규화된 독립 피처에 잘 수렴, 클래스별 확률 분포 출력 | [src/models.py](src/models.py) `make_xgb` |
| 2 | **Stratified 5-Fold CV Bagging** | 5개 fold 각각 FeatureStore + XGBoost를 학습하고 test 확률을 평균 | 단일 학습 대비 분산 감소, OOF Macro F1로 단계별 성능 확인 | [src/models.py](src/models.py) `StageClassifier` |
| 3 | **3-Stage Cascade (MID → HARD 재분류)** | Stage1 전체 21클래스 → MID 9클래스 전용 → HARD 6클래스 전용 모델로 단계적 재분류 | 헷갈리는 클래스끼리만 모아 학습하므로 고난이도 클래스 간 경계를 집중적으로 학습. 베이스라인에서 뭉쳐 있던 0·9·15 등의 오분류 완화 | [src/pipeline.py](src/pipeline.py) `CascadeClassifier` |
| 4 | **Macro F1** | 클래스별 F1의 평균 | 클래스 불균형·난이도 차이에 관계없이 모든 클래스를 동일 비중으로 평가 | `sklearn.metrics.f1_score(average="macro")` |

### 4. 탐색(EDA)과 실험

최종 파이프라인에는 들어가지 않지만 위 설계를 결정하기 위해 수행한 분석입니다.

| Script | 내용 | 얻은 것 |
|---|---|---|
| [eda/run_eda.py](eda/run_eda.py) `importance` | XGBoost feature importance, SHAP summary | 기여도 높은 피처 파악 |
| 〃 `pca` | 2D PCA 투영을 타깃별 색상으로 시각화 | 클래스 군집·겹침 구조 확인 (고난이도 클래스가 서로 겹침) |
| 〃 `corr`, `vif` | 상관 히트맵, \|corr\|≥0.8 쌍, VIF 및 VIF 기반 제거 후보 | 센서 피처 간 다중공선성 확인 → PCA 직교화의 근거 |
| 〃 `dist` | 피처 히스토그램 / 타깃별 boxplot / 피처-타깃 산점도 | 일부 클래스는 단일 변수만으로 분리 가능함을 확인 |
| 〃 `hard` | 고난이도 클래스 한정 pairwise 산점도, `X_01` × 클래스 교차표 | 고난이도 클래스는 단일 변수로 분리 불가 → 전용 분류기 필요성 |
| [experiments/feature_ablation.py](experiments/feature_ablation.py) | 8가지 피처 조합(scaled / PCA / LDA / 조합 / PCA→LDA)을 XGBoost·MLP로 비교, 클래스 구성(`hard`, `hard9`, `mid_vs_hard`) 선택 | 어떤 피처 조합이 고난이도 클래스에 유효한지 검증 |

## 폴더 구조

```
├── docs/                      발표(Solution) PDF
├── src/
│   ├── config.py              시드, 클래스 그룹(MID/HARD), XGBoost 하이퍼파라미터
│   ├── data.py                CSV 로딩
│   ├── preprocessing.py       StandardScaler + PCA
│   ├── features.py            LDA / Gaussian Log-Lik / Mahalanobis → FeatureStore
│   ├── models.py              XGBoost 팩토리, 5-fold bagging StageClassifier
│   └── pipeline.py            3-Stage CascadeClassifier
├── scripts/train_predict.py   학습 → 제출 파일 생성
├── experiments/feature_ablation.py
├── eda/run_eda.py
└── notebooks_original/        정리 전 원본 노트북 (출력 제거)
```

## 실행 방법

```bash
pip install -r requirements.txt

# 대회 데이터(train.csv, test.csv, sample_submission.csv)를 data/ 에 배치
python scripts/train_predict.py --data-dir data --out outputs/submission.csv
python scripts/train_predict.py --data-dir data --n-estimators 100   # 빠른 확인용

python eda/run_eda.py --data-dir data --steps corr vif pca
python experiments/feature_ablation.py --data-dir data --classes hard --model xgb
```

## 참고 사항

- 대회 데이터는 저장소에 포함하지 않았습니다(`.gitignore`).
- 정리된 `src/`의 3-Stage 파이프라인은 발표 자료(13p)의 구조와 `notebooks_original/01`(2-Stage + gating 버전)의 함수들을 바탕으로 재구성한 것입니다. 대회 데이터 없이 합성 데이터로 실행 동작만 확인했으며, 대회 제출 당시 노트북을 그대로 재현한 것은 아니므로 점수(87.793)가 동일하게 재현된다고 보장하지는 않습니다.
- 원본 `01` 노트북에는 신뢰도 gap 기준 gating(`GAP_THR`)과 LightGBM/CatBoost 앙상블 시도가 있었으나, 최종 구조(슬라이드 기준)에서는 XGBoost 단일 모델 + 클래스 기반 라우팅을 사용하므로 정리본에서는 제외했습니다.
