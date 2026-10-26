# Dataset audit (auto-generated)

Generated 2026-10-27T00:03:37 by `python -m scripts.run_audit`. Every number below comes from running that script on the files in `data/raw/`. Do not edit by hand; interpretation lives in `docs/dataset_audit_findings.md`.

## 1. Files

| file | size_mb | sha256 |
|---|---|---|
| application_train.csv | 166.133 | 52e96b895b11 |
| application_test.csv | 26.568 | a36161331d83 |
| bureau.csv | 170.017 | 9d799143423f |
| bureau_balance.csv | 375.593 | 33e09f06174c |
| previous_application.csv | 404.973 | 5046cd657ee0 |
| POS_CASH_balance.csv | 392.703 | 0e13bc573ffa |
| installments_payments.csv | 723.118 | 428c2e2496e4 |
| credit_card_balance.csv | 424.583 | a9cdc48900d5 |
| sample_submission.csv | 0.536 | 95258f8383ad |
| HomeCredit_columns_description.csv | 0.037 | eef766539822 |

## 2. Tables

| table | rows | columns | numeric_cols | non_numeric_cols | memory_mb | cols_with_missing | cols_over_50pct_missing | pct_cells_missing |
|---|---|---|---|---|---|---|---|---|
| application_train | 307,511 | 122 | 106 | 16 | 179.897 | 67 | 41 | 24.396 |
| application_test | 48,744 | 121 | 105 | 16 | 28.469 | 64 | 29 | 23.812 |
| sample_submission | 48,744 | 2 | 2 | 0 | 0.585 | 0 | 0 | 0 |
| bureau | 1,716,428 | 17 | 14 | 3 | 144.181 | 7 | 2 | 13.503 |
| bureau_balance | 27,299,925 | 3 | 2 | 1 | 163.8 | 0 | 0 | 0 |
| previous_application | 1,670,214 | 37 | 21 | 16 | 253.876 | 16 | 4 | 17.977 |
| POS_CASH_balance | 10,001,358 | 8 | 7 | 1 | 300.041 | 2 | 0 | 0.065 |
| installments_payments | 13,605,401 | 8 | 8 | 0 | 680.27 | 2 | 0 | 0.005 |
| credit_card_balance | 3,840,312 | 23 | 22 | 1 | 537.644 | 9 | 0 | 6.654 |

## 3. Target

Official description of `TARGET`: "Target variable (1 - client with payment difficulties: he/she had late payment more than X days on at least one of the first Y installments of the loan in our sample, 0 - all other cases)"

- Rows: 307,511
- TARGET=1: 24,825 (8.07%)
- TARGET=0: 282,686
- Negatives per positive: 11.39

## 4. Train vs test schema

- Columns in train but not test: ['TARGET']
- Columns in test but not train: []
- SK_ID_CURR values shared by train and test: 0
- sample_submission IDs identical to application_test IDs: yes

## 5. Keys and relationships

### Primary keys

| table | key | n_rows | n_null | n_unique | n_duplicates | is_primary_key |
|---|---|---|---|---|---|---|
| application_train | SK_ID_CURR | 307,511 | 0 | 307,511 | 0 | yes |
| application_test | SK_ID_CURR | 48,744 | 0 | 48,744 | 0 | yes |
| sample_submission | SK_ID_CURR | 48,744 | 0 | 48,744 | 0 | yes |
| bureau | SK_ID_BUREAU | 1,716,428 | 0 | 1,716,428 | 0 | yes |
| previous_application | SK_ID_PREV | 1,670,214 | 0 | 1,670,214 | 0 | yes |

### Foreign keys

| child | parent | key | child_rows | child_unique_keys | orphan_child_keys | pct_child_keys_in_parent | pct_parent_with_child | rows_per_key_median | rows_per_key_max |
|---|---|---|---|---|---|---|---|---|---|
| bureau | applications | SK_ID_CURR | 1,716,428 | 305,811 | 0 | 100 | 85.84 | 4 | 116 |
| bureau_balance | bureau | SK_ID_BUREAU | 27,299,925 | 817,395 | 43,041 | 94.734 | 45.114 | 26 | 97 |
| previous_application | applications | SK_ID_CURR | 1,670,214 | 338,857 | 0 | 100 | 95.116 | 4 | 77 |
| POS_CASH_balance | applications | SK_ID_CURR | 10,001,358 | 337,252 | 0 | 100 | 94.666 | 22 | 295 |
| POS_CASH_balance | previous_application | SK_ID_PREV | 10,001,358 | 936,325 | 37,422 | 96.003 | 53.82 | 10 | 96 |
| installments_payments | applications | SK_ID_CURR | 13,605,401 | 339,587 | 0 | 100 | 95.321 | 25 | 372 |
| installments_payments | previous_application | SK_ID_PREV | 13,605,401 | 997,752 | 38,847 | 96.107 | 57.412 | 10 | 293 |
| credit_card_balance | applications | SK_ID_CURR | 3,840,312 | 103,558 | 0 | 100 | 29.069 | 22 | 192 |
| credit_card_balance | previous_application | SK_ID_PREV | 3,840,312 | 104,307 | 11,372 | 89.098 | 5.564 | 21 | 96 |

### Coverage of training applicants

| table | pct_train_applicants_with_records |
|---|---|
| bureau | 85.685 |
| previous_application | 94.649 |
| POS_CASH_balance | 94.125 |
| installments_payments | 94.84 |
| credit_card_balance | 28.261 |


bureau_balance: 817,395 bureau credits have monthly records; 811,206 have 2 or more months.

## 6. Missing values

### application_train: 25 most-missing columns

| column | dtype | n_missing | pct_missing |
|---|---|---|---|
| COMMONAREA_AVG | float64 | 214,865 | 69.872 |
| COMMONAREA_MEDI | float64 | 214,865 | 69.872 |
| COMMONAREA_MODE | float64 | 214,865 | 69.872 |
| NONLIVINGAPARTMENTS_AVG | float64 | 213,514 | 69.433 |
| NONLIVINGAPARTMENTS_MEDI | float64 | 213,514 | 69.433 |
| NONLIVINGAPARTMENTS_MODE | float64 | 213,514 | 69.433 |
| FONDKAPREMONT_MODE | category | 210,295 | 68.386 |
| LIVINGAPARTMENTS_AVG | float64 | 210,199 | 68.355 |
| LIVINGAPARTMENTS_MEDI | float64 | 210,199 | 68.355 |
| LIVINGAPARTMENTS_MODE | float64 | 210,199 | 68.355 |
| FLOORSMIN_AVG | float64 | 208,642 | 67.849 |
| FLOORSMIN_MEDI | float64 | 208,642 | 67.849 |
| FLOORSMIN_MODE | float64 | 208,642 | 67.849 |
| YEARS_BUILD_AVG | float64 | 204,488 | 66.498 |
| YEARS_BUILD_MEDI | float64 | 204,488 | 66.498 |
| YEARS_BUILD_MODE | float64 | 204,488 | 66.498 |
| OWN_CAR_AGE | float64 | 202,929 | 65.991 |
| LANDAREA_AVG | float64 | 182,590 | 59.377 |
| LANDAREA_MEDI | float64 | 182,590 | 59.377 |
| LANDAREA_MODE | float64 | 182,590 | 59.377 |
| BASEMENTAREA_AVG | float64 | 179,943 | 58.516 |
| BASEMENTAREA_MEDI | float64 | 179,943 | 58.516 |
| BASEMENTAREA_MODE | float64 | 179,943 | 58.516 |
| EXT_SOURCE_1 | float64 | 173,378 | 56.381 |
| NONLIVINGAREA_AVG | float64 | 169,682 | 55.179 |

## 7. Temporal fields (days/months relative to the application, day 0)

| table | column | n_placeholder_365243 | min | max | n_positive |
|---|---|---|---|---|---|
| application_train | DAYS_BIRTH | 0 | -25,229 | -7,489 | 0 |
| application_train | DAYS_EMPLOYED | 55,374 | -17,912 | 0 | 0 |
| application_train | DAYS_REGISTRATION | 0 | -24,672 | 0 | 0 |
| application_train | DAYS_ID_PUBLISH | 0 | -7,197 | 0 | 0 |
| application_train | DAYS_LAST_PHONE_CHANGE | 0 | -4,292 | 0 | 0 |
| application_test | DAYS_BIRTH | 0 | -25,195 | -7,338 | 0 |
| application_test | DAYS_EMPLOYED | 9,274 | -17,463 | -1 | 0 |
| application_test | DAYS_REGISTRATION | 0 | -23,722 | 0 | 0 |
| application_test | DAYS_ID_PUBLISH | 0 | -6,348 | 0 | 0 |
| application_test | DAYS_LAST_PHONE_CHANGE | 0 | -4,361 | 0 | 0 |
| bureau | DAYS_CREDIT | 0 | -2,922 | 0 | 0 |
| bureau | DAYS_CREDIT_ENDDATE | 0 | -42,060 | 31,199 | 602,603 |
| bureau | DAYS_ENDDATE_FACT | 0 | -42,023 | 0 | 0 |
| bureau | DAYS_CREDIT_UPDATE | 0 | -41,947 | 372 | 17 |
| bureau_balance | MONTHS_BALANCE | 0 | -96 | 0 | 0 |
| previous_application | DAYS_DECISION | 0 | -2,922 | -1 | 0 |
| previous_application | DAYS_FIRST_DRAWING | 934,444 | -2,922 | -2 | 0 |
| previous_application | DAYS_FIRST_DUE | 40,645 | -2,892 | -2 | 0 |
| previous_application | DAYS_LAST_DUE_1ST_VERSION | 93,864 | -2,801 | 2,389 | 224,392 |
| previous_application | DAYS_LAST_DUE | 211,221 | -2,889 | -2 | 0 |
| previous_application | DAYS_TERMINATION | 225,913 | -2,874 | -2 | 0 |
| POS_CASH_balance | MONTHS_BALANCE | 0 | -96 | -1 | 0 |
| installments_payments | DAYS_INSTALMENT | 0 | -2,922 | -1 | 0 |
| installments_payments | DAYS_ENTRY_PAYMENT | 0 | -4,921 | -1 | 0 |
| credit_card_balance | MONTHS_BALANCE | 0 | -96 | -1 | 0 |

## 8. Data-quality checks on application_train

- Duplicate rows (ignoring SK_ID_CURR): 0
- DAYS_EMPLOYED = 365243 placeholder: 55,374 rows (18.01%). TARGET rate 5.40% vs 8.66% for other rows.
- Age range: 20.5 to 69.1 years
- AMT_CREDIT / AMT_INCOME_TOTAL: median 3.27, 99th percentile 13.03
- Rows with all three EXT_SOURCE scores missing: 172


Income type of DAYS_EMPLOYED placeholder rows:

| value | count | pct |
|---|---|---|
| Pensioner | 55,352 | 99.96 |
| Unemployed | 22 | 0.04 |

### Numeric profile

| column | n_missing | min | p01 | median | p99 | max |
|---|---|---|---|---|---|---|
| AMT_INCOME_TOTAL | 0 | 25,650 | 45,000 | 147,150 | 472,500 | 117,000,000 |
| AMT_CREDIT | 0 | 45,000 | 76,410 | 513,531 | 1,854,000 | 4,050,000 |
| AMT_ANNUITY | 12 | 1,616 | 6,183 | 24,903 | 70,006 | 258,026 |
| AMT_GOODS_PRICE | 278 | 40,500 | 67,500 | 450,000 | 1,800,000 | 4,050,000 |
| DAYS_BIRTH | 0 | -25,229 | -24,419 | -15,750 | -8,263 | -7,489 |
| DAYS_EMPLOYED | 0 | -17,912 | -10,895 | -1,213 | 365,243 | 365,243 |
| CNT_CHILDREN | 0 | 0 | 0 | 0 | 3 | 19 |
| CNT_FAM_MEMBERS | 2 | 1 | 1 | 2 | 5 | 20 |
| EXT_SOURCE_1 | 173,378 | 0.015 | 0.084 | 0.506 | 0.889 | 0.963 |
| EXT_SOURCE_2 | 660 | 0 | 0.029 | 0.566 | 0.783 | 0.855 |
| EXT_SOURCE_3 | 60,965 | 0.001 | 0.065 | 0.535 | 0.833 | 0.896 |

### Categorical columns

| column | n_unique | n_missing | top_values |
|---|---|---|---|
| NAME_CONTRACT_TYPE | 2 | 0 | Cash loans (90.5%); Revolving loans (9.5%) |
| CODE_GENDER | 3 | 0 | F (65.8%); M (34.2%); XNA (0.0%) |
| FLAG_OWN_CAR | 2 | 0 | N (66.0%); Y (34.0%) |
| FLAG_OWN_REALTY | 2 | 0 | Y (69.4%); N (30.6%) |
| NAME_TYPE_SUITE | 7 | 1,292 | Unaccompanied (81.2%); Family (13.1%); Spouse, partner (3.7%); Children (1.1%); Other_B (0.6%) |
| NAME_INCOME_TYPE | 8 | 0 | Working (51.6%); Commercial associate (23.3%); Pensioner (18.0%); State servant (7.1%); Unemployed (0.0%) |
| NAME_EDUCATION_TYPE | 5 | 0 | Secondary / secondary special (71.0%); Higher education (24.3%); Incomplete higher (3.3%); Lower secondary (1.2%); Academic degree (0.1%) |
| NAME_FAMILY_STATUS | 6 | 0 | Married (63.9%); Single / not married (14.8%); Civil marriage (9.7%); Separated (6.4%); Widow (5.2%) |
| NAME_HOUSING_TYPE | 6 | 0 | House / apartment (88.7%); With parents (4.8%); Municipal apartment (3.6%); Rented apartment (1.6%); Office apartment (0.9%) |
| OCCUPATION_TYPE | 18 | 96,391 | Laborers (26.1%); Sales staff (15.2%); Core staff (13.1%); Managers (10.1%); Drivers (8.8%) |
| WEEKDAY_APPR_PROCESS_START | 7 | 0 | TUESDAY (17.5%); WEDNESDAY (16.9%); MONDAY (16.5%); THURSDAY (16.5%); FRIDAY (16.4%) |
| ORGANIZATION_TYPE | 58 | 0 | Business Entity Type 3 (22.1%); XNA (18.0%); Self-employed (12.5%); Other (5.4%); Medicine (3.6%) |
| FONDKAPREMONT_MODE | 4 | 210,295 | reg oper account (75.9%); reg oper spec account (12.4%); not specified (5.8%); org spec account (5.8%) |
| HOUSETYPE_MODE | 3 | 154,297 | block of flats (98.2%); specific housing (1.0%); terraced house (0.8%) |
| WALLSMATERIAL_MODE | 7 | 156,341 | Panel (43.7%); Stone, brick (42.9%); Block (6.1%); Wooden (3.5%); Mixed (1.5%) |
| EMERGENCYSTATE_MODE | 2 | 145,755 | No (98.6%); Yes (1.4%) |

## 9. Candidate group attributes (counts and observed TARGET rate)

These are descriptive base rates, not model results or fairness findings.

### CODE_GENDER

| CODE_GENDER | count | target_rate |
|---|---|---|
| F | 202,448 | 0.07 |
| M | 105,059 | 0.101 |
| XNA | 4 | 0 |

### NAME_FAMILY_STATUS

| NAME_FAMILY_STATUS | count | target_rate |
|---|---|---|
| Married | 196,432 | 0.076 |
| Single / not married | 45,444 | 0.098 |
| Civil marriage | 29,775 | 0.099 |
| Separated | 19,770 | 0.082 |
| Widow | 16,088 | 0.058 |
| Unknown | 2 | 0 |

### NAME_EDUCATION_TYPE

| NAME_EDUCATION_TYPE | count | target_rate |
|---|---|---|
| Secondary / secondary special | 218,391 | 0.089 |
| Higher education | 74,863 | 0.054 |
| Incomplete higher | 10,277 | 0.085 |
| Lower secondary | 3,816 | 0.109 |
| Academic degree | 164 | 0.018 |

### NAME_INCOME_TYPE

| NAME_INCOME_TYPE | count | target_rate |
|---|---|---|
| Working | 158,774 | 0.096 |
| Commercial associate | 71,617 | 0.075 |
| Pensioner | 55,362 | 0.054 |
| State servant | 21,703 | 0.058 |
| Unemployed | 22 | 0.364 |
| Student | 18 | 0 |
| Businessman | 10 | 0 |
| Maternity leave | 5 | 0.4 |

### AGE_BAND_YEARS

| AGE_BAND_YEARS | count | target_rate |
|---|---|---|
| [0, 25) | 12,233 | 0.123 |
| [25, 35) | 72,429 | 0.107 |
| [35, 45) | 84,261 | 0.084 |
| [45, 55) | 70,190 | 0.07 |
| [55, 65) | 60,522 | 0.054 |
| [65, 120) | 7,876 | 0.037 |

## 10. Side-table categorical values

### bureau.CREDIT_ACTIVE

| value | count | pct |
|---|---|---|
| Closed | 1,079,273 | 62.879 |
| Active | 630,607 | 36.739 |
| Sold | 6,527 | 0.38 |
| Bad debt | 21 | 0.001 |

### bureau.CREDIT_CURRENCY

| value | count | pct |
|---|---|---|
| currency 1 | 1,715,020 | 99.918 |
| currency 2 | 1,224 | 0.071 |
| currency 3 | 174 | 0.01 |
| currency 4 | 10 | 0.001 |

### bureau.CREDIT_TYPE

| value | count | pct |
|---|---|---|
| Consumer credit | 1,251,615 | 72.92 |
| Credit card | 402,195 | 23.432 |
| Car loan | 27,690 | 1.613 |
| Mortgage | 18,391 | 1.071 |
| Microloan | 12,413 | 0.723 |
| Loan for business development | 1,975 | 0.115 |
| Another type of loan | 1,017 | 0.059 |
| Unknown type of loan | 555 | 0.032 |
| Loan for working capital replenishment | 469 | 0.027 |
| Cash loan (non-earmarked) | 56 | 0.003 |
| Real estate loan | 27 | 0.002 |
| Loan for the purchase of equipment | 19 | 0.001 |
| Loan for purchase of shares (margin lending) | 4 | 0 |
| Interbank credit | 1 | 0 |
| Mobile operator loan | 1 | 0 |

### bureau_balance.STATUS

| value | count | pct |
|---|---|---|
| C | 13,646,993 | 49.989 |
| 0 | 7,499,507 | 27.471 |
| X | 5,810,482 | 21.284 |
| 1 | 242,347 | 0.888 |
| 5 | 62,406 | 0.229 |
| 2 | 23,419 | 0.086 |
| 3 | 8,924 | 0.033 |
| 4 | 5,847 | 0.021 |

### previous_application.NAME_CONTRACT_STATUS

| value | count | pct |
|---|---|---|
| Approved | 1,036,781 | 62.075 |
| Canceled | 316,319 | 18.939 |
| Refused | 290,678 | 17.404 |
| Unused offer | 26,436 | 1.583 |

### previous_application.NAME_CONTRACT_TYPE

| value | count | pct |
|---|---|---|
| Cash loans | 747,553 | 44.758 |
| Consumer loans | 729,151 | 43.656 |
| Revolving loans | 193,164 | 11.565 |
| XNA | 346 | 0.021 |

### POS_CASH_balance.NAME_CONTRACT_STATUS

| value | count | pct |
|---|---|---|
| Active | 9,151,119 | 91.499 |
| Completed | 744,883 | 7.448 |
| Signed | 87,260 | 0.872 |
| Demand | 7,065 | 0.071 |
| Returned to the store | 5,461 | 0.055 |
| Approved | 4,917 | 0.049 |
| Amortized debt | 636 | 0.006 |
| Canceled | 15 | 0 |
| XNA | 2 | 0 |

### credit_card_balance.NAME_CONTRACT_STATUS

| value | count | pct |
|---|---|---|
| Active | 3,698,436 | 96.306 |
| Completed | 128,918 | 3.357 |
| Signed | 11,058 | 0.288 |
| Demand | 1,365 | 0.036 |
| Sent proposal | 513 | 0.013 |
| Refused | 17 | 0 |
| Approved | 5 | 0 |


_Audit runtime: 61 s._
