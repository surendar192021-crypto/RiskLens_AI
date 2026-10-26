# Data dictionary (auto-generated)

Generated 2026-10-27T00:11:56 by `python -m scripts.build_dataset` (dataset version `homecredit_v1`). Do not edit by hand.

- **pct_missing_train** and **univariate_auc_train** are computed on the training split only. Validation, calibration and test rows are not used.
- Univariate AUC is the ROC-AUC of the single feature on its non-missing rows. 0.5 means no ranking signal on its own; below 0.5 means higher values go with *lower* TARGET rate. It is a screening statistic, not a model result.
- `*_COUNT` history features are 0 when the applicant has no records. Other history features stay missing (see the `HAS_*` flags).

| feature | source | type | pct_missing_train | univariate_auc_train | description |
|---|---|---|---|---|---|
| CNT_CHILDREN | application (raw) | numeric | 0.000 | 0.518 | Number of children the client has |
| AMT_INCOME_TOTAL | application (raw) | numeric | 0.000 | 0.480 | Income of the client |
| AMT_CREDIT | application (raw) | numeric | 0.000 | 0.481 | Credit amount of the loan |
| AMT_ANNUITY | application (raw) | numeric | 0.003 | 0.498 | Loan annuity |
| AMT_GOODS_PRICE | application (raw) | numeric | 0.094 | 0.466 | For consumer loans it is the price of the goods for which the loan is given |
| REGION_POPULATION_RELATIVE | application (raw) | numeric | 0.000 | 0.467 | Normalized population of region where client lives (higher number means the client lives in more populated region) |
| DAYS_REGISTRATION | application (raw) | numeric | 0.000 | 0.543 | How many days before the application did client change his registration |
| DAYS_ID_PUBLISH | application (raw) | numeric | 0.000 | 0.553 | How many days before the application did client change the identity document with which he applied for the loan |
| OWN_CAR_AGE | application (raw) | numeric | 65.986 | 0.559 | Age of client's car |
| FLAG_MOBIL | application (raw) | numeric | 0.000 |  | Did client provide mobile phone (1=YES, 0=NO) |
| FLAG_EMP_PHONE | application (raw) | numeric | 0.000 | 0.533 | Did client provide work phone (1=YES, 0=NO) |
| FLAG_WORK_PHONE | application (raw) | numeric | 0.000 | 0.521 | Did client provide home phone (1=YES, 0=NO) |
| FLAG_CONT_MOBILE | application (raw) | numeric | 0.000 | 0.500 | Was mobile phone reachable (1=YES, 0=NO) |
| FLAG_PHONE | application (raw) | numeric | 0.000 | 0.480 | Did client provide home phone (1=YES, 0=NO) |
| FLAG_EMAIL | application (raw) | numeric | 0.000 | 0.500 | Did client provide email (1=YES, 0=NO) |
| CNT_FAM_MEMBERS | application (raw) | numeric | 0.001 | 0.507 | How many family members does client have |
| REGION_RATING_CLIENT | application (raw) | numeric | 0.000 | 0.548 | Our rating of the region where client lives (1,2,3) |
| REGION_RATING_CLIENT_W_CITY | application (raw) | numeric | 0.000 | 0.550 | Our rating of the region where client lives with taking city into account (1,2,3) |
| HOUR_APPR_PROCESS_START | application (raw) | numeric | 0.000 | 0.474 | Approximately at what hour did the client apply for the loan |
| REG_REGION_NOT_LIVE_REGION | application (raw) | numeric | 0.000 | 0.501 | Flag if client's permanent address does not match contact address (1=different, 0=same, at region level) |
| REG_REGION_NOT_WORK_REGION | application (raw) | numeric | 0.000 | 0.502 | Flag if client's permanent address does not match work address (1=different, 0=same, at region level) |
| LIVE_REGION_NOT_WORK_REGION | application (raw) | numeric | 0.000 | 0.500 | Flag if client's contact address does not match work address (1=different, 0=same, at region level) |
| REG_CITY_NOT_LIVE_CITY | application (raw) | numeric | 0.000 | 0.522 | Flag if client's permanent address does not match contact address (1=different, 0=same, at city level) |
| REG_CITY_NOT_WORK_CITY | application (raw) | numeric | 0.000 | 0.540 | Flag if client's permanent address does not match work address (1=different, 0=same, at city level) |
| LIVE_CITY_NOT_WORK_CITY | application (raw) | numeric | 0.000 | 0.524 | Flag if client's contact address does not match work address (1=different, 0=same, at city level) |
| EXT_SOURCE_1 | application (raw) | numeric | 56.228 | 0.333 | Normalized score from external data source |
| EXT_SOURCE_2 | application (raw) | numeric | 0.214 | 0.347 | Normalized score from external data source |
| EXT_SOURCE_3 | application (raw) | numeric | 19.866 | 0.320 | Normalized score from external data source |
| APARTMENTS_AVG | application (raw) | numeric | 50.758 | 0.460 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor |
| BASEMENTAREA_AVG | application (raw) | numeric | 58.515 | 0.473 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor |
| YEARS_BEGINEXPLUATATION_AVG | application (raw) | numeric | 48.795 | 0.467 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor |
| YEARS_BUILD_AVG | application (raw) | numeric | 66.484 | 0.473 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor |
| COMMONAREA_AVG | application (raw) | numeric | 69.880 | 0.477 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor |
| ELEVATORS_AVG | application (raw) | numeric | 53.321 | 0.457 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor |
| ENTRANCES_AVG | application (raw) | numeric | 50.375 | 0.473 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor |
| FLOORSMAX_AVG | application (raw) | numeric | 49.781 | 0.447 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor |
| FLOORSMIN_AVG | application (raw) | numeric | 67.860 | 0.465 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor |
| LANDAREA_AVG | application (raw) | numeric | 59.391 | 0.485 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor |
| LIVINGAPARTMENTS_AVG | application (raw) | numeric | 68.387 | 0.465 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor |
| LIVINGAREA_AVG | application (raw) | numeric | 50.212 | 0.451 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor |
| NONLIVINGAPARTMENTS_AVG | application (raw) | numeric | 69.434 | 0.492 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor |
| NONLIVINGAREA_AVG | application (raw) | numeric | 55.175 | 0.484 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor |
| TOTALAREA_MODE | application (raw) | numeric | 48.300 | 0.453 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor |
| OBS_30_CNT_SOCIAL_CIRCLE | application (raw) | numeric | 0.335 | 0.511 | How many observation of client's social surroundings with observable 30 DPD (days past due) default |
| DEF_30_CNT_SOCIAL_CIRCLE | application (raw) | numeric | 0.335 | 0.518 | How many observation of client's social surroundings defaulted on 30 DPD (days past due)  |
| OBS_60_CNT_SOCIAL_CIRCLE | application (raw) | numeric | 0.335 | 0.511 | How many observation of client's social surroundings with observable 60 DPD (days past due) default |
| DEF_60_CNT_SOCIAL_CIRCLE | application (raw) | numeric | 0.335 | 0.516 | How many observation of client's social surroundings defaulted on 60 (days past due) DPD |
| DAYS_LAST_PHONE_CHANGE | application (raw) | numeric | 0.000 | 0.554 | How many days before application did client change phone |
| FLAG_DOCUMENT_2 | application (raw) | numeric | 0.000 | 0.500 | Did client provide document 2 |
| FLAG_DOCUMENT_3 | application (raw) | numeric | 0.000 | 0.538 | Did client provide document 3 |
| FLAG_DOCUMENT_4 | application (raw) | numeric | 0.000 | 0.500 | Did client provide document 4 |
| FLAG_DOCUMENT_5 | application (raw) | numeric | 0.000 | 0.500 | Did client provide document 5 |
| FLAG_DOCUMENT_6 | application (raw) | numeric | 0.000 | 0.485 | Did client provide document 6 |
| FLAG_DOCUMENT_7 | application (raw) | numeric | 0.000 | 0.500 | Did client provide document 7 |
| FLAG_DOCUMENT_8 | application (raw) | numeric | 0.000 | 0.494 | Did client provide document 8 |
| FLAG_DOCUMENT_9 | application (raw) | numeric | 0.000 | 0.499 | Did client provide document 9 |
| FLAG_DOCUMENT_10 | application (raw) | numeric | 0.000 | 0.500 | Did client provide document 10 |
| FLAG_DOCUMENT_11 | application (raw) | numeric | 0.000 | 0.500 | Did client provide document 11 |
| FLAG_DOCUMENT_12 | application (raw) | numeric | 0.000 |  | Did client provide document 12 |
| FLAG_DOCUMENT_13 | application (raw) | numeric | 0.000 | 0.499 | Did client provide document 13 |
| FLAG_DOCUMENT_14 | application (raw) | numeric | 0.000 | 0.499 | Did client provide document 14 |
| FLAG_DOCUMENT_15 | application (raw) | numeric | 0.000 | 0.499 | Did client provide document 15 |
| FLAG_DOCUMENT_16 | application (raw) | numeric | 0.000 | 0.498 | Did client provide document 16 |
| FLAG_DOCUMENT_17 | application (raw) | numeric | 0.000 | 0.500 | Did client provide document 17 |
| FLAG_DOCUMENT_18 | application (raw) | numeric | 0.000 | 0.499 | Did client provide document 18 |
| FLAG_DOCUMENT_19 | application (raw) | numeric | 0.000 | 0.500 | Did client provide document 19 |
| FLAG_DOCUMENT_20 | application (raw) | numeric | 0.000 | 0.500 | Did client provide document 20 |
| FLAG_DOCUMENT_21 | application (raw) | numeric | 0.000 | 0.500 | Did client provide document 21 |
| AMT_REQ_CREDIT_BUREAU_HOUR | application (raw) | numeric | 13.554 | 0.500 | Number of enquiries to Credit Bureau about the client one hour before application |
| AMT_REQ_CREDIT_BUREAU_DAY | application (raw) | numeric | 13.554 | 0.501 | Number of enquiries to Credit Bureau about the client one day before application (excluding one hour before application) |
| AMT_REQ_CREDIT_BUREAU_WEEK | application (raw) | numeric | 13.554 | 0.500 | Number of enquiries to Credit Bureau about the client one week before application (excluding one day before application) |
| AMT_REQ_CREDIT_BUREAU_MON | application (raw) | numeric | 13.554 | 0.497 | Number of enquiries to Credit Bureau about the client one month before application (excluding one week before application) |
| AMT_REQ_CREDIT_BUREAU_QRT | application (raw) | numeric | 13.554 | 0.494 | Number of enquiries to Credit Bureau about the client 3 month before application (excluding one month before application) |
| AMT_REQ_CREDIT_BUREAU_YEAR | application (raw) | numeric | 13.554 | 0.523 | Number of enquiries to Credit Bureau about the client one day year (excluding last 3 months before application) |
| DAYS_EMPLOYED_PLACEHOLDER | application (engineered) | numeric | 0.000 | 0.467 | 1 if DAYS_EMPLOYED held the 365243 placeholder (almost all pensioners) |
| CREDIT_INCOME_RATIO | application (engineered) | numeric | 0.000 | 0.498 | AMT_CREDIT / AMT_INCOME_TOTAL |
| ANNUITY_INCOME_RATIO | application (engineered) | numeric | 0.003 | 0.519 | AMT_ANNUITY / AMT_INCOME_TOTAL (repayment burden) |
| CREDIT_ANNUITY_RATIO | application (engineered) | numeric | 0.003 | 0.469 | AMT_CREDIT / AMT_ANNUITY (rough proxy for loan length) |
| GOODS_CREDIT_RATIO | application (engineered) | numeric | 0.094 | 0.429 | AMT_GOODS_PRICE / AMT_CREDIT |
| AGE_YEARS | application (engineered) | numeric | 0.000 | 0.416 | Age at application in years (-DAYS_BIRTH / 365.25) |
| EMPLOYED_YEARS | application (engineered) | numeric | 17.922 | 0.416 | Current employment length in years (missing for placeholder rows) |
| EMPLOYED_TO_AGE_RATIO | application (engineered) | numeric | 17.922 | 0.429 | EMPLOYED_YEARS / AGE_YEARS |
| INCOME_PER_FAMILY_MEMBER | application (engineered) | numeric | 0.001 | 0.481 | AMT_INCOME_TOTAL / CNT_FAM_MEMBERS |
| EXT_SOURCE_MEAN | application (engineered) | numeric | 0.060 | 0.285 | Mean of available EXT_SOURCE_1-3 |
| EXT_SOURCE_MIN | application (engineered) | numeric | 0.060 | 0.311 | Minimum of available EXT_SOURCE_1-3 |
| EXT_SOURCE_N_MISSING | application (engineered) | numeric | 0.000 | 0.527 | How many of EXT_SOURCE_1-3 are missing (0-3) |
| DOCUMENT_COUNT | application (engineered) | numeric | 0.000 | 0.511 | Number of FLAG_DOCUMENT_* flags set |
| BUREAU_COUNT | bureau + bureau_balance | numeric | 0.000 | 0.476 | Number of credit bureau credits |
| BUREAU_ACTIVE_COUNT | bureau + bureau_balance | numeric | 0.000 | 0.528 | Number of active bureau credits |
| BUREAU_CLOSED_COUNT | bureau + bureau_balance | numeric | 0.000 | 0.445 | Number of closed bureau credits |
| BUREAU_BAD_DEBT_OR_SOLD_COUNT | bureau + bureau_balance | numeric | 0.000 | 0.503 | Number of bureau credits marked 'Bad debt' or 'Sold' |
| BUREAU_DAYS_CREDIT_MAX | bureau + bureau_balance | numeric | 14.361 | 0.582 | Days before application of the most recent bureau credit |
| BUREAU_DAYS_CREDIT_MIN | bureau + bureau_balance | numeric | 14.361 | 0.577 | Days before application of the oldest bureau credit |
| BUREAU_ACTIVE_DEBT_SUM | bureau + bureau_balance | numeric | 32.852 | 0.535 | Total current debt on active bureau credits (currency 1 only) |
| BUREAU_ACTIVE_CREDIT_SUM | bureau + bureau_balance | numeric | 29.428 | 0.501 | Total credit amount of active bureau credits (currency 1 only) |
| BUREAU_OVERDUE_SUM | bureau + bureau_balance | numeric | 14.374 | 0.507 | Total currently overdue amount on bureau credits (currency 1 only) |
| BUREAU_MAX_OVERDUE_MAX | bureau + bureau_balance | numeric | 40.295 | 0.545 | Largest overdue amount ever recorded on a bureau credit (currency 1 only) |
| BUREAU_PROLONG_SUM | bureau + bureau_balance | numeric | 14.361 | 0.503 | Total number of times bureau credits were prolonged |
| BUREAU_ACTIVE_DEBT_RATIO | bureau + bureau_balance | numeric | 33.838 | 0.587 | BUREAU_ACTIVE_DEBT_SUM / BUREAU_ACTIVE_CREDIT_SUM |
| BUREAU_BB_WORST_STATUS_12M | bureau + bureau_balance | numeric | 76.089 | 0.540 | Worst monthly DPD bucket (0-5) on any bureau credit, MONTHS_BALANCE >= -12 |
| BUREAU_BB_DPD_CREDIT_COUNT | bureau + bureau_balance | numeric | 0.000 | 0.512 | Number of bureau credits with any month in DPD bucket 1-5 |
| HAS_BUREAU | indicator | numeric | 0.000 | 0.479 | 1 if the applicant has any credit bureau record |
| PREV_COUNT | previous_application | numeric | 0.000 | 0.521 | Number of previous Home Credit applications |
| PREV_APPROVED_COUNT | previous_application | numeric | 0.000 | 0.478 | Number of previous applications approved |
| PREV_REFUSED_COUNT | previous_application | numeric | 0.000 | 0.556 | Number of previous applications refused |
| PREV_REFUSED_LAST_YEAR_COUNT | previous_application | numeric | 0.000 | 0.540 | Previous applications refused within 365 days before this application |
| PREV_AMT_CREDIT_MEAN | previous_application | numeric | 5.383 | 0.478 | Mean final credit amount of previous applications |
| PREV_AMT_ANNUITY_MEAN | previous_application | numeric | 5.513 | 0.463 | Mean annuity of previous applications |
| PREV_CREDIT_TO_APPLICATION_MEAN | previous_application | numeric | 5.725 | 0.573 | Mean of (granted AMT_CREDIT / requested AMT_APPLICATION) |
| PREV_DAYS_DECISION_MAX | previous_application | numeric | 5.383 | 0.528 | Days before application of the most recent previous decision |
| PREV_CNT_PAYMENT_MEAN | previous_application | numeric | 5.512 | 0.523 | Mean planned number of payments on previous credits |
| PREV_APPROVAL_RATE | previous_application | numeric | 5.383 | 0.440 | PREV_APPROVED_COUNT / PREV_COUNT |
| PREV_REFUSAL_RATE | previous_application | numeric | 5.383 | 0.560 | PREV_REFUSED_COUNT / PREV_COUNT |
| HAS_PREV | indicator | numeric | 0.000 | 0.508 | 1 if the applicant has any previous Home Credit application |
| INST_COUNT | installments_payments | numeric | 0.000 | 0.483 | Number of scheduled previous instalments (partial payments combined) |
| INST_UNPAID_COUNT | installments_payments | numeric | 0.000 | 0.502 | Scheduled instalments with no recorded payment |
| INST_LATE_SHARE | installments_payments | numeric | 5.184 | 0.566 | Share of paid instalments paid after the due day |
| INST_LATE30_SHARE | installments_payments | numeric | 5.184 | 0.512 | Share of paid instalments paid more than 30 days late |
| INST_DAYS_LATE_MEAN | installments_payments | numeric | 5.184 | 0.561 | Mean days late per paid instalment (early = 0) |
| INST_DAYS_LATE_MAX | installments_payments | numeric | 5.184 | 0.551 | Maximum days late on any instalment |
| INST_UNDERPAID_SHARE | installments_payments | numeric | 5.182 | 0.506 | Share of instalments where total paid < amount due (by more than 0.01) |
| INST_SHORTFALL_SUM | installments_payments | numeric | 5.182 | 0.519 | Total unpaid amount across instalments |
| INST_LATE_SHARE_LAST_YEAR | installments_payments | numeric | 29.375 | 0.572 | INST_LATE_SHARE for instalments due within 365 days before application |
| HAS_INST | indicator | numeric | 0.000 | 0.507 | 1 if the applicant has any previous instalment record |
| POS_MONTHS_COUNT | POS_CASH_balance | numeric | 0.000 | 0.469 | Number of monthly POS/cash loan records |
| POS_LOAN_COUNT | POS_CASH_balance | numeric | 0.000 | 0.466 | Number of distinct previous POS/cash loans |
| POS_DPD_MAX | POS_CASH_balance | numeric | 5.898 | 0.524 | Maximum days past due on POS/cash loans |
| POS_DPD_DEF_MAX | POS_CASH_balance | numeric | 5.898 | 0.525 | Maximum days past due (with tolerance) on POS/cash loans |
| POS_DPD_MONTH_SHARE | POS_CASH_balance | numeric | 5.898 | 0.525 | Share of POS/cash months with DPD > 0 |
| POS_DPD_MAX_LAST_YEAR | POS_CASH_balance | numeric | 32.851 | 0.520 | Maximum POS/cash DPD with MONTHS_BALANCE >= -12 |
| HAS_POS | indicator | numeric | 0.000 | 0.506 | 1 if the applicant has any previous POS/cash loan monthly record |
| CC_MONTHS_COUNT | credit_card_balance | numeric | 0.000 | 0.508 | Number of monthly credit card records |
| CC_UTILISATION_MEAN | credit_card_balance | numeric | 71.968 | 0.625 | Mean of balance / credit limit (months with limit > 0) |
| CC_UTILISATION_MAX | credit_card_balance | numeric | 71.968 | 0.606 | Maximum of balance / credit limit |
| CC_UTILISATION_MEAN_LAST_YEAR | credit_card_balance | numeric | 76.772 | 0.635 | Mean utilisation with MONTHS_BALANCE >= -12 |
| CC_DPD_MAX | credit_card_balance | numeric | 71.685 | 0.499 | Maximum days past due on credit cards |
| CC_DPD_MONTH_SHARE | credit_card_balance | numeric | 71.685 | 0.499 | Share of credit card months with DPD > 0 |
| CC_DRAWINGS_MEAN | credit_card_balance | numeric | 71.685 | 0.602 | Mean monthly drawn amount on credit cards |
| HAS_CC | indicator | numeric | 0.000 | 0.513 | 1 if the applicant has any previous credit card monthly record |
| NAME_CONTRACT_TYPE | application (raw) | categorical | 0.000 |  | Identification if loan is cash or revolving |
| CODE_GENDER | application (raw) | categorical | 0.001 |  | Gender of the client |
| FLAG_OWN_CAR | application (raw) | categorical | 0.000 |  | Flag if the client owns a car |
| FLAG_OWN_REALTY | application (raw) | categorical | 0.000 |  | Flag if client owns a house or flat |
| NAME_TYPE_SUITE | application (raw) | categorical | 0.424 |  | Who was accompanying client when he was applying for the loan |
| NAME_INCOME_TYPE | application (raw) | categorical | 0.000 |  | Clients income type (businessman, working, maternity leave,) |
| NAME_EDUCATION_TYPE | application (raw) | categorical | 0.000 |  | Level of highest education the client achieved |
| NAME_FAMILY_STATUS | application (raw) | categorical | 0.001 |  | Family status of the client |
| NAME_HOUSING_TYPE | application (raw) | categorical | 0.000 |  | What is the housing situation of the client (renting, living with parents, ...) |
| OCCUPATION_TYPE | application (raw) | categorical | 31.252 |  | What kind of occupation does the client have |
| WEEKDAY_APPR_PROCESS_START | application (raw) | categorical | 0.000 |  | On which day of the week did the client apply for the loan |
| ORGANIZATION_TYPE | application (raw) | categorical | 0.000 |  | Type of organization where client works |
| FONDKAPREMONT_MODE | application (raw) | categorical | 68.367 |  | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor |
| HOUSETYPE_MODE | application (raw) | categorical | 50.187 |  | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor |
| WALLSMATERIAL_MODE | application (raw) | categorical | 50.858 |  | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor |
| EMERGENCYSTATE_MODE | application (raw) | categorical | 47.431 |  | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor |
