# Replication map

This file shows which code produces each table and figure in the paper.

## How to run

Run from the `replication/` folder after installing `requirements.txt` (see `README.md`): `bash run_all.sh` runs the whole pipeline in five stages, and `bash run_all.sh 3` restarts at stage 3. To run one script, use `python scripts/<script>.py`; each writes a plain-text result table to `output/tables/<name>.txt`. `python scripts/paper_figures.py figN` draws one paper figure to `output/figures/paper/`; it reads numbers from the result tables and does not re-estimate anything, so the upstream scripts must be run first. The paper PDF itself is built from the draft sources, which are not part of this package.

The paper tables are typed into the draft Markdown from the `.txt` outputs. The only exception is Table 1: `table1_descriptives.py` writes `output/tables/table1_descriptives.md`, which is pasted into `essay/draft/03_data.md`.

## Pipeline base

| Step | Script | Raw inputs | Outputs |
|---|---|---|---|
| Panel construction | `data_process.py` | CFPS adult, famconf and famecon files for 2010–2022 (`data/2010/` … `data/2022/`); county-to-prefecture crosswalk `data/CFPS_cleaned/顺序码匹配.dta`; `data/Housing Price.xlsx` (prefecture × month second-hand prices) | `output/panel_individual.csv` (all waves), `output/analysis_data.csv` (married women 20–45), `output/housing_price_city.csv`, `output/housing_price_prov.csv`, `output/data_summary.txt` |
| Exposure (treatment) | `scripts/build_hpr_spillover.py` | Prefecture shapefile `data/ChinaAdminDivisonSHP-master/3. City/city.shp`; regulated-city list and dates `data/hpr_cities.csv` | `output/hpr_treatment_city.csv` (exposed = unregulated prefecture within 250 km of a regulated city; distance; regulated flag) |
| Annual birth panels | `build_hazard_panel()` in `scripts/annual_firstborn_gender.py` | `output/panel_individual.csv`, `output/hpr_treatment_city.csv`, latest famconf rosters in `data/20xx/` | Woman-year one-child sample, built in memory and imported by most fertility scripts |
| Child-level files | `scripts/child_quality_did.py`, `scripts/child_health_investment.py` | CFPS child files in `data/20xx/` | Firstborn-level expenditure panels, built in memory |

## Tables and figures

"UNVERIFIED" means the numbers could not be matched to a saved output file; the best candidate is given.

| Paper item | Panel / row | Script (function / section) | Output file |
|---|---|---|---|
| Figure 1 | First stage in second-hand prices | `paper_figures.py` `fig1()`; reads `analysis_data.csv` and `housing_price_city.csv` (from `data_process.py`) | `output/figures/paper/fig1_first_stage.png` |
| Figure 2 | Raw second-birth rates | `paper_figures.py` `fig2()`; calls `build_hazard_panel()` from `annual_firstborn_gender.py` | `output/figures/paper/fig2_raw_hazards.png` |
| Table 1 | Panels A, B, C | `scripts/table1_descriptives.py` (blocks "Panel A", "Panel B", "Panel C" in `main()`) | `output/tables/table1_descriptives.txt` and `.md` |
| Table 2 | A. Any birth | `scripts/annual_parity_hazard.py` (pooled any-birth block) | `annual_parity_hazard.txt` |
| Table 2 | B. By number of children | `scripts/annual_parity_hazard.py` (first, second, third+ blocks; group trend; event-study pre-trends) | `annual_parity_hazard.txt` |
| Table 2 | C. Alternative estimators | `scripts/sdid_mc.py` (pooled one-child block: DID, SDID, MC) | `sdid_mc.txt` |
| Figure 3 | Event study by first-child sex | `paper_figures.py` `fig3()`; parses the "Annual event study" block of `annual_firstborn_gender.txt` (from `annual_firstborn_gender.py`) | `output/figures/paper/fig3_event_study.png` |
| Table 3 | A. DID and triple difference, trend rows, sample rows | `scripts/annual_firstborn_gender.py` (daughter, son, DDD and group-trend blocks) | `annual_firstborn_gender.txt` |
| Table 3 | A. Wild cluster bootstrap row | `scripts/headline_robustness.py` (section [1]) | `headline_robustness.txt` |
| Table 3 | B. Post-period windows | `scripts/split_post_did.py` (prints to screen; output saved by redirecting) | `split_post_did.txt` |
| Table 4 | A. Norm thirds, continuous norm, urban/income horse race, son-first | `scripts/gender_norm_index.py` ("Tercile gradient" and continuous-triple blocks) | `gender_norm_index.txt` |
| Table 4 | A. High third wild bootstrap p, drop Henan | `scripts/gender_norm_robustness.py` (WCB and drop-one-province blocks) | `gender_norm_robustness.txt` |
| Table 4 | B. Affordability within owners | `scripts/tenure_wealth_cost.py` (owner interaction, house-value/income horse race) | `tenure_wealth_cost.txt` |
| Table 4 | C. 1.5-child eligibility | `scripts/timing_policy_margin.py` (section 3, "1.5-child policy margin") | `timing_policy_margin.txt` |
| Figure 4 | Norm thirds (both indices) | `paper_figures.py` `fig4()`; parses "Tercile gradient" in `gender_norm_index.txt` (from `gender_norm_index.py`) | `output/figures/paper/fig4_norm_terciles.png` |
| Table 5 | A. Household education expenditure | `scripts/eduexp_qq_clean.py` ("CLEAN" block for main rows and by-year rows; "CLEAN+" block for the pre-shock urban row) | `eduexp_qq_clean.txt` |
| Table 5 | B. Child-level education row | `scripts/child_quality_did.py` (child-level expenditure block) | `child_quality_did.txt` |
| Table 5 | B. Medical, insurance, hospitalisation rows | `scripts/child_health_investment.py` | `child_health_investment.txt` |
| Table 5 | B. Wild cluster bootstrap row | `scripts/child_health_wcb.py` | `child_health_wcb.txt` |
| Table 5 | C. Second birth by sex of second child | `scripts/sex_selection.py` | `sex_selection.txt` |
| Table 6 | A. Couple son-ideal gap | `scripts/couple_unify_son_preference.py` (Fact 1 block) | `couple_unify_son_preference.txt` |
| Table 6 | B. Whose ideal governs the next birth | `scripts/couple_intentions_annual.py` (block "y ~ T x (ideal_w, ideal_h)") | `couple_intentions_annual.txt` |
| Table 6 | C. Average/difference rows and pre-shock row | `scripts/couple_avg_vs_diff.py` (2014-cohort block; prefecture and couple clustering) | `couple_avg_vs_diff.txt` |
| Table 6 | C. Wife fewer / agrees / more rows | `scripts/couple_unify_son_preference.py` (Fact 2 block) | `couple_unify_son_preference.txt` |
| Table 7 | Boosted trees rows (default, best of 28) | `scripts/gbm_hazard_grid.py` (grid ranking) | `gbm_hazard_grid.txt` |
| Table 7 | Linear logit and ensemble rows, post-shock AUCs | `scripts/pretrain_postpredict_did.py` (learner table; treated-post fit line) | `pretrain_postpredict_did.txt` |
| Figure 5 | Effects by fifth of predicted likelihood | `paper_figures.py` `fig5()`; parses `prognostic_gates.txt` (from `prognostic_gates.py`) | `output/figures/paper/fig5_gates.png` |
| Figure 6 | SHAP beeswarm | `paper_figures.py` `fig6()`; reads `output/gbm_shap_values.csv` (cached by `gbm_feature_importance.py`) | `output/figures/paper/fig6_shap_beeswarm.png` |
| Figure 7 | Pre-train, post-predict | `paper_figures.py` `fig7()`; re-estimates event studies from `output/pretrain_postpredict_preds.csv` (from `pretrain_postpredict_did.py`) and `output/prognostic_scores.csv` (from `prognostic_gates.py`) | `output/figures/paper/fig7_pretrain_postpredict.png` |
| Figure 8 | Confidence in the future | `paper_figures.py` `fig8()`; parses `welfare_margin_triangulation.txt` (from `welfare_margin_triangulation.py`) | `output/figures/paper/fig8_welfare.png` |
| Table 8 | Main estimate; inference (WCB) | `scripts/headline_robustness.py` (sections [0], [1]) | `headline_robustness.txt` |
| Table 8 | Trends and timing: exposed-group trend | `scripts/annual_firstborn_gender.py` (group-trend block) | `annual_firstborn_gender.txt` |
| Table 8 | Trends and timing: false post 2012, 2013 | `scripts/headline_robustness.py` (section [2]) | `headline_robustness.txt` |
| Table 8 | Trends and timing: post from 2016, from 2017, 2016–17 dropped | `scripts/timing_policy_margin.py` (Post=2016, Post=2017, drop 2016–17 blocks) | `timing_policy_margin.txt` |
| Table 8 | Controls: age × duration cell FE | `scripts/saturated_controls_twfe.py` (row [2]) | `saturated_controls_twfe.txt` |
| Table 8 | Distance: 300 km, 200 km, continuous exposure | `scripts/headline_robustness.py` (sections [3] continuous dose, [4] alternative cutoffs) | `headline_robustness.txt` |
| Table 8 | Distance: ring rows | `scripts/distance_ring_profile.py` (ring × post, reference ≥ 450 km) | `distance_ring_profile.txt` |
| Table 8 | Distance: comparison group ≥ 450 km only | `scripts/donut_price_link.py` (block "control = >=450 only") | `donut_price_link.txt` |
| Table 8 | Sample: owners only; women ever in another prefecture dropped | `scripts/renter_mover_robustness.py` (blocks [B] tenure, mover) | `renter_mover_robustness.txt` |
| Table 8 | Sample: only women reported married in some wave | `scripts/headline_ever_married.py` | `headline_ever_married.txt` |
| Table 8 | Sample: only women first observed by 2014 | `scripts/timing_policy_margin.py` (baseline-wave block) | `timing_policy_margin.txt` |
| Table 8 | Measurement: roster miss rate | `scripts/roster_underreport_did.py` | `roster_underreport_did.txt` |
| Table 8 | Attrition: present next wave | `scripts/migration_margin.py` (T:post:dghtr row) | `migration_margin.txt` |
| Table 8 | Prediction-based counterfactual | `scripts/pretrain_postpredict_did.py` (ens_top3 residual DID) | `pretrain_postpredict_did.txt` |
| Table A1 | T-learner effect proxy | `scripts/cddf_spine.py` | `cddf_spine.txt` |
| Table A1 | Causal forest | UNVERIFIED: `scripts/causal_forest_hazard.py` (writes `causal_forest_hazard.txt`, not on disk) with `scripts/causal_forest_scan_c.py` | `causal_forest_scan_c.txt` |
| Table A1 | LASSO | `scripts/hd_lasso_hte.py` | `hd_lasso_hte.txt` |
| Table A1 | Beam search | `scripts/effect_subset_scan.py` | `effect_subset_scan.txt` |
| Table A1 | Three typologies | `scripts/semisup_typology.py`, `scripts/semisup_methods_compare.py`, `scripts/trajectory_typology.py` (verbal summary, no single number) | `semisup_typology.txt`, `semisup_methods_compare.txt`, `trajectory_typology.txt` |
| Table A1 | Response-profile shrinkage | `scripts/response_profile_typology.py` (verbal summary) | `response_profile_typology.txt` |
| Table A1 | Pre/post classifier | `scripts/separability_did.py` | `separability_did.txt` |
| Table B1 | All rows | `scripts/sdid_mc.py` (pooled, daughter and son blocks; summary grid) | `sdid_mc.txt` |
| Table B2 | Drop one province | `scripts/gender_norm_robustness.py` (drop-one-province block) | `gender_norm_robustness.txt` |
| Table B3 | Crude five-year conversion column | `scripts/inverted_u_tests.py` (Test 2 structural test and WCB; cohort structural K=4) | `inverted_u_tests.txt` |
| Table B3 | Survival-curve column | `scripts/calibrated_survival_density.py` (RAW measure, WCB block) | `calibrated_survival_density.txt` |
| Table B3 | Tuned-model column | `scripts/gbm_hazard_grid.py` (sections 4 and 5, TUNED) | `gbm_hazard_grid.txt` |
| Table B4 | Distance rings | `scripts/distance_ring_profile.py` (ring description table; blocks (a)–(c)) | `distance_ring_profile.txt` |
| Table C1 | Regulated cities and dates | Hand-compiled from Liu and Zhang (2025) Tables A.1–A.2; stored in `data/hpr_cities.csv` | `data/hpr_cities.csv` |
| Table C2 | Provincial fertility rules | Hand-compiled from Gu, Wang, Guo and Zhang (2007); the same classification is coded in `timing_policy_margin.py` section 3 | none |
| Table C3 | Questionnaire codes | Hand-compiled from the 2012/2014 questionnaire audit; the masking and wave-specific coding are implemented in `data_process.py` (`build_homeowner()`, 2012 masks) | none |
| Table D1 | First stage and income-share rows | `scripts/couple_mechanism_annual.py` | `couple_mechanism_annual.txt` |
| Table D1 | Caregiver rows, both moderators together | `scripts/couple_caregiving_channel.py` | `couple_caregiving_channel.txt` |
| Table D2 | All couples, composite rows, honest half-splits | `scripts/couple_wx_scan.py` (and `couple_avg_vs_diff.py` for the all-couples row) | `couple_wx_scan.txt` |
| Table D2 | Norm halves, norm thirds, high composite × high norm | `scripts/norm_decision_weight.py` | `norm_decision_weight.txt` |
| Table E1 | 28 model settings | `scripts/gbm_hazard_grid.py` (grid ranking); linear-logit row from `scripts/pretrain_postpredict_did.py` | `gbm_hazard_grid.txt`, `pretrain_postpredict_did.txt` |
| Table E2 | Ten learners | `scripts/pretrain_postpredict_did.py` (learner table) | `pretrain_postpredict_did.txt` |
| Figure E1 | Calibration of the score | `scripts/fig_pretrain_calibration.py`; reads `output/pretrain_postpredict_preds.csv` (from `pretrain_postpredict_did.py`) | `output/figures/pretrain_calibration.png` |
| Table E3 | K = 3, 4 and 5 columns | `scripts/inverted_u_tests.py` (Test 1, "K=4", "K=3", "K=5" cohort-DID-by-quintile blocks) | `inverted_u_tests.txt` |
| Table E4 | Ten fold assignments | `scripts/prognostic_gates.py` (seed-stability block) | `prognostic_gates.txt` |
| Table E5 | Top versus bottom fifth | `scripts/prognostic_gates.py` (CLAN block) | `prognostic_gates.txt` |
| Table E6 | Models without motive variables | `scripts/prognostic_no_motive.py` | `prognostic_no_motive.txt` |

All `.txt` files listed above are in `output/tables/`.
