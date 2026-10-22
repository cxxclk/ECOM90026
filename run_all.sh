#!/usr/bin/env bash
# Runs the full pipeline in order. Run from the replication/ directory:
#   bash run_all.sh            (everything)
#   bash run_all.sh 3          (start at stage 3; earlier outputs must exist)
#   bash run_all.sh 1 2        (run stages 1 to 2 only)
set -euo pipefail
cd "$(dirname "$0")"
PY="${PY:-python3}"
START="${1:-1}"
STOP="${2:-5}"
mkdir -p output/tables output/figures/paper

if [ "$START" -le 1 ] && [ 1 -le "$STOP" ]; then
  echo "=== Stage 1. Build panels"
  echo "[1] data_process.py"; $PY data_process.py
  echo "[2] scripts/build_hpr_spillover.py"; $PY scripts/build_hpr_spillover.py
  echo "[3] data_process.py"; $PY data_process.py
  echo "[4] scripts/build_couples.py"; $PY scripts/build_couples.py
fi
if [ "$START" -le 2 ] && [ 2 -le "$STOP" ]; then
  echo "=== Stage 2. Cached scores used downstream"
  echo "[5] scripts/prognostic_gates.py"; $PY scripts/prognostic_gates.py
  echo "[6] scripts/cddf_spine.py"; $PY scripts/cddf_spine.py
  echo "[7] scripts/semisup_typology.py"; $PY scripts/semisup_typology.py
  echo "[8] scripts/pretrain_postpredict_did.py"; $PY scripts/pretrain_postpredict_did.py
  echo "[9] scripts/gbm_feature_importance.py"; $PY scripts/gbm_feature_importance.py
fi
if [ "$START" -le 3 ] && [ 3 -le "$STOP" ]; then
  echo "=== Stage 3. Main-text tables"
  echo "[10] scripts/table1_descriptives.py"; $PY scripts/table1_descriptives.py
  echo "[11] scripts/annual_parity_hazard.py"; $PY scripts/annual_parity_hazard.py
  echo "[12] scripts/annual_firstborn_gender.py"; $PY scripts/annual_firstborn_gender.py
  echo "[13] scripts/sdid_mc.py"; $PY scripts/sdid_mc.py
  echo "[14] scripts/headline_robustness.py"; $PY scripts/headline_robustness.py
  echo "[15] scripts/split_post_did.py"; $PY scripts/split_post_did.py > output/tables/split_post_did.txt
  echo "[16] scripts/gender_norm_index.py"; $PY scripts/gender_norm_index.py
  echo "[17] scripts/gender_norm_robustness.py"; $PY scripts/gender_norm_robustness.py
  echo "[18] scripts/tenure_wealth_cost.py"; $PY scripts/tenure_wealth_cost.py
  echo "[19] scripts/timing_policy_margin.py"; $PY scripts/timing_policy_margin.py
  echo "[20] scripts/eduexp_qq_clean.py"; $PY scripts/eduexp_qq_clean.py
  echo "[21] scripts/child_quality_did.py"; $PY scripts/child_quality_did.py
  echo "[22] scripts/child_health_investment.py"; $PY scripts/child_health_investment.py
  echo "[23] scripts/child_health_wcb.py"; $PY scripts/child_health_wcb.py
  echo "[24] scripts/sex_selection.py"; $PY scripts/sex_selection.py
  echo "[25] scripts/couple_unify_son_preference.py"; $PY scripts/couple_unify_son_preference.py
  echo "[26] scripts/couple_intentions_annual.py"; $PY scripts/couple_intentions_annual.py
  echo "[27] scripts/couple_avg_vs_diff.py"; $PY scripts/couple_avg_vs_diff.py
  echo "[28] scripts/gbm_hazard_grid.py"; $PY scripts/gbm_hazard_grid.py
  echo "[29] scripts/welfare_margin_triangulation.py"; $PY scripts/welfare_margin_triangulation.py
  echo "[30] scripts/saturated_controls_twfe.py"; $PY scripts/saturated_controls_twfe.py
  echo "[31] scripts/distance_ring_profile.py"; $PY scripts/distance_ring_profile.py
  echo "[32] scripts/donut_price_link.py"; $PY scripts/donut_price_link.py
  echo "[33] scripts/renter_mover_robustness.py"; $PY scripts/renter_mover_robustness.py
  echo "[34] scripts/headline_ever_married.py"; $PY scripts/headline_ever_married.py
  echo "[35] scripts/roster_underreport_did.py"; $PY scripts/roster_underreport_did.py
  echo "[36] scripts/migration_margin.py"; $PY scripts/migration_margin.py
fi
if [ "$START" -le 4 ] && [ 4 -le "$STOP" ]; then
  echo "=== Stage 4. Appendix tables and prose-quoted results"
  echo "[37] scripts/causal_forest_hazard.py"; $PY scripts/causal_forest_hazard.py
  echo "[38] scripts/causal_forest_scan_c.py"; $PY scripts/causal_forest_scan_c.py
  echo "[39] scripts/hd_lasso_hte.py"; $PY scripts/hd_lasso_hte.py
  echo "[40] scripts/effect_subset_scan.py"; $PY scripts/effect_subset_scan.py
  echo "[41] scripts/semisup_methods_compare.py"; $PY scripts/semisup_methods_compare.py
  echo "[42] scripts/trajectory_typology.py"; $PY scripts/trajectory_typology.py
  echo "[43] scripts/response_profile_typology.py"; $PY scripts/response_profile_typology.py
  echo "[44] scripts/separability_did.py"; $PY scripts/separability_did.py
  echo "[45] scripts/inverted_u_tests.py"; $PY scripts/inverted_u_tests.py
  echo "[46] scripts/calibrated_survival_density.py"; $PY scripts/calibrated_survival_density.py
  echo "[47] scripts/couple_mechanism_annual.py"; $PY scripts/couple_mechanism_annual.py
  echo "[48] scripts/couple_caregiving_channel.py"; $PY scripts/couple_caregiving_channel.py
  echo "[49] scripts/couple_wx_scan.py"; $PY scripts/couple_wx_scan.py
  echo "[50] scripts/norm_decision_weight.py"; $PY scripts/norm_decision_weight.py
  echo "[51] scripts/prognostic_no_motive.py"; $PY scripts/prognostic_no_motive.py
  echo "[52] scripts/replicate_liu_zhang.py"; $PY scripts/replicate_liu_zhang.py
  echo "[53] scripts/intentions_did.py"; $PY scripts/intentions_did.py
  echo "[54] scripts/youth_ideal_scarring.py"; $PY scripts/youth_ideal_scarring.py
  echo "[55] scripts/norm_erosion_did.py"; $PY scripts/norm_erosion_did.py
fi
if [ "$START" -le 5 ] && [ 5 -le "$STOP" ]; then
  echo "=== Stage 5. Figures"
  echo "[56] scripts/paper_figures.py"; $PY scripts/paper_figures.py
  echo "[57] scripts/fig_pretrain_calibration.py"; $PY scripts/fig_pretrain_calibration.py
fi
echo "Done. Tables in output/tables/, figures in output/figures/paper/."
