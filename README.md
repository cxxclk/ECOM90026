# Replication package

**Paper:** Housing Prices and the Contested Birth: Son Preference, Couple Decisions, and the Margin of Fertility in China, by Xiaxiao Chen (ECOM90027 Econometrics Research Report, University of Melbourne, 2026). Repository: <https://github.com/cxxclk/ECOM90026>.

This package has the code that builds every table and figure in the paper and every number the text quotes, together with the result files those runs produced. The CFPS microdata and the purchased county-to-prefecture file are not included (see "Data").

## Contents

| Path | What it is |
|---|---|
| `run_all.sh` | Runs the whole pipeline in order (5 stages, 57 steps) |
| `data_process.py` | Builds the analysis panels from the raw CFPS files |
| `scripts/` | All other code. Every file starts with a header giving its tier, the paper items it produces, its inputs and outputs, the local modules it imports and its step number in `run_all.sh` |
| `REPLICATION_MAP.md` | Paper item → script → output file |
| `requirements.txt` | Python package versions used |
| `data/Housing Price.xlsx` | Prefecture × month second-hand house prices collected by the author from Anjuke (see Data, item 3) |
| `data/hpr_cities.csv` | The 22 regulated cities and their restriction dates (hand-compiled from Liu and Zhang 2025, Tables A.1–A.2; Appendix Table C1) |
| `data/ChinaAdminDivisonSHP-master/` | Prefecture boundary shapefile (MIT licence, see below) |
| `advisory_record/` | English record of the communication with the advisor and the subject coordinator (meeting notes, emails, feedback on drafts; Appendix F of the paper) |
| `reference_output/` | The result tables (`tables/*.txt`), paper figures (`figures/`) and processing log (`data_summary.txt`) behind the submitted paper, so a re-run can be checked line by line |

### Script tiers

- **CORE (34 scripts):** the data pipeline and everything in the main-text tables and figures (Tables 1–8, Figures 1–8).
- **SUPPLEMENTARY (22 scripts):** the appendix tables (A1–E6, Figure E1) and five results quoted in the text without a table: the Liu and Zhang replication (Section 5.1), stated ideals and youth ideals (Section 6.1), and the stability of the norm index (Section 2.3).
- **HELPER (5 scripts):** modules other scripts import functions from (for example `paper_style.py`, the figure style). They are not run on their own.

Exploratory scripts that produced no reported result are left out.

## Data

Put the files below in `data/` with these names. The scripts expect exactly this layout.

**1. China Family Panel Studies (CFPS), waves 2010–2022.** Institute of Social Science Survey (ISSS), Peking University. Access is free, but users must register and apply on the CFPS data platform (https://www.isss.pku.edu.cn/cfps/download/login). Peking University owns the copyright, and users may not redistribute the data or post them on journal websites or third-party platforms (https://www.isss.pku.edu.cn/cfps/en/faq/PublishwithCFPSData/index.htm). For this reason the data are not in this package. Files used:

```
data/2010/ ecfps2010adult_201906.dta  ecfps2010child_201906.dta  ecfps2010famconf_nat072016.dta  ecfps2010famecon_201906.dta
data/2012/ ecfps2012adult_202505.dta  ecfps2012child_201906.dta  ecfps2012famconf_092015.dta   ecfps2012famecon_201906.dta
data/2014/ ecfps2014adult_201906.dta  ecfps2014child_201906.dta  ecfps2014famconf_170630.dta   ecfps2014famecon_201906.dta
data/2016/ ecfps2016adult_201906.dta  ecfps2016child_201906.dta  ecfps2016famconf_201804.dta   ecfps2016famecon_201807.dta
data/2018/ ecfps2018person_202012.dta ecfps2018childproxy_202012.dta ecfps2018famconf_202008.dta ecfps2018famecon_202101.dta
data/2020/ ecfps2020person_202306.dta ecfps2020child_202306.dta  ecfps2020famconf_202306.dta   ecfps2020famecon_202306.dta
data/2022/ ecfps2022person_202410.dta ecfps2022child_202410.dta  ecfps2022famconf_202410.dta   ecfps2022famecon_202410.dta
```

The suffix in each name is the CFPS release date. Other releases of the same wave may differ slightly.

**2. County-to-prefecture crosswalk** `data/CFPS_cleaned/顺序码匹配.dta`. Maps each CFPS `countyid` to a prefecture code and name (162 counties → 127 prefectures).
Source: purchased by the author from an online vendor. The vendor's terms do not allow redistribution, so the file is not included; its use for this subject project was approved by the subject coordinator (program director). Without it, prefecture-level geography cannot be attached to the CFPS records.

**3. Prefecture house prices** `data/Housing Price.xlsx`. Prefecture × month second-hand residential prices (CNY per m²), 626 prefectures, 2010–2025. Used only for Figure 1, Table 1 Panel B and the price columns of the distance checks; the exposure definition does not use prices.
Source: collected by the author by web-scraping the second-hand listing-price pages of Anjuke (anjuke.com), a Chinese real-estate platform. The file is included in this package (`data/Housing Price.xlsx`). The prices are the averages Anjuke displays publicly; Anjuke's terms of use apply, and the file is shared only so that this report can be replicated.

**4. Prefecture boundaries** `data/ChinaAdminDivisonSHP-master/3. City/city.shp`, from GaryBikini/ChinaAdminDivisonSHP on GitHub (MIT licence, included). It is used to compute each prefecture's distance to the nearest regulated city.

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash run_all.sh          # everything
bash run_all.sh 3        # restart at stage 3 (stages 1–2 must have run)
bash run_all.sh 1 2      # run stages 1–2 only
PY=/path/to/python bash run_all.sh   # interpreter (default python3)
```

Stages:

1. **Build panels.** `data_process.py` → `scripts/build_hpr_spillover.py` (exposure, needs `output/analysis_data.csv`) → `data_process.py` again (merges exposure) → `scripts/build_couples.py` (links spouses).
2. **Cached scores** used by later scripts: prediction scores (`prognostic_gates.py` → `output/prognostic_scores.csv`), `cddf_spine.py`, `semisup_typology.py`, the pre-train/post-predict predictions and the SHAP values.
3. **Main-text tables.**
4. **Appendix tables and quoted results.**
5. **Figures.** `paper_figures.py` reads numbers from the result tables and the cached score files. It does not re-estimate anything.

Each script writes a plain-text table to `output/tables/<name>.txt`. The exception is `split_post_did.py`, which prints to the screen; `run_all.sh` saves that output to `output/tables/split_post_did.txt`. The paper tables were typed from these files. Table 1 is written as Markdown (`output/tables/table1_descriptives.md`).

**Run time.** Several scripts run for more than ten minutes (the wild cluster bootstraps use 4,999 replications, and the machine-learning scripts use repeated cross-fitting). The machine-learning results depend on random splits. The scripts fix their seeds, but results may differ in the last digit across library versions or numbers of threads.

## Checking results

Compare `output/tables/` with `reference_output/tables/`. `REPLICATION_MAP.md` gives the block of each file that each paper item comes from. One map entry is marked UNVERIFIED: the causal-forest row of Table A1, whose original output file (`causal_forest_hazard.txt`) was not saved; `causal_forest_scan_c.txt` is the closest saved output.

## Changes from the working copies

The scripts are copies of the project code. The only edits are (i) the provenance header at the top of each file and (ii) absolute paths (`/home/...`) replaced by paths relative to the script's location, in `data_process.py`, `build_hpr_spillover.py`, `build_couples.py`, `eduexp_qq_clean.py` and `split_post_did.py`. All files compile under Python 3.12.

## AI use

The analysis code and this package were written with help from Claude (Anthropic) through Claude Code. The author designed the research, checked the results and is responsible for the content.
