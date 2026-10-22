*English translation of the author's notes for this meeting; original (Chinese) in meetings/advisor_meeting_ml_20260807.md.*

# Discussion outline for the ML econometrics advisor — 8/7

This is not a progress report; it is a conversation built around a methodological question. The other party leans towards ML econometrics, so the focus is not on the story itself but on "the full history of our attempt to find heterogeneity with ML" and a few technical questions I genuinely want advice on. He may not be familiar with the project details, so I will open by spending two minutes setting out the setup.

---

## 0. Two-minute version of the project setup (for the opening)

In 2016, China's housing purchase restrictions pushed housing demand into surrounding unrestricted cities (the spillover design of Liu & Zhang 2025). Using the CFPS panel, we ask whose fertility this house-price shock suppressed. The shock is a city-level binary treatment (≤250km of a regulated city); the outcome is the woman × year second-birth hazard (about 7–8% per year), with city+year FE, clustered by city, about 114 prefecture clusters and 40,000 woman-years.

The theory-driven answer is already locked: the average effect is ≈0, but the effect is fully concentrated in families where "the firstborn is a daughter × the city has traditional gender norms × affordability is low" (daughter arm −2.4pp, p=0.024; son-first families are a built-in placebo that is flat year by year). The mechanism is intra-couple bargaining: the shock redistributes fertility decision power from the husband to the wife, and the births suppressed are in families where the husband wants to try for a son and the wife does not.

Outside this locked main line, from late June to early August we ran a complete ML side line: **a dozen or so data-driven methods and four classification units (individuals / strata / rules / cities), systematically letting the data find "who responds" by itself**. The full record of this line is in `essay/ml_explorations_summary_20260805.md`, and it is what I want to discuss this time.

---

## 1. Core issue: given a dataset, subgroup discovery should in theory be fully automatable — why is it not in practice?

My starting point is this intuition: once the dataset is given, the variation is already there. In a DID setting, once the shock is given and treat/control are cleanly organised, the remaining work is essentially estimating τ(x). So using ML to find the most affected subgroup directly should in theory be fully feasible, and the theory-driven approach of "first think through who will respond, then test it" is just a manual version of the search.

We executed this intuition seriously, and the result is a very clean counterexample: **the three main formal HTE method families (causal forest, CDDF generic-ML, HD-LASSO) plus seven or eight home-made classification schemes never once independently rediscovered from the data the daughter × norm structure, which really exists and was confirmed ex post as very robust**. In the causal forest, firstborn gender ranks 5/7 or even 9/11 in importance; the BLP of the effect proxy (T-learner score) is consistently inverted; the free rule search ultimately lands back on the theory composite, but after accounting for search multiplicity p=0.159. Conversely, once theory tells you where to look, the same structure is confirmed under four estimator families.

Looking back, the statement "the variation is already there" runs into walls in three places. I would like to hear how he sees this from an ML econometrics perspective:

**1. The effective sample size is at the cluster level, not the row level.** The treatment is at the city level; after FE residualisation, the variation that identifies τ(x) is really only ~114 clusters × the post contrast. The 40,000 woman-year rows at the individual level are enough for predicting μ(x), but for effect estimation the effective N is two orders of magnitude lower. So "there is variation in the data" is correct, but **the variation for causal contrasts is much thinner than the variation for prediction**. Does this mean that under a city-level treatment, all individual-level τ(x) learning methods are bound to degenerate?

**2. When noise overwhelms signal, the effect proxy degenerates into a prognostic index.** Our most systematic failure: the four-cell T-learner effect proxy has a consistently inverted BLP across 20 random splits — the "most affected end" ranked by the score is actually fertility propensity itself (differential shrinkage: in high-propensity cells the variance of μ̂₁−μ̂₀ scales with propensity). With an effect of ~2pp and an outcome that is a rare binary event at 7%, the signal-to-noise ratio of τ is so low that the score contains only μ, not τ. **I want to ask: is this a known pathology of low-signal regions? Would the orthogonalised scores of the DR-learner / R-learner do better in this regime, or would they likewise die of variance?** Is there literature we should cite?

**3. The shape of the true heterogeneity is unfriendly to search algorithms.** In hindsight, the true structure has two features that make it a needle rather than a haystack: (i) it is an interaction of "motive variable × city-level cultural variable" — firstborn gender alone has no main-effect gradient, and the norm is an aggregate-level moderator; neither is a direction on the SES manifold, while the high-quality variables ML sees are all SES; (ii) along the propensity dimension the effect is an **inverted U** (the shock kills the "hesitant marginal birth" at Pc≈0.5, while the certain-to-give-birth G5 does not move at all) — a whole week of linear moderator scans came up empty, and only later did we understand that linear specifications are structurally blind to quadratic shapes (linear benchmark p=0.64 vs density specification WCB p=0.015). A forest can in theory learn nonlinearity, but under the power described in points 1 and 2 above it cannot learn it.

There is also a fourth point, more fundamental: **ML cannot propose a placebo structure**. Half of the credibility of the daughter arm comes from the theory-specified comparison group, son-first families, being zero year after year; the question "which subgroup should not respond" is not part of the problem definition of τ(x) estimation.

My current revised version of my original proposition is: "automatic discovery" holds under two conditions — the effect signal-to-noise ratio is sufficient (basically not met when the effective N is the number of clusters), and the true structure lies within the hypothesis space of the search (a needle of interaction + nonlinearity is not in a linear scan). I would like to hear whether he thinks this revision is right, and **whether there are HTE methods we have not run that are designed specifically for low signal-to-noise + cluster-level treatment**.

An important counterpoint, and also the most interesting detail of this line: ML did not fail entirely. **The only new finding that graduated (the inverted U) came precisely from a division of labour of "ML for prognosis, theory for the effect"** — the prognostic score (GBM learning baseline fertility propensity on pre-period data, a supervised learning problem with ample signal) is of very good quality (OOF calibration ECE 0.0035); stratifying GATES by it exposed the shape, and then a theory-driven structural test (effect ∝ Pc(1−Pc)) made it graduate. This is exactly the risk-based HTE route that Kent–Hayward advocate in the medical literature. So the empirical lesson may be: **hand to ML whatever supervised learning can learn (μ, propensity, prognosis); hypothesis generation for the effect step still has to come from theory** — I would like to hear whether he agrees with this division of labour, or thinks we simply did not choose the right methods.

---

## 1′. A live experiment: what happens to the same ML machinery under a different design (done on the evening of 8/6)

To test the revised proposition above, we turned a thought experiment into practice: working backwards from "discoverability", we designed a topic — **"the price of a son"**: in son-first families with a second child, the sex of the second child is naturally random (in our data the sex ratio of recent cohorts is 0.498–0.505 and pre-determined variables are all balanced, so a randomisation certificate can be obtained); treatment = the second child is also a son (= randomly taking on one more marriage-housing obligation); outcomes are housing/wealth (the richest coverage layer in CFPS). The treatment moves from the city level (G=114) to the individual level (G≈5,000 mothers); everything else is the same CFPS.

One afternoon's prototype gave three layers of results: (1) **the ATE layer is immediately cheap** — mothers in two-son families of the older cohort (sons already adult, marriages already held) have homeowner −2.7pp (p=0.008) and housing assets about −22% (p=0.027); woman-level OLS is enough, and the direction points to "the marriage-housing bill is settled after the son marries and is paid out of the parents' housing wealth"; (2) moving the CDDF machinery of `cddf_spine.py` over unchanged, **its behaviour normalises** — GATES is monotone and the CLAN profile is interpretable (most affected = older, less-educated mothers, i.e. the group that has already paid the bill), with none of the inverted-BLP pathology seen on HPR; (3) but **the HTE layer still does not pass the threshold** (BLP median p 0.36–0.55 vs 0.67 for HPR) — all directions are right, but the inference falls just short.

From this, the "discoverability map" is sharpened into a tiered proposition: **each step up the question hierarchy (ATE → HTE → mechanism decomposition) costs one more tier of power**. HPR: the ATE is barely affordable, HTE is unaffordable; individual-randomisation design: the ATE is cheap, HTE is at the threshold; for HTE also to be cheap requires G≈20,000+ (the administrative-data regime). The same code, three regimes, three fates — I would like his view on this tiered statement, and on how the MDE for HTE detection should be formally computed.

(An incidental data-methods finding: in the CFPS roster, the "second-child sex ratio" of older cohorts in the son-first arm is as high as 0.56 — biologically impossible; it is a recording bias from mothers in later life underreporting adult daughters, relevant to anyone who reconstructs historical fertility from the roster. Scripts `son_price_prototype.py` / `son_price_ml.py`, tables with the same names in `output/tables/`.)

## 2. Several specific technical questions (in order of priority)

**1. The boundaries of how to state the "completeness certificate".** After excluding the known moderator families, the CDDF BLP median p=0.67 (148-variable wide menu p=0.83). We want to write this in the paper as evidence that "there is no undiscovered ranked heterogeneity". But strictly speaking it is only "not detected" — **is there a standard way to quantify minimum detectable heterogeneity / power for the BLP, so that the sentence is upgraded from "we did not find it" to "heterogeneity above this size can be ruled out"?** This is a key sentence of the paper's discovery appendix, and I want the wording to be solid.

**2. Is permuting the entire search process the correct multiplicity correction?** In the effect subset scan we beam-searched 264 interpretable rules, then placed **the whole search pipeline** inside 200 city-level permutations and reported a scan-adjusted p (best rule nominal p=0.015 → 0.159 after correction). We put this procedure together ourselves; intuitively it resembles a rule-space version of maxT/Westfall-Young. I want to confirm: is this a standard operation in the selective inference literature? Or should we use conditional inference such as the winner's curse correction (Andrews–Kitagawa–McCloskey)? Would the two give very different answers?

**3. Is the "graduation" standard for the inverted U sufficient?** For a shape discovered post hoc (in GATES, G4 carries 66% of implied lost births and G5 does not move), we used two tests to promote it: (i) the U shape reproduces in the cohort cumulative DID, which is robust to dynamic selection (G4 −24.3pp p=0.035); (ii) we wrote it as a structural prediction, "effect ∝ marginal density Pc(1−Pc)", and tested it directly (daughter arm WCB p=0.015, son placebo clean, linear benchmark entirely empty). **By his standards, does this count as confirmatory, or should it still be labelled quantification/descriptive?** This determines whether the inverted U is a section in the main text or in the appendix.

**4. Can MC choosing rank-0 serve as a TWFE certificate?** Matrix completion on the city×year hazard panel chose rank 0 (pure two-way FE) by cross-validation; we interpret this as "there is no low-rank confounding structure beyond city+year FE", issuing a certificate for the TWFE specification. Does this interpretation hold? In addition, the SDID unit weights learned are close to uniform (the pre-period is too noisy to learn anything) — in this case, does the p=0.04 reported by SDID still count as independent evidence, or is it essentially DID in different clothes?

**5. How to report a lead that recurs but fails its confirmatory test.** A distributional lead, "the traditional-poor corner loses income", has surfaced in five mutually independent exploratory methods (the only FDR survivor, the only organising dimension under EB shrinkage, the direction of classifier loadings…), but the formal continuous-gradient confirmatory test gives WCB p=0.139. We decided to downgrade it to suggestive. **Does recurrence across methods count as an informal accumulation of evidence? Is there a formal framework for handling an evidence structure where "several correlated explorations all point the same way"** (similar to a meta-analytic approach), or should we go by the strictest confirmatory test?

**6. A small methodological exhibit, asking for his judgement.** HistGBM on this binary hazard is already nearly perfectly calibrated OOF (ECE 0.0035); wrapping it in CalibratedClassifierCV isotonic actually breaks it (ECE 0.0252, top decile squashed). We write this as an anti-pattern in a methods footnote. Another: full-sample LASSO followed by post-selection OLS selected subpopulation10 (a pure sampling-design variable) at p=0.001, and everything died after an honest split — a classroom-grade cautionary example. Are these two appropriate for the paper, or do they look like showing off?

---

## 3. Two projects not yet done; asking for advice before starting

Two remain on the list (`essay/ml_projects_brainstorm_20260802.md`):

**Couple decision-weight w(x) scan.** Our mechanism layer found that the shock redistributes fertility decision power from the husband to the wife (`T:post:diff` +0.027, p=0.055/0.025, diff = wife's ideal − husband's ideal). What we want to do is make this redistribution coefficient itself a function of covariates, w(x), and scan which types of couples have the largest shift in weight — because the channel (income, caregiving, grandparents, debt all null/weak) has not been pinned down, this may be the last lead. **Question: this is estimation of a "heterogeneous coefficient on an interaction", one interaction layer deeper than a standard CATE. At our power, is there a reliable approach** (interacted lasso? a causal forest with diff×post as the treatment?), or, given the lessons from Part 1, should we not hold out hope?

**Newborn-sex predictability index** (bonus project): use pre-period data to learn the "predictability of the second child's sex" as a proxy for the intensity of sex selection, and see whether the shock lowers it. Low priority; just asking in passing whether there is a smarter construction.

In addition, the birth-boundary SVM (the selection-sharpening corollary) was null in all three designs; we judged this a power ceiling, not a refutation (permutation noise 0.03–0.08 AUC, far larger than the compositional change that −2.9pp × the 20% contested stratum could produce). **Is there a more formal way to express this kind of judgement, "null because it cannot be measured"** (for example, computing the minimum detectable compositional shift and putting it in a footnote)?

---

## 4. If he asks "what exactly did you run" — scoreboard quick reference

| Method | Date | Verdict |
|---|---|---|
| Couple GMM clustering | 6/22 | Exploratory hint (single-income couples), not promoted |
| Causal forest (DiD-CATE, two rounds) | 6/22, 7/1 | Negative screen; the strongest known moderator does not rank near the top |
| DML / DR-DiD | 7/1 | Direction robust; 68% lack overlap, precision actually worse than FE-DiD |
| Semi-sup typology + sweep of 15 method combinations + trajectory clustering | 7/16 | No discrete latent classes (HDBSCAN all noise, ARI 0.02–0.30); the fertility effect is not organised under any typology |
| CDDF generic-ML (BLP/GATES/CLAN) | 7/21 | Completeness certificate: after excluding known families, BLP median p=0.67/0.83 |
| Prognostic GATES (Kent–Hayward) | 7/21 | ★ Source of the inverted U; later graduated via two structural tests |
| Effect subset scan (264 rules + full-search permutation) | 7/21 | Best rule = theory composite; scan-adjusted p=0.159 |
| Response-profile typology (EB shrinkage) | 7/21 | Fertility/education tau=0; only income organises |
| HD-LASSO (150 variables / 600-column dictionary) | 7/21 | Looks good in sample, all killed by honest city-split |
| City price-response classification | 7/21 | Dose reversed (stronger suppression in low-growth cities); motive geography |
| Separability DID (classifier two-sample test) | 8/2 | Distributional footprint real (p=0.005) but zero fertility loading; balanced panel null |
| Birth-boundary SVM | 8/2 | All three designs null; power ceiling |
| Roster underreporting classifier + DID | 8/3 | Underreporting 20–35% but orthogonal to treatment; caveat closed |
| SDID + matrix completion | 8/3 | Layer 2 confirmed by a fourth estimator family (perm p=0.040); MC rank-0 |
| Calibrated survival density | 8/3 | Cohort version stronger; isotonic anti-pattern |

One-sentence summary (consistent with the wording used with the main advisor): we ran all three main HTE method families plus ten classification schemes, and the data repeatedly gave the same answer — the heterogeneity is daughter × norm × affordability, and nothing else; along the way we picked up one new finding that passed structural tests (the inverted U), and a set of certificates that pin the main line down more firmly.
