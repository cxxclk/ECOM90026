# Advisor Meeting Memos — ECOM90026 (Housing Prices & Fertility)

Running log of advisor consultations for the research project *"The Effect of House Prices
on Fertility in China"* (Difference-in-Differences using the 2016 Housing Purchase
Restriction as the shock). Four meetings to date: **2026-03-26, 2026-04-21, 2026-06-11,
2026-06-25**. Most recent first.

Base paper throughout: Ziqian Liu (UW Foster) & Yu Zhang (PKU Guanghua), *"The Effect of
House Prices on Fertility: Evidence from House Purchase Restrictions"* (2025) — the HPR
capital-spillover identification design (`original paper.pdf`).

---

## Meeting 4 — 2026-06-25

**Context.** Fourth consultation. By now the city-level identification is in place and the
locked four-layer result exists; the discussion was about *which original angle to commit to*
and how to discipline that choice with the data.

**What I presented.**
- A **data-completeness reverse-engineering** approach to topic selection: rather than
  starting from a hypothesis and hoping the data supports it, first build a **content-aligned
  cross-wave variable catalog** (align by question *content*, not variable number, because CFPS
  renames variables and reuses numbers across waves), then choose a topic only among constructs
  that are (a) present in most/all waves and (b) carry real variation.
- The catalog's load-bearing finding — CFPS coverage is **three-tier**: fertility *outcomes*
  (births/parity/child sex) are available in all 7 waves; fertility *intentions/preferences*
  are **never** present in ≥6 waves (they rotate), which structurally starves any
  intention-driven design; housing/wealth and *subjective housing-pressure* variables are rich
  (6–7 waves).

**Advisor endorsed.** The data-completeness reverse-engineering logic — choosing the topic from
what is consistently and well-measured, rather than forcing a design onto thin data.

**Advisor's new suggestions (moderators / mechanisms to explore).**
1. **Traditional-culture / son-continuation intensity as a moderator.** Where the local
   "carry-on-the-family-line" (传宗接代) norm is strong, fertility should be *less* price-elastic
   — those families will pursue the (son) birth regardless of housing cost. Predicts the housing
   shock bites hardest where traditional son-continuation pressure is weaker, and is muted where
   it is strong. (Feasible proxy: the CFPS attitude item *"women should bear at least one boy to
   continue the family line"*, available 2014 & 2022, aggregated to a baseline locality-level
   norm index; complements the locked son-chasing-suppression result directly.)
2. **Household-chore / care responsibility allocation — especially the husband's willingness.**
   Frame fertility response through who bears the child-rearing burden and the husband's stated
   willingness to share it. Connects to the couple decision-weight line already prototyped
   (housing pressure shifts decision weight toward the lower-earning/caregiving spouse) and to
   the household-time-use data (2014 daily housework hours).

**Decisions / next steps.**
- Keep the theme **housing × fertility**; pick one direction over the next few days from the
  menu in `essay/topic_directions_20260625.md` (8 options: A1–A3 extend the locked main line,
  B4–B8 standalone).
- Two data-cleaning prerequisites flagged regardless of choice: (i) the "net family income"
  series silently switches deflation across waves (`fincome2` comparable-to-2010 in 2012–18 vs
  `fincome1` nominal in 2020/22); (ii) the only 7-wave "education" series `eec` lumps education
  + entertainment and must not be confused with pure-education `fp510`.
- Artifacts produced: `scripts/build_variable_catalog.py`,
  `output/catalog/variable_catalog.csv`, `output/catalog/catalog_digest.md`.

---

## Meeting 3 — 2026-06-11

**Context.** Third consultation, the first after the city-level identification problem was
solved. Detailed Chinese write-up with full model specifications and the agenda/questions is in
`meetings/advisor_meeting_20260611.md`; this is the English summary.

**What I presented.**
- Identification unblocked: found the `顺序码匹配.dta` crosswalk (county→prefecture, 162→127
  cities, 99.9% validated, ~85% sample coverage), and rebuilt the **HPR spillover treatment**
  per the base paper (`treat=1` for unregulated prefectures ≤250 km of a regulated HPR city; the
  22 regulated source cities dropped). The earlier fallback shock (棚改 monetised resettlement)
  was therefore downgraded to optional robustness.
- The **locked four-layer result**: the price shock does *not* suppress fertility on average;
  it crushes the **son-chasing second-birth margin of firstborn-daughter one-child families**,
  who redirect resources into the existing child's education (quantity→quality substitution).
- A **replication of Liu & Zhang's Table 3** (~60% of their magnitude; sign and robustness
  confirmed; their significance is conditional on linear pre-trend adjustment) and a **design
  upgrade** to an annual second-birth-hazard panel (daughter DID −2.4pp, p=0.024; flat annual
  pre-trends), which removed the thin-cell caveat.

**Topics discussed.** Possible original angles now that the index exists; how to differentiate
from Liu & Zhang (their margins are rural marriage-market entry and education costs of
non-owners — ours is within-marriage parity progression × son preference × Q→Q substitution).

**Decisions / next steps.** Migrate fertility layers to the annual-hazard design as the headline
(wave-level as robustness); inference strategy for thin post-period cells; framing the concurrent
2016 universal two-child policy as the object of study rather than a nuisance confounder.

---

## Meeting 2 — 2026-04-21

**Context.** Second consultation. The binding constraint at this point was data/identification,
not theory.

**What was discussed.**
- The **difficulty of obtaining a usable city index**: CFPS county codes are privacy-masked, so
  a clean city-level merge of the city-specific HPR policy (and city housing prices) was not yet
  possible. This was the main obstacle and the reason a province-level proxy / an alternative
  province-resolution shock (棚改 monetised resettlement) were being considered.

**Advisor's suggestion.** Approach the question from **couple behaviour** rather than only the
woman's fertility outcome — in particular **fertility *timing* / postponement**, which can be
inferred from the **birth-date information** in the family roster (e.g., spacing and the timing
of the next birth relative to the shock), so that a delayed-fertility (tempo) response is
identifiable even without a perfect city-price merge.

**Decisions / next steps.** Pursue the county→city crosswalk to unblock city-level matching;
keep the couple-behaviour and birth-timing angles as candidate original contributions. (Both
later materialised: the crosswalk was found before Meeting 3, and the couple decision-weight and
tempo/quantum analyses were prototyped subsequently.)

---

## Meeting 1 — 2026-03-26

**Context.** First consultation / project kickoff.

**What was discussed.**
- A brief introduction to the **project scope** and my own topic preference — the causal effect
  of **housing prices on fertility** in China, using the 2016 HPR as an exogenous shock in a DID
  framework.
- A focused discussion of the **base paper** (Liu & Zhang 2025): its capital-spillover
  identification (unregulated prefectures within 250 km of a regulated HPR city), its main
  channels (marriage market and child-education costs, concentrated among rural individuals with
  no urban homeownership), and what an original contribution would need to differ on.

**Decisions / next steps.** Adopt the HPR-spillover DID as the identification backbone; find an
original angle distinct from the base paper's rural/marriage/education framing; secure the CFPS
data and a feasible treatment assignment.

---

### Arc across the four meetings
Constraint → resolution: **(M1)** topic + base-paper framing → **(M2)** data/identification was
the blocker; advisor steered toward couple behaviour & birth-timing → **(M3)** identification
solved, headline result locked, replication done → **(M4)** topic selection disciplined by a
data-completeness catalog; advisor added traditional-culture-as-moderator and chore/husband-
willingness angles. Open item: commit to one direction from `topic_directions_20260625.md`.
