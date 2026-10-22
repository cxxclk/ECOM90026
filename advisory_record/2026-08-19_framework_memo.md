*English translation of the author's notes for this meeting; original (Chinese) in meetings/framework_memo_20260819.md.*

# Paper framework (2026-08-19)

## 1. Overall sections

1. Abstract
2. Introduction
    - motivation
    - research question
    - lit review
    - 
3. Institutional Background
    - Housing Purchase Restrictions
    - 1.5-child policy / universal two-child policy
    - Lineage (clan) culture
4. Data
5. Empirical Strategy
6. Results
    - Result 1: zero effect + the margin is pinned to the second birth
    - Result 2: gender / culture / affordability + Q→Q
    - Mechanism: intra-couple decision power
7. Discussion
    - Welfare
    - robustness check
    - ML validation: inverted U + completeness
8. Conclusion
    - limitation/caveat
    - future direction
9. Appendix

## 2. Content sketched so far

### Introduction

1. - Driven by the fact that fertility -> productivity, and China's fertility rate / demographic dividend is disappearing
    - Second child policy (even multi later) released, but its effect does not appear obvious
    - There must be some reasons constraining it; cost is one of the most likely -> housing is the largest cost item

2. - intro to the Housing Purchase Restrictions and why it's a plausible exogenous shock (from original paper)
3. - The overall effect is 0, but this is not the answer: - married sample (original paper)
                            - logically, some families should not respond in the first place
                            - basic data pattern (including sex ratio, etc)
4. - 4-layer results representation

### Institutional Background

1. Housing Purchase Restrictions: - end of 2016: ~22 large cities reinstated purchase restrictions (eligibility / number-of-units limits)
          - demand spilled over to nearby unrestricted cities -> house prices pushed up exogenously (first stage: +9.7 log-pts, p=0.011)
          - why we study the spillover cities rather than the restricted cities themselves; treatment assigned by baseline place of residence
2. 1.5-child policy / universal two-child policy: - 1.5-child: rural families with a firstborn daughter were the only legal second-birth margin before 2016 -> why the effect concentrates in daughter families
                        - the universal two-child policy took effect in 2016.1, at the same time as the shock -> the confounder becomes the object of study (house prices determined who responded to the policy)
3. Lineage culture: - the norm of continuing the family line provides the motive for a second birth (trying for a son)
            - sets up the city norm index in §6 (a measurable, stable city characteristic)
            - [New] argument that the norm is a stable city characteristic: the 2014 vs 2022 city index correlation is 0.748; at the individual level attitudes are sticky (within-person Δ≈0), and the aggregate decline is generational replacement, not individual change

### Results

1. Result 2: [New] child-level education spending replicates Q→Q (DDD +0.37, p=0.043) -> resolves the old caveat that "education spending is only at the household level"
2. Mechanism: [New] where the redistribution sits: concentrated in the urban / homeowner / high-education-husband half of the sample (+0.060, p=0.007); the channel is still open
3. Welfare: [New] youth ideal suppressed by 8.2pp (WCB p=0.004, with the married group exactly zero as a comparison) -> plan to add a new subsection, "demand destruction in the next generation"
    - The rise in CES-D depression serves as clinical corroboration, concentrated in families that still want a second child
4. The other new results (SDID reconfirmation, roster orthogonality, migration/kinship all zero, demolition mirror, etc.) are all at the level of one sentence in robustness/footnote/discussion and do not change the structure
