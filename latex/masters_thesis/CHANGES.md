# CHANGES — Plan A Restructure (branch `restructure-plan-a`)

All changes are LaTeX-only in `main.tex`. No figures regenerated, no analysis code touched,
no numbers altered. Every new/rewritten sentence is wrapped in `\changed{}` (or a
`\begingroup\color{red}` block for list environments); moved-verbatim blocks are unwrapped and
carry a `% MOVED(agent)` marker. All `\KOT{}` reviewer comments preserved.

## 1. Structure map (old → new)

| Old | New |
|---|---|
| 1.3 five RQs, four contributions | 1.3 three RQs (Measurement/Explanation/Generation), five contributions (judge added as #4) |
| 1.4 roadmap (judge in Ch5) | 1.4 roadmap (Ch4 = collection → analysis → judge; Ch5 = representation → dataset → DPO → results) |
| 2.4.1 ERA text + `fig:era-grid` results figure | 2.4.1 ERA text + one-line pointer to `fig:eda-expressive-range`; figure and intro paragraph deleted |
| 2.6.1 RLHF: 4-item enumerate + paragraph | 2.6.1 single condensed paragraph (all 4 citations, KL penalty, ROUGE point retained) |
| 3 System Design | untouched |
| 4 "User Study and Exploratory Data Analysis" | 4 "User Study: Measuring and Modelling Player Preference" |
| 4.1 Study Protocol and Dataset | 4.1 + NEW `tab:eda-timeline` + canonical clarifications (a) & (b) |
| 4.2 RQ1: Generator Preference Ranking | 4.2 Generator Preference Ranking |
| 4.5 RQ4: User Preference Heterogeneity | 4.3 Player Engagement and Preference Consistency (MOVED UP) |
| 4.3 RQ2: Static Expressive Range | 4.4 Static Expressive Range (+ implication closer) |
| 4.4 RQ3: Gameplay Trajectory Fingerprints | 4.5 Gameplay Trajectory Fingerprints (+ implication closer) |
| 4.6 RQ5: Tag Semantics and Failure Modes | 4.6 Tag Semantics and Failure Modes (+ implication closer) |
| 5.3 Judge Function Development | 4.7 A Computational Judge of Level Preference (MOVED; label `sec:dpo-judge` kept) |
| 4.7 Summary of Findings | 4.8 Summary: A Reliable but Sparse Signal |
| 5.1 Motivation and Approach | 5.1 Motivation: From Sparse Votes to a Training Signal |
| 5.2 Column-Major Tokenisation + 5.4.1 SFT | 5.2 Level Representation and Base Model (two subsections; new label `subsec:dpo-representation`) |
| 5.4 Training Pipeline (wrapper) | deleted; `\label{sec:dpo-training}` reattached to 5.3 |
| 5.4.2 Preference Dataset Construction | 5.3 Preference Dataset Construction (promoted; carries `sec:dpo-training` + `subsec:dpo-dataset`) |
| 5.4.3 DPO Training | 5.4 DPO Fine-Tuning and Inference-Time Filtering (promoted) |
| 5.5/5.6/5.7 Implementation/Results/Discussion | unchanged positions; edited (below) |
| 6 Conclusions ("tbd" ×3) | 6 full skeleton drafted (`\changed` + `\TODO`) |

## 2. Rewritten passages (one-line rationale each)

- **PL + EN abstracts**: judge half-sentence (r_s = 0.736) inserted before MarioDPO sentence — judge is now a standalone contribution.
- **§1.3 RQ block**: canonical 3-RQ block — matches the measure→explain→generate arc.
- **§1.3 contributions**: judge contribution inserted as item 4 — brief §6.2.
- **§1.4 roadmap**: mirrors the new chapter contents.
- **§2.3.2 generator count**: "includes 15" → "hosts \TODO{confirm: 14 or 15} …, 14 of which appear in the analyses" — resolves 14/15 inconsistency pending author confirmation.
- **§2.4.1 ERA**: deleted duplicate results figure; Background keeps method only.
- **§2.6.1 RLHF**: condensed to one paragraph — remove history-lesson bloat, keep the metric-mismatch argument that §5.1 depends on.
- **Ch4 opener**: 5-RQ enumerate → RQ1/RQ2 roadmap; exploratory caveat kept verbatim.
- **§4.1**: timeline table + clarifications (a) "trained once, deployed unchanged" and (b) "May export = larger sample of the same systems" — canonical ground truth §2.1.
- **§4.2**: "meaningful for two reasons" → sanity-check + deferral of MarioDPO interpretation to §5.6 (claim reinstated there, verified).
- **§4.3**: opening/closing framing — consistency as the license for aggregation (→§4.2) and training (→Ch5).
- **§§4.4–4.6 closers**: each maps its findings to specific judge terms (forward-refs to §4.7).
- **§4.7 opener**: correlates recap → operationalisation; notes experiments ran on the January snapshot (Table ref); sparsity argument relocated to §5.1.
- **§4.7 fixes**: "neuro-symbolic in the sense that" (definition now lives here); "Three findings from the preceding sections (§§4.4–4.6)"; hazard "Section 4.5 showed"; style "The DPO stage (Chapter 5)"; RLAIF closer → one-line bridge to Ch5.
- **`tab:dpo-judge-weights`**: Evidence column added — finding→design bridge for all five weights (experiment subsection + motivating EDA section).
- **§4.8**: retitled; findings reordered to section order; judge finding added; bridge paragraph (reliable per §4.3, sparse: 473 pairs, judge as the missing piece).
- **§5.1**: three-assets recap → sparsity argument (473 vs. 64K, \citet{stiennon2020learning}) → three-ingredient strategy → component list with judge *referenced* not introduced; pipeline-figure caption cross-refs §4.7.
- **§5.2**: SFT absorbed; one transition sentence added.
- **§5.3 opener**: names both supervision sources and cites the Chapter-4 judge.
- **§5.5**: experiment-module sentences cross-ref §4.7.
- **§5.6**: subsections renamed "Post-deployment evaluation (January 2026 snapshot)" / "Full arena evaluation (May 2026 export)"; clarification (c) train/eval-disjointness inserted; deferred interpretation received; "controlled" → "post-deployment" (incl. table caption).
- **§5.7**: judge design-choice cross-refs §4.7.
- **Ch6**: contributions per RQ, three new limitation bullets (baseline vote overlap, judge bias, engagement concentration), four future bullets.

## 3. Open questions for Antoni

1. **14 vs. 15 generators** — §2.3.2 said 15; `tab:generators` has 14 rows; EDA reports 14 IDs;
   note `tab:eval-comparison`'s bottom row also says "15 generators (all paradigms)" (left untouched).
   Confirm whether a 15th generator exists on the platform but not in the analyses, then resolve the `\TODO`.
2. **Exact January 2026 dates** — training-snapshot extraction, deployment, post-deployment snapshot: `\TODO` in `tab:eda-timeline`.
3. **Title page `\date{March 2026}`** — likely stale (data runs to May 2026). Flagged only.
4. **Chapter 4 title** — "User Study: Measuring and Modelling Player Preference" per brief; variants welcome.
5. **Polish abstract clause** — agent-drafted PL sentence needs a native read (existing `\KOT{todo -- przeczytać PL potem}` covers this).
6. **M2 DoD interpretation** — the Ch4 roadmap intentionally names (new-scheme) RQ1/RQ2, per the brief's task text; the "no RQ tokens in Ch4" grep DoD was read as "no *stale* five-RQ references". All 11 remaining RQ hits are new-scheme (Ch1 block/roadmap, Ch4 roadmap, Ch6 recap).

## 4. `\TODO{}` markers inserted (11 total)

| Location | Marker |
|---|---|
| §2.3.2 generator count | `\TODO{confirm: 14 or 15}` |
| `tab:eda-timeline`, post-deployment row | `\TODO{author: exact date}` |
| §6.1 opening recap | `\TODO{author: review and expand opening recap}` |
| §6.1 RQ1 / RQ2 / RQ3 / contributions paragraphs | `\TODO{author: review}` ×4 |
| §6.3 four future bullets | `\TODO{author: keep/expand}` ×4 |

## 5. Verification (final)

- duplicate labels: none · dangling refs: none · `fig:era-grid`: 0 hits
- chapter-relative phrasing: 2 hits, both correct post-move ("next chapter" at end of Ch4; "previous chapter" at start of Ch5)
- all `ch:eda`/`ch:mariodpo`/"this chapter"/"above" hits in Ch4–5 audited — none self-referential/broken
- pdflatex ×2: exit 0, 85 pages, 0 undefined references, 0 multiply-defined labels
- overfull hboxes: all pre-existing (long `\KOT` URLs), none introduced
