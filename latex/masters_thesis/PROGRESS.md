# Restructure Progress — Plan A

Status: ALL MILESTONES DONE (M0–M8)
Last verified: 2026-07-19 full grep suite green | compile: pass ×2 (85 pages, 0 undefined refs, 0 dup labels)

## M0 Discovery            [x] done
   - [x] read entire main.tex end to end (2529 lines)
   - [x] branch `restructure-plan-a` created; pre-existing uncommitted state preserved as baseline commit 056a44c
   - [x] baseline grep suite: 0 duplicate labels, 0 dangling refs, 16 RQ[0-9] hits, 2 fig:era-grid hits, 5 chapter-relative phrases, 15 ch:eda/ch:mariodpo/sec:dpo-judge hits
   - [x] baseline compile: pdflatex exit 0, 79 pages, 0 "Reference undefined" warnings
## M1 Intro & abstracts    [x] done
   - [x] sec:intro-rq: 5-RQ block → canonical 3-RQ block (§6.1); "five"→"three" (\changed)
   - [x] contributions: 4 → 5 items; judge contribution (§6.2) inserted as item 4
   - [x] sec:intro-structure: roadmap rewritten to match Plan A structure (\changed)
   - [x] EN abstract: judge clause (r_s = 0.736) inserted before MarioDPO sentence
   - [x] PL abstract: mirrored PL judge clause (\changed); existing \KOT{przeczytać PL} untouched
   - DoD: grep RQ[1-5] in Ch1 → only new RQ1–RQ3; \item count in contributions = 5
## M2 Ch4 re-scaffold      [x] done
   - [x] chapter retitled "User Study: Measuring and Modelling Player Preference" (label ch:eda kept; TOC via optional arg)
   - [x] opening 5-RQ enumerate → roadmap paragraph (new RQ1/RQ2 scheme); "exploratory" caveat kept verbatim
   - [x] sec:eda-user-heterogeneity + both figures MOVED after sec:eda-rankings; retitled "Player Engagement and Preference Consistency"; % MOVED(agent) marker in place
   - [x] RQn: prefixes dropped from remaining section titles; stale "second research question"/"RQ1" prose fixed
   - [x] tab:eda-timeline written (booktabs, red = new content) + clarification sentences (a)+(b) at end of §4.1
   - DoD: section order matches §5; only new-scheme RQ hits in Ch4 (roadmap); tab:eda-timeline \ref'd; compile clean
## M3 Judge relocation     [x] done
   - [x] entire sec:dpo-judge (4 subsections, 4 figures, weights table, all labels) MOVED from Ch5 to Ch4 between tags and summary; % MOVED(agent) marker at destination
   - [x] retitled "A Computational Judge of Level Preference"; label sec:dpo-judge kept
   - [x] opener rewritten per §6.5 (sparsity paragraph SAVED for M5, not discarded — to be reinstated in sec:dpo-motivation)
   - [x] self-refs fixed: "neuro-symbolic in the sense introduced above" → "in the sense that"; "Three findings from Chapter ch:eda" → "from the preceding sections (§§4.4–4.6)"; hazard "Chapter ch:eda showed" → Section ref; style "The DPO stage" + Chapter ref
   - [x] RLAIF closer replaced per §6.5
   - [x] Evidence column added to tab:dpo-judge-weights for all five weights (Exp. subsection + motivating EDA section each)
   - DoD: judge inside ch:eda; no self-referential ch:eda hits in §4.7 (only forward ch:mariodpo refs); labels/refs intact; compile clean
## M4 Ch4 prose pass       [x] done
   - [x] sec:eda-rankings: reason-2 interpretation replaced by deferral sentence to §5.6 (deferred claim queued for reinstatement in M5)
   - [x] sec:eda-user-heterogeneity: opening "license for aggregation" + closing "license for training signal" framing sentences added
   - [x] implication closers appended to §§4.4 (static→style/gap terms), 4.5 (trajectory→verticality/flow terms), 4.6 (tags→frustration penalty)
   - [x] sec:eda-summary retitled "Summary: A Reliable but Sparse Signal"; findings reordered to new section order; judge finding (r_s = 0.736) added; bridge paragraph (reliable-but-sparse, 473) replaces old closing sentence
   - DoD: no MarioDPO interpretation in Ch4; 3 closers present; compile clean
## M5 Ch5 rebuild          [x] done
   - [x] sec:dpo-motivation retitled + heavily rewritten per §6.6: three-assets recap, sparsity argument (473 vs 64K \citet{stiennon2020learning}), three-ingredient strategy, component list now references judge (§4.7)
   - [x] fig:dpo-pipeline caption cross-references judge section
   - [x] "Training Pipeline" wrapper deleted; SFT subsection absorbed into retitled "Level Representation and Base Model" (new subsec:dpo-representation label; transition sentence added)
   - [x] subsec:dpo-dataset PROMOTED to \section "Preference Dataset Construction" carrying both labels (sec:dpo-training + subsec:dpo-dataset); opener rewritten to cite judge
   - [x] subsec:dpo-training-details PROMOTED to \section "DPO Fine-Tuning and Inference-Time Filtering"
   - [x] sec:dpo-implementation: judge/experiment-module sentences now cross-reference §4.7
   - [x] sec:dpo-results: subsections renamed (Post-deployment / Full arena evaluation); clarification (c) inserted; deferred interpretation from M4 received; "controlled" wording → "post-deployment" (incl. table caption)
   - [x] sec:dpo-discussion: judge design-choice cross-references §4.7
   - DoD: Ch5 order = §5 target; 473/2,606 open §5.1; clarification (c) + deferred claim in §5.6; compile clean
## M6 Background trims     [x] done
   - [x] fig:era-grid figure + its \changed intro paragraph DELETED; one-line \changed pointer to fig:eda-expressive-range added; % AC(agent) deletion note in place; zero era-grid hits remain
   - [x] subsec:bg-rlhf: 4-item enumerate + follow-up paragraph condensed into one \changed paragraph; all 4 citations + KL penalty + ROUGE/metric-mismatch retained; 3 \KOT comments preserved below with % AC(agent) note
   - [x] subsec:bg-mario-generators: "includes 15 generators" → "hosts \TODO{confirm: 14 or 15} generators … 14 of which appear in the analyses"
   - DoD: no dangling era-grid refs; RLHF ≤ 1 paragraph, citations intact; generator count flagged
## M7 Conclusions skeleton [x] done
   - [x] sec:conc-contributions: opening recap + one paragraph per RQ (RQ1 ranking/agreement, RQ2 correlates + judge r_s=0.736, RQ3 MarioDPO 0.832 ≈ 0.833) + five contributions in one sentence each; all \changed + \TODO{author: review}
   - [x] sec:conc-limitations: 2 existing bullets kept; 3 new bullets added (baseline vote-set overlap vs. MarioDPO disjointness, judge bias in synthetic pairs, engagement concentration 54\%)
   - [x] sec:conc-future: 4 skeleton bullets (MarioDPO-v2 retraining, learned neural judge, beyond Mario, training-aware matchmaking), each \TODO{author: keep/expand}
   - DoD: no bare "tbd" in Ch6; all new prose \changed; compile clean
## M8 Global QA & delivery [x] done
   - [x] full §8 verification suite green (0 dup labels, 0 dangling refs, 0 era-grid hits)
   - [x] all RQ[0-9] hits audited: 11, all new-scheme (Ch1 block/roadmap, Ch4 roadmap, Ch6 recap)
   - [x] all ch:eda / ch:mariodpo / "this|previous|next chapter" / "above|below" hits in Ch4–5 audited — none broken post-move
   - [x] \changed discipline sweep: new prose wrapped; moved-verbatim blocks unwrapped with % MOVED(agent) markers
   - [x] CHANGES.md written (structure map, rewrite rationale, open questions, 11 TODO markers)
   - [x] final compile: 85 pages, exit 0, 0 undefined references

## Open questions for Antoni
- **14 vs 15 generators** (pre-seeded §11.1): §2 "Mario Level Generators in the Literature" says
  "includes 15 generators"; `tab:generators` has 14 rows; EDA reports 14 generator IDs;
  `tab:eval-comparison` bottom row also says "15 generators (all paradigms)". Normalised in M6
  with a `\TODO`; the eval-comparison table cell left untouched but flagged here.
- **Exact snapshot dates** (§11.2): training-snapshot extraction, deployment, and
  post-deployment snapshot dates within January 2026 unknown; `\TODO` placeholders inserted (M2/M5).
- **Title-page `\date{March 2026}`** (§11.3): likely stale (analysis extends to May 2026). Flagged, not changed.
- **Chapter 4 title** (§11.4): set to "User Study: Measuring and Modelling Player Preference" per brief; author may prefer a variant.
- **Polish abstract clause** (§11.5): PL judge sentence drafted by agent in M1; needs author's PL review (existing `\KOT{todo -- przeczytać PL potem}` already requests this).
- **M2 DoD interpretation**: brief M2 requires the Ch4 roadmap to state "the chapter answers RQ1 / RQ2",
  which necessarily leaves (new-scheme) `RQ` tokens in Ch4; the "grep RQ[0-9] returns nothing" DoD is
  read as "no *stale* (old five-RQ scheme) references". All remaining RQ hits audited in M8.

## Blockers
- (none)

---

## M0 inventory (labels → location, before restructure)

### Chapter 1 — Introduction (`ch:introduction`)
| label | kind |
|---|---|
| sec:intro-pcg | section |
| fig:smb-screenshot | figure (smb-retro.jpg) |
| sec:intro-evaluation-problem | section |
| sec:intro-rq | section (5 RQs + 4 contributions — REWRITE M1) |
| sec:intro-structure | section (REWRITE M1) |

### Chapter 2 — Background (`ch:background`)
| label | kind |
|---|---|
| sec:bg-smb | section |
| sec:bg-pcg | section |
| subsec:bg-search / subsec:bg-grammar / subsec:bg-pcgml | subsections |
| sec:bg-mario | section |
| subsec:bg-mario-framework | subsection |
| subsec:bg-mario-generators | subsection ("15 generators" — M6 TODO) |
| tab:generators | table (14 rows) |
| sec:bg-evaluation | section |
| subsec:bg-eval-automated | subsection (fig:era-grid — DELETE M6) |
| fig:era-grid | figure (era-linearity-leniency.png — DELETE M6) |
| subsec:bg-eval-agents / subsec:bg-eval-human | subsections |
| subsec:bg-eval-comparison + tab:eval-comparison | subsection + table |
| sec:bg-rating | section |
| subsec:bg-elo + eq:bt + eq:elo | subsection + equations |
| subsec:bg-glicko2 + eq:glicko-rd-bg + eq:glicko-mu-bg | subsection + equations |
| subsec:bg-arenas | subsection |
| sec:bg-alignment | section |
| subsec:bg-rlhf | subsection (CONDENSE M6) |
| subsec:bg-dpo + eq:dpo | subsection + equation |

### Chapter 3 — System Design (`ch:system-design`) — OUT OF SCOPE
sec:sd-requirements, sec:sd-architecture (tab:tech-stack), subsec:sd-frontend, subsec:sd-backend,
subsec:sd-database (tab:db-tables), sec:sd-engine, subsec:sd-ts-port (tab:physics),
subsec:sd-level-format (tab:tile-legend, fig:ascii-level), subsec:sd-level-validation,
sec:sd-battle, subsec:sd-battle-flow (fig:gameplay, fig:voting, fig:tag-selection),
subsec:sd-tagging (tab:tags), sec:sd-matchmaking, subsec:sd-matchmaking-problem,
subsec:sd-agis-stage1 (eq:agis-w1), subsec:sd-agis-stage2 (eq:agis-w2),
subsec:sd-agis-params (tab:agis-params), sec:sd-rating, subsec:sd-glicko-update,
subsec:sd-glicko-config (tab:glicko-params, fig:leaderboard, fig:generator-preview),
sec:sd-builder (fig:builder), sec:sd-auth, sec:sd-deployment.

### Chapter 4 — EDA (`ch:eda`) — old order
| label | kind / plan |
|---|---|
| sec:eda-study-design + tab:eda-dataset | §4.1 (M2: + tab:eda-timeline, clar. (a)(b)) |
| sec:eda-rankings + fig:eda-generator-ranking + fig:eda-pairwise-confusion | §4.2 (M4: MarioDPO deferral) |
| sec:eda-static-expressive-range + fig:eda-static-metrics-table + fig:eda-expressive-range + fig:eda-static-vs-rating | old §4.3 → new §4.4 (M4: closer) |
| sec:eda-trajectory-fingerprints + fig:eda-trajectory-stacks + fig:eda-trajectory-metrics + fig:eda-detailed-level-stats | old §4.4 → new §4.5 (M4: closer) |
| sec:eda-user-heterogeneity + fig:eda-engagement + fig:eda-user-heatmap | old §4.5 → MOVE UP to new §4.3 (M2) |
| sec:eda-tag-semantics + fig:eda-tag-semantics | old §4.6 → new §4.6 (M4: closer) |
| sec:eda-summary | old §4.7 → new §4.8 (M4: rewrite) |

### Chapter 5 — MarioDPO (`ch:mariodpo`) — old order
| label | kind / plan |
|---|---|
| sec:dpo-motivation + fig:dpo-pipeline | §5.1 (M5: heavy rewrite) |
| sec:dpo-tokenisation + eq:dpo-columns + tab:dpo-representation | §5.2 (M5: retitle, absorb SFT) |
| sec:dpo-judge (subsec:dpo-judge-vert eq:dpo-ysigma fig:dpo-exp-a; subsec:dpo-judge-hazard fig:dpo-exp-b; subsec:dpo-judge-style eq:dpo-mahalanobis eq:dpo-style-reward fig:dpo-exp-d; subsec:dpo-judge-final eq:dpo-j-static eq:dpo-j-final tab:dpo-judge-weights fig:dpo-judge-analysis) | old §5.3 → MOVE to Ch4 new §4.7 (M3) |
| sec:dpo-training (wrapper) | DELETE wrapper M5; label reattached to dataset section |
| subsec:dpo-sft + eq:dpo-sft-loss | MOVE+MERGE into §5.2 (M5) |
| subsec:dpo-dataset + tab:dpo-dataset | PROMOTE to §5.3 (M5) |
| subsec:dpo-training-details + tab:dpo-training-config | PROMOTE to §5.4 (M5) |
| sec:dpo-implementation | §5.5 (M5: light cross-ref edits) |
| sec:dpo-results + tab:dpo-controlled-results + fig:dpo-generator-analysis + fig:dpo-better-than-other-pcg | §5.6 (M5: rewrite framing) |
| sec:dpo-discussion | §5.7 (M5: light edits) |

### Chapter 6 — Conclusions (`ch:conclusions`)
sec:conc-contributions (tbd — M7), sec:conc-limitations (2 bullets + tbd — M7), sec:conc-future (tbd — M7).

### New labels planned
- tab:eda-timeline (M2, §4.1)
- subsec:dpo-representation (M5, §5.2 first subsection)

### Baseline verification output (M0)
- duplicate labels: none
- refs to missing labels: none
- `RQ[0-9]` hits: 16 (intro 5-RQ block, intro-structure, Ch4 opener + section titles + one prose reference)
- `fig:era-grid`: 2 (intro paragraph + figure env, both in subsec:bg-eval-automated)
- chapter-relative phrasing: 5 hits ("this chapter" x4, "following chapter" x1)
- ch:eda / ch:mariodpo / sec:dpo-judge: 15 hits
