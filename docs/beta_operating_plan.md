# Beta coverage, publishing and budget

Decision date: 2026-11-01. Companion to the [product specification](product_spec.md) and [implementation plan](transformation_plan.md). Values below are initial product limits and budget allocations, not provider quotes or demonstrated capacity.

## Starting audience and coverage

Begin with five invited users and 25 planned companies. Selection is a working product decision based on official reporting sources and useful comparison scenarios, not an investment recommendation or evidence of user demand. Pilot interviews may justify substitutions before launch. The expanded roster spans technology, banking, consumer businesses, autos, pharma, telecom, energy, utilities and industrials. It is still a selected universe, not comprehensive coverage of Indian equities.

Official reporting entry points were identified through web research on 2026-11-01. This verifies source discovery only; no two-period document set has yet been extracted or licensed for redistribution. All 25 start at `planned`, not `covered`. As of 2026-11-02, TCS and Infosys have unpublished drafts; none has passed the full activation gate yet.

| Company | NSE symbol | Initial research focus | Official source entry point |
| --- | --- | --- | --- |
| Tata Consultancy Services | TCS | Revenue, operating margin, order commentary | [Investor relations](https://www.tcs.com/investor-relations) |
| Infosys | INFY | Revenue guidance, margins, reported versus constant-currency growth | [Financials and filings](https://www.infosys.com/investors/reports-filings.html) |
| HCL Technologies | HCLTECH | Revenue guidance, services/software mix, margins | [Quarter results](https://www.hcltech.com/investor-relations/quarter-results) |
| Wipro | WIPRO | IT-services growth, guidance, operating margin | [Quarterly results](https://www.wipro.com/investors/quarterly-results/) |
| Hindustan Unilever | HINDUNILVR | Revenue/volume commentary and margins | [Results and presentations](https://www.hul.co.in/investors/results-and-presentations/) |
| ITC | ITC | Segment performance and margins; comparability after structural changes | [Quarterly results](https://itcportal.com/investors/quarterly-results.html) |
| Asian Paints | ASIANPAINT | Sales, volume commentary, margins and input costs | [Investors](https://www.asianpaints.com/more/investors.html) |
| Titan Company | TITAN | Segment growth, margins and expansion commentary | [Quarterly results](https://www.titancompany.in/investors/investor-relations/quarterly-results) |
| Maruti Suzuki India | MARUTI | Volumes, revenue, profitability and capacity commentary | [Investor entry point](https://www.marutisuzuki.com/corporate/investors); [example official results release](https://www.marutisuzuki.com/corporate/media/press-releases/2026/october/maruti-suzuki-india-limited-financial-results-q2-and-h1-fy-2026-27) |
| Larsen & Toubro | LT | Orders, revenue, execution and segment margins | [Quarterly results archive](https://investors.larsentoubro.com/Quarterly-Results-Archives.aspx) |
| Reliance Industries | RELIANCE | Segment revenue, EBITDA, capex and debt; avoid mixing segment/group measures | [Financial reporting](https://www.ril.com/investors/financial-reporting?page=1) |
| Bharti Airtel | BHARTIARTL | ARPU, subscriber trends, regional EBITDA and capex | [Quarterly results](https://www.airtel.in/about-bharti/equity/results) |
| Sun Pharmaceutical Industries | SUNPHARMA | Regional sales, specialty business and profitability | [Investor information](https://sunpharma.com/investors/) |
| Cipla | CIPLA | Regional revenue, product commentary and margins | [Quarterly results](https://www.cipla.com/investors/quarterly-results) |
| HDFC Bank | HDFCBANK | NII, NIM, loan/deposit growth and asset quality | [Financial results](https://www.hdfc.bank.in/about-us/investor-relations/financial-results) |
| ICICI Bank | ICICIBANK | NII, NIM, deposits, credit costs and asset quality | [Investor relations](https://www.icici.bank.in/about-us/invest-relations) |
| State Bank of India | SBIN | NII, loan/deposit growth, provisions and asset quality | [Quarterly results](https://sbi.bank.in/en/web/investor-relations/reports) |
| Axis Bank | AXISBANK | NIM, deposits, advances, provisions and asset quality | [Shareholder information and results](https://www.axis.bank.in/shareholders-corner/other-information) |
| Kotak Mahindra Bank | KOTAKBANK | NIM, deposits, advances and asset quality; separate exceptional gains | [Quarterly results](https://www.kotak.bank.in/en/investor-relations/financial-results/quarterly-results.html) |
| NTPC | NTPC | Generation, capacity commissioning, receivables and capex | [Financial results](https://ntpc.co.in/investors/financial-performance/financial-results) |
| Power Grid Corporation of India | POWERGRID | Capitalization, capex, regulated earnings and receivables | [Quarterly results](https://www.powergrid.in/en/annual-quarterly-results) |
| UltraTech Cement | ULTRACEMCO | Volumes, realization, unit profitability and capacity | [Financials](https://www.ultratechcement.com/corporate/investors-/financials-) |
| Bajaj Auto | BAJAJ-AUTO | Volumes, exports, margins and subsidiary impacts | [Financial and operational performance](https://www.bajajauto.com/investors/financial-and-operational-performance) |
| Hero MotoCorp | HEROMOTOCO | Volumes, margins, product mix and EV investment | [Investor relations](https://www.heromotocorp.com/en-in/company/investors/investor-relations.html) |
| Nestlé India | NESTLEIND | Sales, volume commentary and margins; verify fiscal-period comparability | [Financial results](https://www.nestle.in/investors/stockandfinancials/financialresults) |

Build the first full ingestion example with Infosys, then Wipro, before scaling the same reviewed workflow to the rest. Metrics above are topics to inspect, not guaranteed available fields or default user beliefs. Symbols and entity mappings must be checked during catalog seeding.

## Staged activation under the unchanged budget

Scope lock (2026-11-01): launch with 25 covered companies. Keep the selected roster as the implementation baseline; any pre-launch substitution must have a documented source-quality or pilot-demand reason and preserve the 25-company target.

After launch, consider expansion toward 50 in batches of five, prioritized by requests for unsupported companies. Expand only after two consecutive source-check/publication cycles meet the stated cadence, evidence checks remain reliable, and measured processing costs plus projected maintenance fit the remaining budget. A requested company is not automatically covered. Fifty is an optional future target, not a commitment or permission to increase spending.

The public-beta target is 25 covered companies. Internal/pilot activation happens in three batches; this sequencing does not reduce the release target:

1. **Batch A — original 10:** TCS, Infosys, HCLTech, Wipro, HUL, ITC, Asian Paints, Titan, Maruti Suzuki and L&T.
2. **Batch B — 10 additions:** Reliance Industries, Bharti Airtel, Sun Pharma, Cipla, NTPC, Power Grid, UltraTech Cement, Bajaj Auto, Hero MotoCorp and Nestlé India.
3. **Batch C — 5 banks:** HDFC Bank, ICICI Bank, SBI, Axis Bank and Kotak Mahindra Bank, after bank-specific comparisons pass fixtures.

A minimum two-period baseline now means 50 company-period sets rather than 20. Results plus supporting releases/presentations can mean more than 50 individual documents. At least three checked comparisons per company means at least 75 checked comparisons overall. These are workload counts, not API-cost estimates.

Keep the five-user pilot, inference quotas and INR 2,000 total ceiling unchanged. Extract public evidence once and reuse it. Benchmark actual processing cost and operator time after the first two companies, then after each batch; reserve funds for ongoing questions and updates before activating the next batch. If projected spend exceeds the available allocation, pause ingestion, use manually prepared evidence where feasible, or extend the rollout. Never silently increase spend or mark an unprocessed company covered. The existing 20-entry per-user watchlist limit remains independent of the 25-company research universe.

Check all 25 covered source pages on each twice-weekly source-check cycle (50 company checks per week). Only process changed material. Test whether this manual cadence is sustainable during the pilot; disclose delays rather than claim unattended monitoring.

Bank fixtures must distinguish net interest income from revenue, NIM from operating margin, gross from net NPAs, period-end from average balances, and standalone-bank from consolidated-group results. Do not apply industrial-company debt/margin heuristics to banks. Other sector-specific metrics require explicit definitions and comparable periods too.

## Activation gate for every company (Step 5)

1. Select two latest available comparable quarterly periods at ingestion time. Record exact period end dates; do not assume all companies have released the same quarter.
2. Obtain official financial results and a supporting investor release/presentation; include an official transcript only when available. A scheduled-call notice is not results evidence.
3. Check retrieval conditions and permitted storage/excerpt use. An official public page is not permission to republish its full documents. Link to originals by default; resolve reuse conditions before hosting excerpts or originals. Keep redistributed sample fixtures synthetic until cleared.
4. Extract and inspect numerical units, accounting basis, tables, page/section references and any restatement or restructuring effects.
5. Hand-check at least three substantive comparisons per company, including limitations and a missing/incomparable-evidence example where applicable. Fetch prior-year material if needed for a YoY comparison; never substitute QoQ silently.
6. Publish a reviewed baseline and record the operator/reviewer, sources, version and check dates. The same sole operator may review during the beta; do not claim independent audit.
7. Set coverage to `covered` only after these checks. If blocked, leave `planned` and explain the gap. 25 activated companies is the release target; do not claim completion with fewer.

States: `planned`, `covered`, `paused`, `watchlist_only`. The small initial catalog supports manual watchlist-only entries outside the 25; broad exchange search is not promised. Pause coverage when sources or operator capacity prevent maintenance, retaining historical research and disclosing the pause.

## Operator publication routine

- Manually check all covered sources twice weekly, targeted for Monday and Thursday. These are operating targets, not scheduled automations already created.
- Aim to publish a checked update within three business days of detecting a relevant release. If unable, display a pending/delayed status rather than imply freshness.
- Ingest results, investor presentations/releases and available official transcripts. Broad news, social feeds and intraday events are outside coverage.
- Draft → numeric/citation check → operator approval → versioned publication → relevant inbox records.
- Require comparable evidence before labeling a change. Preserve a missing-data flag rather than filling gaps with model guesses.
- Deduplicate by company/document fingerprint and comparison version. Retried publication must not duplicate inbox entries.
- Corrections create a new version with a visible explanation; previous user decisions remain attached to the evidence originally reviewed.
- Store last attempted source check, last successful source check, latest source publication, reporting period and update publication separately.
- Mark a source check overdue after seven days without a successful check. Display failed checks immediately. Document age alone is not evidence that no newer report exists.
- Show “No new published updates” rather than “Nothing important happened.” If a check failed, say so alongside that state.

The operator supplies the maintenance effort; this is hands-off for the user within the stated coverage, not autonomous infrastructure. Before inviting more users, measure operator time for two publishing cycles and confirm the cadence is sustainable.

## Initial configurable limits

| Resource | Beta limit | Exhaustion behavior |
| --- | --- | --- |
| Invited accounts | 5 initially | Further sign-ups join an invitation queue |
| Watchlist entries | 20 per user | Explain limit; allow removal |
| Holdings CSV | 1 MB, 100 rows | Reject before parsing; preview accepted format |
| Confirmed assumptions | 5 per company per user | Notes remain available; explain structured-assumption limit |
| Follow-up questions | 5 per user per day; 25 globally per day | Return explicit quota message before any model call |
| Other private inference (suggestions/matching) | 25 calls globally per day | Persist pending items for retry; do not relabel them complete |
| Model context/output | Up to 12,000 input and 1,500 output tokens per request | Retrieve relevant excerpts; abstain if context is insufficient |
| Operator ingestion | Up to 20 MB and 100 pages per document | Split into reviewed sections or mark unsupported; no silent truncation |
| Automated retry | At most 1 retry per model request | Both attempts consume budget; preserve failure state |

These are maximum allowances, not promised capacity. Budget controls override every quota. Reserve expected input plus maximum output cost before dispatch, reconcile actual usage afterward, and persist reservations transactionally so concurrent requests cannot overspend the same balance. Use current provider rates and a conservative tax/FX allowance before enabling a model. No unmetered fallback provider.

Pause new model requests when the application ledger cannot be read or when the cost estimate is unavailable. Reset daily counts at 00:00 Asia/Kolkata. Keep user reads, notes, decisions, history and exports working during inference outages. Cache shared public extraction by document/version; isolate private answers and matching by workspace/revision. Never send holdings quantities or purchase costs when not needed to answer a question.

## Spending ceiling

Assumption: INR 2,000 total initial launch funds, excluding the existing laptop, connectivity and development tools. This is not a monthly commitment or a perpetual hosting guarantee.

| Allocation | Ceiling | Rule |
| --- | ---: | --- |
| AI experiments and beta processing | INR 800 | Warn at 600; stop dispatch when projected reserved spend reaches 800 |
| Deployment/storage contingency | INR 400 | Unspent until an actual requirement and eligible service are verified |
| Reserve | INR 800 | No automatic transfer or recurring subscription |
| Domain | INR 0 | Use provider subdomain |
| Initial hosting/database/auth | INR 0 target | Verify plan eligibility, quotas and authentication delivery requirements during Step 2 |
| Total | INR 2,000 | Stop expansion when the ceiling is reached |

Record actual debits, including tax and provider credit purchases; credits are not a reason to ignore cash spend. Application cost estimates cannot account perfectly for all provider-side billing: reconcile against provider usage and use provider hard limits where available. Budget alerts alone are not a hard cap. No paid resource is being purchased in Step 1.

Do not silently enroll in usage-based overages. Disable model-dependent features if the available budget runs out; tell users the research maintenance status. Free hosting/provider quotas are constraints, not a guarantee of indefinite capacity. Review measured costs and user behavior at Step 9 before expanding beyond the invited beta.

## Pilot and public-release evidence

Record onboarding completion, time to first useful update, accepted/rejected findings, citation corrections, repeat visits, model cost per processed document, cost per answered question and operator maintenance time. Do not equate five sign-ups or successful model responses with useful research.

Public-beta gate: all 25 companies activated; seeded no-key sample; working sign-in/import/review/history/export/deletion; verified private data isolation; source-check failures and AI budget exhaustion tested; restore procedure tested; license chosen; hosting eligibility checked. Email digests and paid services are not required to pass this gate.


## Implemented maintenance slice — 2026-11-02

The first two pipeline examples are TCS and Infosys. They are stored as unapproved Supabase drafts. The current reader maps `covered` to a **Published** label and uses **Withdrawn** for a withdrawn latest version; a failed/overdue check is shown separately. Broader paused/watchlist-only workflows are not implemented. The locked 25-company roster is the only supported holdings/research universe in the current app.

The batch checks explicitly configured documents, not an entire exchange or automatic new-release feed. Initial direct retrieval returned 403; operator text snapshots were used with original retrieval dates and explicit provenance. Every company still requires original-document inspection and publication before being considered covered. Run two real maintenance cycles and measure operator time before expanding the two examples through the remaining batches. See [batch operations](../scripts/research/README.md).


## Monitoring implementation update — 2026-11-05

The local worker now schedules checks of the two configured TCS/Infosys document sets on Monday/Thursday at 18:00 IST. These are actual enabled schedules stored in Supabase, unlike the original operating targets above. Both first attempts were denied by the official sites with HTTP 403; automatic collection is not yet demonstrated. Human original-source review and publication remain required, and the remaining 23 companies are not scheduled. Local uptime and a permitted unattended retrieval method remain release constraints. This build adds no paid infrastructure or AI usage.

## Infosys source acquisition and discovery — 2026-11-05

The SEC route resolves unattended document acquisition for Infosys on the local runner. It discovers recent results filings, validates the newest package’s identity/period/units/table structure, and cross-checks exact quarterly facts across separate exhibits. Fresh validated evidence enters the existing unapproved review queue. Missing or changed exhibits fail visibly; no older-quarter fallback, inferred prior-quarter figures or automatic publication is allowed. Discovery persists separately from successful retrieval in migration 007 and appears on Monitoring.

TCS remains on blocked pinned company URLs. The remaining 23 companies are not activated. This is not yet a fully hands-off 25-company release. Next gates: human review of the new Infosys source originals, two scheduled maintenance cycles, a reliable TCS acquisition route, and gradual company-specific adapter expansion. No paid service or model was added.
