# IndianAPI evaluation

## Setup

Choose the free Indian Stock Market API plan at
https://indianapi.in/indian-stock-market. Save the dashboard API key in
`web/.env.local` as `INDIANAPI_API_KEY=...`. Keep the file private (0600),
Git-ignored and outside NEXT_PUBLIC variables. Do not paste keys into chat or shell
arguments. No paid plan is required to attempt this evaluation; actual endpoint
entitlement must be verified with the account.

From `web/`, with Node 22–24:

```sh
npm run indianapi:benchmark
```

This makes one TCS quarterly-financial request if no cached response exists. After
reviewing TCS and the account's remaining credits, use
`npm run indianapi:benchmark -- --all` for the five-company pilot. Cached companies
consume no further calls. The pinned Node command is:

```sh
./node_modules/node/bin/node --env-file-if-exists=.env.local --import tsx scripts/indianapi-benchmark.ts
```

## Scope and limits

The fixed free-tier host is `stock.indianapi.in`; the evaluator never switches to
paid hosts. It uses GET `/historical_stats?stock_name=TCS&stats=quarter_results` and
the documented X-Api-Key header. Requests have a 25-second timeout, 5 MB limit,
no redirects and no automatic retries. Any request/schema error stops the batch.
401/403 may mean the key or endpoint entitlement is wrong; 429 means credits or
rate limits need attention. Neither triggers an upgrade.

Private snapshots and the JSON report are stored in `web/.local/indianapi/`.
Original JSON text and a SHA-256 fingerprint allow offline schema inspection and
replay. Reflected credentials and raw error responses are never saved or logged.
An exclusive lock prevents concurrent runs. If interrupted, confirm the process
has stopped before removing its stale lock. Cached responses are reused indefinitely
for evaluation, with their original retrieval timestamps; they do not imply freshness.
At most one request is made by default, or five with --all (fewer with cache hits).
Provider credit limits are not assumed; check the account dashboard before expanding.

The documented quarterly response is a dictionary of provider labels and periods.
Missing periods remain missing. Currency/scale, company identity, consolidated vs
standalone basis, accounting standards and profit definitions are not inferred.
The narrow TCS revenue check compares June and September 2026 with both official
standalone and consolidated series. A numerical match is not publication approval.
The reference is restricted to those exact quarters and that company.

This evaluator has no database or publication access. Existing research, source
monitoring, Upstox and Fincrux integrations are unchanged. Before production use,
verify original filings, update/correction behavior and storage/display rights.

## Official references

- Product, free-plan host and key setup: https://indianapi.in/indian-stock-market
- Quarterly endpoint and sample schema: https://indianapi.in/documentation/indian-stock-market
- Header authentication reference: https://pro.indianapi.in/ (documentation only;
  the evaluator uses the free host, never this paid host).
- TCS reference, physical pages 8 and 17:
  https://www.tcs.com/content/dam/tcs/pdf/discover-tcs/investor-relations/corporate-actions/2026-27/financial-results-qtr-sep-30-2026.pdf

Status on 2026-11-08 IST: five authenticated requests succeeded on the free-tier
host. TCS, INFY, HCLTECH, WIPRO and HINDUNILVR returned all three requested metrics
for June/September 2026. Snapshots are cached; no further calls are needed to replay.

TCS Sales matches consolidated revenue (72,275 → 73,188 INR crore). Net Profit
matches total profit including non-controlling interests (13,420 → 13,934), whereas
shareholder-attributable profit is 13,349 → 13,884 (physical page 8). Operating Profit
(18,556 → 18,815) differs from reported operating income (16,376 → 17,301, page 9).
These are definition mismatches, not evidence that the returned numbers are wrong.
The evaluator now records these dated diagnostics without permitting direct mapping.

The other four companies were reconciled on 2026-11-08 against official consolidated
Ind AS statements for June/September 2026. All 24 values match either an explicit
statement row or a documented derivation, with Wipro requiring conversion/rounding.
This supports the dated interpretations below; it does not establish the provider's
semantics for every period or company. TCS Operating Profit numerically matches
total segment results before unallocable expenses (page 9), not reported operating income.

## Dated mapping decisions

Pairs below run June → September 2026, in INR crore. `Sales` matches revenue from
operations for all five pilot companies. `Net Profit` matches total profit including
non-controlling interests, not necessarily profit attributable to shareholders.

| Company | Revenue | API Operating Profit | Total profit | Owner-attributable profit from filing |
|---|---:|---:|---:|---:|
| TCS | 72275 → 73188 | 18556 → 18815 | 13420 → 13934 | 13349 → 13884 |
| Infosys | 48211 → 49244 | 11409 → 11689 | 7775 → 7838 | 7769 → 7823 |
| HCLTech | 34579 → 35620 | 6870 → 7439 | 4626 → 5132 | 4624 → 5132 |
| Wipro | 24478.6 → 24752.7 | 4633 → 4863 | 3356.3 → 3199.7 | 3352 → 3200.6 |
| HUL | 17341 → 18226 | 3947 → 4089 | 2680 → 2796 | 2673 → 2790 |

- **Infosys:** API operating profit equals revenue − expenses + finance costs + D&A.
  It is a derived pre-depreciation measure, not reported operating income.
  [Official consolidated statement](https://www.sec.gov/Archives/edgar/data/1067491/000106749126000042/exv99w03.htm).
- **HCLTech:** the same formula must additionally subtract the September associate
  loss of 38. Reported segment operating results are 5831 → 6465. Do not label the
  API series EBIT or uniform EBITDA. Tax benefits also affect reported profit.
  [Official filing, pages 2–3 and 9](https://www.hcltech.com/sites/default/files/documents/investor-reports/FinancialResults_4.pdf).
- **Wipro:** filing amounts are INR millions; divide by 10. API rounds to whole
  crores: revenue 24479 → 24753, total profit 3356 → 3200. Operating profit derives
  from expenses with finance and D&A/impairment added back: 4633.2 → 4862.7 before
  rounding. It excludes the separate associate/JV result. Preserve filing precision.
  [Official filing, consolidated Ind AS table](https://www.sec.gov/Archives/edgar/data/1123799/000119312526426839/d118455dex991.htm).
- **HUL:** revenue includes other operating revenue; turnover is 17184 → 18071.
  The expense-based derivation reconciles operating profit; September also matches
  disclosed EBITDA. Total profit includes exceptional items. Continuing and total
  profit coincide in these quarters because discontinued profit is nil, but not in
  all historical periods. Do not infer a permanent continuing-profit mapping.
  [Official filing, pages 7–8 and 11](https://www.hul.co.in/files/hul-sq-26-results.pdf).

The executable references in `scripts/lib/indianapi-references.ts` preserve native
units, formula operands, quarter scope, source locators and SHA-256 fingerprints.
Public source originals are retained privately in `.local/indianapi/originals/`;
metadata sidecars record retrieval times. Changed references require a new review.

Replay all five cached snapshots without credentials or network calls:

```sh
./node_modules/node/bin/node --import tsx scripts/indianapi-benchmark.ts --all --cached-only
```

The report separates exact, rounded, missing and mismatched values. Missing caches
stop offline mode; no fallback request is made. References deliberately do not
validate later quarters. The CLI reports a nonzero exit for missing values or failed
four-company reconciliations; an exit of zero does not mean production acceptance
(in particular TCS's segment-result definition remains incompatible with reported operating income).

## Acceptance still pending

Endpoint responses have no explicit identity, basis, units, source links or update
timestamps. Do not wire these fields directly into published research. Before
public display, confirm storage/display rights and test correction/freshness behavior.
Private staged ingestion now preserves explicit definitions and original-filing
evidence while retaining the existing publication gates. No Supabase writes or publication approvals occurred in
the original evaluation; no additional IndianAPI requests were needed for reconciliation.

## Private import workflow

Migration `008_provider_imports.sql` adds operator-only immutable raw snapshots,
request history, five scheduling targets and links to unpublished research drafts.
Visit `/imports` from the operator link on `/monitoring`. Non-operators cannot read
these tables or use the API. Raw response text is not returned by the review API.

Run `npm run indianapi:seed` once to stage the existing five cached evaluations.
It is idempotent, preserves original retrieval dates, and counts prior requests in
the rolling quota. It performs no network retrieval. Identical response bytes share
one immutable import; later successful retrievals appear separately in attempt history.
The current response is selected by its latest successful observation, including
when the provider returns to earlier response bytes. Original snapshot dates stay unchanged.

Run `npm run indianapi:worker` alongside Next.js, or use `npm run dev:monitored`,
which starts the worker when its private token file is configured. Queue a single
refresh or enable daily refresh per company. Five attempts per rolling 24 hours
are serialized in Postgres across workers; failures and interrupted requests count.
There are no automatic immediate retries, and each company has a 24-hour request
window. The worker must stay running on an awake, connected computer. Scheduling
starts paused. The provider dashboard remains authoritative for calls outside this
workflow, including the standalone benchmark or other applications.

The review screen retains provider labels, official references, rounding differences,
missing references, observation dates and definition differences. Preparing a draft
uses checked official revenue and owner-profit values, not relabelled API profits.
TCS additionally uses reported IFRS operating income from the downloaded workbook,
including the June cell-comment adjustment. Other companies start with two metrics.
Original source retrieval dates are retained; preparation records no successful source
check, attestation or publication. Existing gates require a supporting source, at
least three comparable metrics, a current check of exact original bytes and human
approval. Wipro SEC sources are restricted to its issuer path, just as Infosys is.

No API-hosted data is added to investor pages. Usage permissions remain unconfirmed.
New quarters and numerical corrections without matching checked references require
investigation before draft preparation. Existing published research is not replaced.

Validation: `npm test`, `npm run typecheck`, `npm run build`; local scheduler/RLS
integration via `npm run test:imports` (requires the local test database).

The 8 November setup applied this migration to hosted Supabase, seeded all five
cached responses, and enabled daily refresh for all five targets. The signed-in
operator flow prepared a Wipro draft and verified its missing-source/metric gates;
it did not approve or publish it. Unit tests (64), type checking, the production
build and local database isolation/scheduler regressions passed.
