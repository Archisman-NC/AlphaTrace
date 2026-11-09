# Shared research batch

Source monitoring supports **TCS pinned Q2 FY27/Q1 FY27 documents and Infosys SEC result discovery**. It prepares evidence; it never approves or publishes. The 25-company release target is unchanged.

## Current state — 2026-11-08

The [IndianAPI evaluation](../../web/scripts/INDIANAPI.md) completed five live
free-tier requests and returned all requested June/September 2026 metric pairs.
TCS Sales matches official consolidated revenue. However, Net Profit includes
non-controlling interests and Operating Profit differs from reported operating
income; neither can be mapped directly to existing AlphaTrace metrics. Other
companies' definitions and basis remain unverified. Private cached reports include
these diagnostics; no research publication or database mutation was made.

The [Fincrux evaluation](../../web/scripts/FINCRUX.md) completed five live company
requests on the free tier. All returned the requested June/September quarterly
fields, but TCS Sales matches official **standalone** revenue rather than the
consolidated basis used by AlphaTrace. All five quarterly Raw PDF rows were empty;
unit/basis metadata was absent. Reports remain private and unpublished. The API is
not accepted as a replacement for the current evidence pipeline; other companies'
accounting basis and metric definitions still require verification.

An evaluation-only Upstox connector is available via `npm run upstox:benchmark`
from `web/`; see [setup and acceptance checks](../../web/scripts/UPSTOX.md).
Its public instrument download resolved TCS, INFY, HCLTECH, WIPRO and HINDUNILVR.
Authenticated financial retrieval awaits an operator-supplied Analytics Token in
the ignored `web/.env.local`. It produces private comparison reports, has no
database access, and does not change the scheduled source monitor or publication
rules. Passing API coverage is not original-source verification.

TCS's reconciled three-file evidence is published as version 2 (`90edcbae-825e-46f9-a073-31a69f120f95`) after the operator's approval/publication. Infosys version 1 remains published. Automated TCS access remains blocked. Older execution notes below are historical.

The Monitoring page now separates the last evidence verification (which may include manual files) from the last successful automated check. Retry guidance uses the durable attempt counter and failure class, not a nearby schedule date. Blocked access, interrupted runs, offline workers and validation failures have specific operator guidance. Historical draft-creation messages no longer assert that a draft is still awaiting review.

The operator-only **Import official TCS files** form accepts the pinned Q2 FY27 workbook and Q1/Q2 consolidated IFRS USD PDFs, with explicit original-file confirmation and actual download times in IST. It prepares a draft without approving or publishing. It uses the same checked parser as the CLI; this server must have the repository's Python importer and Poppler installed. This is not a general document upload feature and is not yet a serverless-hosted ingestion service.

Live verification: an operator-triggered Infosys run completed successfully on 8 November IST. It created draft `34c0afe4-fa5c-4621-9a98-c290ad6a4b38` because both original document fingerprints/byte counts differ from the published version; the extracted comparisons and title are unchanged. The draft remains unapproved and unpublished. A second independent direct acquisition returned identical document fingerprints. After normalizing both payloads through the application validator, the hashes matched; re-ingestion reused the draft, created no duplicate, and preserved publication history. This is a live check plus repeat acquisition, not two completed scheduled maintenance cycles.

Validation: 40 TypeScript unit tests, 17 Python tests, isolated monitor integration (including two direct-success completions, bounded retries, expired leases, and manual/automatic separation), actual-original-file import/replay in the isolated local database, TypeScript and production build passed. Authenticated browser inspection verified the status labels, official links and file form before that tab closed; the replacement preview requires sign-in.

## TCS operator-file fallback

Prefer the operator form at `/monitoring` when the local app is running. The protected `/api/monitor/manual-tcs` endpoint authenticates and checks operator membership before parsing the multipart body. Each file is capped at 20 MB, aggregate request size is bounded, extensions and file signatures are checked, and a process-local gate permits one extraction at a time. The parser runs with a two-minute timeout and without database/auth credentials. Temporary files are removed; original bytes remain in the ignored fingerprinted cache. Unchanged imports reuse existing drafts/publications and show the appropriate review or published-evidence link. No database migration or new credential is needed.

From the repository root, run `python3 scripts/research/tcs_workbook.py --help`. Supply the downloaded workbook and its actual download timestamp via `--retrieved-at`. For a reconciled draft, also supply `--q1-results`, `--q1-retrieved-at`, `--q2-results`, and `--q2-retrieved-at`. All four PDF arguments are required together. This adapter is deliberately pinned to the observed FY27 Q1/Q2 layout, not a general spreadsheet importer.

The command writes `.local/research/manual/TCS-workbook.json` or `TCS-reconciled.json`; original bytes are fingerprinted and retained under ignored `.local/research/cache`. It does not submit, approve, or publish. An enrolled operator may submit the generated bundle through the existing authenticated ingestion endpoint. No new credentials or database migration are required. Original downloads remain unchanged. Do not commit the cache or pass database credentials to the parser.

Files remain labelled `operator_file`, never `direct`. Download timestamps are preserved and files older than seven days are rejected. The completed three-file check uses the final download timestamp, distinct from the earlier workbook-only check. Replaying identical evidence produces no new draft. Scheduled completion rejects operator-supplied files. The UI shows exact worksheet/cell locations, PDF pages, original-byte fingerprints and the manual retrieval limitation.

The parser bounds compressed/expanded size, rejects changed dates/units/basis/labels, and validates only the fixed workbook's supported simple formulas against their literal inputs. It runs no arbitrary formulas or macros. The original PDFs must match their known page counts, IFRS accounting notes, quarter-column structure and shareholder-profit reconciliation. Financial rows are cross-checked against the workbook, including its explicitly reported Q1 cell comments. No quarter is calculated by subtracting half-year totals. The workbook-only draft intentionally withholds profit growth and cannot pass publication requirements. The reconciled draft uses reported profits from the two original statements and preserves the unresolved adjustment-footnote discrepancy as a limitation.

Validation: 17 offline Python tests, 37 TypeScript unit tests, isolated local research/monitor integration suites, and production build. Integration verifies hidden/deduplicated manual drafts, rejection of incomplete publication, and rejection of manual files as successful scheduled retrieval.

TCS publication v1 remains available after human approval. Infosys now has a direct-byte SEC adapter: it reads the official issuer submissions feed, selects the newest recent results filing, follows its declared fact-sheet and consolidated IFRS USD exhibits, and cross-checks quarterly figures. Migration 007 records discovery independently of extraction success. A newly discovered unsupported package fails visibly instead of silently substituting an older quarter. Human review is still required before any Infosys publication.

The company-site URLs returned HTTP 403. Infosys's SEC feed and exhibits were accessible from this machine; TCS remains blocked and pinned to its configured documents. Neither source access nor two scheduled companies means all 25 companies are covered. The initial company-site drafts used explicitly labelled operator snapshots; they are retained as historical unapproved evidence (TCS's approved version remains unchanged).

## Infosys release discovery

The adapter uses `https://data.sec.gov/submissions/CIK0001067491.json` and only `www.sec.gov/Archives/edgar/data/1067491/{accession}/...htm` documents. Issuer CIK/name/ticker, accession, dates, actual exhibit links, consolidated IFRS/USD basis, table boundaries, quarter headers, row uniqueness and cross-document agreement are checked. Financial values are parsed as exact decimal strings; prior-quarter numbers come directly from the fact sheet, not half-year subtraction. A change to an unsupported table layout requires an adapter update and human review.

Discovery scans at most 300 recent entries for results-of-operations 6-K/6-K/A filings within 180 days. It chooses the latest filing by filing date/accession, even when the full package is not yet available. A missing fact sheet, wrong period, changed currency or inaccessible exhibit records a failed check while preserving valid discovery metadata. This is a deliberately limited SEC results feed, not all company announcements, exchange filings or historical backfill. SEC publication can lag the original company release. Filing dates in source cards refer to the SEC filing, not a guessed announcement date.

The runner makes four sequential bounded requests for a normal Infosys check (feed, cover, fact sheet, statements), separated by one second, with a declared AlphaTrace user agent, TLS verification, no redirects and existing size/time limits. It does not use browser sessions, search tools, API keys or paid data services at runtime. Original bytes are hashed and cached privately. Discovery history includes the feed fingerprint, accession, filing date, reporting period and check time. A later feed failure does not erase the last observed filing; its separate timestamp remains visible. Repeated identical evidence deduplicates; discovery/retrieval timestamps do not create new drafts.

## Run the local monitor

Apply migrations first. The worker uses `DATABASE_URL` with the restricted `alphatrace_app` role, verified database TLS, and the enrolled `ALPHATRACE_INGEST_USER_ID`. Credentials stay in the ignored environment file. Python 3 and Poppler must be on PATH. The Python child receives no database, authentication or ingestion credentials.

From `web/`:

```sh
# Keep the existing Next.js server running; start the sidecar worker.
npm run monitor
# Or, after stopping existing local processes, launch both together.
npm run dev:monitored
# Run only currently due work and exit.
npm run monitor:once
```

Use the operator controls at `/monitoring` to resume/pause a company or request a check. Initial command-line activation is also available with `npm run monitor -- --enable`. Ordinary invited users can read status, not change schedules. Stopping the local runner does not delete the schedule; queued and overdue work stays in Supabase and is picked up when it returns. This is a local process, not an installed startup service or always-on hosted worker. Keep the computer awake and connected; after a restart run `npm run dev:monitored` again. The page marks heartbeat age over three minutes as offline. No Codex agent automation, paid hosting, inference or external scheduler is involved.

Each claim holds a five-minute database lease. Workers use row locks and `SKIP LOCKED`, so concurrent workers cannot claim the same company. Abandoned leases produce failed-check records; late completions cannot overwrite newer runs. Each process limits source execution to three minutes; each download remains bounded at 20 MB and 100 PDF pages. Transient network/timeout/429/selected-5xx failures get one retry after five minutes. Access denial, extraction changes and other validation failures wait for the next regular slot. Repeated missed schedule slots coalesce into one current attempt; there is no historical catch-up storm.

Results are validated and ingested in the same transaction as run completion and rescheduling. New content creates an unapproved draft; unchanged content deduplicates through the existing draft hash. The worker never calls approve or publish. A successful run must contain direct document-byte sources retrieved after the run started; old outputs and operator snapshots cannot masquerade as a fresh automated check. A changed HTML fingerprint may include nonfinancial page changes, so the operator must inspect the draft before publication.

Temporary per-run output is removed after processing. Source bytes still use the existing ignored local cache and are never served by the app. Logs contain company/status only; stored failures use fixed messages rather than raw source/error contents. Recent history shows 20 attempts; full monitor history remains in the database.

## Run a batch

Requires Python 3.11+ and, for direct PDF retrieval, Poppler (`pdfinfo`, `pdftotext`). It uses only Python’s standard library; the legacy Python environment is untouched. Run from the repository root:

```sh
python3 scripts/research/batch.py
python3 scripts/research/batch.py --symbol INFY
python3 -m unittest discover -s scripts/research -p 'test_*.py'
```

Outputs go to ignored `.local/research/output/{SYMBOL}.json`; source bytes/text stay in ignored `.local/research/cache/`. A failure writes a failed check without a draft and exits nonzero. It never reuses an old output as if a download succeeded. Requests allow exact official hosts, verify TLS, refuse redirects, cap each input at 20 MB / 100 PDF pages and use timeouts. Changed dates, ambiguous rows, missing anchors or cross-document disagreements stop extraction. Text is treated as data; no model or source-provided instructions are executed.

To submit to the running app, from `web/` with Node 22:

```sh
node --env-file=.env.local scripts/research-batch.mjs --submit
```

The private `ALPHATRACE_INGEST_TOKEN` authorizes only draft/check submission. `ALPHATRACE_INGEST_USER_ID` must identify an enrolled operator. It cannot read the research API, approve, publish or withdraw. Use HTTPS for a deployed endpoint; loopback HTTP is allowed for development. Do not put the token in command-line arguments or commit environment files.

### Explicit snapshot fallback

When permitted direct retrieval is unavailable, an operator may supply a current extracted text snapshot from the exact official URL. Do not fabricate source contents, hashes or retrieval times. A snapshot is a private JSON file named `{source-id}.snapshot.json`, with `url`, `retrievedAt` (timezone-aware ISO timestamp) and `text`. PDF snapshots must retain the browsing tool’s zero-based `L…@P…:` page markers. This format supports the currently checked text views; other extraction formats need an explicit adapter.

```sh
# From repository root: validate without transmitting anything.
python3 scripts/research/batch.py --snapshots .local/research/sources
# From web/: submit through the protected endpoint.
node --env-file=.env.local scripts/research-batch.mjs --snapshots .local/research/sources --submit
```

Replaying a snapshot preserves its retrieval timestamp and does not refresh source age. Snapshots older than seven days are rejected. Fingerprints and retrieval mode are retained in every immutable payload. Raw originals and full extracted text are never served by the app; the shared payload contains numerical facts, locations, source links and original explanatory notes.

## Review and publication

1. Sign in through `/workspace` in the same browser tab, then open `/research?admin=1`. Your portfolio header links to research. Only an enrolled operator sees the review desk; other invited users see published research.
2. Choose a company marked **Draft ready**. Open the official sources and inspect every value, physical PDF page, period, unit, accounting basis and footnote. Review the full original table visually, not only flattened text.
3. Inspect the calculated changes and limitations. For example, TCS’s prior operating-profit column excludes exceptional items, so the current draft deliberately withholds a growth rate. TCV and employee counts are operating context, not substitutes for an earnings assessment.
4. Confirm evidence and comparison basis, and write a substantive review note. Request correction if necessary; correct extraction rules/metadata and submit a new draft. Unchanged source fingerprints/content deduplicate on retry; new evidence creates a separate immutable payload.
5. Approve, then publish. At least three comparable metrics plus results and supporting sources are required. Approval and publication also require a successful source check within seven days that matches every source fingerprint. A newer failed or different check blocks publication.
6. Corrections create a new numbered version; explain the correction in the new payload’s limitations and review note. The previous version remains inspectable. Withdrawal marks a version unreliable without deleting it; the catalog does not silently fall back to an older version.

Publication is transactional and serialized per company. Repeat publish requests return the same version. Database RLS hides drafts from ordinary users and restricts review/publication writes to operators. The runtime role cannot rewrite published payloads. The reader shows the latest 20 versions; all versions remain stored. The review desk loads the latest 50 drafts. These bounded pilot views need pagination before larger volumes.

The review desk prioritizes actionable drafts within its 50-record limit. An unpublished draft is superseded when a later-prepared draft has been published for the same company and the same or a later reporting period. This is derived from publication history without rewriting the earlier draft's review record. Superseded drafts are excluded from pending counts and available in a read-only history view; review/publish requests for them are rejected. Drafts prepared after the published draft remain actionable. Withdrawal does not reactivate earlier drafts. After publication, the UI opens the exact published version; source-check logs are presented as historical records, separate from current publication status.

## Enroll the operator / apply hosted migration

The existing private `web/.env.local` setup and runtime settings are required. From `web/`:

```sh
node scripts/setup-research.mjs INVITED_CONFIRMED_USER_UUID
```

This applies checksum-checked migrations, runs the disposable research integration suite, verifies the invited confirmed Auth user, enrolls that operator and saves a random ingestion credential privately. It does not publish any research. The active hosted runtime never receives the administrator connection. For an isolated local database, an administrator may insert the existing development UUID into `alphatrace.research_operators`; local access still requires explicit loopback development mode.

## Maintenance and remaining rollout

The TCS source manifest is intentionally pinned to known quarters. Check its official investor result index and update manifest dates, assertions and extraction rules deliberately. Infosys uses the SEC discovery adapter instead of the manifest for direct runs; `--snapshots` explicitly uses the legacy manifest. Re-fetching an old TCS URL does not prove no newer filing exists. Successful checks, failed attempts, source publication dates and reporting periods are distinct. Failed attempts show immediately; successful checks become overdue after seven days.

Complete Infosys original-document review, then measure two maintenance cycles before expanding through the remaining 8 / 10 / 5 companies. Add company-specific adapters and bank-specific fixtures before enabling banks; generic industrial ratios are rejected for bank payloads. No paid inference, worker or recurring automation was introduced. Operator maintenance time and full 25-company throughput are not yet benchmarked. Private notes and the review inbox already exist; new shared updates still require human publication.

Validation for this stage: 36 TypeScript unit tests, 12 offline Python tests, isolated local monitor integration (including discovery persistence, cross-accession rejection and repeated-evidence deduplication), publication-gate regression suite and production build. Live direct SEC retrieval produced four validated metrics. No new research is automatically approved or published.

Live verification on 2026-11-05: the existing scheduled worker completed an Infosys run and saved one new unapproved SEC-backed draft in Supabase. A second independent acquisition returned identical document fingerprints and content hash. The local integration test confirmed a repeat completion creates no duplicate draft. Infosys has no published version; the existing TCS publication is unchanged. The authenticated Monitoring and review desk show the correct filing and source links. No human-review checkbox or approval was submitted.
