# AlphaTrace web app

The Next.js App Router app includes invited-account authentication, private portfolios and theses, reviewed shared research, a personal review inbox and journal exports. `/` redirects to the real-account workspace at `/workspace`; the browser-local illustrative sample is available at `/demo`; `/workspace?view=companies` manages holdings, watchlists and CSV imports. Human-reviewed TCS, Infosys and Wipro research is available to invited users. Recovery and a second real Auth account still need testing. No automatic invitations or broker integration is active. The legacy Python app also reads the consolidated local environment.

## Guided onboarding

New empty accounts see a three-step setup at `/workspace`. Existing accounts can open `/workspace?setup=1` or use the Guided setup link. Setup saves to the account after each Continue action, resumes after refresh, and can be skipped or reopened.

1. Follow companies or preview a holdings CSV. Existing follows and unrelated holdings are preserved; explicitly confirmed imports replace quantities for included symbols.
2. Optionally choose research interests. These prioritize the first available comparison and never create an investment thesis, filter the inbox, or enable notifications.
3. Open a published comparison in the review inbox, or finish with a clear coverage gap. Opening research does not mark it reviewed.

Apply migration `015_guided_onboarding.sql` before deploying this feature. It adds owner-isolated progress and a narrow database function exposing only monitoring symbol/enabled status. Provider data and operator controls retain their existing policies. `GET/POST /api/onboarding` require an invited account. Concurrent setup and portfolio edits are version checked and saved atomically.

Validate with `npm test`, `node scripts/test-local.mjs tests/onboarding.integration.ts`, `npm run test:portfolio`, and `npm run build`. Browser checks use synthetic data in isolated local storage; no real account's preferences are prefilled.

## Issue reporting

Published comparisons and individual metrics now include “Report an issue”. `/issues` provides private report history and an operator queue with Open, In review and Resolved states. Resolutions can explain the evidence, link a newer published correction or confirm a withdrawal. Research publication remains a separate human-reviewed action. Apply migration 017 and run `npm run test:issues`; see [issue reporting](docs/issue-reporting.md).

## Account & data controls

`/settings` now includes password changes, a complete private-data JSON export, and explicitly confirmed account closure. Operator accounts require handover before deletion. Apply migration 018 before deploying the shared account gate, then deploy the account-cleanup worker and website. See [account controls](docs/account-controls.md); `npm run test:accounts` uses isolated synthetic users. Live recovery email verification remains pending.

## Reliability and service health

Recoverable page/root errors, a missing-page view and connection/session notices are implemented. Operators can use **Service health** at `/monitoring` to inspect database availability, recorded worker/job failures, review backlog and coverage without consuming provider requests. Authentication service outages return 503 rather than a misleading expired-session response. Apply migration 019 before deployment; see [reliability](docs/reliability.md) and run `npm run test:reliability`.

## Local setup

Personal update emails are implemented at `/settings`: explicit opt-in, optional deferred-review reminders, unsubscribe and delivery history. Resend delivery through Trigger.dev remains disabled until a verified sending domain and signed webhook are configured. Apply migration 016 before deploying; see [personal email setup](docs/personal-emails.md) and run `npm run test:emails` against isolated local storage.

Requirements: Node 22 (see `.nvmrc`), npm, Docker Compose or PostgreSQL 18.3. The project pins dependencies and commits its npm lockfile. A project-local Node 22 binary is also included as a development dependency to support machines whose global Node is older; npm scripts use it automatically after installation.

From `web/`:

```sh
npm ci
cp .env.example .env.local
# Replace ALPHATRACE_DEV_TOKEN with at least 32 random characters.
docker compose up -d --wait
npm run db:migrate
docker compose exec -T db psql -U postgres -d alphatrace -v ON_ERROR_STOP=1 -f - < scripts/local-role.sql
npm run dev
```

Open http://127.0.0.1:3000. Create or rename a workspace, refresh, then restart the web process: the saved record remains. The database volume survives `docker compose down`. Do not use `down -v` unless deliberately deleting local data. `.env.local` and generated files are ignored; example passwords are exclusively for loopback development.

For an existing PostgreSQL server, create a separate `alphatrace` database, update the two URLs, run the migration, then execute `scripts/local-role.sql` against that database with the migration connection. Keep the listener on loopback. No SQLite or in-memory fallback masks unavailable PostgreSQL.

## Security boundary

- The local page/form is enabled only with `NODE_ENV=development`, `ALPHATRACE_LOCAL_DEV=true`, a valid configured user UUID, and a loopback Host header. The dev server binds to 127.0.0.1. Never tunnel or publicly proxy development mode: Host validation is defense in depth, not authentication.
- Local API access additionally requires the random development bearer token. It is never rendered into the page or sent in client JavaScript.
- Production ignores local development access even if the flag is set. The page displays a closed preview. There is no anonymous workspace fallback.
- The API accepts a Supabase access token only after server-side `auth.getUser(token)` verification using `SUPABASE_URL` and `SUPABASE_PUBLISHABLE_KEY`. Owner IDs are derived from that verified identity, never from request JSON. The Step 4 invitation allowlist requires a confirmed account; browser sessions use per-tab sessionStorage and API requests use verified bearer tokens.
- `DATABASE_URL` must use the dedicated `alphatrace_app` role without superuser/RLS bypass. `MIGRATION_DATABASE_URL` is only used by the migration CLI. On hosted environments supply the migration credential only to a migration job, not the web runtime.
- Ownership policies use `app.user_id` set transaction-locally by the trusted server. This model requires keeping runtime database credentials private: it is not intended for browser/direct-client SQL access. No public Supabase Data API privileges are granted.
- Migrations are transactional, checksum-verified and serialized by an advisory lock. Alter applied schemas through a new migration, not by editing an applied file.
- Health is liveness only. An authenticated workspace request verifies storage; storage failures return a generic 503 without exposing connection details.

## Managed PostgreSQL / Supabase configuration

The free AlphaTrace Supabase project has been created in Mumbai and public signups have been disabled for the invited beta. Migrations 001–005 and the workspace, portfolio, shared-research, private-research and inbox integration suites have passed against the hosted project. The local preview now uses the restricted hosted database connection; pilot sign-in and authenticated watchlist save/reload have passed; recovery verification remains pending. For managed deployment:

1. Apply migrations with the project administrator connection using the CLI.
2. Set a strong password and LOGIN on `alphatrace_app` in a secure administrator session. Never use the local-role password remotely.
3. Use a provider-supported direct/session connection for that role as runtime DATABASE_URL. Confirm provider pooler support before changing connection mode. Enable certificate-verified TLS (`DATABASE_SSL=true`) with the provider trust chain; never disable certificate verification to make a connection work.
4. Configure the Supabase project URL and publishable key. Keep database-administrator credentials out of deployed runtimes. The server-only Supabase secret is used for private document storage; browser auth uses only the publishable key.
5. Disable local access, complete Step 4 authentication/invitations, and verify two-user isolation against the hosted project before public access.

The hosted Auth settings endpoint is verified reachable with email authentication enabled and public signup disabled. No public website deployment is claimed. Hosting eligibility, free-tier limits and actual authentication email delivery must be checked before release.

### Prepared hosted setup

All local configuration lives in one Git-ignored `web/.env.local` (mode 0600), with placeholders in `.env.example`. Service-specific env files and the old local database backup have been consolidated. Provider commands, setup commands, Next.js and deployment helpers read this file.

`DATABASE_URL` remains the active restricted hosted connection. `TEST_*` preserves the isolated loopback database configuration; `npm run test:*` selects only those settings and refuses remote test URLs. `TRIGGER_DEV_SECRET_KEY` is used by development commands; `TRIGGER_SECRET_KEY` is the Production key used by the website. Database setup credentials remain local and are excluded from cloud exports.

`node scripts/setup-supabase.mjs` verifies TLS, applies migrations and configures/reuses the restricted application role. Supply `SUPABASE_DB_HOST`, `SUPABASE_DB_PASSWORD`, project URL and publishable key in `.env.local`. Existing active connections are checked before reuse. Do not run setup against the wrong database. `scripts/setup-research.mjs <operator-uuid>` enrolls an already invited and confirmed operator.

Supply the trusted certificate through `DATABASE_SSL_CA_PEM` for cloud runtimes; `DATABASE_SSL_CA_FILE` is supported locally. Never disable TLS verification. The deployment helpers export only named runtime variables, not the whole local file. See [Vercel deployment](docs/vercel.md) for preparation and release checks.

## API

- `GET /api/health`: liveness, no secrets/configuration output.
- `GET /api/workspaces`: current user's workspace or null; bearer required.
- `POST /api/workspaces`: JSON `{ "name": "My research" }`; idempotently creates/renames the user's one workspace. Limits JSON to 1 KB and name to 80 characters. No owner field is accepted as identity.

All workspace responses use `Cache-Control: no-store`. API keys should be supplied by a local client from environment, not pasted into shared command history. The local form provides the easiest persistence demonstration.

## Checks

```sh
npm run typecheck
npm test
npm run db:migrate
npm run db:migrate  # second run must be a no-op
npm run test:db
npm run build
```

Database tests use random user IDs and remove only their own fixture rows. They verify persistence across new connections, idempotent updates, cross-user read/write denial, owner spoof rejection, rollback, missing identity and privileged-runtime-role rejection. GitHub Actions runs these checks with an isolated PostgreSQL service. It has not run remotely until changes are pushed.

## Boundaries and next milestone

The UI now implements the dark finance design from the supplied prototype: responsive overview, company pages, evidence reviews, history, settings and CSV watchlist onboarding. Sample decisions and assumptions persist in browser storage, separately from the PostgreSQL workspace name. Step 4 now adds private PostgreSQL portfolios and configured Supabase sign-in/recovery; hosted verification is pending. Company coverage remains planned. Batch processing is reserved under `../scripts/research/`; no ingestion or automatic spending runs in this step.

## This checkout's verification environment

The local checks used Homebrew PostgreSQL 18.3 with its data under the repository's ignored `.local/postgres`, listening only on port 55432. Temporary servers were stopped after verification. To reopen that saved test workspace on this machine, run from the repository root:

```sh
/opt/homebrew/opt/postgresql@18/bin/pg_ctl -D .local/postgres -l .local/postgres.log -o '-h 127.0.0.1 -p 55432 -k /private/tmp' start
npm --prefix web run dev
```

This machine-specific path is not required for other contributors; use the Docker setup above. Do not run both database instances on the same port.

During setup, an initial npm invocation from the manifest-free repository root resolved to the user's ancestor package directory. It was stopped and rerun explicitly in `web/`. The ancestor package.json and lockfile retained their original timestamps; that interrupted attempt may have changed packages in the ancestor node_modules directory. No ancestor files were deleted or restored speculatively. All application dependencies and the committed lockfile are under `web/`.

## Product UI build — 2026-11-02

- Search the locked 25-company universe; follow up to 20 companies. Five companies have fictional research fixtures; others honestly show planned coverage.
- Inspect evidence, keep/revise/defer/dismiss updates, edit assumptions and export sample history as JSON. Each decision preserves its previous assumption and evidence text. Revised assumptions show reassessment pending.
- Browser-local persistence uses `alphatrace.sample.v1`, validates stored data, reports storage failures and offers reset/export in Settings. It is not account storage.
- CSV preview parses `symbol,quantity`, rejects invalid quantities, flags duplicates/unsupported symbols, and enforces 1 MB/100-row/20-company limits. Quantities are validated but not retained; this is watchlist onboarding, not a portfolio ledger.
- Geist and Geist Mono Latin font assets are extracted from the user-supplied prototype and served locally; the OFL license is included under `public/fonts/`.
- No inference, real source collection, source freshness checks, broker access or cloud synchronization is represented as active. Questions are explicitly unavailable pending ingestion.

Verification: production build and TypeScript passed; all six tests passed using `node --import tsx --test tests/*.test.ts` (the tsx CLI IPC socket is sandbox-restricted). Browser verified search, company view, source inspection, decision revision, history and reload persistence; desktop 1440px and mobile 390px layouts inspected, with no document overflow on the mobile review screen. CSV parser was unit-tested; OS file selection was not exercised. Existing database integration was not rerun because the database layer is unchanged.

## Accounts and private portfolios — Step 4

`/` remains the browser-local sample. `/workspace` now stores private holdings and watchlist membership in PostgreSQL. The loopback development mode uses the configured local identity; production disables this mode. Never infer hosted readiness from a successful local test.

Local startup from the repository root (after initial database setup above):

```sh
/opt/homebrew/opt/postgresql@18/bin/pg_ctl -D .local/postgres -l .local/postgres.log -o '-h 127.0.0.1 -p 55432 -k /private/tmp' start
npm --prefix web run db:migrate
npm --prefix web run dev
```

Use the existing Docker/manual setup instead of the Homebrew command on other systems. Open `http://127.0.0.1:3000/workspace`. The local database needs all current migrations. No demo decisions or sample holdings are copied into the private portfolio.

### Hosted authentication setup (first pilot verified)

1. Configure a Supabase project with email/password authentication; disable public signups for the invited beta. Provision up to five pilot user records through the project’s admin interface. This app does not expose signup or grant access based on client-supplied emails.
2. Put the project URL and **publishable or legacy anon key** into `SUPABASE_URL` and `SUPABASE_PUBLISHABLE_KEY`. A service-role/secret key is never exposed to the browser. Set `ALPHATRACE_INVITED_USER_IDS` to the comma-separated user UUIDs approved by the operator. Empty, malformed or more-than-five entries deny hosted access. The server also requires a verified email and validates each bearer token using `auth.getUser`.
3. Configure the deployed Site URL and allowed redirect URL `https://YOUR_HOST/workspace?recovery=1` in Supabase. Configure email delivery and test it before inviting users. Existing pilot users can establish/reset their passwords via “Forgot your password?”. Recovery links open a password-update form; the user enters their own password. Provider password rules still apply.
4. Apply migrations using the separate migration connection. Configure the dedicated `alphatrace_app` runtime connection and trusted TLS; use `DATABASE_SSL=true` for the managed database. Do not use an owner/admin connection at runtime.
5. Build/run in production with local development access disabled. Test sign-in, refresh, sign-out, expired recovery links and two separate pilot accounts against the hosted project before public deployment. The free database project and first confirmed pilot account are configured; email delivery and password recovery remain unverified.

The browser SDK stores its auth session in sessionStorage (per tab); holdings are never stored in browser storage. Sign-out clears the rendered portfolio. APIs accept bearer credentials, not cookie identity or owner IDs from JSON. Local server actions require development mode, loopback Host and same-origin requests. Real research remains unavailable until the shared pipeline phase.

The auth implementation follows [Supabase password authentication](https://supabase.com/docs/guides/auth/passwords) and [password recovery](https://supabase.com/docs/reference/javascript/auth-resetpasswordforemail).

### Portfolio semantics and checks

- One exact decimal quantity per owner/company, with up to six decimal places. Saving/importing sets an absolute quantity; it never adds to an old quantity.
- Holdings and watchlist membership are independent. Removing a quantity preserves a followed company; unfollowing preserves its quantity. A workspace may follow up to 20 companies.
- CSV files stay in the browser. Map symbol and quantity columns and explicitly confirm selected rows. Only normalized supported holdings reach the API. Maximum 1 MB / 100 source rows. Duplicate, unsupported and invalid rows are flagged and excluded; duplicate rows must be resolved in the CSV. Unselected and unrelated holdings are preserved.
- Every mutation locks the owning workspace and checks its portfolio version. Stale writers receive HTTP 409 and must reload; imports are transactional. RLS protects reads, inserts, updates and deletes. API bodies are capped at 32 KB.

From `web/`:

```sh
node --import tsx --test tests/*.test.ts
npm run test:db
npm run test:portfolio
# Requires the local dev server; uses .env.local without printing the token:
node --env-file-if-exists=.env.local --import tsx tests/portfolio-http.integration.ts
npm run typecheck
npm run build
```

Verified locally: all 12 unit tests, both real-PostgreSQL integration suites, migration replay, API authentication/media/size/validation/conflict checks and production denial of the development token. Browser verified exact-quantity manual entry, reload persistence, mapped CSV preview, unsupported-row exclusion, absolute replacement and independent watchlists. Hosted Supabase sign-in/recovery and email delivery remain unverified without project configuration.

### First hosted pilot verification — 2026-11-02

The user created their own confirmed pilot account and signed in through Chrome. The account was added to the server allowlist without copying its password. Verified an authenticated empty portfolio load, followed INFY through the UI, checked that the record belonged to the pilot in Supabase, and confirmed that the signed-in session and followed-company count survived a full page reload. Restored the original empty watchlist with an exact-state/version-checked update. No holdings were fabricated.

Fixed duplicate browser Auth client construction under React Strict Mode by sharing one client per tab and cleaning up only component subscriptions. TypeScript, all 13 unit tests and the production build passed. Recovery redirects/email delivery, token expiry/sign-out and a second real Auth account remain separate release checks. The site is still a localhost preview backed by hosted Supabase, not a public deployment.


## Shared research — Step 5 first slice

Open `/research` from the private portfolio header. The 25-company catalog distinguishes published, planned and withdrawn coverage and reports failed/overdue source checks. An enrolled operator can switch to the review desk, inspect source-linked exact-decimal comparisons, approve a draft, and publish an immutable version. Historical versions and withdrawals remain visible. Supabase currently holds two unpublished drafts (TCS and Infosys), not 25 active baselines.

Migration `003_shared_research.sql` introduces shared updates/checks and operator-only drafts/review writes. `GET /api/research` accepts optional `symbol` and `admin=1`; `POST /api/research` accepts operator approve/reject/publish/withdraw commands. Both require the existing invited bearer identity. `POST /api/research/ingest` uses a separate draft-only credential and caps JSON at 128 KB. Every response is no-store. Local server actions enforce development/loopback/same-origin access.

Read the [research batch instructions](../scripts/research/README.md) for setup, commands, snapshot provenance and publication gates. The initial official endpoints blocked direct downloads; the drafts use explicitly labeled text snapshots and still require original-document review. No unattended acquisition or personal inbox matching is claimed.

Tests: `npm run test:research` requires separate runtime/admin database URLs, like the existing integration suites. `python3 -m unittest discover -s ../scripts/research -p 'test_*.py'` runs offline extraction safeguards. CI includes both. The production build and 18 unit tests pass; local and hosted research integration plus HTTP credential-scope checks passed on 2026-11-02. The full browser flow was tested with synthetic data on an isolated local database, not by publishing real drafts.

## Private investment research — Step 6 manual path

On `/research`, select a company and open **Write your thesis**. Save your investment case, reasons to reconsider and optional confirmed assumptions. A later save requires a revision reason; history retains the original wording. Restoring an earlier version only creates an editable draft. Notes do not require a publication or AI provider. They stay in owner-isolated PostgreSQL storage.

Migration `004_private_research.sql` adds owner-scoped heads and append-only revisions. `GET /api/thesis?symbol=INFY` returns the current revision plus 20 history entries; `before=<version>` paginates older history. `POST /api/thesis` accepts `{symbol, version, commandId, payload, reason, confirmed}`. Payload contains `thesis`, `risks`, `assumptions`, and up to five published `evidenceIds`. Each assumption has a stable UUID, statement, optional metric/baseline/horizon/reviewCondition and an explicit direction. Assumptions require confirmation. The endpoint bounds JSON at 128 KB and returns no-store responses.

A repeated command ID with identical content is idempotent. Stale writers receive 409; private wording is never added to shared research. Runtime permissions prevent changing old revisions. Links use `/research?symbol=INFY&update=<publication-uuid>` to open the exact version, including versions outside the most recent 20. Withdrawn anchors remain visible in old revisions but cannot be newly attached.

`npm run test:thesis` requires the separate runtime and migration database URLs, like the other integration tests. It uses random test owners and removes its synthetic fixtures afterward. Local and hosted integration passed, along with 22 unit tests and the production build. The isolated local browser workflow passed; pilot-browser verification, private export/deletion and automated matching remain future work. No AI or scheduled spending was introduced.

## Personal review inbox — Step 7

Open `/inbox` from the portfolio or research header. It checks published research for held/followed companies, preserving one private review item per exact publication. All previously unseen published versions are eligible, including existing research when a company is first followed. Each catch-up adds at most 50 versions; “Check next batch” resumes larger backlogs. Withdrawn versions are not newly delivered, but existing withdrawn items retain visible warnings and history. Unfollowing does not erase old reviews.

Topic matching uses only the confirmed assumption's `metric` field and a fixed alias list in `src/lib/inbox-contract.ts`. It snapshots the current thesis when the inbox item is created—not when the source was originally published. Blank or unmatched topics produce a general company update. It does not evaluate direction, time horizon, numerical thresholds or sentiment. There are no paid AI calls or background workers.

Responses: keep view, dismiss, defer, reopen and record a revised thesis. Every response requires a short reason. Deferred items return to the pending view at 9 am IST on the specified date. A revised response requires a newer saved private thesis revision linked to the exact publication; the saved wording is copied into the immutable response event. Keeping/dismissing never edits beliefs. Unsaved responses block filter/item changes; uncertain saves can retry the same command without appending duplicates.

API (invited bearer identity, no-store):

- `GET /api/inbox?filter=pending|deferred|reviewed|all&cursor=...`: 25 items, global current-state counts, optional next cursor. Preserve the opaque cursor verbatim.
- `GET /api/inbox?update=<uuid>&before=<response-version>`: exact owned item, original private snapshot, publication/source status and 20 response events per page.
- `POST /api/inbox` with `{action:"sync"}`: resumable, bounded discovery.
- `POST /api/inbox` with `{updateId,commandId,version,action,note,deferUntil,thesisVersion}`: versioned response. Action is `keep`, `dismiss`, `defer`, `revise` or `reopen`; `deferUntil` is a calendar date required only for defer; `thesisVersion` is required only for revise. The body limit is 16 KB. Owner IDs are never accepted from the caller.

Migration `005_review_inbox.sql` adds forced-owner RLS, unique publication delivery, append-only response events and immutable original snapshots. Shared publication advisory locks coordinate reader decisions/evidence links with withdrawal without granting operator access.

Run `npm run test:inbox` with separate runtime/admin database URLs. It uses synthetic publications and random owners, then removes its fixtures. It covers matching snapshots, deduplication, RLS, concurrency, retry, deferral, corrections, withdrawal, revision linkage and pagination. The same suite passed on Supabase. Existing publication/thesis regression suites pass locally; 26 unit tests and the production build pass. Browser checks exercised the full synthetic workflow and mobile layout; real pilot walkthrough and public deployment remain pending. The first two real research drafts still await original-document human review.


## Private journal downloads (Step 8)

The portfolio (`/workspace`) and review inbox (`/inbox`) include **Download journal** with Markdown and JSON formats. The authenticated `GET /api/export?format=md` (or `json`) exports only the verified account's saved portfolio, all thesis revisions, original inbox snapshots, review events and linked published evidence. JSON has `schemaVersion: 1`, `exportedAt`, `complete: true` and named collections. The format does not include tokens, owner identifiers, internal command IDs, operator drafts or other users' data. It is not a restore/import API.

Downloads are private/no-store attachments. Database reads use a read-only repeatable-read transaction under the restricted runtime role. Limits are 5,000 rows per collection, 10 MiB projected database rows and 20 MiB final content; exceeding a limit returns 413 without a partial file. Unknown formats return 400, unauthenticated requests return 401 and an unavailable database returns 503 without database details. No provider or inference calls occur. Unsaved edits are not exported.

`npm test` includes format, escaping and size tests; `npm run test:inbox` also verifies journal isolation, full history, withdrawn evidence, exact quantities, concurrent snapshot consistency and read-only/limit behavior using synthetic fixtures that it removes afterward. These checks passed on both local PostgreSQL and Supabase. Actual Markdown and JSON downloads were verified in an isolated local browser session. The new controls still need the real pilot-account walkthrough.


## Private overview and evidence review — 2026-11-05

The overview combines the existing authenticated portfolio, research catalog and inbox endpoints. Opening it catches up at most 50 published versions; incomplete catch-up and load failures are explicit. Counts include historical reviews retained after unfollowing. Tracked-company coverage deduplicates holdings and follows, and distinguishes unpublished/withdrawn research from failed, overdue or changed source checks. Refreshing the overview discovers publications; it does not fetch original company sources.

Inbox filter links are addressable with `?filter=pending|deferred|reviewed|all`, and `update` pins an exact review. Nonempty filters select their first item when no item is requested. The evidence view shows exact decimal values, both period/source links, comparison basis and suppressed growth for noncomparable figures. Section navigation connects evidence, the saved thesis snapshot, response and immutable history. Shared navigation links the overview, company management, inbox and research library. No database migration, provider or paid service is needed.

Validation: 32 unit tests, TypeScript and production build; authenticated browser checks of dashboard counts, published versus missing coverage, exact-publication navigation, reviewed filter, unsaved response protection and discard. The existing pilot response was left intact. New coverage and rendering tests check stale/failed/withdrawn/changed sources, deduplication, exact decimals, comparison suppression and escaping.

Responsive verification for this slice: inspected overview and evidence layouts at 1280px and 390px, with no horizontal overflow; browser console reported no errors. The inbuilt browser was returned to its normal viewport on the overview.


## Source monitoring — 2026-11-05

`/monitoring` shows the local runner heartbeat, per-company schedule, current/failed checks, and 20 recent attempts. Operators can resume/pause or queue a check; ordinary invited users have read-only access. A compact monitor panel also appears on the overview. Migration 006 adds shared monitoring tables with forced RLS and operator-only writes. No private portfolio or thesis data is sent to sources.

`npm run monitor` starts the sidecar; `npm run dev:monitored` starts both the app and worker after existing instances are stopped. The computer must stay awake and the process must remain running; this is not installed as a startup service. The configured cadence is Monday/Thursday 18:00 IST, with one bounded retry for transient failures. Supported source sets are TCS and Infosys only. TCS rechecks pinned documents. Infosys now discovers recent results filings from the official SEC submissions feed and validates the latest filing’s fact sheet and IFRS USD statements. Migration 007 stores discovery separately from extraction success.

The first live hosted runs for both companies returned HTTP 403. The UI accurately reports failed retrieval; successful-source dates and the published TCS version were preserved. That initial blockage remains for TCS. The subsequent Infosys SEC adapter successfully retrieves direct document bytes without a paid service; it creates unapproved drafts only. See [monitor operations](../scripts/research/README.md).

Validation: 35 TypeScript unit tests, six offline Python tests, isolated local monitor integration (concurrent claims, operator-only writes, snapshot/stale-source rejection, retry limits, lease recovery, pause during a run and late/duplicate completion), existing research integration and production build. Hosted runtime verification checks actual failed run records, unchanged publication/successful-source timestamps, anonymous API denial and no-store responses. Run `npm run test:monitor` only against an isolated local database with the worker stopped; it deliberately manipulates shared test schedules and restores them.

Final browser verification: operator pause/resume persisted against Supabase, the resumed Infosys check recorded another real 403, and both schedules remain enabled. The combined local launcher was exercised by restarting both services; the existing sign-in and saved research survived. Monitoring and the overview were checked at 1280px and 390px without horizontal overflow or console errors. The monitor presents the last successful source-check timestamp separately from the latest attempt.

## SEC discovery update — 2026-11-05

Monitoring now shows Infosys’s last discovered reporting period, SEC filing date, accession, official link and feed-check time. A new unextractable filing remains visible; the worker never silently substitutes an older quarter. A later feed failure keeps the last discovery with its original timestamp. Sources are restricted to Infosys’s SEC accession path. Private account data is never included in source requests.

Validation: 36 unit tests, 12 Python tests, local monitor/research integration and production build. Live acquisition found the September 2026 quarter and cross-checked revenue, operating profit, owner net profit and operating margin. The new evidence still requires human publication review. Coverage remains 2 scheduled companies, with TCS direct access blocked and the other 23 unimplemented. SEC release timing can lag company announcements; the laptop worker must remain running.

### Evidence completion and corrections

The operator research desk now includes an **Evidence workbench**. Open a pending
company draft (or the latest published version) and choose **Edit evidence**:

1. Attach an official company URL, or supply an original PDF/HTML file with its
   official URL, publication date and document role. Files are limited to 20 MB;
   PDFs require `pdfinfo` on the application server and are limited to 100 pages.
   Original bytes are hashed on the server. Uploads are inspected and discarded;
   keep your original files. Existing workbook evidence remains editable, while
   new workbook import/reverification uses the TCS workflow under Monitoring.
2. Add or correct metrics, exact decimal values, dates, units, accounting basis,
   source locations and limitations. Empty values explicitly mean unavailable.
3. Inspect the change list, give a revision reason and save. Every correction is
   a new unapproved draft. The prior payload and review remain unchanged, and the
   prior draft cannot subsequently be approved or published. Concurrent conflicting
   saves must reload the successor; replaying the same save returns that successor.
4. Choose **Recheck saved sources**. Direct sources must return the exact original
   bytes; redirects are refused. Operator files require a matching attachment
   receipt from the last seven days. This checks fingerprints, not factual accuracy.
   A failed check is recorded and blocks approval. Unchanged reattachments preserve
   original evidence timestamps and can be checked without creating a new revision.
5. Resolve readiness gaps before human review. Saving or checking evidence never
   supplies the operator's attestations and never publishes automatically.

Migration `009_research_editing.sql` adds operator-only revision lineage and
owner-scoped attachment receipts. Browser writes require an authenticated operator;
new retrieval metadata must match an existing source or a server-created receipt.
No provider API allowance is consumed by editing research.

Checks: `npm test`, `npm run typecheck`, `npm run build`, and
`npm run test:research-edit` against the isolated local database. The 8 November
browser verification saved a title-only Wipro revision, preserving its figures
and leaving supporting evidence, a third metric and human approval outstanding.

### Wipro pilot review package

`scripts/wipro-review.ts` completes the June/September 2026 Wipro draft from three
fingerprint-pinned official HTML originals: the consolidated Ind AS statement and
the two quarterly IFRS releases. It requires `.local/wipro-review/` originals and
metadata sidecars (or `--directory=/absolute/path`). The default previews only;
`--prepare` records attachment receipts, saves an unapproved revision through the
normal correction service, and checks its exact source fingerprints. It never
approves or publishes. Run with the enrolled operator environment:

```sh
./node_modules/node/bin/node --env-file=.env.local --import tsx scripts/wipro-review.ts
./node_modules/node/bin/node --env-file=.env.local --import tsx scripts/wipro-review.ts --prepare
```

The package preserves INR-million-to-crore conversion, owner-attributable profit,
and a separately scoped IT-services IFRS margin (16.0% → 16.5%, +0.50 percentage
points). The release's rounded IFRS net income decline is not substituted for the
Ind AS owner-profit change. Source locators and limitations explain this distinction.

Evidence points now optionally carry `scope`. Different business scopes are not
comparable; omitted scope retains existing whole-company behavior. Scope survives
validation, editing, research/inbox rendering and journal export.

The SEC response changed tracking scripts between retrievals while its statement
text stayed identical. Original-file snapshots are used explicitly; successful
file checks do not assert live-site freshness. Keep the downloaded originals for
human review. This dated package refuses other quarters or unrecognized files.

### Foreground publication updates

Overview and Inbox check the authenticated inbox API every minute while visible
and online. Focus/reconnection can initiate a check, with a cooldown and bounded
failure backoff (up to five minutes). Requests do not overlap within a refresh
loop; late responses after navigation/account changes are ignored. Reads and
inbox catch-up requests have a 20-second timeout. Existing review-save retry
semantics are unchanged.

Overview refreshes its review counts/feed without a full-page loading state.
Inbox announces new or changed reviews and keeps the current selection, pages
and unsaved response intact until “Load updates”. That action is disabled while
editing. Existing database synchronization remains idempotent and adds at most
50 publications per check. This path makes no financial-provider calls.

This is periodic delivery of already-published research, not source discovery,
live prices, a push subscription, or unattended ingestion. Coverage/source-monitor
panels retain their own refresh controls. Cloud workers use the bounded Trigger.dev dispatcher described below. Tests in `tests/auto-refresh.test.ts` cover suspended
checks, backoff, overlapping requests, cleanup and changed-publication detection.

### Trigger.dev development

Project: `proj_azibgotclaclvmjiifqv`. The SDK and build extension were initialized
with Trigger.dev 4.8.3. `trigger.config.ts` uses Node 22 (matching the local app),
`src/trigger`, a 60-second compute limit and no automatic development retries.
The exported `alphatrace-health-check` task returns a timestamp and `status: ok`.
It performs no financial-provider requests or database writes.

From `web/`, authenticate with `npx trigger.dev@latest login` if needed. Save the
Development environment key as `TRIGGER_DEV_SECRET_KEY` in private `.env.local` (Git ignored),
then run `npm run trigger:dev` in a terminal that stays open. In a second terminal,
run `npm run trigger:check` to enqueue the health check. The dashboard's Development
Tasks and Runs screens show registration and completion. Keep keys server-side;
never add a `NEXT_PUBLIC_` prefix or log the key.

The CLI command is version-matched to the installed SDK; update them together.
Development tasks execute locally while the dev CLI is connected. The monitoring
cron is Production-only, so connecting a development worker does not enable it.


### Cloud monitoring

`alphatrace-scheduled-monitoring` is a Production-only hourly task in
IST (UTC cron `30 * * * *`), with a 60-minute scheduling window to comply with
the free plan. UTC avoids a server-side rejection of the Asia/Kolkata alias.
This is a dispatcher: it does not fetch every company every hour. Supabase keeps
eligibility, leases, failures and quota across process restarts. It runs at most
two official checks and five provider requests per invocation, sequentially,
with concurrency one, a ten-minute compute limit and no task-level retries.

- TCS checks pinned documents; Infosys discovers recent SEC results filings.
  Regular work becomes eligible Monday/Thursday at 18:00 IST and runs on the next
  dispatcher. One transient-failure retry becomes eligible after five minutes;
  it also waits for a dispatcher. Denied access requires operator attention.
- Five IndianAPI companies refresh no more than once per rolling 24 hours, under
  a shared five-attempt cap. Failed attempts consume allowance. Imports remain
  private and do not constitute verification of an official filing.
- Changed, valid official evidence may create a draft. This task never approves
  evidence or publishes research. Paused targets stay paused.
- An idle cloud worker is normal. The UI marks it overdue after 150 minutes
  without a check-in; this includes the hourly cadence and scheduling window.

Apply migration `010_cloud_monitoring.sql` before deployment. From `web/`, run
`npm run trigger:deploy`. The build validates local hosted configuration, packages
only `batch.py`, `sec_infy.py`, `tcs_workbook.py` and `sources.json`, and installs Python/Poppler.
Per-run document caches use temporary storage and are removed after processing.
Python receives no database or provider credentials.

The deployment syncs these monitoring settings as secrets: `DATABASE_URL` (the existing
`alphatrace_app` role), `DATABASE_SSL`, `DATABASE_SSL_CA_PEM`,
`ALPHATRACE_INGEST_USER_ID`, and `INDIANAPI_API_KEY`. It also syncs `SUPABASE_URL` and `SUPABASE_SECRET_KEY` for private document processing when configured. It reads only private `.env.local`; it never uploads that file, test settings, Trigger API keys or the database-admin password. TLS verification stays enabled.
The inline CA replaces a certificate path that exists only on the laptop.

After a successful cloud verification, set `ALPHATRACE_CLOUD_MONITORING=true` in
private `.env.local`. This prevents local monitor/import workers from starting;
`dev:monitored` then runs only the website. To return to local operation, deactivate
the Production schedule in Trigger.dev first, wait for active runs to finish,
remove this flag, then start local workers. Database pause/resume controls apply
to both environments; do not pause targets merely to stop a local process.

Verify the deployed task by triggering it in Production, then inspect its run
output and `/monitoring` and `/imports`. A successful dispatcher with failed
source results is not a successful source verification. A manual run does not
prove the recurring schedule has fired: check the dashboard's active schedule
and a subsequent scheduled run separately. The Next.js website has its own
hosting lifecycle; deploying Trigger.dev does not publish the website.


Cloud rollout verified on 9 November 2026 IST:

- Production version `20261108.2`, deployment
  [krqixdeg](https://cloud.trigger.dev/projects/v3/proj_azibgotclaclvmjiifqv/deployments/krqixdeg).
- Active Production schedule `sched_82uebxygn8bu3sjcyg6hv`, cron `30 * * * *`,
  UTC, one-hour spreading window. The API reported the first effective scheduled
  time as 9 November 03:11 IST (8 November 21:41 UTC).
- Manual cloud verification
  [run_06grppdhckjid0kq4iash6ila2](https://cloud.trigger.dev/projects/v3/proj_azibgotclaclvmjiifqv/runs/run_06grppdhckjid0kq4iash6ila2)
  completed in 17.64 seconds. Supabase recorded both source results and cloud
  heartbeats for all seven targets. TCS denied retrieval with HTTP 403; Infosys
  discovery found a filing but its documents returned HTTP 403. These are source
  failures, not fresh evidence. No research was published by the worker.
- IndianAPI allowance was 5/5, so the worker made no provider requests. The next
  provider refresh and the first recurring run were not yet observed during
  this deployment verification. Local source/import startup is disabled using
  the private cloud-monitoring flag; the website remains locally hosted.
- Validation: 82 TypeScript unit tests, 17 Python tests, source/provider database
  integration suites, TypeScript check and Next.js production build passed.


### Provider change detection and operator review queue

The `/imports` screen now includes **Needs review** and **Reviewed** queues with
20-item cursor pagination. Migration `011_provider_review_queue.sql` adds an
operator-only observation head per company, immutable comparisons, and append-only
triage actions. Successful imports and queue detection commit in one transaction.

Detection covers the provider's Sales, Operating Profit and Net Profit series:
new quarter labels, added older history, changed exact decimal values, values that
become missing, and removed quarters. Removed history may be a provider retention
change, not a financial restatement. Initial observations create a baseline item
showing the latest two available quarters; they do not claim a newly announced
company event. Units and accounting definitions still need official evidence.

JSON ordering, equivalent decimal formatting and untracked fields do not create
alerts. Each meaningful transition gets its own sequence, so A → B → A produces
two changes even when A reuses an existing stored response. Repeated observations
and stale cache seeds cannot move the detection head backwards. Failed requests
remain in Import attempts and do not create financial change observations.

Operators can inspect exact before/after values and snapshot hashes, follow dated
official references, add an optional 500-character note, mark an item reviewed,
and reopen it. Optimistic versions prevent one operator session overwriting
another. Each action records its actor, time, version and note. Reviewing an old
item does not resolve a newer change. Triage never verifies official evidence,
prepares a draft, publishes research, or sends an investor update.

Deploy migration 011, then `npm run imports:backfill-reviews` to process existing
successful observations without API calls or freshness changes. Deploy the cloud
worker with `npm run trigger:deploy`, then rerun backfill to catch observations
completed by the prior worker during rollout. The backfill is idempotent.

Rollout on 9 November 2026 IST: five real baseline items created in Supabase,
zero marked reviewed. Trigger.dev Production version `20261108.3`
([deployment](https://cloud.trigger.dev/projects/v3/proj_azibgotclaclvmjiifqv/deployments/8iu5kdg9))
includes automatic detection for subsequent imports. The scheduled five-company
refresh was still delayed at verification time; no new company change is claimed.

Validation: 87 unit tests, import/review database integration suites, TypeScript
check and Next.js production build. A disposable local operator browser test
verified review/reopen, unsaved-note preservation during refresh, desktop and
390px mobile rendering. No production item was reviewed by that test. Anonymous
requests to `/api/imports/reviews` return 401; ordinary readers are denied by
application checks and database RLS. Run `npm run test:import-reviews` only on the
empty isolated local test database.

### Import outcomes and cloud run traceability — 9 November 2026

Migration `012_import_outcomes.sql` records each attempt's comparison result and
exact review ID in the same transaction as its saved response. The cloud worker
also records its Trigger.dev run ID. `/imports` shows the latest batch's saved
responses, created reviews, unchanged comparisons and failed requests, with a
link to the cloud run. Counts cover claimed requests, not every scheduled company
or the overall task's infrastructure status.

Each import attempt distinguishes a baseline, meaningful changes, no tracked
changes, an older observation, a failed request, or an older import whose result
was not recorded. Legacy attempts remain explicitly unclassified. Links open the
exact observation through `?review=<id>`, including completed and superseded
reviews; operator authorization and unsaved-note protection still apply.

Migration 012 is applied to Supabase. Trigger.dev Production version
`20261108.4` includes this tracking
([deployment](https://cloud.trigger.dev/projects/v3/proj_azibgotclaclvmjiifqv/deployments/tnfnygf5)).
Apply the migration before deploying the updated application or worker. The
website still runs locally; cloud task deployment does not host the Next.js app.

Validation: 89 unit tests, both isolated import/review database suites, TypeScript
and a production build passed. Browser tests verified exact links to changed,
completed and older reviews, note preservation, review/reopen, populated batch
counts, and desktop/390px layouts using disposable data. No real financial review
was recorded by these tests. Source-access failures remain separate from successful
provider retrieval and comparison.

Live verification at 02:54 IST on 9 November: delayed run
[`run_06grpr4i4po9ipqvf8k8c81ha2`](https://cloud.trigger.dev/projects/v3/proj_azibgotclaclvmjiifqv/runs/run_06grpr4i4po9ipqvf8k8c81ha2)
completed on version `20261108.4`. Supabase records five successful requests
(TCS, INFY, HCLTECH, WIPRO, HINDUNILVR), five unchanged comparisons, zero failed
requests and zero new review items, all linked to that run. The five earlier
baseline reviews remain pending. This verifies the delayed cloud refresh;
the recurring hourly schedule remains active but its first recurring execution
has not yet been observed here.

### IndianAPI-primary monitoring and evidence handoff — 9 November 2026

Operators now land on IndianAPI monitoring at `/monitoring`; automatic official
source checks and their history sit in a separate expandable section. The TCS and
Infosys official adapters were paused in Supabase at the user's request. Their
IndianAPI schedules remain enabled with the existing five-attempt rolling-day
limit. No extra provider requests were made for this change. Pausing a source
adapter does not erase failures or mark evidence as checked.

Select an import observation and choose **Prepare evidence review**. Migration
013 stores an immutable operator-only link to its initial research draft.
Preparation is idempotent and rejects an unlinked superseded observation. Checked
reference documents, or an existing company evidence template, provide the draft
structure; all comparison values start blank. The provider's exact before/after
values, periods and definition differences stay in a private context panel beside
the evidence workbench. They are never cited as official reported values.

The `importReview` URL opens the linked work and follows immutable revisions.
Saved edits remain unapproved. Approval/publication reject a linked observation
that has been superseded or a draft with the wrong reporting period, as well as
the existing incomplete-evidence/source-check failures. Publishing exposes only
the reviewed research through the existing follower inbox. It does not fabricate
an import-triage action. The import queue links to the current draft/publication.

Use **Edit evidence** to attach the supporting documents, complete verified values
and dates, save a revision, and recheck exact sources before personal approval.
For a company with neither a checked reference nor any evidence template, its
first official results package must still be ingested. Automatic document access
and 25-company discovery are not implemented by this handoff.

Validation: 89 unit tests; isolated handoff, import, review, research-edit and
research-publication integration checks; TypeScript and production build. The
handoff suite verifies concurrent idempotency, reader isolation, blank values,
stale observations, wrong periods, revision lineage and one-time follower delivery
after synthetic approval/publication. A disposable browser test verified the
monitoring layout, exact handoff, revision/reload and 390px layout. Migration 013
is applied to Supabase; no real financial approval or publication was performed.
Run `npm run test:import-handoff` against the empty isolated local database.
