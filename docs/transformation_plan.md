# AlphaTrace — revised ten-step transformation

## Product development direction — 8 November 2026

The user wants a product people return to for timely, personal research, before introducing pricing. The core journey is: follow companies → receive relevant developments → inspect the change beside a saved assumption → record a response → revisit unresolved decisions. A scheduled runner is one component, not the complete product.

Build order from the current three-company publication pilot:

1. **Automatic delivery in the open app:** foreground checks for published updates in Overview and Inbox, connection/freshness status, recovery after interruptions, and no replacement of an unsaved response. Implemented locally on 8 November: one-minute polling of the existing authenticated inbox API, hidden/offline suspension, bounded backoff, automatic overview refresh and an explicit “Load updates” notification in the inbox. This is publication delivery, not a live market feed or faster source discovery.
2. **Cloud operation:** migrate finite source/import tasks to Trigger.dev, retain durable Supabase run history and quota enforcement, and replace continuous-worker heartbeat assumptions. Deployed and manually verified on 9 November IST: Trigger.dev Production version `20261108.2`, hourly dispatcher with a free-plan 60-minute window, verified Supabase cloud heartbeats, and local financial-data workers disabled. Both source checks recorded HTTP 403; IndianAPI correctly skipped requests at the five-attempt cap. Recurring execution and the next provider refresh still need their first scheduled observation. See `web/README.md` for deployment and rollback.
3. **Useful discovery:** Provider snapshot change detection and an operator review queue shipped on 9 November: exact value comparisons, new quarters, missing data, semantic deduplication and review/reopen history. Five saved baselines were queued; future successful imports feed the queue automatically. This does not resolve official-source HTTP 403 failures. Next, add source adapters and new-quarter/correction detection for covered companies, with visible access failures and duplicate prevention. Existing source monitoring supports TCS configured documents and Infosys SEC discovery; Wipro publication does not establish automatic Wipro discovery.
4. **Personal relevance:** make confirmed assumptions structured enough to evaluate comparable metrics and explicit thresholds, preserve original revisions and explain each match. Current matching is topic-only; never label it an investment conclusion.
5. **Investigation:** one coherent company timeline with source comparisons, corrections, saved reasoning, and evidence-grounded follow-up questions that abstain when unsupported. Model use requires separate quota and private-data disclosure work.
6. **Return workflow:** dependable deferred reviews, concise in-app briefs, onboarding that reaches a useful first review quickly, mobile usability and separately opted-in notifications when delivery infrastructure is ready.
7. **Release validation:** hosted end-to-end operation with the operator laptop off, duplicate/retry and isolation checks, measured discovery-to-publication-to-delivery latency, five independent pilot journeys and measured operating cost.

Human approval remains required for financial research publication. Faster UI refresh does not change source age, guarantee accuracy, activate 25-company coverage, or imply that all company events are monitored. No pricing, billing or revenue forecast is included in this work.

9 November follow-up: import attempts now retain their comparison outcome and exact review link, and `/imports` summarizes the latest cloud batch. Migration 012 is applied to Supabase and Trigger.dev Production `20261108.4` records cloud run identifiers. Legacy imports explicitly lack per-attempt comparison results; they are not labelled unchanged. Browser checks cover completed/older review links and mobile layouts. The Next.js website remains local and official-source access failures still need resolution.

At 02:54 IST, the queued five-company cloud refresh completed on that version: five requests succeeded, all five comparisons were unchanged, and no duplicate review items were added. Supabase retains the exact cloud run ID on each attempt. The five baseline reviews remain pending. The hourly schedule is active; its first recurring execution remains a separate observation to verify.

9 November user-directed change: IndianAPI is the primary TCS/Infosys monitoring feed. Their automatic official-source adapters are paused; supporting evidence stays in the manual review workflow. The monitoring page prioritizes provider changes. Each observation can now start an unapproved evidence template with its provider context preserved separately, and an immutable link follows revisions through publication. Approval still requires checked official evidence and matching reporting periods; provider values are not automatically mapped into reported metrics. Migration 013 is applied to Supabase. End-to-end synthetic tests verify delivery to followers once after approval/publication; no real financial review was performed in this rollout.

The original ten-step plan below remains the historical baseline; the status entries are dated and should not be read as a current feature inventory.

Updated 2026-11-03. Replaces the earlier FastAPI/always-on-worker architecture and upload-first MVP. Product: [specification](product_spec.md). Operations and budget: [beta operating plan](beta_operating_plan.md). Legacy runtime findings: [baseline audit](baseline_audit.md).

## Milestones

Launch scope locked on 2026-11-01: 25 companies, five invited pilot users, INR 2,000 total launch budget. Expansion toward 50 is deferred until after launch and subject to demand, quality, maintenance capacity and measured cost. Step 2 now connects to hosted Supabase; Step 3 has a responsive sample app and Step 4 has private portfolios verified against the hosted database. The first pilot sign-in and authenticated save/reload are verified; recovery and remaining release checks are pending.

| Step | Deliverable | Completion gate | Status |
| --- | --- | --- | --- |
| 1. Scope and coverage | Budget-fit specification, 25-company roster, journey, operating policy and release gates | Concrete scope, source entry points, private/shared boundaries and spending ceiling recorded | Complete (2026-11-01); documents not yet ingested |
| 2. Next.js foundation | TypeScript app, managed PostgreSQL/auth integration, migrations, environment examples, isolated batch script boundary | Preview/local app persists a migration-backed record; clean setup documented; privileged data not exposed | Hosted PostgreSQL and Auth endpoint connected (2026-11-02); first pilot sign-in verified in Step 4 |
| 3. Product experience | Responsive landing/demo, onboarding, workspace, company, inbox, history/settings | Complete navigable sample journey with empty/loading/error/stale states | In progress (2026-11-02): main responsive sample journey implemented; landing and remaining service-state designs pending |
| 4. Accounts and portfolios | Invited sign-in/recovery, watchlists, manual holdings, CSV preview/mapping/deduplication | Two users cannot access one another's data; repeat import is safe | Hosted persistence and two-owner isolation verified (2026-11-02); first pilot sign-in/save/reload verified; recovery and second Auth account pending |
| 5. Shared research pipeline | Source acquisition, local extraction, citations, checked comparisons and admin publication | 25-company target activated in 10/10/5 batches; two comparable periods per company; bank-specific fixtures; measured costs within cap | In progress: review/publication system implemented; TCS v1 published after pilot approval (2026-11-05); Infosys review, acquisition/maintenance checks and 23-company rollout pending |
| 6. Personal research | Notes, confirmed assumptions, optional AI suggestions and revisions | Manual path works without inference; original beliefs remain inspectable | Manual path implemented and verified locally/on Supabase (2026-11-03); optional AI suggestions deferred |
| 7. Review inbox | General updates plus personalized findings, relevance mapping and grouped reviews | Published updates yield traceable, nonduplicate items; no invented user beliefs | Implemented and verified locally/on Supabase; authenticated pilot journey passed with labelled test wording (2026-11-05) |
| 8. Investigation and decisions | Quota-limited grounded questions, accept/reject/defer/annotate, revisions, history and export | Full research-to-review workflow persists; quotas and unavailable-provider states work | Response/history and private journal export implemented and verified (2026-11-03); grounded questions and inference quotas pending |
| 9. Pilot and validation | Approximately five invited users, fixtures, privacy tests, cost and usability measurements | Critical failures fixed; independent task completion; measured spend fits limits | Pending |
| 10. Release and maintenance | Public demo, invited app, eligible hosting, monitoring, backups/restore, deletion and license | Working deployed journey, reproducible setup and maintainable publication routine | Pending |

## Budget-fit architecture

- `web/`: Next.js UI and server-side application/domain logic. No separate hosted API at launch.
- Managed PostgreSQL/auth: persistent accounts, private workspaces, public research, revisions and publication metadata, with ownership enforcement.
- Private storage: permitted source assets and processing artifacts. Public readers use approved excerpts/source links; no unrestricted raw-document bucket.
- `scripts/research/`: isolated Python batch ingestion and draft generation, run on the operator's machine. No always-on compute dependency.
- `tests/fixtures/`: clearly labeled synthetic and permission-checked source examples.
- Existing `app/` and `research/`: preserved legacy analytics and RL application.

Exact provider, Node version, authentication method and dependencies are selected and pinned in Step 2. Do not freeze the global Python environment. Verify hosting eligibility and actual auth-email requirements rather than assume every free tier supports this product.

## Publishing and personalization boundary

Local scripts acquire/check documents and write drafts through an authenticated admin path. The operator reviews and publishes an immutable update version. Database-backed pending review records are created idempotently per relevant workspace. Expensive extraction is shared; private reasoning is isolated and budgeted.

For the small pilot, private matching can run in bounded server-side batches triggered by publication or inbox visits. Persist state and cursor so interrupted runs resume. An inbox must remain readable if matching is paused; never require a permanently running Python process. Avoid loading all company documents for each user question.

## Step 2 first slice

Create the app, migrations and environment examples; persist a development workspace through server-side application logic; verify restart persistence. Establish a protected preview before auth is complete; do not expose a shared unauthenticated workspace publicly. Add CI checks appropriate to the new stack and document the local startup path.

## Gates throughout implementation

- Reference the operating plan for hard limits; no unlimited inference endpoints.
- Build ownership, citations and revision history alongside functionality rather than postponing them to Step 9.
- Run deterministic local fixtures separately from opt-in provider/network tests.
- Do not interpret legacy verification output as proof of new product behavior.
- No paid provisioning, broker credential collection or public private-data exposure during scaffolding.
- Each completed milestone records validation and remaining limitations here.

## Step 2 verification — 2026-11-01

Implemented under `web/`: pinned Next.js/React/TypeScript dependencies, PostgreSQL migration with checksum ledger, dedicated runtime role and ownership policies, server-verified Supabase bearer-token adapter, production-disabled local access, workspace form/API, liveness endpoint, CI configuration and setup instructions. Local batch-processing boundary is documented under `scripts/research/`.

Verified: TypeScript check; three unit tests; database persistence across new connections; idempotent upsert; cross-user read/update denial; owner-spoof insertion rejection; rollback and missing-identity isolation; privileged-runtime-role rejection; migration replay; production build; browser workspace creation and persistence after a PostgreSQL restart; HTTP authentication/validation/body-size checks; production rejection of local tokens and no private workspace rendered. The database was tested with a dedicated local PostgreSQL 18.3 cluster. No hosted resources or paid services were provisioned.

Limitations: Supabase hosted identity verification and database connectivity require a configured project and were not exercised; sign-in/invitations remain Step 4. CI is configured but has not run on GitHub. The current page now includes the Step 3 sample journey described below. See [web setup](../web/README.md).

## Step 3 UI build — 2026-11-02

Replaced the foundation screen with a dark finance workspace based on the supplied prototype. Implemented overview, 25-company search/watchlists, five sample research workspaces, review queue, source excerpts, assumption editing, keep/revise/defer/dismiss actions, history/export, CSV preview and settings. Mobile panels stack and the review queue scrolls horizontally. Sample state persists only in the browser; the Step 2 database/auth boundary is preserved.

Validated production compilation, six unit tests, desktop/mobile rendering and the browser review-to-history/reload path. Remaining Step 3 work: public landing experience, dedicated source loading/stale/failure previews and broader usability checks. Backend research, private account persistence, real holdings and grounded questions remain later milestones.

## Step 4 accounts and portfolios — 2026-11-02

Added `/workspace` alongside the unchanged sample demo. It includes private PostgreSQL holdings and independent watchlists, manual exact-decimal entry, CSV column mapping/preview and explicit import confirmation. Imports set absolute quantities and preserve unrelated holdings and watchlist membership. Duplicate/unsupported/invalid rows are excluded before confirmation; no broker connectivity or live valuations are implied. All mutations check the current portfolio version under a per-workspace database lock and run transactionally.

Added migration `002_portfolios.sql`, forced owner RLS for reads/inserts/updates/deletes, authenticated no-store `/api/portfolio`, bounded JSON bodies, local-only same-origin server actions and a shared 25-company roster. Production auth verifies Supabase bearer tokens and verified email, then checks an operator-controlled allowlist of at most five user IDs. Added sign-in, sign-out and password-recovery/update screens; no public signup or automatic invitations. Auth keys are limited to public publishable/anon configuration before being sent to the browser.

Validation: 12 unit tests, existing workspace DB suite, new portfolio DB suite (exact quantities, cross-user access, spoofing, repeat imports, concurrent saves, cap and rollback), idempotent migration replay, HTTP validation/auth/conflict checks and production rejection of development identity. Browser verified manual save/reload, actual CSV picker/mapping/unsupported exclusion/absolute replacement, and independent watchlist persistence; desktop and 390px mobile inspected with no document overflow or console errors. Synthetic browser fixtures were removed after verification.

Remaining at the initial local milestone: connect a configured Supabase project and email delivery, then verify invited sign-in, recovery, expiry/sign-out and two hosted accounts. Hosted connection progress is recorded below. Step 3 landing/service-state work is still outstanding; Step 5 shared research ingestion has not begun.

## Supabase connection verification — 2026-11-02

Created the free AlphaTrace project in Mumbai, disabled public signup, and connected through the session pooler using the official Supabase root certificate with hostname/certificate verification enabled. Applied migrations 001 and 002. The generated runtime role has no superuser or RLS bypass; the administrator credential is excluded from the app runtime.

Both database integration suites passed against Supabase using random owner IDs, with their fixture rows cleaned afterward: workspace persistence, exact quantities, cross-owner read/write/delete denial, spoof rejection, missing identity, repeat imports, concurrent save conflicts, watchlist cap and atomic rollback. The preview now uses hosted configuration with development identity disabled. Smoke checks confirmed email Auth availability, disabled signup, the sign-in page, and rejection of anonymous, invalid and former development tokens. TypeScript passed.

The user created the first confirmed pilot account and its UUID was enabled in the server allowlist. Chrome sign-in, authenticated portfolio load, a temporary INFY follow, owner-specific hosted persistence and full-page reload/session persistence were verified. The original empty watchlist was restored after the check. Fixed duplicate browser Auth clients during development; TypeScript, 13 unit tests and the production build passed. A second real Auth account, recovery redirects, email delivery and sign-out/expiry remain unverified. No public website or research pipeline has been deployed.


## Step 5 first slice — 2026-11-02

Implemented `/research`, coverage links in `/workspace`, exact-decimal quarter comparisons, source metadata and locations, failed/stale/changed source states, operator review/approval, transactional publication, immutable numbered versions and withdrawal history. Migration 003 is applied locally and on Supabase. The invited pilot user is enrolled as an operator. Ingestion uses a separate draft-only credential; ordinary invited users cannot see drafts or make publication writes.

The local Python batch has a period-pinned manifest for TCS and Infosys, bounded official-source retrieval, source fingerprints, deterministic extraction, cross-document checks for selected Infosys values, failure records and explicit operator-snapshot support. TCS's exceptional-item mismatch withholds its operating-profit growth calculation. Initial direct downloads returned HTTP 403; browsing-tool text snapshots were retained privately and labeled as extracted-text fingerprints, not original PDF bytes. Automatic acquisition is therefore not verified.

Current real data: two unapproved drafts, zero publications, 25 companies still planned for user-facing coverage. The other 23 companies, bank-specific positive fixtures/adapters, original PDF visual/source-rights review, two measured maintenance cycles and activation of all 25 remain required before Step 5 is complete. No scheduled task or paid inference was introduced. Personal inbox creation is Step 7, not part of this publication slice.

Validation: 18 TypeScript unit tests; five offline Python tests; local and hosted research integration; anonymous/scoped-token denial, body cap and idempotent HTTP submission; production build; isolated browser approval → publication → failed-source warning → withdrawal path. Synthetic publications were cleaned. The signed-in hosted operator browser flow has not been re-tested; use the same tab/session as the verified `/workspace` sign-in.

## Step 6 manual research — 2026-11-03

Implemented private investment cases, risks, user-confirmed structured assumptions, explicit revision reasons, immutable revision history, restore-as-draft and exact published-version evidence links inside `/research`. Notes work for all 25 companies before shared coverage exists. Blank assumption details remain unknown; nothing is sent to an AI provider. This is the manual personal-research path; automated matching and review inbox creation remain Step 7.

Migration 004 is applied locally and on Supabase. Runtime ownership policies isolate notes and revisions; API identity comes from the verified invited account. Saves use optimistic version checks and idempotent command IDs. The editor retains failed edits, supports safe retries, and blocks company switching while edits are unsaved. Evidence links reject drafts, other companies and withdrawn publications; previously saved withdrawn references remain visible.

Validation: 22 unit tests, production build, local and hosted private-research integration (two-owner isolation, spoof rejection, immutable revisions, concurrent saves, retry idempotency, source-version links, withdrawal and history pagination). Isolated browser testing verified save/reload, explicit assumption confirmation, unsaved company-switch protection and original revision wording. Test fixtures are synthetic and separate from the pilot account. Hosted database access is verified; the new editor has not yet been exercised through the pilot's authenticated browser session.

TCS/Infosys extracted figures were rechecked against official source text. Their drafts remain unpublished: original-document visual/source-rights review and the product's human publication gate are still outstanding. See the [baseline review packet](baseline_review.md).

## Step 7 review inbox — 2026-11-03

Added `/inbox` with account-specific publication catch-up, pending/later/reviewed filters, evidence comparisons and source links, original thesis/assumption snapshots, explicit topic matches, response notes and immutable decision history. Holding a company or following it is enough to receive general updates. Matching examines only the user-confirmed metric/topic field through a fixed alias list; it does not infer beliefs, evaluate thresholds, label assumptions supported/contradicted or call a model. One item groups all matching assumptions for an exact publication.

Discovery runs on opening the inbox and explicit refresh, up to 50 previously unseen published versions per transaction. Unseen-item queries plus the unique owner/publication key provide a durable resumable cursor without skipping late commits. No scheduled worker or emails. Existing items survive unfollowing. New corrected versions get separate reviews; withdrawals and newer versions are flagged without rewriting old decisions. Source checking failures, overdue checks and changed fingerprints remain visible.

Users can keep their view, dismiss, defer to 9 am IST on a chosen future date, reopen, or record an explicitly saved thesis revision. Revising follows the exact research link, saves and confirms a new private revision with that evidence attached, then returns to the same inbox item. The response snapshots that saved revision and never silently changes a thesis. Deferrals return to the pending filter when due without erasing their scheduling events.

Migration 005 is applied locally and on Supabase. Owner RLS, immutable snapshots/events, bounded APIs, version-conflict checks, idempotent response commands, saved-note protection and paginated history are implemented. Integration testing exposed an existing ordinary-reader evidence-link issue: PostgreSQL row locks invoked the operator-only update policy. Readers now use shared advisory locks coordinated with withdrawal; operator publication privileges remain unchanged. The private-thesis path uses the same fix.

Validation: 26 unit tests; production build; local and hosted inbox integration covering two-owner isolation, no private leakage, general updates, duplicate/concurrent discovery, immutable history, response retries/conflicts, defer/due behavior, explicit revision links, corrections/withdrawal, unfollow retention, 52-item catch-up pagination and 22-response history pagination. Existing local shared-research and thesis integration suites also pass. HTTP checks cover authentication, no-store, invalid bodies/media types, size caps, stale versions and hosted rejection of development tokens. Browser verification used isolated synthetic fixtures: keep → revise thesis/link evidence → record revision → defer, then reload. Desktop and 390px mobile layout inspected with no horizontal overflow or console errors. Synthetic fixtures were removed.

No real TCS/Infosys draft was published. The live pilot inbox will remain empty until followed/held companies have reviewed publications. Actual pilot-browser verification is still pending; this is a localhost app backed by hosted Supabase, not a public deployment.


## Step 8 private journal export — 2026-11-03

Added a download panel to the private portfolio and review inbox. Users can download a readable Markdown journal or structured JSON containing saved holdings/watchlist membership, every thesis revision, original review snapshots, all saved responses and the exact published evidence referenced by those records. Quantities remain decimal strings. Source locations, comparison basis, retrieval provenance, withdrawal reasons and newer-version references remain visible. Unpublished drafts, unrelated publications, other users' private data, credentials and internal request commands are excluded.

Exports use the existing invited identity and forced row-level security, within a read-only repeatable-read transaction. Concurrent edits cannot mix revision states in one file. The API returns an attachment with private/no-store caching. Local preview access remains loopback-only and requires same-origin server actions. Exports contain saved records only; unsaved edits are excluded. Files are snapshots, not an import/restore mechanism. No AI provider or paid service is called.

Bounds: at most 5,000 records per category, 10 MiB projected source rows and 20 MiB rendered output; excess data returns an explicit error and no partial download. Queries have a 15-second statement timeout. An operator-assisted full export is needed beyond these limits; asynchronous large exports remain future work.

Verified: 29 unit tests, TypeScript, production build; local and hosted database tests for complete history beyond UI pagination (52 reviews, 22 responses), two-owner privacy, empty accounts, exact decimals, historical/withdrawn publications, exclusion of drafts/internal identities, concurrent snapshot consistency, read-only enforcement and oversize failure. HTTP checks cover authentication rejection, caller-supplied owner isolation, attachment/cache headers and invalid formats. Isolated browser downloads produced readable Markdown and parseable JSON; desktop and 390px mobile layouts had no horizontal overflow or console errors. Temporary database fixtures were removed. Grounded questions, provider configuration, daily quotas and spending reservations remain pending; Step 8 is not fully complete. The real TCS/Infosys publications still await operator review.


## First human-approved publication — 2026-11-05

The pilot operator saved TCS approval with the note “reviewed personally”. Published that approved draft as version 1 (`d35fb9b6-c948-48b8-b4b2-2d890b922d36`) through the application publication checks, preserving the operator's review note/time. TCS is now published in the user-facing catalog; Infosys remains draft/planned. This does not establish automatic retrieval, independent audit, source redistribution permission or a complete 25-company baseline. The real pilot inbox/thesis walkthrough is next; no holdings or watchlist selections were changed as part of publication.


## Authenticated pilot walkthrough — 2026-11-05

Completed the first real-account browser journey against hosted Supabase and TCS publication v1. TCS was followed; a clearly labelled `WORKFLOW TEST ONLY` research entry was saved as revision 1 with the exact published evidence link. No investment assumptions were asserted. Opening the inbox created one general company update with the original research snapshot. A labelled test dismissal was saved through the browser, persisted across reload and appeared in the downloaded Markdown journal alongside the publication/source references. Reload catch-up added zero duplicate review items; the account had one reviewed item and one response at verification.

The test entry and response remain visibly labelled in the pilot account's immutable history; they are not evidence of an investment belief or user demand. This verifies authenticated save → publication discovery → response → reload → export. It does not verify assumption matching with real user beliefs, recovery/email delivery, a second real Auth account, independent usability or all review actions through the pilot browser. Infosys remains unapproved. No paid service was used.


## Product experience: overview → evidence → response — 2026-11-05

Built the private overview at `/workspace`, using real account data rather than the sample demo. It includes pending/due review counts, tracked companies, published coverage, reviewed updates, recent inbox items, deferred reviews and company-level freshness/gaps. Holdings and watchlist overlap is counted once. Portfolio editing remains at `/workspace?view=companies`. Opening/refreshing the overview catches up existing publications, with incomplete-check and failure states; it does not imply automatic source acquisition.

Reworked the inbox evidence page with both periods and source links, exact formatted values, accounting-basis details, explicit noncomparability and limitations. Added section navigation, a saved-response status header, addressable inbox filters, automatic first-item selection for nonempty views and common workspace navigation. The original thesis snapshot, immutable response history, version safeguards and manual publication gate remain in use. Corrected a response-editing guard so detail reloads cannot clear the parent's unsaved-response protection.

Verification: 32 unit tests and production build pass. The signed-in pilot journey verified two tracked companies, one published/one awaiting research, one reviewed TCS update, no duplicate items, exact-publication thesis navigation, filter selection, and response drafting/discard without writing a new decision. This slice adds no migrations or paid services. Guided evidence questions and scheduled source monitoring remain separate upcoming builds; the 25-company baseline is still incomplete.

Responsive verification for this slice: inspected overview and evidence layouts at 1280px and 390px, with no horizontal overflow; browser console reported no errors. The inbuilt browser was returned to its normal viewport on the overview.


## Source monitoring service — 2026-11-05

Implemented a local source-check worker, durable company schedules/run history, operator pause/resume/check-now controls, a monitoring screen and dashboard status. Migration 006 is applied locally and on Supabase. The worker is enabled for the two configured TCS/Infosys source sets at Monday/Thursday 18:00 IST. It rechecks pinned quarterly documents; future-release discovery and the other 23 adapters remain pending. After a machine restart, `npm run dev:monitored` starts both services; no startup daemon or always-on hosting has been provisioned.

Claims are protected by row locks and five-minute leases; crashed work is recovered, stale completions are rejected, and repeated missed slots do not create a backlog of redundant runs. Network execution is bounded to three minutes. Transient failures get at most one retry; permission/validation failures wait for the regular schedule. Completion, source-check ingestion and next scheduling are transactional. Only freshly retrieved direct source bytes can count as automated success. Publication still requires explicit human review through the existing gate.

The first live checks returned HTTP 403 for both companies. The worker recorded those failures and the next attempt, preserving the older successful-source dates and TCS publication v1. This is working monitoring infrastructure, not proof of unattended acquisition: a permitted accessible source method is still required for these two sites. No paid services or model calls were added.

Validation: 35 TypeScript unit tests, six Python tests, local monitor and research integration, production build, hosted run/timestamp/publication verification and anonymous API rejection. The live browser shows real failures and enabled schedules rather than fabricated fresh research.

Final browser verification: operator pause/resume persisted against Supabase, the resumed Infosys check recorded another real 403, and both schedules remain enabled. The combined local launcher was exercised by restarting both services; the existing sign-in and saved research survived. Monitoring and the overview were checked at 1280px and 390px without horizontal overflow or console errors. The monitor presents the last successful source-check timestamp separately from the latest attempt.

## Infosys source acquisition and discovery — 2026-11-05

The SEC route resolves unattended document acquisition for Infosys on the local runner. It discovers recent results filings, validates the newest package’s identity/period/units/table structure, and cross-checks exact quarterly facts across separate exhibits. Fresh validated evidence enters the existing unapproved review queue. Missing or changed exhibits fail visibly; no older-quarter fallback, inferred prior-quarter figures or automatic publication is allowed. Discovery persists separately from successful retrieval in migration 007 and appears on Monitoring.

TCS remains on blocked pinned company URLs. The remaining 23 companies are not activated. This is not yet a fully hands-off 25-company release. Next gates: human review of the new Infosys source originals, two scheduled maintenance cycles, a reliable TCS acquisition route, and gradual company-specific adapter expansion. No paid service or model was added.
