# AlphaTrace — free research companion beta

Status: revised Step 1 complete, 2026-11-01. Specification only; these features are not implemented. Supersedes the PDF-upload-first scope dated 2026-10-31. The historical [baseline audit](baseline_audit.md) remains unchanged.

Launch scope locked by the user on 2026-11-01: 25 covered companies, five invited pilot users, INR 2,000 total launch budget, and a free Next.js web product. Fifty-company coverage is a post-launch expansion option, not a release requirement. Build a configurable catalog without hard-coding the launch count into the application.

## Product promise and first user

Follow the companies you own, keep your investment research organized, and see which developments deserve another look.

Start with approximately five invited individual investors in Indian equities. They can add a watchlist without writing a thesis. The initial website supports 25 selected companies, with maintained research published by the operator. This is a free beta with no billing, paid tiers, or investment-return promises.

The differentiator hypothesis is continuity: new evidence → personal assumption → inspectable review → saved decision. Citations and summaries alone are not unique. Repeat use and reduced research effort must be established during the pilot.

## Complete user journey

1. Explore a clearly labeled, precomputed sample workspace without credentials or an AI key.
2. Sign in and add companies, or preview and confirm a holdings CSV import.
3. See published research for supported companies immediately. Unsupported companies can be recorded as watchlist-only; never imply research coverage for them.
4. Read a dated company update with source evidence and previous-period comparisons.
5. Optionally save a note or confirm an AI-suggested assumption. Manual entry always works.
6. When an operator publishes a new update, receive a relevant item in the in-app review inbox. General company updates work without personal assumptions.
7. Inspect previous/new evidence and ask a limited, document-grounded follow-up question.
8. Accept, reject, annotate, defer, or explicitly revise an assumption. Keep original findings and previous revisions.
9. Return later to unresolved reviews, new updates, and preserved decisions. Export or delete personal data through settings.

Users maintain their holdings after purchases/sales. The product does not infer a sale from an absent CSV row or automatically overwrite confirmed beliefs. There is no routine user PDF-upload requirement.

## Screens and acceptance

| Screen | User outcome | Required states |
| --- | --- | --- |
| Landing/sample | Understand and try the workflow | Precomputed demo, sign-in, invitation required |
| Onboarding | Follow a supported company quickly | Empty watchlist, imported preview, unknown/ambiguous symbol |
| Workspace | See watchlist, holdings and review counts | Covered, planned, watchlist-only, no new updates |
| Company | Read published evidence, save notes/assumptions | Reporting period, last source check, last publication, source unavailable |
| Review inbox/detail | Inspect changes and record a response | Unread, reviewed, deferred, superseded/corrected, processing unavailable |
| History/settings | Recover reasoning and control data | Revisions, export, deletion, quota status |
| Admin | Validate and publish shared updates | Draft, needs correction, approved, published, withdrawn |

Company pages are research pages, not live-price dashboards. Holdings quantity and optional average purchase cost do not imply current valuation, tax reporting, or performance tracking.

## Scope decisions

Included: responsive Next.js website; invited accounts; watchlist; manual/CSV holdings; 25-company research coverage; curated published updates; original-source links and permitted excerpts; notes and confirmed assumptions; evidence-backed inbox; limited follow-up questions; review history; export; admin publishing; no-key demo.

Deferred: broker linking; live prices; automatic market-wide filing ingestion; continuous monitoring; unrestricted AI; end-user arbitrary PDF uploads; OCR; arbitrary spreadsheets; valuation/scenario calculators; collaboration; push/email notifications; trading; and investment-performance attribution. Email digests may be considered after the measured pilot budget allows them, not as a launch dependency.

Existing Streamlit, HMM, signals, backtests, and PPO tools stay separate in the legacy Research Lab. They do not assess document-based investment assumptions.

## Shared and private data

| Shared research | Private per workspace |
| --- | --- |
| Company identifiers and coverage status | Watchlist and quantities/purchase costs |
| Official source metadata and permitted document material | Notes and thesis revisions |
| Checked financial observations and published company updates | Confirmed assumptions and personal findings |
| Versioned citations and public corrections | Questions, answers, dispositions, deferred reviews and exports |

Shared research caches must never contain private prompts or notes. Private inference cache keys must include workspace and relevant revision IDs. Unpublished drafts and raw admin ingestion assets are not publicly readable. Model credentials and database privileged credentials stay server-side.

## Assumption and finding contracts

An assumption stores user-confirmed wording, optional baseline/metric, expected direction, time horizon and review condition. Unknown values remain unknown. Suggestions are never treated as a user's belief before confirmation.

A personal finding references a published update version, confirmed assumption revision, evidence locations, interpretation and limitations. Assessments: `supports`, `challenges`, `mixed`, `insufficient_evidence`. A general company update is not a personal finding and must not invent an assumption for the user.

User disposition is separate: `unreviewed`, `accepted`, `rejected`, `deferred`. A deferred finding has an optional revisit date. Accepting a finding does not change a thesis; revising creates a new revision. All review events carry timestamps and reference the version actually reviewed.

Preserve periods, currency, units, consolidation basis, reported-versus-guided status, document URL, page/section, fingerprint and publication date when known. Compare like with like; distinguish QoQ from YoY and quarter from year-to-date. Missing evidence is not a failed thesis. Use code for arithmetic, not model-generated calculations.

## Coverage and operations

The [beta operating plan](beta_operating_plan.md) defines the 25-company roster, official source entry points, publication cadence, numerical limits and spending controls. Companies are not marked covered until two comparable periods and their citations are checked. Publication is human-reviewed; no real-time or continuous-monitoring claim.

Show last successful source check separately from reporting period and last published update. A failed check cannot produce an “all clear.” Late or corrected material produces a versioned update and visibly preserves the prior review history.

## Trust and release gates

- Two-user isolation tests cover direct record access, exports, private answers and privileged publishing endpoints.
- Non-insufficient findings have valid evidence locations and traceable supporting text. Follow-up answers abstain when sources are insufficient.
- Documents are untrusted data, never instructions to change application behavior or execute actions.
- Users see external-model processing disclosure before private text is sent. Holdings/cost basis are not sent to the model when irrelevant.
- Data deletion removes personal records, stored prompts/answers and private caches; backup retention and expiry are disclosed.
- CSV import previews before mutation; ambiguous identifiers require resolution; repeated import cannot silently duplicate positions.
- Publish/retry operations are idempotent. A correction creates a new version, not a silent rewrite.
- Fixtures cover unit/period/basis mismatch, missing disclosures, mixed evidence, malicious source instructions and retrospective leakage.
- AI budget exhaustion preserves read access, notes and review actions; displays a clear unavailable state for new inference.
- Five invited users can independently finish onboarding and review; record time, relevance, corrections and repeat usage without collecting private content unnecessarily.

## Step 1 completion and remaining decisions

Scope, journey, screen map, initial coverage, cadence, private/shared boundaries, usage limits and budget are now recorded. The expanded 25-company source roster is recorded, with activation in 10/10/5 batches and bank-specific validation. Source discovery is complete at the reporting-page level; actual document extraction, rights review and company activation belong to Step 5. No fresh runtime audit or application implementation was performed in this revised Step 1.

Step 2 chooses compatible pinned dependencies and an eligible hosting/auth setup. Open-source license selection and production service terms must be resolved before release. No paid accounts or services have been provisioned.
