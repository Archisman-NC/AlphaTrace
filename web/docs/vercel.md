# Vercel deployment

Deploy the existing Next.js app with **Root Directory `web`**, framework Next.js and Node 22. `vercel.json` selects Mumbai (`bom1`) and runs configuration preflight before the production build. Keep the Vercel project restricted to this invited beta. Vercel Hobby permits personal/noncommercial use; confirm eligibility for the actual project before selecting it. [Vercel Hobby documentation](https://vercel.com/docs/plans/hobby).

## Configuration

There is one private local file, `.env.local`, and one committed placeholder file, `.env.example`. Do not upload the entire local file: it includes setup credentials and isolated test settings. `scripts/lib/vercel-environment.ts` defines the exact Production allowlist:

- `DATABASE_URL` using the restricted `alphatrace_app` role
- `DATABASE_SSL=true`, `DATABASE_SSL_CA_PEM` with the complete trusted certificate
- `SUPABASE_URL`, `SUPABASE_PUBLISHABLE_KEY`
- `SUPABASE_SECRET_KEY` (server-only private storage access)
- `TRIGGER_SECRET_KEY` (Production `tr_prod_…` key)
- `ALPHATRACE_INGEST_USER_ID`, `ALPHATRACE_INVITED_USER_IDS`
- `ALPHATRACE_LOCAL_DEV=false`
- `ALPHATRACE_CLOUD_MONITORING=true`
- `ALPHATRACE_DOCUMENT_JOBS=true`

Configure these only for Vercel **Production**, not Preview. Preview builds intentionally fail the preflight without a separately provisioned environment. Never add `NEXT_PUBLIC_` prefixes to secrets. Never upload `SUPABASE_DB_PASSWORD`, `MIGRATION_DATABASE_URL`, `TEST_*`, `TRIGGER_DEV_SECRET_KEY`, provider keys or the laptop CA path to Vercel. Trigger.dev has its own explicit seven-setting worker allowlist. Development commands select `TRIGGER_DEV_SECRET_KEY` from the same local file.

## Prepare and release

1. Run `npm run typecheck`, `npm test`, `npm run test:documents`, `npm run test:research-edit`, and `npm run build`. Local integration tests use only loopback `TEST_*` settings.
2. Apply migrations through `019_operator_health.sql` with the administrative migration process, never the application connection. Keep prior checksums intact.
3. Run `npm run storage:setup` to create/configure the private `alphatrace-evidence-jobs` bucket. It limits objects to 20 MB and adds no public/anonymous access policies.
4. Deploy `npm run trigger:deploy`; verify the active `alphatrace-account-cleanup` schedule and keep personal emails paused until sender readiness. Also verify `alphatrace-process-document` plus the Production-only `alphatrace-clean-document-uploads` schedule. The worker packages Python and Poppler; Vercel does not need them.
5. Run `npm run vercel:check`. Connect the repository to Vercel with Root Directory `web`, or deploy that directory using the CLI. Push the prepared app to GitHub before importing the repository into Vercel. Keep unrelated root artifacts out of the deployment.
6. Add the allowlisted Production settings, deploy, then set the exact public origin as Supabase Auth Site URL and allow its password-recovery return URL `/workspace?recovery=1`. Keep public signups disabled for this invited beta.
7. Verify `/api/health` (web liveness only), operator `/api/health/readiness`, recovery screens, sign-in, workspace persistence, research, inbox, direct document upload, completion after refresh, journal downloads, issue submission/history/operator review and password recovery on the final HTTPS origin. Confirm two-user isolation before wider invitations. No website is considered released merely because its build passed.

## Document processing

PDF/HTML attachments and the supported TCS workbook import upload directly from the browser to private Supabase Storage through expiring signed URLs. The website accepts only a small metadata command, so files do not cross Vercel's 4.5 MB function request limit. The authenticated API checks operator membership and job ownership; it triggers a queued task with an idempotency key. The worker checks exact byte size and SHA-256 before inspecting the document. [Vercel function limits](https://vercel.com/docs/functions/limitations).

A job can be resumed from the same browser tab after refresh. Incomplete uploads require reselecting the same files. The app stores only operation IDs and input fingerprints in sessionStorage, never storage tokens or document contents. Discarding the local result does not cancel a running operation. If a worker terminates after saving a draft/receipt, inspect the review desk before repeating the operation; no automatic publication occurs.

The shared cap is 10 document operations per rolling 24 hours, separate from IndianAPI's five requests. Each file is limited to 20 MB, with up to three files for TCS. Upload submission expires after one hour. Running claims expire after ten minutes; task execution is bounded to five minutes. Signed upload URLs last two hours, so cleanup starts only after four hours and runs hourly with a scheduling window. Failed cleanup retries on the next scheduled pass; monitor cleanup failures because retention can extend during outages. Extractor temporary caches are removed when processing ends.

JSON/Markdown journal downloads use streamed responses. Ordinary research revision saves stay synchronous. File processing and rechecks prepare evidence only; human review and publication remain explicit actions.

To keep using the local Python path before the worker is verified, leave `ALPHATRACE_DOCUMENT_JOBS=false` locally. Vercel always enables the cloud path. Set the local flag true only after the cloud task is ready. The monitoring schedule and existing published research are independent of website deployment.

## Verified preparation — 9 November 2026

- Migration 014 applied locally and to hosted Supabase; private bucket configured.
- Trigger.dev Production version `20261109.1` deployed, including the two document tasks. [Deployment](https://cloud.trigger.dev/projects/v3/proj_azibgotclaclvmjiifqv/deployments/k2j44ovp).
- Live synthetic upload/worker test `run_06gruuf9p2uhslj28b9q42p8a2` passed: signed upload, public-read denial, exact hash, unapproved result and active cleanup schedule. Test data was removed. This verifies the HTML attachment path; a real TCS workbook import on the final website remains a release check.
- 92 unit tests, 17 Python tests, document-job and evidence-edit integration tests, TypeScript and the production Next.js build passed. A browser test with mocked cloud responses verified direct upload, progress, refresh/resume and mobile layout.
- Local cloud document mode enabled. The website is deployed at https://alpha-trace-beryl.vercel.app/. Public-page, health and signed-out API checks passed; the pilot confirmed live sign-in and saved-data loading after refresh. Password recovery and a real TCS workbook import on the final origin still need verification.
