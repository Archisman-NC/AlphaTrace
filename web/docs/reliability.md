# Recovery screens and operational health

## User-facing recovery

The application has route-level and root-layout error boundaries plus a custom missing-page view. Next.js 16.4's `retry()` refetches and renders the failed segment. The UI never prints thrown exception text or stack traces. It reminds users that an interrupted submission may have completed and that unsaved edits may need re-entry; it does not claim a crash preserved form state.

Application API calls use `apiFetch`: reads default to a 20-second timeout unless the caller specifies another limit. Existing mutation timeouts are retained. Transport/timeout failures become `TypeError`, which existing idempotent forms treat as an unconfirmed result, preserving their command UUID. Requests are never automatically retried. Direct private-storage uploads keep their dedicated upload behavior.

A nonblocking connection notice distinguishes offline status, session rejection and failed services. It does not clear sessions, navigate, store private drafts or replay writes. Users can keep visible unfinished text before choosing to sign in. A rendered form survives a rejected API request; a genuine SDK sign-out or page crash can unmount it, so this feature does not promise recovery of unsaved data in all circumstances.

## Authentication failures

The server rejects missing/malformed bearer tokens, revoked/expired identities, unconfirmed or uninvited users, and closed accounts. Supabase transport errors, rate limits and service errors, missing authentication configuration, and database failures return a safe `503` through the shared HTTP service boundary. These are no longer incorrectly reported as expired logins. Supabase verification requests have a 10-second timeout; account-state checks and readiness reads use a 5-second statement timeout after acquiring the connection. The existing database connection timeout remains 10 seconds.

Unexpected boundary failures return safe JSON and a random reference mirrored in `X-Request-ID`. Boundary logs contain only the reference and error category, never request contents, original exception text, tokens, notes or upstream URLs. Existing expected validation/conflict responses retain their current status and message. Local development actions retain their existing guards and errors.

## Operator service health

The top of `/monitoring` provides an explicit **Check health** action. It reads saved records only: no market-data calls, document downloads, emails or job dispatches. A checked timestamp is always shown; after roughly a minute, or after a failed refresh, the earlier snapshot is marked stale. The page does not silently substitute an earlier success for a failed check.

- Website/authentication: this operator request was verified and reached the application. Password-recovery email is not tested.
- Database: restricted-role access to operational tables succeeded.
- Official and IndianAPI monitoring: enabled/queued targets, worker check-ins, latest failures, expired leases and delayed pickup. Cloud check-ins allow 150 minutes; local check-ins allow 3 minutes. Exhausting the recorded five-request quota is distinguished from worker failure.
- Documents: queued jobs older than 30 minutes, expired running leases, failures/expirations in the last day, and file cleanups more than 150 minutes past eligibility.
- Account cleanup: pending removals and those older than 150 minutes. Counts only; no account identifiers.
- Personal emails: website configuration availability, recorded failures and overdue deliveries. Worker configuration and sender readiness still require independent verification. Disabled delivery is shown as disabled, not a failed worker.
- Human review: pending import observations and those older than one day.
- Coverage: latest non-withdrawn publications, withdrawals and missing/failed/older-than-seven-days source checks. This is not proof that the latest company quarter has been discovered or published.

`GET /api/health` remains anonymous, cheap web-process liveness. Its response now explicitly says database and workers were **not checked**. `GET /api/health/readiness` requires an invited research operator. Ordinary users receive `403`; no private operational data is returned. A successful read with operational warnings returns `200` plus `health.status: attention`; inability to perform the read returns `503`. Neither response is a public launch certification.

Migration `019_operator_health.sql` adds a read-only security-definer aggregate function that explicitly checks operator membership. It returns no private notes, recipients, account IDs, provider bodies or document contents. Shared and owner-specific records remain under their existing policies. The panel is a manual operational view, not an external alerting service or live Trigger.dev status query; an empty queue does not prove a worker is running.

## Validation and release

Run `npm test`, `npm run test:reliability`, `npm run typecheck` and `npm run build`. Regression checks cover database/portfolio/thesis/onboarding, inbox/journal, emails, issues, accounts and document jobs. The production-mode browser harness verifies client, server and root-layout exceptions and retry, absence of private exception text, stale health snapshots, mobile layout, expired-session draft retention, and a committed issue whose HTTP response is lost: retrying its preserved UUID must produce one record/event.

Apply migration 019 before deploying the website. No Trigger.dev redeployment or new credentials are required. Do not create real failures, publish test research or consume provider quota to validate production. Use read-only hosted readiness checks and signed-out live-page/API checks. Password recovery email and a second real pilot account remain human release checks.
