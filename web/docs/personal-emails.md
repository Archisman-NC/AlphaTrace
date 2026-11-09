# Personal update emails

`/settings` contains opt-in research digests, optional deferred-review reminders and recent delivery records. Onboarding links here after the final step. Existing accounts are never enrolled automatically.

## Behavior

- The Production Trigger task `alphatrace-personal-update-emails` checks hourly. New digests are considered from 9 am IST, at most one per account per IST calendar day. Empty days create no message. Retry work may run outside that window.
- Only the latest non-withdrawn published comparison for a currently tracked company is eligible. Publication emails start from the latest opt-in date. Already reviewed or deferred publications do not generate a new-publication email.
- A due deferral generates one reminder per review version. Reminders can be disabled separately.
- Emails contain company, publication title, quarter and review links. They exclude private thesis notes, assumption matches and holding quantities. The worker rechecks the verified Supabase Auth address; it never uses a client-supplied recipient.
- Payloads are frozen in an owner-isolated outbox before sending. A stable Resend idempotency key survives worker/database retries. Attempts stop after five claims or 23 hours from the first claim. Ambiguous results become **Unconfirmed**, not “Delivered.” Do not automatically resend after the provider's 24-hour key retention expires.
- Preference changes, unsubscribe, withdrawals, superseding publications, removed companies and changed recipient addresses cancel queued messages. An in-flight message cannot be recalled. Cancelled item keys remain recorded; later publications or deferral revisions are new events.
- Acceptance and delivery are distinct. A signed delivery callback confirms receiving-server acceptance, not readership. Bounce/complaint/suppression callbacks disable further delivery. A suppressed account needs operator investigation before re-enabling.

## Configuration

Keep one private `web/.env.local` and the tracked `.env.example`.

```dotenv
ALPHATRACE_EMAILS_ENABLED=false
ALPHATRACE_APP_URL=https://alpha-trace-beryl.vercel.app
RESEND_API_KEY=
RESEND_FROM_EMAIL=onboarding@resend.dev
RESEND_WEBHOOK_SECRET=
```

The shared Resend sender is development-only and explicitly rejected by the production availability gate. A verified domain is required for other recipients; the Vercel subdomain cannot supply sender DNS ownership. No domain purchase is needed for development or Resend's simulated-recipient tests.

1. Apply migration `016_personal_emails.sql`.
2. Verify a domain in Resend and set a plain sender address, such as `research@updates.example.com`.
3. Configure `https://alpha-trace-beryl.vercel.app/api/email/webhook` for `email.delivered`, `email.bounced`, `email.complained`, `email.failed`, and `email.suppressed`. Save its signing secret as `RESEND_WEBHOOK_SECRET`.
4. Configure the named email runtime values on Vercel and Trigger.dev. The existing Supabase secret additionally supports read-only Auth account lookup by the worker. The restricted database role stores email records; the database admin password is never deployed.
5. Set `ALPHATRACE_EMAILS_ENABLED=true` only after the sender, API key, HTTPS origin, webhook, account lookup and pilot allowlist are ready. Deployment helpers reject incomplete enabled configurations. Disabled configurations export only the false flag and omit new email credentials.
6. Deploy the website and `npm run trigger:deploy`, then verify the schedule. Have the pilot opt in from Settings. Actual inbox delivery and a live callback remain release checks; fake-provider tests do not prove mailbox delivery.

Setup creates no opt-ins or publications and selects no paid plan. This implementation does not consume open/click events; check the Resend domain's tracking configuration separately.

## Security and verification

Emails include a visible unsubscribe link and RFC 8058 headers. GET only displays confirmation. POST uses a random 32-byte possession token whose hash is indexed in storage. It can disable emails but cannot read private data or subscribe anyone. The handler's fixed internal capability identity has no private records; a narrow database function performs the token-scoped disable.

The webhook verifies original request bytes with Svix before parsing JSON. A narrow function updates known provider IDs; raw bodies and addresses are not logged. Early callbacks for uncommitted IDs receive 503 for retry. Event IDs deduplicate callbacks, and late delivery events cannot clear a bounce or complaint.

Settings require invited-account authentication. User tables have forced owner RLS. Webhook event storage has no direct application-role access. The worker uses the server-owned pilot allowlist and does not log recipients.

Run `npm test`, `npm run test:emails`, and `npm run build`. Tests cover explicit consent, escaping, configuration gates, signatures, isolation, concurrent sends, historical cutoff, quiet days, retries, unsubscribe, reminders, withdrawals, recipient changes and suppression. Integration tests use isolated local PostgreSQL and a fake sender.

References: [idempotency](https://resend.com/docs/dashboard/emails/idempotency-keys), [simulated recipients](https://resend.com/docs/dashboard/emails/send-test-emails), [verified domains](https://resend.com/docs/dashboard/domains/introduction), [unsubscribe headers](https://resend.com/docs/dashboard/emails/add-unsubscribe-to-transactional-emails), [webhook verification](https://resend.com/docs/webhooks/verify-webhooks-requests).
