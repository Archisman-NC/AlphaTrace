# Beta access management

Operators open `/people` from **Service health → Manage beta access**. Ordinary accounts cannot list people or change access. This beta has five seats total, including operators. Active accounts and unexpired invitations reserve seats. Expired, revoked and closed accounts do not.

## Invite and accept

1. Enter the intended recipient's email and create an invitation.
2. Copy the link and share it privately with that person. No email is sent and no sending domain is required. Anyone holding a new-account link can activate the account for its stated email, so do not publish or forward it to others.
3. The recipient opens the link, chooses and confirms a 12–128 character password, and explicitly creates their account. Opening the page alone has no effect. Existing accounts sign in with their existing password to accept; the invitation cannot reset an existing password.
4. Successful activation opens the existing guided onboarding. New members receive no operator privileges. Email preferences remain off by default.

Links expire after three days. They are shown after creation/replacement, retained only in browser component memory, and stored only as SHA-256 hashes in PostgreSQL. The raw token travels in a URL fragment, which is removed on opening; it is posted to the activation endpoint without entering server URLs. Reopening the original link is necessary after refreshing an unfinished activation. Replacing a link invalidates the old one immediately. Successful activation consumes it.

The People screen shows email, membership status, timestamps, completed setup and whether a first response exists. It does not expose holdings, thesis notes or response content. All occupied seats remain visible, plus recent history, up to 100 records.

## Revoke and restore

Revocation requires explicit confirmation and preserves private records. It blocks new application access even for a previously signed-in session. Requests already in progress may finish before revocation commits. A revoked person's Auth login is not deleted; they can be invited back and accept using that login. Operators cannot revoke themselves or other operators here. Account closure clears the membership email and invitation token and frees the seat; a closed membership cannot be restored.

A lost invitation-save response can be retried with the same command without consuming another seat. Interrupted Auth creation reserves a stable user UUID; a retry verifies the previously chosen password instead of replacing it. If activation succeeded but the browser lost its response, sign in with the chosen password. An expired/revoked unfinished invitation can leave a blocked Auth record; reissuing the same invitation record supports recovery without granting access automatically.

## Deploy and bootstrap

1. Apply migrations `020_beta_access.sql` and `021_beta_visibility.sql` with the migration role before deploying code that calls the new functions.
2. Deploy the website and Trigger.dev workers. Existing accounts still use the environment allowlist until bootstrap. Keep public Supabase signup disabled. The existing server-only Supabase secret is needed for invitation account creation; never expose it in client configuration.
3. With the verified existing allowlist and enrolled operator in `.env.local`, run `npm run beta:bootstrap`. It verifies existing Auth accounts, imports them as active members and enables database-managed access atomically. It changes neither passwords nor research operator roles. Rerunning reports current state without reinitializing it.
4. After bootstrap, manage invitations in People; no environment edits or deployment are required for membership changes. Keep `ALPHATRACE_INVITED_USER_IDS` as the bootstrap/legacy configuration for deployment compatibility. It is no longer the live access authority. The email worker reads active members from the database and independently checks their opt-in and confirmed Auth address before delivery.

Do not roll back the application to an allowlist-only version after activating membership management: it would bypass revocations and exclude new members. Fix forward or use a version that retains the database gate. Database outages fail closed. Public unsubscribe and signed email webhook handlers use existing narrow capability functions independently of membership status.

The private-data export includes only the caller's membership metadata and excludes invitation tokens and operator audit history. Account closure retains minimal non-email access/audit records to prevent old sessions from recreating deleted data.

## Verification

- `npm test`: capability/input validation and real Supabase SDK calls against simulated Auth responses, including interrupted creation and existing-password protection.
- `npm run test:beta`: isolated PostgreSQL checks for operator boundaries, five-seat concurrency, link replacement, expiration, one-time activation, retries, revocation, restoration, private export and account closure.
- `npm run test:accounts`, `npm run test:emails`, `npm run test:onboarding`, `npm run test:reliability`: affected regressions.
- Production-build browser checks: synthetic operator/invitee flow against isolated storage, intercepted Auth, mobile sizing, retry, password confirmation and transition into guided onboarding. No external invitations are sent by these checks.
