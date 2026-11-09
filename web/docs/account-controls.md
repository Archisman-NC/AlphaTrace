# Account and data controls

Settings (`/settings`) includes the invited account email, password change, a complete private-data JSON export, account closure and existing email preferences. Email address changes remain operator-managed in this invited beta.

## Passwords and sessions

`POST /api/account/password` requires an invited bearer identity, the current password and a matching 12–128 character new password. The server verifies the bearer identity with Supabase, then reauthenticates the same account with its existing email/password. It never accepts a client-supplied owner or email. Passwords are used only for this request, never persisted or logged. Temporary verification sessions are signed out. A successful update requests global session sign-out and sends the browser back to sign-in. Existing access tokens may remain valid until expiry; this does not promise immediate invalidation of every token.

Expired sessions produce a sign-in action. A network failure during password change does not establish whether the password changed: the UI asks the user to try the new password before repeating. Existing recovery links and the “Forgot your password?” flow remain available; live email delivery and expired-link recovery still require verification after sender setup.

## Personal export

`GET /api/account/export` returns a streamed JSON attachment with an Auth email/creation/confirmation profile plus a transaction-consistent private database snapshot: existing journal, onboarding, email preferences and delivery history, issue reports and all their events, and referenced publications/corrections. Auth profile retrieval is separately verified and is not in the PostgreSQL snapshot. Operator accounts still export only their own private reports. Passwords, session tokens, unsubscribe capabilities, internal commands and shared operator work are excluded. Each category is capped at 5,000 rows, database content at 10 MB, and formatted output at 20 MB; oversize exports fail explicitly instead of returning a partial file. The existing readable journal export is unchanged. Neither format is a restore/import interface.

## Account closure

`DELETE /api/account` requires verified invited identity, current-password reauthentication, exact `DELETE MY ACCOUNT` text and explicit acknowledgement. The server refuses active research operators, the configured monitoring identity and accounts with document files awaiting cleanup. Those users need an operator handover and cleanup first. No application setting can delete another user's account.

Migration `018_account_controls.sql` introduces a closure receipt and narrow database functions. Closure runs under an exclusive account lock, while normal application transactions hold a shared account lock. It records the receipt and deletes the private workspace in one transaction. Cascades remove portfolios, theses/revisions, inbox items/events, issue reports/events, onboarding and email data. Linked email webhook receipts are deleted first; already-cleaned operator document jobs can also be removed. Shared research and its operator audit trail remain. A database trigger prevents workspace recreation, including from an older worker deployment. The central database accessor and verified API identity check block closed accounts.

After private removal, the server calls Supabase Admin `deleteUser(id, false)` using the existing server-only secret. A successful deletion marks the receipt complete. Failure leaves access blocked and the receipt pending. The Production-only Trigger.dev task `alphatrace-account-cleanup` retries hourly (with a scheduling window), claims up to 20 pending IDs, and treats a confirmed `user_not_found` response as success. A one-hour lease prevents overlapping attempts. A crash after Auth deletion but before receipt completion is safely retried. The worker reports failures without logging credentials, addresses or private content. Monitor failed task runs; a persistent provider failure needs operator attention.

The receipt retains account UUID, request/completion times and retry metadata to prevent recreation and support recovery. Account removal does not erase already-delivered emails, user downloads, provider logs or backup copies. These limits are disclosed before confirmation. Email already in flight may arrive. The scheduled email worker skips closed accounts; email delivery remains disabled while the sending domain is pending.

## Release and validation

1. Apply migration 018 using the administrative migration process before deploying code that uses the central closure gate.
2. Deploy Trigger.dev and verify the active account-cleanup schedule. Keep the personal-email schedule paused until sender readiness.
3. Deploy the website. No new environment files or credentials are required: the existing Supabase URL/publishable key/secret and restricted database connection are used.
4. Run `npm test`, `npm run test:accounts`, `npm run typecheck`, and `npm run build`. Regression checks cover database isolation, portfolios, theses, onboarding, inbox/journal, emails, issues and document jobs.
5. Browser tests cover responsive settings, export downloads, password validation/session expiry, typed confirmation, operator guards and uncertain deletion recovery. Auth mutation responses are mocked; actual private-data deletion and retry state are tested only in isolated local PostgreSQL. No real user login or production research is deleted for testing.

Supabase references: [delete user](https://supabase.com/docs/reference/javascript/auth-admin-deleteuser), [update user](https://supabase.com/docs/reference/javascript/auth-updateuser), [user management and deletion limitations](https://supabase.com/docs/guides/auth/managing-user-data).
