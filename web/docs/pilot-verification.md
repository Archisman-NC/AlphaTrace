# Pilot verification

## Verified locally

On 9 November 2026, the onboarding database and browser checks passed: first-use setup, skip/restart, resume after refresh, companies with no published research, CSV validation and exact quantities, topic selection and handoff to a published comparison. Database tests use separate synthetic owners and verify that one cannot read or mutate another owner's setup. They do not replace a two-account production walkthrough.

Recovery browser checks run against a production Next build with intercepted synthetic Supabase responses. They cover invalid email input, rate limits, expired callback errors and rejected callback tokens while an older session exists, password confirmation, successful reset and sign-out, the return-to-sign-in message, ambiguous updates and failure to sign out after a successful password change. They send no real email and change no real password.

## Live pilot walkthrough

Use a separate email controlled by the operator, not the existing operator login. Keep passwords, bearer tokens and recovery links out of logs, screenshots and chat.

1. Create a confirmed Supabase test user only after the address is supplied. Add its UUID to the existing server-owned pilot allowlist, preserving existing IDs and the five-user limit. Grant no research operator role. Deploy the updated website environment. Keep public signups disabled.
2. In a separate browser context, sign in and confirm that the account starts with no private holdings, thesis or decisions. Verify that operator endpoints reject it.
3. Follow TCS or another company with published research. Choose topics, refresh midway, then finish and open the first comparison. Inspect the saved original links and coverage labels; opening evidence must not record a review decision.
4. Save an explicitly labelled test thesis and response. Refresh and sign out/in, then confirm persistence. Verify that the operator's private data is absent from the test account and the test records are absent from the operator's workspace. Record only pass/fail outcomes, never the other account's private contents.
5. Request one recovery email through the actual website. The mailbox owner confirms receipt and opens the newest link. Set a new password, sign in again and confirm that the same private data remains. Reusing the consumed link must show the expired-link path.
6. Record the exact deployment and completed checks. Leave the test account for the user to inspect; remove it only when requested. Never publish synthetic research to test this flow.

## Email dependency

Supabase Auth recovery email is separate from AlphaTrace's Resend update-email worker. A Resend key in the website does not configure Supabase SMTP. Supabase's default sender restricts recipients to project-team addresses and is not a production delivery setup. Configure custom SMTP with a verified sender for outside pilots, and verify the Site URL and exact `/workspace?recovery=1` redirect on the deployed origin. Keep personal update-email delivery disabled until its own sender and webhook checks pass.

References: [Supabase custom SMTP](https://supabase.com/docs/guides/auth/auth-smtp), [recovery flow](https://supabase.com/docs/reference/javascript/auth-resetpasswordforemail), [sign-out and token expiry](https://supabase.com/docs/reference/javascript/auth-signout).

## Hosted verification, 9 November 2026

The separate user-supplied test address was created as a confirmed non-operator account. The website allowlist now contains the existing operator and this pilot. Vercel deployment `dpl_3Su2WFLP5WWBqaRiW1oCQZTwJ7ZA` passed and was assigned to production, running commit `3bb489a`.

The actual hosted browser walkthrough passed: fresh empty workspace, TCS selection, topic preferences, refresh/resume, mobile layout, first published comparison without an automatic review decision, explicitly labelled test thesis and response, and persistence after sign-out/sign-in. Operator-only readiness, imports and research endpoints returned `403`; a monitoring command was also denied. Hosted row-level-security checks prevented reads of the other account's private rows in both directions, and the operator's private portfolio/research/onboarding/review fingerprint remained unchanged. This was one real pilot browser session plus database isolation checks; it does not claim a second simultaneous operator browser session.

No research was published and no market-data request was made. Generated test credentials are in the private, ignored `web/.local/pilot-login.txt`, outside version control. The test account and its clearly labelled private records remain available for inspection.

**Live recovery verified:** the initial check found that Supabase redirected recovery to `http://localhost:3000/`. The operator corrected the Auth Site URL and allowed `https://alpha-trace-beryl.vercel.app/workspace?recovery=1`. The next completed live walkthrough verified the deployed reset form using a real Supabase recovery link for the test account: password update, global sign-out, rejection of the old password, successful sign-in with the new password, persistence of the saved test response, and rejection of the consumed link including after refresh. Only the test account's password changed. The private login file contains its current generated password.

One recovery email was then requested through the actual website for that test address. Supabase returned HTTP `200`. **Mailbox delivery confirmed on 9 November 2026:** the user reported receipt and supplied a screenshot showing the password-reset email from Supabase Auth in Gmail's Inbox. This closes the pilot's recovery-delivery check for this address; it does not establish deliverability to every recipient. The reset flow itself was verified separately with a real Supabase recovery link as described above. Personal update emails remain disabled while their verified sending domain is pending.
