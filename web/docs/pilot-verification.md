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

**Pending:** the second test email address, actual hosted two-account walkthrough, mailbox delivery and final live password-reset confirmation. Automated UI tests or an admin-generated recovery link do not prove inbox delivery.
