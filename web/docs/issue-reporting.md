# Research issue reporting

Invited users can report an entire published comparison or a specific metric from Research or the Review inbox. Categories are incorrect value, wrong quarter/units, source link problem and unclear explanation. The form attaches the stored publication ID/version, metric, quarter, source metadata and the selected comparison. Client-supplied ownership or evidence snapshots are ignored. Drafts cannot be reported through this flow.

`/issues` shows the author's private history and responses. Research operators can switch to the operator queue. Reports begin Open, can move to In review, then Resolved. Operators may reopen a resolved report with a reason. Active duplicate reports for the same author/publication/metric/category link to the existing record; the second note is not added. A resolved report does not prevent a later new concern.

Resolution choices:

- **Evidence explained:** record what was checked and why the original evidence is correct.
- **Published correction:** select a newer, non-withdrawn publication for the same company and explain how it addresses the concern. The UI asks the operator to inspect and confirm the selected publication. Create, review and publish corrections separately in the existing research desk.
- **Publication withdrawn:** first withdraw the unreliable publication in the research desk; then explain that outcome here.

No report action edits, approves, publishes or withdraws research. Resolution is a recorded operator response, not a new automated verification of the evidence. Later withdrawal of a linked correction is visibly flagged; the original resolution history is retained. The queue requires an operator to check it; no email or background notification is sent by this feature.

## Storage and access

Apply migration `017_research_issues.sql` before deployment. No new credentials, providers or scheduled tasks are needed. `GET/POST /api/issues` require a verified invited identity. Local server actions retain the loopback-only development guard and same-origin check.

Forced row-level security allows authors to read only their own reports and research operators to read the queue and update status. Publication attachment and initial note are immutable. A database trigger derives the evidence context and appends events; the application role cannot edit events or delete reports. Operator identity comes from the transaction's verified account context. Private notes are rendered as text and are never included in shared research.

Commands use actor-bound UUIDs and normalized payloads for safe retries. Conflicting versions require reload; unconfirmed requests retain the same command in the mounted form. Closing the page discards the in-memory retry state, so inspect report history before resubmitting after navigation. Duplicate checks and a 20-new-report rolling 24-hour limit per account protect the beta queue. Lists use 25-row timestamp/UUID cursor pages, and the detail shows the latest 50 audit events; older events remain stored. Correction selection lists the latest 25 eligible newer publications.

## Validation

Run `npm test`, `npm run typecheck`, `npm run test:issues`, and `npm run build`. Integration tests must use isolated local `TEST_*` storage. They verify two-user isolation, operator-only status changes, immutable context/history, retry/duplicate handling, concurrent updates, correction eligibility, withdrawal, reopening conflicts, rate limits, pagination and persistence after reconnect.

Browser verification uses synthetic publications in isolated local storage: report the Revenue metric, save and reload, start review, discard an unsaved response, resolve with an explanation, reopen, link a published correction, check duplicate feedback, and verify desktop/mobile layouts. Never generate or resolve synthetic reports against real user publications just to test production.
