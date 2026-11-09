# Fincrux evaluation

## Live result — 2026-11-08 IST

Five successful authenticated requests retrieved TCS, INFY, HCLTECH, WIPRO and
HINDUNILVR. All responses contain Sales, Operating Profit and Net Profit for June
and September 2026: 15 pairs / 30 values. This confirms field coverage, not accuracy.
The five-call free allowance was used; subsequent local runs reuse the snapshots.

The live quarterly format is an array of arrays with a Category/date header, under
the misspelled `quaterly_results` field. The adapter validates column lengths,
duplicate dates/labels and exact decimal strings. The API responses lack explicit
currency/unit, statement-basis and accounting-standard metadata. Every tested
quarterly Raw PDF row is empty. Update timestamps have no timezone.

### TCS acceptance finding

TCS's official September 2026 filing, physical page 17, labels its figures as
standalone INR crore. Fincrux Sales matches standalone revenue for both quarters
(June 59,553; September 60,441). Consolidated revenue on physical page 8 is
72,275 and 73,188 respectively. The retrieved series therefore cannot replace
AlphaTrace's consolidated evidence. This identifies a basis mismatch, not proof
that the standalone numbers are incorrect. The other four companies' basis and
profit definitions remain unverified.

Source: https://www.tcs.com/content/dam/tcs/pdf/discover-tcs/investor-relations/corporate-actions/2026-27/financial-results-qtr-sep-30-2026.pdf

Do not enable this provider for production research until consolidated selection,
profit definitions, original source links, currency/units and storage/display
rights are established. No paid upgrade or provider-support message was sent.

## Commands and credentials

Save `FINCRUX_API_KEY` in the Git-ignored `web/.env.local` (mode 0600).
Do not put the key in chat, command arguments or a NEXT_PUBLIC variable.
From `web/` with Node 22–24:

```sh
npm run fincrux:benchmark
npm run fincrux:benchmark -- --all
```

Default is TCS only; `--all` checks the five companies. The pinned Node equivalent:

```sh
./node_modules/node/bin/node --env-file-if-exists=.env.local --import tsx scripts/fincrux-benchmark.ts --all
```

Snapshots and reports are under ignored `web/.local/fincrux/`. Successful JSON
responses are stored before interpretation, so schema fixes can be replayed without
API calls. Exact response text and its SHA-256 are retained. Cached files are reused
indefinitely for this evaluation and must not be represented as fresh data.

A local exclusive lock prevents concurrent evaluations; a rolling 24-hour ledger
caps attempts at five, including failed requests. This does not count calls made
outside this script or establish Fincrux's reset timezone. There are no automatic
retries or force_update requests. Any error stops the batch. If interrupted, inspect
the lock and confirm the process has ended before removing that lock. Do not delete
the quota ledger to retry sooner. HTTP 401/403/429 require account/quota attention.

The API key is sent only to the fixed official HTTPS endpoint using the documented
query parameter. URLs with the key, response errors and request exceptions are
never logged. Redirects are disabled and reflected keys are rejected before storage.
The evaluator has no database, publication, portfolio or trading dependencies.

## Documentation checked

- https://fincrux.org/docs/company-financials
- https://fincrux.org/docs/api-key
- https://fincrux.org/pricing

The free Hobby plan advertises five calls/day. No subscription was purchased.
