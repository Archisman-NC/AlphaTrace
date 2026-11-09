# Upstox financial-data evaluation

This is a local, read-only evaluation of five companies: TCS, Infosys, HCLTech,
Wipro and Hindustan Unilever. It has no database imports and cannot approve or
publish research. The existing source monitor continues unchanged.

Status on 2026-11-08: live public instrument retrieval resolved all five companies.
The 47-test TypeScript unit suite (including seven new adapter tests) and TypeScript
checking passed. No Analytics Token is configured yet, so authenticated financial
retrieval, comparison to originals and provider acceptance remain pending.

## Setup

1. Open https://account.upstox.com/developer/apps and choose Analytics → Generate
   Token. Use an **Analytics Token**, not an API secret or trading access token.
2. Save it as `UPSTOX_ANALYTICS_TOKEN=...` in `web/.env.local`. This file is
   ignored by Git; keep permissions at `0600`. Do not paste credentials into chat,
   CLI arguments or a `NEXT_PUBLIC_` variable. The web app does not load this
   separate credential file.
3. From `web/`, run `npm run upstox:benchmark` with Node 22–24. With the repository's
   pinned Node, run `./node_modules/node/bin/node --env-file-if-exists=.env.local --import tsx scripts/upstox-benchmark.ts`.

For a credential-free connectivity and company-mapping check, append
`-- --instruments` to the npm command (or `--instruments` to the direct Node command).

## What the benchmark does

- Resolves exact NSE equity ISINs from Upstox's official instrument master; rejects
  ambiguous matches rather than guessing identifiers.
- Fetches consolidated quarterly income statements including full line items,
  sequentially, with request timeouts and response-size limits. No redirects,
  automatic retries, portfolio calls or trading calls are made.
- Requests the two latest completed calendar quarters. Missing quarters remain
  missing; the benchmark never silently substitutes older quarters.
- Validates provider units, period labels, statement basis and numeric values.
  Missing metrics remain gaps, not zeroes. Bank metrics are outside this evaluation.
- Saves private, timestamped/fingerprinted responses and `report.md`/`report.json`
  under ignored `web/.local/upstox/run-*/` directories. Errors show safe fixed
  messages, not raw provider error bodies or credentials. Each run has its own folder.
- Stops subsequent authenticated requests after 401, 403 or 429. Exit code 1 means
  missing configuration, a failed request or incomplete requested-quarter coverage.
  Exit code 0 means only that retrieval/coverage succeeded, **not** financial accuracy.

## Acceptance before scheduled ingestion

For every company's three metrics and two quarters, compare against official INR
consolidated originals; retain source URLs and page references in the review.
Establish accounting standard, exceptional-item treatment, revenue versus total
income, operating-profit definition and whether net profit is attributable to
owners. Do not map the provider's `net_profit` label directly to AlphaTrace's
owner-attributable metric. Do not convert existing USD evidence to create a match.

Check freshness and restatements with repeat retrievals. Fetch timestamps are not
provider update timestamps. Confirm contractual permission for local retention and
user-facing display before production use. The first benchmark covers income
statements; balance sheet, cash flow, prices and company-specific KPIs are later
extensions. Original filing evidence and human publication review remain required.

## Official documentation checked 2026-11-08

- https://upstox.com/developer/api-documentation/analytics-token/
- https://upstox.com/developer/api-documentation/get-income-statement/
- https://upstox.com/developer/api-documentation/instruments/

Upstox documents free read-only Analytics Tokens valid for one year, with
fundamentals calls allowed without a static IP. Live account entitlement, actual
financial responses and redistribution rights still require verification.
