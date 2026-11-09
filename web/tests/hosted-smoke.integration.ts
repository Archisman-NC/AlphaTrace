// Run explicitly against the configured project and the running localhost preview.
// Does not create Auth users or send emails. Full sign-in/recovery needs a pilot.
import assert from 'node:assert/strict';
import pg from 'pg';
import { databaseConfig } from '../src/lib/database-config.mjs';

const client = new pg.Client(databaseConfig(process.env.DATABASE_URL));
await client.connect();
try {
  assert.equal((await client.query('SELECT current_user AS role')).rows[0].role, 'alphatrace_app');
  assert.equal((await client.query('SELECT * FROM alphatrace.portfolio_entries')).rowCount, 0);
} finally { await client.end(); }
const settingsResponse = await fetch(`${process.env.SUPABASE_URL}/auth/v1/settings`, {
  headers: { apikey: process.env.SUPABASE_PUBLISHABLE_KEY! }, signal: AbortSignal.timeout(10000),
});
assert.equal(settingsResponse.status, 200);
const settings = await settingsResponse.json();
assert.equal(settings.disable_signup, true);
assert.equal(settings.external.email, true);
const origin = process.env.ALPHATRACE_TEST_URL || 'http://127.0.0.1:3000';
const page = await fetch(`${origin}/workspace`);
assert.equal(page.status, 200);
const html = await page.text();
assert.ok(html.includes('A home for your research.'));
assert.ok(!html.includes('Local development identity'));
for (const token of ['', 'invalid', process.env.ALPHATRACE_PREVIOUS_DEV_TOKEN].filter(t => t !== undefined)) {
  const response = await fetch(`${origin}/api/portfolio`, { headers: token ? { Authorization: `Bearer ${token}` } : {} });
  assert.equal(response.status, 401);
  assert.equal(response.headers.get('cache-control'), 'no-store');
}
console.log('PASS: hosted runtime role, identity-free RLS, email Auth availability, closed signup, hosted sign-in page and unauthenticated token rejection. Pilot sign-in/recovery remains a separate check.');
