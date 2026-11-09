import test from 'node:test';
import assert from 'node:assert/strict';
import { fetchFincrux, inspectFinancials, inspectQuarterMatrix, reserveCall } from '../scripts/lib/fincrux';

const fixture = () => ({ success: 'true', trading_symbol: 'TCS', data: { quaterly_results: [{ label: 'Synthetic test row', 'Jun 2025': '0' }], last_updated_at: '2025-07-01T10:00:00' } });
const fake = (fn: (url: string, init?: RequestInit) => Response) => ((url, init) => Promise.resolve(fn(String(url), init))) as typeof fetch;
test('observed matrix schema extracts exact strings while rejecting duplicate periods, labels and malformed numbers', () => {
  const rows = [['Category', 'Jun 2026', 'Sep 2026'], ['Sales', '1,23,456.00', '123,457.50'], ['Operating Profit', '0', '-50'], ['Raw PDF', '', '']];
  const report = inspectQuarterMatrix(rows, new Date('2026-11-08'));
  assert.equal(report.comparison[0].previous, '123456.00'); assert.equal(report.comparison[0].current, '123457.50');
  assert.equal(report.comparison[1].previous, '0'); assert.equal(report.comparison[1].current, '-50');
  assert.equal(report.comparison[2].coverage, 'missing'); assert.equal(report.sourceLinksPresent, false);
  assert.throws(() => inspectQuarterMatrix([...rows, rows[1]], new Date('2026-11-08')));
  assert.throws(() => inspectQuarterMatrix([['Category', 'Jun 2026', 'Jun 2026']], new Date('2026-11-08')));
  assert.throws(() => inspectQuarterMatrix([rows[0], ['Sales', '12,34', '123']], new Date('2026-11-08')));
  assert.throws(() => inspectQuarterMatrix([rows[0], ['Sales', '123']], new Date('2026-11-08')));
  assert.throws(() => inspectQuarterMatrix(rows, new Date('2026-07-01')));
});
test('Fincrux inventory accepts documented spellings without inventing basis or financial values', () => {
  const report = inspectFinancials(fixture(), 'TCS');
  assert.equal(report.quarterlyRowCount, 1); assert.equal(report.unit, 'unverified'); assert.equal(report.basis, 'unverified');
  assert.equal(report.providerUpdatedAtRaw, '2025-07-01T10:00:00');
  assert.deepEqual(report.quarterlyRowFields, ['label', 'Jun 2025']);
  assert.equal(inspectFinancials({ success: true, trading_symbol: 'TCS', data: { quarterly_results: [] } }, 'TCS').quarterlyRowCount, 0);
  assert.throws(() => inspectFinancials(fixture(), 'INFY'), { code: 'identity' });
  assert.throws(() => inspectFinancials({ ...fixture(), success: 'false' }, 'TCS'));
  assert.throws(() => inspectFinancials({ ...fixture(), data: { quarterly_results: [], quaterly_results: [] } }, 'TCS'));
  assert.throws(() => inspectFinancials({ ...fixture(), data: { quarterly_results: {} } }, 'TCS'));
});
test('local quota cap counts all attempts in a rolling day and rejects corrupt history', () => {
  const now = 200_000_000;
  assert.equal(reserveCall([now - 86_400_000], now).length, 1);
  assert.equal(reserveCall([now, now, now, now], now).length, 5);
  assert.throws(() => reserveCall([now, now, now, now, now], now), { code: 'quota' });
  for (const bad of [null, {}, ['100'], [NaN], [-1], [now + 1]]) assert.throws(() => reserveCall(bad, now), { code: 'ledger' });
});
test('Fincrux requests use only the documented endpoint with no redirect or forced refresh', async () => {
  const result = await fetchFincrux('TCS', 'test-key-123', fake((url, init) => {
    const parsed = new URL(url);
    assert.equal(parsed.origin, 'https://api.fincrux.org'); assert.equal(parsed.pathname, '/api/financials/TCS');
    assert.equal(parsed.searchParams.get('api_key'), 'test-key-123'); assert.equal(parsed.searchParams.has('force_update'), false);
    assert.equal(init?.method, 'GET'); assert.equal(init?.redirect, 'error');
    return Response.json(fixture());
  }));
  assert.equal(result.endpoint, 'https://api.fincrux.org/api/financials/TCS');
  assert.equal(JSON.stringify(result).includes('test-key-123'), false); assert.match(result.sha256, /^[a-f0-9]{64}$/);
  await assert.rejects(fetchFincrux('../../user', 'test-key-123'), { code: 'symbol' });
});
test('provider errors, echoed keys and unbounded responses cannot expose credentials', async () => {
  for (const status of [401, 403, 429, 500]) await assert.rejects(fetchFincrux('TCS', 'test-key-123', fake(() => new Response('test-key-123', { status }))),
    (e: Error) => !e.message.includes('test-key-123'));
  await assert.rejects(fetchFincrux('TCS', 'test-key-123', fake(() => Response.json({ echo: 'test-key-123' }))), { code: 'credential_echo' });
  await assert.rejects(fetchFincrux('TCS', 'key+123', fake(() => Response.json({ echo: 'key%2B123' }))), { code: 'credential_echo' });
  await assert.rejects(fetchFincrux('TCS', 'test-key-123', fake(() => { throw Error('test-key-123'); })), { code: 'network' });
  await assert.rejects(fetchFincrux('TCS', 'test-key-123', fake(() => new Response('x'.repeat(5 * 1024 * 1024 + 1)))), { code: 'size' });
  await assert.rejects(fetchFincrux('TCS', 'test-key-123', fake(() => new Response('<html>oops</html>'))), { code: 'schema' });
});
