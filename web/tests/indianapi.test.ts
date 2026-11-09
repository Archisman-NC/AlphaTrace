import test from 'node:test';
import assert from 'node:assert/strict';
import { fetchIndianStats, inspectIndianStats, indianStatsUrl, tcsRevenueCheck, tcsProfitChecks } from '../scripts/lib/indianapi';
import { officialReferenceChecks } from '../scripts/lib/indianapi-references';

const now = new Date('2026-11-08T00:00:00Z');
const fixture = () => ({ Sales: { 'Jun 2026': 100, 'Sep 2026': 120 }, 'Operating Profit': { 'Jun 2026': 0, 'Sep 2026': -5 } });
const fake = (fn: (url: string, init?: RequestInit) => Response) => ((url, init) => Promise.resolve(fn(String(url), init))) as typeof fetch;
test('official checks distinguish Wipro unit rounding from actual discrepancies', () => {
  const report = inspectIndianStats({ Sales: { 'Jun 2026': 24479, 'Sep 2026': 24753 },
    'Operating Profit': { 'Jun 2026': 4633, 'Sep 2026': 4863 }, 'Net Profit': { 'Jun 2026': 3356, 'Sep 2026': 3200 } }, 'WIPRO', now);
  const ref = officialReferenceChecks(report)!;
  assert.ok(ref.checks.every(check => check.results.every(r => r === 'rounded') && !check.directMappingAllowed));
  assert.deepEqual(ref.ownersProfitInrCrore, [3352, 3200.6]);
  report.rows[0].current = '24752.7';
  assert.equal(officialReferenceChecks(report)!.checks[0].results[1], 'exact');
  report.rows[0].current = '24752.8';
  assert.equal(officialReferenceChecks(report)!.checks[0].results[1], 'mismatch');
  report.rows[0].current = '247527';
  assert.equal(officialReferenceChecks(report)!.checks[0].results[1], 'mismatch');
});
test('HCL associate losses and owner-attributable profit cannot silently change definitions', () => {
  const report = inspectIndianStats({ Sales: { 'Jun 2026': 34579, 'Sep 2026': 35620 },
    'Operating Profit': { 'Jun 2026': 6870, 'Sep 2026': 7439 }, 'Net Profit': { 'Jun 2026': 4626, 'Sep 2026': 5132 } }, 'HCLTECH', now);
  const ref = officialReferenceChecks(report)!;
  assert.ok(ref.checks.every(check => check.results.every(r => r === 'exact')));
  assert.deepEqual(ref.operandsInSourceUnit.associateLossIncluded, [0, 38]);
  assert.deepEqual(ref.ownersProfitInrCrore, [4624, 5132]);
  report.rows[1].current = '7477'; // Excluding the associate loss is a different metric.
  assert.equal(officialReferenceChecks(report)!.checks[1].results[1], 'mismatch');
});
test('dated references never validate other periods/companies, missing data or HUL turnover as revenue', () => {
  const report = inspectIndianStats({ Sales: { 'Jun 2026': 17184, 'Sep 2026': 18071 } }, 'HINDUNILVR', now);
  const ref = officialReferenceChecks(report)!;
  assert.deepEqual(ref.checks[0].results, ['mismatch', 'mismatch']);
  assert.deepEqual(ref.checks[1].results, ['missing', 'missing']);
  assert.equal(officialReferenceChecks({ ...report, requestedSymbol: 'UNKNOWN' }), null);
  assert.equal(officialReferenceChecks({ ...report, periods: ['2026-09-30', '2026-12-31'] }), null);
  const infy = officialReferenceChecks({ ...report, requestedSymbol: 'INFY' })!;
  assert.deepEqual(infy.checks[1].expectedInrCrore, [11409, 11689]);
  assert.deepEqual(infy.ownersProfitInrCrore, [7769, 7823]);
});
test('quarterly parser preserves missing data and leaves identity, basis and definitions unverified', () => {
  const report = inspectIndianStats(fixture(), 'TCS', now);
  assert.equal(report.rows[1].previous, '0'); assert.equal(report.rows[1].current, '-5');
  assert.equal(report.rows[2].coverage, 'missing'); assert.equal(report.rows[2].current, null);
  assert.equal(report.basis, 'unverified'); assert.equal(report.identityCheck, 'pending');
  for (const value of [true, '12%', '12,34', Infinity, NaN]) assert.throws(() => inspectIndianStats({ Sales: { 'Jun 2026': value } }, 'TCS', now));
  for (const period of ['TTM', 'Feb 2026', 'Dec 2026']) assert.throws(() => inspectIndianStats({ Sales: { [period]: 10 } }, 'TCS', now));
  assert.throws(() => inspectIndianStats({ message: 'error' }, 'TCS', now));
  assert.throws(() => inspectIndianStats({ Sales: [] }, 'TCS', now));
});
test('TCS reference compares only the documented quarters and never approves a financial series', () => {
  const reference = (a: unknown, b: unknown, symbol = 'TCS') => tcsRevenueCheck(inspectIndianStats({ Sales: { 'Jun 2026': a, 'Sep 2026': b } }, symbol, now));
  assert.equal(reference('59,553', '60,441').result, 'matches_standalone_revenue');
  assert.equal(reference(72275, 73188).result, 'matches_consolidated_revenue');
  assert.equal(reference(1, 2).result, 'matches_neither_reference');
  assert.equal(reference(null, 73188).result, 'not_checked');
  assert.equal(reference(72275, 73188, 'INFY').result, 'not_checked');
  const otherPeriod = inspectIndianStats({ Sales: { 'Jun 2026': 72275, 'Sep 2026': 73188 } }, 'TCS', new Date('2027-01-01'));
  assert.equal(tcsRevenueCheck(otherPeriod).result, 'not_checked');
});
test('free-tier request fixes the host, keeps keys in headers and disables redirects', async () => {
  const result = await fetchIndianStats('TCS', 'private-test-key', fake((url, init) => {
    assert.equal(url, 'https://stock.indianapi.in/historical_stats?stock_name=TCS&stats=quarter_results');
    assert.equal(init?.redirect, 'error'); assert.equal(init?.method, 'GET');
    assert.equal((init?.headers as Record<string, string>)['X-Api-Key'], 'private-test-key');
    return Response.json(fixture());
  }));
  assert.equal(JSON.stringify(result).includes('private-test-key'), false);
  assert.match(result.sha256, /^[a-f0-9]{64}$/);
  assert.throws(() => indianStatsUrl('TCS&redirect=evil'), { code: 'symbol' });
});
test('TCS total profit and operating-profit labels never silently become AlphaTrace reported metrics', () => {
  const report = inspectIndianStats({ 'Net Profit': { 'Jun 2026': 13420, 'Sep 2026': 13934 }, 'Operating Profit': { 'Jun 2026': 18556, 'Sep 2026': 18815 } }, 'TCS', now);
  const checks = tcsProfitChecks(report);
  assert.equal(checks[0].result, 'matches_segment_results_before_unallocable_expenses');
  assert.equal(checks[1].result, 'matches_total_profit_including_non_controlling_interests');
  assert.ok(checks.every(c => !c.directMappingAllowed));
  assert.equal(tcsProfitChecks({ ...report, requestedSymbol: 'INFY' }).length, 0);
  assert.equal(tcsProfitChecks({ ...report, periods: ['2026-09-30', '2026-12-31'] }).length, 0);
  assert.equal(tcsProfitChecks(inspectIndianStats({ Sales: {} }, 'TCS', now))[0].result, 'missing');
});
test('HTTP errors, oversized bodies and echoed or thrown keys are handled without credential disclosure', async () => {
  for (const status of [401, 403, 429, 500]) await assert.rejects(fetchIndianStats('TCS', 'private-test-key', fake(() => new Response('private-test-key', { status }))),
    (e: Error) => !e.message.includes('private-test-key'));
  await assert.rejects(fetchIndianStats('TCS', 'private-test-key', fake(() => Response.json({ echo: 'private-test-key' }))), { code: 'credential_echo' });
  await assert.rejects(fetchIndianStats('TCS', 'private-test-key', fake(() => { throw Error('private-test-key'); })), { code: 'network' });
  await assert.rejects(fetchIndianStats('TCS', 'private-test-key', fake(() => new Response('x'.repeat(5 * 1024 * 1024 + 1)))), { code: 'size' });
  await assert.rejects(fetchIndianStats('TCS', 'private-test-key', fake(() => new Response('invalid'))), { code: 'schema' });
});
