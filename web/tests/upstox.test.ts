import test from 'node:test';
import assert from 'node:assert/strict';
import { gzipSync } from 'node:zlib';
import { selectInstruments, parseIncomeStatement, quarterEnd, quarterPair, comparisonRows, fetchIncome, fetchInstruments, UpstoxError } from '../scripts/lib/upstox';

const instrument = { symbol: 'TCS', isin: 'INE467B01029', instrumentKey: 'NSE_EQ|INE467B01029' };
const master = [{ trading_symbol: instrument.symbol, isin: instrument.isin, instrument_key: instrument.instrumentKey, segment: 'NSE_EQ', instrument_type: 'EQ' }];
// Synthetic test values, deliberately not company financial evidence.
const fixture = () => ({ status: 'success', data: { type: 'consolidated', time_period: 'quarterly', units_in: 'crore', income_statement: [
  { category: 'revenue', history: [{ period: 'Sep 2025', value: 110 }, { period: 'Jun 2025', value: 100 }] },
  { category: 'operating_profit', history: [{ period: 'Jun 2025', value: 0 }, { period: 'Sep 2025', value: -5 }] },
  { category: 'net_profit', history: [{ period: 'Jun 2025', value: 8 }, { period: 'Sep 2025', value: 9 }] },
] } });
const fake = (handler: (url: string, init?: RequestInit) => Response | Promise<Response>) => ((url, init) => Promise.resolve(handler(String(url), init))) as typeof fetch;

test('instrument mapping rejects missing, ambiguous or cross-market matches', () => {
  assert.deepEqual(selectInstruments(master, ['TCS']), [instrument]);
  assert.throws(() => selectInstruments(master, ['INFY']), UpstoxError);
  assert.throws(() => selectInstruments([...master, ...master], ['TCS']), UpstoxError);
  assert.throws(() => selectInstruments([{ ...master[0], instrument_key: 'BSE_EQ|INE467B01029' }], ['TCS']), UpstoxError);
  assert.deepEqual(selectInstruments([...master, { ...master[0], segment: 'NSE_FO' }], ['TCS']), [instrument]);
});
test('quarter selection is explicit across year boundaries; invalid and future periods fail closed', () => {
  assert.deepEqual(quarterPair(new Date('2026-11-08T00:00:00Z')), ['2026-06-30', '2026-09-30']);
  assert.deepEqual(quarterPair(new Date('2026-01-01T00:00:00Z')), ['2025-09-30', '2025-12-31']);
  assert.equal(quarterEnd('Mar 2026'), '2026-03-31');
  for (const value of ['TTM', '2026-09-30', 'Sept 2026', 'Feb 2026']) assert.throws(() => quarterEnd(value));
  assert.throws(() => parseIncomeStatement(fixture(), new Date('2025-07-01')));
});
test('quarterly parser rejects wrong basis, units, frequency and ambiguous histories', () => {
  for (const mutation of [{ type: 'standalone' }, { units_in: 'million' }, { time_period: 'yearly' }]) {
    const f = fixture(); Object.assign(f.data, mutation); assert.throws(() => parseIncomeStatement(f));
  }
  const duplicate = fixture(); duplicate.data.income_statement[0].history.push({ period: 'Jun 2025', value: 999 });
  assert.throws(() => parseIncomeStatement(duplicate));
  const groups = fixture(); groups.data.income_statement.push(groups.data.income_statement[0]);
  assert.throws(() => parseIncomeStatement(groups));
  for (const value of ['100', NaN, Infinity, undefined]) {
    const f = fixture(); Object.assign(f.data.income_statement[0].history[0], { value });
    assert.throws(() => parseIncomeStatement(f));
  }
});
test('missing observations stay missing, zero and negative values survive, and coverage never means verified', () => {
  const f = fixture(); f.data.income_statement = f.data.income_statement.slice(0, 2);
  const rows = comparisonRows(parseIncomeStatement(f), ['2025-06-30', '2025-09-30']);
  assert.equal(rows[1].previous, 0); assert.equal(rows[1].current, -5);
  assert.equal(rows[2].current, null); assert.equal(rows[2].coverage, 'missing');
  assert.ok(rows.every(row => row.officialEvidenceCheck === 'pending' && row.definitionCheck === 'pending'));
  const empty = fixture(); Object.assign(empty.data.income_statement[0].history[0], { value: null });
  assert.equal(comparisonRows(parseIncomeStatement(empty), ['2025-06-30', '2025-09-30'])[0].coverage, 'missing');
});
test('authenticated request uses fixed financial endpoint and never follows redirects', async () => {
  const result = await fetchIncome(instrument, 'test-secret', fake((url, init) => {
    assert.equal(url, 'https://api.upstox.com/v2/fundamentals/INE467B01029/income-statement?type=consolidated&time_period=quarterly&fs=true');
    assert.equal(init?.method, 'GET'); assert.equal(init?.redirect, 'error'); assert.equal(init?.cache, 'no-store');
    assert.equal((init?.headers as Record<string, string>).Authorization, 'Bearer test-secret');
    return Response.json(fixture());
  }));
  assert.equal(result.observations.length, 6); assert.match(result.sha256, /^[a-f0-9]{64}$/);
  assert.ok(!JSON.stringify(result).includes('test-secret'));
});
test('provider failures, reflected credentials, malformed and oversized responses cannot leak secrets', async () => {
  for (const status of [401, 403, 429, 500]) {
    await assert.rejects(fetchIncome(instrument, 'test-secret', fake(() => new Response('test-secret', { status }))),
      (error: unknown) => error instanceof UpstoxError && error.code === `http_${status}` && !error.message.includes('test-secret'));
  }
  await assert.rejects(fetchIncome(instrument, 'test-secret', fake(() => Response.json({ echo: 'test-secret' }))), { code: 'credential_echo' });
  await assert.rejects(fetchIncome(instrument, 'test-secret', fake(() => new Response('invalid JSON'))), { code: 'schema' });
  await assert.rejects(fetchIncome(instrument, 'test-secret', fake(() => new Response('x'.repeat(2 * 1024 * 1024 + 1)))), { code: 'size' });
  await assert.rejects(fetchIncome(instrument, 'test-secret', fake(() => { throw Error('test-secret'); })), { code: 'network', message: 'Could not reach Upstox securely.' });
});
test('public instrument download carries no authorization and fingerprints the compressed source', async () => {
  const bytes = gzipSync(JSON.stringify(master));
  const result = await fetchInstruments(['TCS'], fake((_url, init) => {
    assert.equal((init?.headers as Record<string, string>).Authorization, undefined);
    return new Response(bytes);
  }));
  assert.deepEqual(result.instruments, [instrument]); assert.match(result.sha256, /^[a-f0-9]{64}$/);
  await assert.rejects(fetchInstruments(['TCS'], fake(() => new Response('not gzip'))), { code: 'schema' });
});
