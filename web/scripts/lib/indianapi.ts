import { createHash } from 'node:crypto';
import { quarterEnd, quarterPair } from './upstox';

export const indianApiSymbols = ['TCS', 'INFY', 'HCLTECH', 'WIPRO', 'HINDUNILVR'] as const;
export class IndianApiError extends Error {
  constructor(public code: string, message: string) { super(message); }
}
const invalid = (): never => { throw new IndianApiError('schema', 'Unexpected IndianAPI financial schema. Inspect the private response before mapping data.'); };
function object(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return invalid();
  return value as Record<string, unknown>;
}
function decimal(value: unknown): string | null {
  if (value === null || value === undefined || value === '' || value === '-' || value === '—') return null;
  if (typeof value === 'number') {
    if (!Number.isFinite(value) || Math.abs(value) > Number.MAX_SAFE_INTEGER) return invalid();
    return String(value);
  }
  if (typeof value !== 'string' || !/^-?(?:\d+|\d{1,3}(?:,\d{3})+|\d{1,2}(?:,\d{2})*,\d{3})(?:\.\d+)?$/.test(value.trim())) return invalid();
  return value.trim().replaceAll(',', '');
}
export function inspectIndianStats(input: unknown, symbol: string, now = new Date()) {
  const data = object(input), periods = quarterPair(now);
  if (!['Sales', 'Operating Profit', 'Net Profit'].some(label => Object.hasOwn(data, label))) return invalid();
  const rows = ['Sales', 'Operating Profit', 'Net Profit'].map(providerLabel => {
    const history = data[providerLabel] === undefined ? {} : object(data[providerLabel]);
    const points = new Map<string, string | null>();
    for (const [period, value] of Object.entries(history)) {
      const end = quarterEnd(period);
      if (end > now.toISOString().slice(0, 10) || points.has(end)) return invalid();
      points.set(end, decimal(value));
    }
    const previous = points.get(periods[0]) ?? null, current = points.get(periods[1]) ?? null;
    return { providerLabel, previous, current, coverage: previous === null || current === null ? 'missing' : 'present' };
  });
  // The documented response contains no company identity, unit or statement basis.
  // Preserve the requested symbol without claiming the provider echoed it.
  return { requestedSymbol: symbol, periods, rows, identityCheck: 'pending', unit: 'unverified', basis: 'unverified',
    metricDefinitions: 'unverified', officialEvidenceCheck: 'pending' };
}

export function tcsRevenueCheck(report: ReturnType<typeof inspectIndianStats>) {
  const sourceUrl = 'https://www.tcs.com/content/dam/tcs/pdf/discover-tcs/investor-relations/corporate-actions/2026-27/financial-results-qtr-sep-30-2026.pdf';
  const sales = report.rows.find(row => row.providerLabel === 'Sales');
  let result = 'not_checked';
  if (report.requestedSymbol === 'TCS' && report.periods.join(',') === '2026-06-30,2026-09-30' && sales && sales.previous !== null && sales.current !== null) {
    const same = (previous: number, current: number) => Number(sales.previous) === previous && Number(sales.current) === current;
    result = same(72275, 73188) ? 'matches_consolidated_revenue' : same(59553, 60441) ? 'matches_standalone_revenue' : 'matches_neither_reference';
  }
  return { result, sourceUrl, consolidatedPage: 8, standalonePage: 17, referenceUnit: 'INR crore',
    limitation: 'Numerical revenue cross-check only; does not verify company identity, other metrics, source provenance or publication readiness.' };
}

/** A dated diagnostic, never a mapping rule or a publication attestation. */
export function tcsProfitChecks(report: ReturnType<typeof inspectIndianStats>) {
  if (report.requestedSymbol !== 'TCS' || report.periods.join(',') !== '2026-06-30,2026-09-30') return [];
  const references = [
    { providerLabel: 'Operating Profit', officialLabel: 'Reported consolidated operating income', previous: '16376', current: '17301', page: 9 },
    { providerLabel: 'Net Profit', officialLabel: 'Profit attributable to shareholders', previous: '13349', current: '13884', page: 8 },
  ];
  return references.map(reference => {
    const row = report.rows.find(row => row.providerLabel === reference.providerLabel);
    const available = !!row && row.previous !== null && row.current !== null;
    const matches = available && Number(row.previous) === Number(reference.previous) && Number(row.current) === Number(reference.current);
    const matchesTotalProfit = reference.providerLabel === 'Net Profit' && available && Number(row.previous) === 13420 && Number(row.current) === 13934;
    const matchesSegmentTotal = reference.providerLabel === 'Operating Profit' && available && Number(row.previous) === 18556 && Number(row.current) === 18815;
    return { ...reference, result: !available ? 'missing' : matches ? 'numerical_match_only' : matchesTotalProfit ? 'matches_total_profit_including_non_controlling_interests' : matchesSegmentTotal ? 'matches_segment_results_before_unallocable_expenses' : 'differs_from_official_metric',
      sourceUrl: tcsRevenueCheck(report).sourceUrl, directMappingAllowed: false };
  });
}

export const indianStatsUrl = (symbol: string) => {
  if (!indianApiSymbols.includes(symbol as typeof indianApiSymbols[number])) throw new IndianApiError('symbol', 'Choose one of the five evaluation companies.');
  return `https://stock.indianapi.in/historical_stats?stock_name=${symbol}&stats=quarter_results`;
};
export async function fetchIndianStats(symbol: string, key: string, fetcher: typeof fetch = fetch) {
  const endpoint = indianStatsUrl(symbol);
  if (!key || key.length > 8192 || /\s/.test(key)) throw new IndianApiError('configuration', 'Save a valid INDIANAPI_API_KEY in web/.env.local.');
  const signal = AbortSignal.timeout(25_000), maxBytes = 5 * 1024 * 1024;
  try {
    const response = await fetcher(endpoint, { method: 'GET', redirect: 'error', cache: 'no-store', signal,
      headers: { Accept: 'application/json', 'X-Api-Key': key } });
    if (!response.ok) {
      await response.body?.cancel();
      throw new IndianApiError(`http_${response.status}`, response.status === 429 ? 'IndianAPI credits or rate limit reached. No retry was made.'
        : [401, 403].includes(response.status) ? 'IndianAPI denied access. Check the key and free-plan entitlement for this endpoint.' : `IndianAPI returned HTTP ${response.status}.`);
    }
    if (!response.body) return invalid();
    if (Number(response.headers.get('content-length')) > maxBytes) { await response.body.cancel(); throw new IndianApiError('size', 'IndianAPI response exceeded the size limit.'); }
    const reader = response.body.getReader(), chunks: Uint8Array[] = []; let size = 0;
    try {
      while (true) {
        const part = await reader.read(); if (part.done) break;
        size += part.value.byteLength;
        if (size > maxBytes) { await reader.cancel(); throw new IndianApiError('size', 'IndianAPI response exceeded the size limit.'); }
        chunks.push(part.value);
      }
    } finally { reader.releaseLock(); }
    const bytes = Buffer.concat(chunks), responseText = bytes.toString('utf8');
    let raw: unknown;
    try { raw = JSON.parse(responseText); } catch { return invalid(); }
    if ([key, encodeURIComponent(key)].some(secret => responseText.includes(secret) || JSON.stringify(raw).includes(secret))) {
      throw new IndianApiError('credential_echo', 'IndianAPI response contained a credential; discarded without storage.');
    }
    return { endpoint, requestedSymbol: symbol, fetchedAt: new Date().toISOString(), sha256: createHash('sha256').update(bytes).digest('hex'), responseText };
  } catch (error) {
    if (error instanceof IndianApiError) throw error;
    throw new IndianApiError(signal.aborted ? 'timeout' : 'network', signal.aborted ? 'IndianAPI request timed out.' : 'Could not reach IndianAPI securely.');
  }
}
