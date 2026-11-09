import { createHash } from 'node:crypto';
import { gunzipSync } from 'node:zlib';

// Evaluation-only adapter. No database, publication or trading dependencies.
export const instrumentUrl = 'https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz';
export const pilotSymbols = ['TCS', 'INFY', 'HCLTECH', 'WIPRO', 'HINDUNILVR'] as const;
export const providerMetrics = ['revenue', 'operating_profit', 'net_profit'] as const;
export type ProviderMetric = typeof providerMetrics[number];
export type Instrument = { symbol: string; isin: string; instrumentKey: string };
export type Observation = { metric: ProviderMetric; periodEnd: string; value: number | null };
export class UpstoxError extends Error {
  constructor(public code: string, message: string) { super(message); }
}
const invalid = (): never => { throw new UpstoxError('schema', 'Unexpected Upstox data; inspect the provider schema before using it.'); };
function object(v: unknown): Record<string, unknown> {
  if (!v || typeof v !== 'object' || Array.isArray(v)) return invalid();
  return v as Record<string, unknown>;
}
export const sha256 = (bytes: Uint8Array) => createHash('sha256').update(bytes).digest('hex');

export function selectInstruments(input: unknown, symbols: readonly string[]): Instrument[] {
  if (!Array.isArray(input)) return invalid();
  return symbols.map(symbol => {
    const matches = input.filter(v => v && v.segment === 'NSE_EQ' && v.instrument_type === 'EQ' && v.trading_symbol === symbol);
    if (matches.length !== 1) throw new UpstoxError('instrument', `Expected exactly one NSE equity instrument for ${symbol}.`);
    const row = object(matches[0]);
    if (typeof row.isin !== 'string' || !/^IN[A-Z0-9]{9}[0-9]$/.test(row.isin) || row.instrument_key !== `NSE_EQ|${row.isin}`) return invalid();
    return { symbol, isin: row.isin, instrumentKey: row.instrument_key as string };
  });
}

export function quarterEnd(period: unknown): string {
  if (typeof period !== 'string') return invalid();
  const match = /^(Mar|Jun|Sep|Dec) (20\d{2})$/.exec(period);
  if (!match) return invalid();
  const month = { Mar: 3, Jun: 6, Sep: 9, Dec: 12 }[match[1]]!;
  return new Date(Date.UTC(Number(match[2]), month, 0)).toISOString().slice(0, 10);
}
export function quarterPair(now = new Date()): [string, string] {
  const currentQuarterStart = Math.floor(now.getUTCMonth() / 3) * 3;
  return [new Date(Date.UTC(now.getUTCFullYear(), currentQuarterStart - 3, 0)).toISOString().slice(0, 10),
    new Date(Date.UTC(now.getUTCFullYear(), currentQuarterStart, 0)).toISOString().slice(0, 10)];
}

export function parseIncomeStatement(input: unknown, now = new Date()): Observation[] {
  const root = object(input), data = object(root.data);
  if (root.status !== 'success' || data.type !== 'consolidated' || data.time_period !== 'quarterly' || data.units_in !== 'crore' || !Array.isArray(data.income_statement)) return invalid();
  const result: Observation[] = [], categories = new Set<string>();
  for (const entry of data.income_statement) {
    const group = object(entry);
    if (!providerMetrics.includes(group.category as ProviderMetric)) continue;
    const metric = group.category as ProviderMetric;
    if (categories.has(metric) || !Array.isArray(group.history)) return invalid();
    categories.add(metric);
    const periods = new Set<string>();
    for (const entry of group.history) {
      const row = object(entry), periodEnd = quarterEnd(row.period);
      if (periods.has(periodEnd) || periodEnd > now.toISOString().slice(0, 10)) return invalid();
      if (row.value !== null && (typeof row.value !== 'number' || !Number.isFinite(row.value) || Math.abs(row.value) > Number.MAX_SAFE_INTEGER)) return invalid();
      periods.add(periodEnd);
      result.push({ metric, periodEnd, value: row.value as number | null });
    }
  }
  // Missing categories/quarters remain visible gaps, never zero-filled.
  return result;
}

/** Fixed destinations only; bearer tokens never follow redirects or enter logs. */
async function download(url: string, token: string | undefined, maxBytes: number, fetcher: typeof fetch): Promise<Buffer> {
  const signal = AbortSignal.timeout(25_000);
  try {
    const response = await fetcher(url, { method: 'GET', redirect: 'error', cache: 'no-store', signal,
      headers: { Accept: token ? 'application/json' : 'application/gzip', ...(token ? { Authorization: `Bearer ${token}` } : {}) } });
    if (!response.ok) {
      await response.body?.cancel();
      const message = response.status === 401 ? 'Upstox rejected the token. Generate or renew the Analytics Token.'
        : response.status === 403 ? 'Upstox denied access to this endpoint. Check account and API permissions.'
        : response.status === 429 ? 'Upstox rate limit reached. Retry later; no automatic retry was made.'
        : `Upstox returned HTTP ${response.status}.`;
      throw new UpstoxError(`http_${response.status}`, message);
    }
    if (Number(response.headers.get('content-length')) > maxBytes) { await response.body?.cancel(); throw new UpstoxError('size', 'Upstox response exceeded the size limit.'); }
    if (!response.body) return invalid();
    const reader = response.body.getReader(), chunks: Uint8Array[] = [];
    let size = 0;
    try {
      while (true) {
        const part = await reader.read();
        if (part.done) break;
        size += part.value.byteLength;
        if (size > maxBytes) { await reader.cancel(); throw new UpstoxError('size', 'Upstox response exceeded the size limit.'); }
        chunks.push(part.value);
      }
    } finally { reader.releaseLock(); }
    const bytes = Buffer.concat(chunks);
    if (token && bytes.includes(Buffer.from(token))) throw new UpstoxError('credential_echo', 'Provider response unexpectedly contained a credential; response discarded.');
    return bytes;
  } catch (error) {
    if (error instanceof UpstoxError) throw error;
    throw new UpstoxError(signal.aborted ? 'timeout' : 'network', signal.aborted ? 'Upstox request timed out.' : 'Could not reach Upstox securely.');
  }
}
function json(bytes: Uint8Array): unknown {
  try { return JSON.parse(Buffer.from(bytes).toString('utf8')); } catch { return invalid(); }
}
export async function fetchInstruments(symbols: readonly string[], fetcher: typeof fetch = fetch) {
  const bytes = await download(instrumentUrl, undefined, 20 * 1024 * 1024, fetcher);
  let expanded: Buffer;
  try { expanded = gunzipSync(bytes, { maxOutputLength: 100 * 1024 * 1024 }); } catch { return invalid(); }
  return { fetchedAt: new Date().toISOString(), url: instrumentUrl, sha256: sha256(bytes), instruments: selectInstruments(json(expanded), symbols) };
}
export async function fetchIncome(instrument: Instrument, token: string, fetcher: typeof fetch = fetch) {
  if (!token || token.length > 8192 || /\s/.test(token)) throw new UpstoxError('configuration', 'Save a valid UPSTOX_ANALYTICS_TOKEN in .env.local.');
  if (!/^IN[A-Z0-9]{9}[0-9]$/.test(instrument.isin)) return invalid();
  const url = `https://api.upstox.com/v2/fundamentals/${instrument.isin}/income-statement?type=consolidated&time_period=quarterly&fs=true`;
  const bytes = await download(url, token, 2 * 1024 * 1024, fetcher);
  const raw = json(bytes), observations = parseIncomeStatement(raw);
  return { url, fetchedAt: new Date().toISOString(), sha256: sha256(bytes), raw, observations };
}

export function comparisonRows(observations: Observation[], periods: [string, string]) {
  return providerMetrics.map(metric => {
    const previous = observations.find(v => v.metric === metric && v.periodEnd === periods[0])?.value ?? null;
    const current = observations.find(v => v.metric === metric && v.periodEnd === periods[1])?.value ?? null;
    return { metric, previous, current, coverage: previous === null || current === null ? 'missing' : 'present',
      officialEvidenceCheck: 'pending', definitionCheck: 'pending' };
  });
}
