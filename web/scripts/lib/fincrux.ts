import { createHash } from 'node:crypto';
import { quarterEnd, quarterPair } from './upstox';

export const fincruxSymbols = ['TCS', 'INFY', 'HCLTECH', 'WIPRO', 'HINDUNILVR'] as const;
export class FincruxError extends Error {
  constructor(public code: string, message: string) { super(message); }
}
const invalid = (): never => { throw new FincruxError('schema', 'Unexpected Fincrux response. Inspect the private snapshot before mapping financial data.'); };
function record(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return invalid();
  return value as Record<string, unknown>;
}

/** Inventory only: the docs do not define units, accounting basis or row structure. */
export function inspectFinancials(input: unknown, symbol: string) {
  const root = record(input);
  if (root.success !== true && root.success !== 'true') return invalid();
  if (root.trading_symbol !== symbol) throw new FincruxError('identity', 'Fincrux returned a different company; response cannot be used.');
  const data = record(root.data);
  // Both spellings occur in the official docs. Never silently choose between two tables.
  if (data.quarterly_results !== undefined && data.quaterly_results !== undefined) return invalid();
  const key = data.quarterly_results !== undefined ? 'quarterly_results' : data.quaterly_results !== undefined ? 'quaterly_results' : null;
  const table = key ? data[key] : undefined;
  if (table !== undefined && !Array.isArray(table)) return invalid();
  const rows = table as unknown[] | undefined;
  const matrix = rows?.length && Array.isArray(rows[0]) ? inspectQuarterMatrix(rows) : null;
  return {
    symbol, quarterlyField: key, quarterlyRowCount: rows?.length ?? 0,
    quarterlyRowFields: matrix ? matrix.columns : rows?.length ? Object.keys(record(rows[0])) : [],
    quarterlyComparison: matrix?.comparison ?? null,
    requestedPeriods: matrix?.periods ?? quarterPair(),
    sourceLinksPresent: matrix?.sourceLinksPresent ?? false,
    availableTables: Object.entries(data).filter(([, value]) => Array.isArray(value)).map(([name, value]) => ({ name, rows: (value as unknown[]).length })),
    providerUpdatedAtRaw: typeof data.last_updated_at === 'string' ? data.last_updated_at : null,
    unit: 'unverified', basis: 'unverified', metricDefinitions: 'unverified', officialEvidenceCheck: 'pending',
  };
}

/** Observed live matrix schema; preserve provider labels and leave financial meaning unverified. */
export function inspectQuarterMatrix(rows: unknown[], now = new Date()) {
  const header = rows[0];
  if (!Array.isArray(header) || header[0] !== 'Category' || header.length < 2 || !header.every(v => typeof v === 'string')) return invalid();
  const periodsInTable = header.slice(1).map(quarterEnd);
  if (new Set(periodsInTable).size !== periodsInTable.length || periodsInTable.some(p => p > now.toISOString().slice(0, 10))) return invalid();
  const values = new Map<string, unknown[]>();
  for (const row of rows.slice(1)) {
    if (!Array.isArray(row) || row.length !== header.length || typeof row[0] !== 'string' || values.has(row[0])) return invalid();
    values.set(row[0], row.slice(1));
  }
  const periods = quarterPair(now);
  const decimal = (v: unknown): string | null => {
    if (v === undefined || v === null || v === '' || v === '-' || v === '—') return null;
    if (typeof v !== 'string' || !/^-?(?:\d+|\d{1,3}(?:,\d{3})+|\d{1,2}(?:,\d{2})*,\d{3})(?:\.\d+)?$/.test(v.trim())) return invalid();
    return v.trim().replaceAll(',', '');
  };
  const comparison = ['Sales', 'Operating Profit', 'Net Profit'].map(providerLabel => {
    const row = values.get(providerLabel);
    const previous = decimal(row?.[periodsInTable.indexOf(periods[0])]);
    const current = decimal(row?.[periodsInTable.indexOf(periods[1])]);
    return { providerLabel, previous, current, coverage: previous === null || current === null ? 'missing' : 'present', definitionCheck: 'pending' };
  });
  return { columns: header as string[], periods, comparison,
    sourceLinksPresent: (values.get('Raw PDF') ?? []).some(v => typeof v === 'string' && /^https:\/\//.test(v)) };
}

/** Only the official financial endpoint is allowed. The query key never reaches logs. */
export async function fetchFincrux(symbol: string, key: string, fetcher: typeof fetch = fetch) {
  if (!fincruxSymbols.includes(symbol as typeof fincruxSymbols[number])) throw new FincruxError('symbol', 'Choose one of the five evaluation companies.');
  if (!key || key.length > 8192 || /\s/.test(key)) throw new FincruxError('configuration', 'Save a valid FINCRUX_API_KEY in web/.env.local.');
  const endpoint = `https://api.fincrux.org/api/financials/${symbol}`;
  const url = new URL(endpoint); url.searchParams.set('api_key', key);
  const signal = AbortSignal.timeout(25_000), maxBytes = 5 * 1024 * 1024;
  try {
    const response = await fetcher(url.href, { method: 'GET', redirect: 'error', cache: 'no-store', signal, headers: { Accept: 'application/json' } });
    if (!response.ok) {
      await response.body?.cancel();
      throw new FincruxError(`http_${response.status}`, response.status === 429 ? 'Fincrux quota reached. Wait for the provider reset; no retry was made.'
        : [401, 403].includes(response.status) ? 'Fincrux rejected access. Check the API key and free-plan access.' : `Fincrux returned HTTP ${response.status}.`);
    }
    if (!response.body) return invalid();
    if (Number(response.headers.get('content-length')) > maxBytes) { await response.body.cancel(); throw new FincruxError('size', 'Fincrux response exceeded the size limit.'); }
    const reader = response.body.getReader(), chunks: Uint8Array[] = []; let size = 0;
    try {
      while (true) {
        const part = await reader.read(); if (part.done) break;
        size += part.value.byteLength;
        if (size > maxBytes) { await reader.cancel(); throw new FincruxError('size', 'Fincrux response exceeded the size limit.'); }
        chunks.push(part.value);
      }
    } finally { reader.releaseLock(); }
    const bytes = Buffer.concat(chunks), text = bytes.toString('utf8');
    let raw: unknown;
    try { raw = JSON.parse(text); } catch { return invalid(); }
    // JSON decoding also catches escaped reflections. Do not retain credential-bearing responses.
    const decoded = JSON.stringify(raw);
    if ([key, encodeURIComponent(key), url.searchParams.toString().slice('api_key='.length)].some(secret => text.includes(secret) || decoded.includes(secret))) {
      throw new FincruxError('credential_echo', 'Fincrux response contained a credential; response discarded.');
    }
    return { endpoint, fetchedAt: new Date().toISOString(), sha256: createHash('sha256').update(bytes).digest('hex'), responseText: text };
  } catch (error) {
    if (error instanceof FincruxError) throw error;
    throw new FincruxError(signal.aborted ? 'timeout' : 'network', signal.aborted ? 'Fincrux request timed out.' : 'Could not reach Fincrux securely.');
  }
}

/** Conservative local rolling-day cap; failed requests count too. */
export function reserveCall(history: unknown, now = Date.now()): number[] {
  if (!Array.isArray(history) || !history.every(n => typeof n === 'number' && Number.isFinite(n) && n >= 0 && n <= now)) throw new FincruxError('ledger', 'Invalid local Fincrux quota history. Inspect it before making more requests.');
  const recent = (history as number[]).filter(n => now - n < 86_400_000);
  if (recent.length >= 5) throw new FincruxError('quota', 'Five local Fincrux requests were reserved in the last 24 hours. Use cached responses or wait.');
  return [...recent, now];
}
