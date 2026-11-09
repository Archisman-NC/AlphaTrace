import { mkdir, open, readFile, writeFile, rename, unlink } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { join } from 'node:path';
import { createHash } from 'node:crypto';
import { fetchIndianStats, inspectIndianStats, tcsRevenueCheck, tcsProfitChecks, indianApiSymbols, indianStatsUrl, IndianApiError } from './lib/indianapi';
import { officialReferenceChecks } from './lib/indianapi-references';

const root = fileURLToPath(new URL('../.local/indianapi/', import.meta.url));
async function optionalJson(path: string): Promise<unknown | undefined> {
  try { return JSON.parse(await readFile(path, 'utf8')); } catch (error) { if ((error as NodeJS.ErrnoException).code === 'ENOENT') return undefined; throw error; }
}
async function save(name: string, value: unknown) {
  const path = join(root, name), temp = `${path}.${process.pid}.tmp`;
  await writeFile(temp, JSON.stringify(value, null, 2) + '\n', { mode: 0o600 }); await rename(temp, path);
}
async function main() {
  const args = process.argv.slice(2);
  if (new Set(args).size !== args.length || args.some(arg => !['--all', '--cached-only'].includes(arg))) throw new IndianApiError('arguments', 'Usage: npm run indianapi:benchmark [-- --all --cached-only]. Default: one TCS request.');
  const symbols = args.includes('--all') ? indianApiSymbols : ['TCS'];
  await mkdir(root, { recursive: true, mode: 0o700 });
  const lockPath = join(root, 'evaluation.lock'); let lock;
  try { lock = await open(lockPath, 'wx', 0o600); } catch { throw new IndianApiError('lock', 'An IndianAPI evaluation is running, or a stale lock needs inspection.'); }
  try {
    const companies: Array<ReturnType<typeof inspectIndianStats> & {
      fetchedAt: string; sha256: string; cached: boolean; revenueReference: ReturnType<typeof tcsRevenueCheck>; profitReferences: ReturnType<typeof tcsProfitChecks>;
      officialReferences: ReturnType<typeof officialReferenceChecks>;
    }> = [];
    // Replace any older report with an explicit in-progress record before requests.
    await save('report.json', { evaluationOnly: true, requestedSymbols: symbols, completed: false, companies });
    for (const symbol of symbols) {
      let snapshot = await optionalJson(join(root, `${symbol}.json`)) as Awaited<ReturnType<typeof fetchIndianStats>> | undefined;
      const cached = !!snapshot;
      if (!snapshot) {
        if (args.includes('--cached-only')) throw new IndianApiError('cache_missing', `No cached response for ${symbol}; offline evaluation made no request.`);
        const key = process.env.INDIANAPI_API_KEY?.trim();
        if (!key) throw new IndianApiError('configuration', 'Save INDIANAPI_API_KEY in web/.env.local, then rerun. Do not paste it into chat.');
        snapshot = await fetchIndianStats(symbol, key);
        await save(`${symbol}.json`, snapshot);
      }
      if (snapshot.requestedSymbol !== symbol || snapshot.endpoint !== indianStatsUrl(symbol) || typeof snapshot.responseText !== 'string'
        || createHash('sha256').update(snapshot.responseText).digest('hex') !== snapshot.sha256) throw new IndianApiError('cache', 'Cached IndianAPI response failed integrity checks.');
      const evaluation = inspectIndianStats(JSON.parse(snapshot.responseText), symbol);
      const officialReferences = officialReferenceChecks(evaluation);
      companies.push({ ...evaluation, fetchedAt: snapshot.fetchedAt, sha256: snapshot.sha256, cached, revenueReference: tcsRevenueCheck(evaluation), profitReferences: tcsProfitChecks(evaluation), officialReferences });
      console.log(`${symbol}: ${cached ? 'cached response reused' : 'response saved'}; ${evaluation.rows.filter(row => row.coverage === 'present').length}/3 metric pairs present. ${officialReferences ? officialReferences.checks.map(check => `${check.providerLabel}: ${check.results.join('/')}`).join('; ') : tcsRevenueCheck(evaluation).result}.`);
      await save('report.json', { provider: 'indianapi', generatedAt: new Date().toISOString(), evaluationOnly: true,
        completed: companies.length === symbols.length, requestedSymbols: symbols, companies, displayAndStorageRights: 'unconfirmed' });
      if (evaluation.rows.some(row => row.coverage !== 'present')) process.exitCode = 1;
      if (officialReferences?.checks.some(check => check.results.some(result => result === 'mismatch' || result === 'missing'))) process.exitCode = 1;
      if (!cached && symbol !== symbols.at(-1)) await new Promise(resolve => setTimeout(resolve, 1_100));
    }
    const lines = ['# IndianAPI evaluation — unpublished', '',
      'Values use provider labels. Currency/scale, accounting basis, identity and profit definitions are not supplied by this endpoint and need independent verification.',
      'Retrieval timestamps do not establish provider freshness. Cached data is reused without API calls.', '',
      '| Company | Provider label | Previous quarter | Current quarter | Coverage |', '|---|---|---:|---:|---|'];
    for (const company of companies) for (const row of company.rows) lines.push(`| ${company.requestedSymbol} | ${row.providerLabel} | ${row.previous ?? 'Missing'} | ${row.current ?? 'Missing'} | ${row.coverage} |`);
    lines.push('', '## TCS reference checks', '');
    for (const company of companies.filter(c => c.requestedSymbol === 'TCS')) {
      lines.push(`Revenue: ${company.revenueReference.result}. Reference units: INR crore.`, '');
      for (const check of company.profitReferences) lines.push(`- ${check.providerLabel}: ${check.result}. Official ${check.officialLabel}: ${check.previous} → ${check.current}; PDF page ${check.page}.`);
      lines.push('', `[Official TCS filing](${company.revenueReference.sourceUrl})`, '',
        'TCS Operating Profit matches total segment results before unallocable expenses (page 9). Profit fields must not be substituted directly into AlphaTrace reported metrics.');
    }
    lines.push('', '## Other companies: dated official-statement reconciliation', '',
      'References apply only to June/September 2026. Exact/rounded matches support these inferred definitions, not provider provenance or publication approval. All amounts below are INR crore.', '');
    for (const company of companies) {
      const ref = company.officialReferences;
      if (!ref) continue;
      lines.push(`### ${company.requestedSymbol}`, '', `[Official statement](${ref.sourceUrl}) — ${ref.locator}. Original unit: ${ref.sourceUnit}.`, '',
        '| Provider label | Official/derived June | Official/derived September | June check | September check |', '|---|---:|---:|---|---|');
      for (const check of ref.checks) lines.push(`| ${check.providerLabel} | ${check.expectedInrCrore[0]} | ${check.expectedInrCrore[1]} | ${check.results[0]} | ${check.results[1]} |`);
      lines.push('', ref.note, '', `Owner-attributable profit: ${ref.ownersProfitInrCrore.join(' → ')}.`, '',
        'Derived Operating Profit formula: revenue − total expenses + finance costs + depreciation/amortisation/impairment − specified associate loss. Operands and source SHA-256 are in report.json.', '');
    }
    lines.push('', '## Snapshot times', '');
    for (const company of companies) lines.push(`- ${company.requestedSymbol}: ${company.periods.join(' → ')}; retrieved ${company.fetchedAt}.`);
    lines.push('', 'No Supabase writes, research approvals, publications or subscriptions were made. Storage/display rights remain unconfirmed.');
    await writeFile(join(root, 'report.md'), lines.join('\n') + '\n', { mode: 0o600 });
    console.log(`Private evaluation report: ${join(root, 'report.md')}`);
  } finally { await lock.close(); await unlink(lockPath); }
}
main().catch(error => {
  console.error(error instanceof IndianApiError ? error.message : 'IndianAPI evaluation stopped. Inspect the private response/configuration; no raw errors or credentials are logged.');
  process.exitCode = 1;
});
