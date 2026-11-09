import { mkdir, readFile, writeFile, open, unlink, rename } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { join } from 'node:path';
import { createHash } from 'node:crypto';
import { fetchFincrux, inspectFinancials, fincruxSymbols, FincruxError, reserveCall } from './lib/fincrux';

const root = fileURLToPath(new URL('../.local/fincrux/', import.meta.url));
const args = process.argv.slice(2);
const missing = (error: unknown) => (error as NodeJS.ErrnoException).code === 'ENOENT';
async function optionalJson(path: string): Promise<unknown | undefined> {
  try { return JSON.parse(await readFile(path, 'utf8')); } catch (error) { if (missing(error)) return undefined; throw error; }
}
async function save(name: string, value: unknown) {
  const path = join(root, name), temp = `${path}.${process.pid}.tmp`;
  await writeFile(temp, JSON.stringify(value, null, 2) + '\n', { mode: 0o600 });
  await rename(temp, path);
}
async function main() {
  if (args.length > 1 || (args.length === 1 && args[0] !== '--all')) throw new FincruxError('arguments', 'Usage: npm run fincrux:benchmark [-- --all]. Default: TCS only; cached responses are reused.');
  const symbols = args[0] === '--all' ? fincruxSymbols : ['TCS'];
  await mkdir(root, { recursive: true, mode: 0o700 });
  const lockPath = join(root, 'evaluation.lock');
  let lock;
  try { lock = await open(lockPath, 'wx', 0o600); }
  catch { throw new FincruxError('lock', 'A Fincrux evaluation is running, or a stale lock needs inspection. No request was made.'); }
  try {
    const companies: Array<ReturnType<typeof inspectFinancials> & { fetchedAt: string; sha256: string; cached: boolean }> = [];
    for (const symbol of symbols) {
      let snapshot = await optionalJson(join(root, `${symbol}.json`)) as Awaited<ReturnType<typeof fetchFincrux>> | undefined;
      const cached = !!snapshot;
      if (!snapshot) {
        const key = process.env.FINCRUX_API_KEY?.trim();
        if (!key) throw new FincruxError('configuration', 'Save FINCRUX_API_KEY in web/.env.local, then rerun. Do not paste it into chat.');
        const history = await optionalJson(join(root, 'requests.json')) ?? [];
        await save('requests.json', reserveCall(history));
        // No forced refresh, retries, discovery calls or paid upgrade actions.
        snapshot = await fetchFincrux(symbol, key);
        await save(`${symbol}.json`, snapshot);
      }
      if (typeof snapshot.responseText !== 'string' || createHash('sha256').update(snapshot.responseText).digest('hex') !== snapshot.sha256
        || snapshot.endpoint !== `https://api.fincrux.org/api/financials/${symbol}`) throw new FincruxError('cache', 'Cached response failed integrity checks. Inspect it before continuing.');
      let evaluation;
      try { evaluation = inspectFinancials(JSON.parse(snapshot.responseText), symbol); }
      catch (error) {
        console.log(`${symbol}: response retained privately; schema or company identity requires inspection.`);
        throw error;
      }
      companies.push({ ...evaluation, fetchedAt: snapshot.fetchedAt, sha256: snapshot.sha256, cached });
      console.log(`${symbol}: ${cached ? 'cached response reused (no API call)' : 'response saved'}; ${evaluation.quarterlyRowCount} quarterly rows. Units, basis and originals still need verification.`);
      await save('report.json', { provider: 'fincrux', generatedAt: new Date().toISOString(), evaluationOnly: true, requestedSymbols: symbols,
        completed: companies.length === symbols.length, companies, financialMapping: 'provider_labels_only', displayAndStorageRights: 'unconfirmed' });
      if (symbol !== symbols.at(-1)) await new Promise(resolve => setTimeout(resolve, 1_100));
    }
    const lines = ['# Fincrux evaluation — unpublished provider data', '',
      'Units, consolidated/standalone basis and metric definitions are unverified in these responses. No growth rates or investment conclusions are inferred.',
      'Cached responses retain their original retrieval times. Provider update timestamps lack a timezone and are preserved without reinterpretation.', '',
      '| Company | Provider label | Previous quarter value | Current quarter value | Coverage |', '|---|---|---:|---:|---|'];
    for (const company of companies) {
      for (const row of company.quarterlyComparison ?? []) lines.push(`| ${company.symbol} | ${row.providerLabel} | ${row.previous ?? 'Missing'} | ${row.current ?? 'Missing'} | ${row.coverage} |`);
    }
    lines.push('', '## Retrieval and provenance', '');
    for (const company of companies) lines.push(`- ${company.symbol}: quarters ${company.requestedPeriods.join(' → ')}; retrieved ${company.fetchedAt}; Raw PDF links ${company.sourceLinksPresent ? 'present but unchecked' : 'absent'}.`);
    lines.push('', 'No original-document approval, publication or Supabase write was performed. See web/scripts/FINCRUX.md for acceptance checks.');
    await writeFile(join(root, 'report.md'), lines.join('\n') + '\n', { mode: 0o600 });
    console.log(`Private evaluation report: ${join(root, 'report.md')}`);
  } finally { await lock.close(); await unlink(lockPath); }
}
main().catch(error => {
  console.error(error instanceof FincruxError ? error.message : 'Fincrux evaluation stopped. Inspect local configuration or cache without exposing credentials.');
  process.exitCode = 1;
});
