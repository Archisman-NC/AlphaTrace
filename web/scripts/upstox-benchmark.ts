import { mkdir, mkdtemp, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { join } from 'node:path';
import { fetchInstruments, fetchIncome, pilotSymbols, quarterPair, comparisonRows, UpstoxError } from './lib/upstox';

const args = process.argv.slice(2);
const instrumentsOnly = args.length === 1 && args[0] === '--instruments';
async function main() {
  if (args.length && !instrumentsOnly) throw new UpstoxError('arguments', 'Usage: npm run upstox:benchmark [-- --instruments]');
  const token = process.env.UPSTOX_ANALYTICS_TOKEN?.trim();
  if (!instrumentsOnly && !token) throw new UpstoxError('configuration', 'Add your Analytics Token to UPSTOX_ANALYTICS_TOKEN in web/.env.local, then rerun. Do not paste it into chat.');
  const root = fileURLToPath(new URL('../.local/upstox/', import.meta.url));
  await mkdir(root, { recursive: true, mode: 0o700 });
  const directory = await mkdtemp(join(root, 'run-'));
  const save = (name: string, value: unknown) => writeFile(join(directory, name), JSON.stringify(value, null, 2) + '\n', { mode: 0o600 });
  const instruments = await fetchInstruments(pilotSymbols);
  await save('instruments.json', instruments);
  console.log(`Resolved ${instruments.instruments.length} NSE equity ISINs from Upstox's public instrument file.`);
  if (instrumentsOnly) { console.log(`Instrument mapping: ${join(directory, 'instruments.json')}`); return; }
  const periods = quarterPair();
  const companies: Array<Record<string, unknown>> = [];
  const lines = ['# Upstox evaluation — not published research', '', `Requested quarters: ${periods.join(' → ')}. Consolidated; INR crore.`,
    'Values below use provider labels. Accounting standard, exceptional-item treatment and owner-attributable profit are unverified.',
    'Retrieval time is not the provider update time. A missing recent quarter may reflect reporting or provider delay.', '',
    '| Company | Provider metric | Previous | Current | Coverage | Official evidence |', '|---|---|---:|---:|---|---|'];
  let failed = false;
  for (let index = 0; index < instruments.instruments.length; index++) {
    const instrument = instruments.instruments[index];
    try {
      const result = await fetchIncome(instrument, token!);
      await save(`${instrument.symbol}.json`, { instrument, ...result });
      const rows = comparisonRows(result.observations, periods);
      const complete = rows.every(row => row.coverage === 'present');
      if (!complete) failed = true;
      companies.push({ ...instrument, status: complete ? 'coverage_present' : 'coverage_gap', url: result.url, fetchedAt: result.fetchedAt, sha256: result.sha256, rows });
      for (const row of rows) lines.push(`| ${instrument.symbol} | ${row.metric} | ${row.previous ?? 'Missing'} | ${row.current ?? 'Missing'} | ${row.coverage} | Pending |`);
      console.log(`${instrument.symbol}: ${complete ? 'both quarters returned; original-source verification pending' : 'missing data in requested quarters'}.`);
    } catch (error) {
      failed = true;
      const safe = error instanceof UpstoxError ? error : new UpstoxError('local', 'Local evaluation failed.');
      companies.push({ ...instrument, status: 'failed', code: safe.code, message: safe.message });
      lines.push(`| ${instrument.symbol} | — | — | — | ${safe.code} | Not checked |`);
      console.log(`${instrument.symbol}: ${safe.message}`);
      if (['http_401', 'http_403', 'http_429'].includes(safe.code)) {
        for (const remaining of instruments.instruments.slice(index + 1)) {
          companies.push({ ...remaining, status: 'skipped', reason: safe.code });
          lines.push(`| ${remaining.symbol} | — | — | — | Skipped after ${safe.code} | Not checked |`);
        }
        break;
      }
    }
    if (index < instruments.instruments.length - 1) await new Promise(resolve => setTimeout(resolve, 1_100));
  }
  await save('report.json', { provider: 'upstox', schemaVersion: 1, evaluationOnly: true, periods, unit: 'INR crore', basis: 'consolidated',
    accountingStandard: 'unverified', metricDefinitions: 'unverified', displayAndStorageRights: 'unconfirmed', companies });
  lines.push('', '## Checks before integration', '',
    '- Match both quarters to official INR consolidated statements. Do not convert the existing USD evidence to force a match.',
    '- Establish revenue versus total income, operating-profit definition, owner-attributable profit and exceptional-item treatment.',
    '- Record the original source URL and page for each comparison; API snapshots do not replace official evidence.',
    '- Confirm provider update timing, correction handling and permission to cache/display data to AlphaTrace users.',
    '- Existing monitoring and human-reviewed publications are unchanged. This evaluation has no database access.');
  await writeFile(join(directory, 'report.md'), lines.join('\n') + '\n', { mode: 0o600 });
  console.log(`Evaluation report: ${join(directory, 'report.md')}`);
  if (failed) process.exitCode = 1;
}
main().catch(error => {
  console.error(error instanceof UpstoxError ? error.message : 'Upstox evaluation could not complete. Check local file access and configuration.');
  process.exitCode = 1;
});
