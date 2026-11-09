import { readFileSync, writeFileSync, chmodSync } from 'node:fs';
import { parseEnv } from 'node:util';

export function readLocalEnvironment(directory = process.cwd()) {
  return parseEnv(readFileSync(`${directory}/.env.local`, 'utf8'));
}
export function writeLocalEnvironment(values, directory = process.cwd()) {
  // Actual newlines keep PEM certificates compatible with both Node and Next dotenv.
  const text = Object.entries(values).sort(([a], [b]) => a.localeCompare(b)).map(([key, value]) => {
    if (!/^[A-Z][A-Z0-9_]*$/.test(key) || /['$]/.test(value)) throw Error('Environment value requires explicit encoding before saving.');
    return `${key}='${value}'`;
  }).join('\n') + '\n';
  const path = `${directory}/.env.local`;
  writeFileSync(path, text, { mode: 0o600 });
  chmodSync(path, 0o600);
}
export function isolatedTestEnvironment(values) {
  const test = Object.fromEntries(Object.entries(values).filter(([key]) => key.startsWith('TEST_')).map(([key, value]) => [key.slice(5), value]));
  for (const key of ['DATABASE_URL', 'MIGRATION_DATABASE_URL']) {
    if (!test[key] || !['127.0.0.1', 'localhost', '[::1]'].includes(new URL(test[key]).hostname)) throw Error('Integration tests require loopback TEST_DATABASE_URL and TEST_MIGRATION_DATABASE_URL.');
  }
  return test;
}
