import { spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { readLocalEnvironment, isolatedTestEnvironment } from './lib/local-environment.mjs';

const file = process.argv[2];
if (!file || !/^(tests\/[a-z-]+\.integration\.ts|scripts\/migrate\.mjs|\.local\/[a-z-]+\.ts)$/.test(file)) throw Error('Choose a local integration test, migration or preview script.');
const test = isolatedTestEnvironment(existsSync('.env.local') ? readLocalEnvironment() : process.env);
// Deliberately do not forward hosted credentials or provider keys to isolated tests.
const env = { PATH: process.env.PATH, HOME: process.env.HOME, TMPDIR: process.env.TMPDIR, LANG: process.env.LANG, ...test };
const result = spawnSync(process.execPath, ['--import', 'tsx', file, ...process.argv.slice(3)], { env, stdio: 'inherit' });
process.exitCode = result.status ?? 1;
