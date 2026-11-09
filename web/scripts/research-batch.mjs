import { spawnSync } from 'node:child_process';
// Invoke with Node --env-file=.env.local; no credentials in command-line arguments.
const result=spawnSync('python3',['scripts/research/batch.py',...process.argv.slice(2)],{cwd:new URL('../../',import.meta.url),env:process.env,stdio:'inherit'});
process.exitCode=result.status??1;
