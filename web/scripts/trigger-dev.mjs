import { spawn } from 'node:child_process';
import { readLocalEnvironment } from './lib/local-environment.mjs';
const env = { ...process.env, ...readLocalEnvironment() };
if (!env.TRIGGER_DEV_SECRET_KEY?.startsWith('tr_dev_')) throw Error('Save TRIGGER_DEV_SECRET_KEY in .env.local before starting the development worker.');
env.TRIGGER_SECRET_KEY = env.TRIGGER_DEV_SECRET_KEY;
const child = spawn('npx', ['trigger.dev@4.8.3', 'dev'], { env, stdio: 'inherit' });
child.on('error', () => { console.error('Could not start the Trigger.dev development worker.'); process.exitCode = 1; });
child.on('exit', code => { process.exitCode = code ?? 1; });
for (const signal of ['SIGINT', 'SIGTERM']) process.on(signal, () => child.kill(signal));
