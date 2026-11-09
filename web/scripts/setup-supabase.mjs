// Run from web/. Reads private setup input; never logs passwords or connection URLs.
import { readLocalEnvironment, writeLocalEnvironment } from './lib/local-environment.mjs';
import { randomBytes } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import pg from 'pg';
import { databaseConfig } from '../src/lib/database-config.mjs';

async function main() {
  const input = readLocalEnvironment();
  if (!input.SUPABASE_DB_PASSWORD) throw new Error('Add SUPABASE_DB_PASSWORD to .env.local before running setup.');
  const project = new URL(input.SUPABASE_URL);
  if (!/^https:\/\/[a-z]{20}\.supabase\.co\/?$/.test(project.href)) throw new Error('Expected the Supabase project URL.');
  const ref = project.hostname.split('.')[0];
  const host = input.SUPABASE_DB_HOST;
  if (!/^aws-\d+-[a-z]+-[a-z]+-\d+\.pooler\.supabase\.com$/.test(host || '')) throw new Error('Copy the session pooler host from Dashboard > Connect into SUPABASE_DB_HOST.');
  if (!/^sb_publishable_[A-Za-z0-9_-]+$/.test(input.SUPABASE_PUBLISHABLE_KEY || '')) throw new Error('Expected a public publishable key.');
  function connection(role, password) {
    // Encode every non-unreserved character, including $, so Next dotenv expansion
    // and shell syntax cannot reinterpret passwords when the file is loaded.
    const encoded = encodeURIComponent(password).replace(/[!'()*]/g, c => `%${c.charCodeAt(0).toString(16).toUpperCase()}`);
    return `postgresql://${role}.${ref}:${encoded}@${host}:5432/postgres`;
  }
  const adminUrl = connection('postgres', input.SUPABASE_DB_PASSWORD);
  const tls = { DATABASE_SSL: 'true', DATABASE_SSL_CA_FILE: input.DATABASE_SSL_CA_FILE || '', DATABASE_SSL_CA_PEM: input.DATABASE_SSL_CA_PEM || '' };
  const admin = new pg.Client(databaseConfig(adminUrl, tls));
  await admin.connect();
  try {
    const identity = await admin.query('SELECT current_user AS name');
    if (identity.rows[0]?.name !== 'postgres') throw new Error('Setup requires the project administrator connection.');
    console.log('Verified TLS connection to project administrator.');
    Object.assign(process.env, tls, { MIGRATION_DATABASE_URL: adminUrl });
    await import('./migrate.mjs');
    const role = (await admin.query("SELECT rolcanlogin, rolsuper, rolbypassrls FROM pg_roles WHERE rolname='alphatrace_app'")).rows[0];
    if (!role || role.rolsuper || role.rolbypassrls) throw new Error('Runtime role security check failed.');
    let runtime = input.DATABASE_URL ? { ...input } : undefined;
    if (runtime) {
      const previous = new URL(runtime.DATABASE_URL);
      if (previous.hostname !== host || previous.username !== `alphatrace_app.${ref}` || previous.pathname !== '/postgres' || previous.search) throw new Error('Existing hosted file targets another connection; inspect it before continuing.');
    } else {
      if (role.rolcanlogin) throw new Error('Runtime role already has a login; supply its existing connection in .env.local rather than rotating it.');
      runtime = {
        DATABASE_URL: connection('alphatrace_app', randomBytes(32).toString('hex')),
        ...tls,
        ALPHATRACE_LOCAL_DEV: 'false',
        SUPABASE_URL: project.origin,
        SUPABASE_PUBLISHABLE_KEY: input.SUPABASE_PUBLISHABLE_KEY,
        ALPHATRACE_INVITED_USER_IDS: '',
      };
      // Save before activating the login so an interrupted setup can be resumed.
      writeLocalEnvironment({ ...input, ...runtime });
    }
    if (!role.rolcanlogin) {
      const password = decodeURIComponent(new URL(runtime.DATABASE_URL).password);
      await admin.query(`ALTER ROLE alphatrace_app LOGIN PASSWORD ${pg.escapeLiteral(password)}`);
    }
    const env = { ...process.env, ...runtime, ...tls, MIGRATION_DATABASE_URL: adminUrl };
    for (const file of ['tests/database.integration.ts', 'tests/portfolio.integration.ts']) {
      const result = spawnSync(process.execPath, ['--import', 'tsx', file], { env, encoding: 'utf8', timeout: 120000 });
      if (result.status !== 0) throw new Error(`Hosted verification failed in ${file}; check the local configuration.`);
      console.log(`Passed hosted ${file}`);
    }
    console.log('Hosted persistence and user isolation verified. Runtime and setup settings are in the private .env.local. Configure pilot identities before activation.');
  } finally { await admin.end(); }
}
main().catch(error => {
  // Avoid dumping driver errors, queries, or connection strings into shared logs.
  const safe = /^(Add SUPABASE|Expected |Copy the |Setup requires|Runtime role |Existing hosted |Hosted verification)/.test(error.message || '');
  console.error(safe ? error.message : `Supabase setup failed (${error.code || 'connection/setup error'}). Credentials were not logged; Inspect the local configuration before retrying.`);
  process.exitCode = 1;
});
