import { readdir, readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import pg from 'pg';
import { databaseConfig } from '../src/lib/database-config.mjs';
if (!process.env.MIGRATION_DATABASE_URL) throw new Error('MIGRATION_DATABASE_URL is required');
const client = new pg.Client(databaseConfig(process.env.MIGRATION_DATABASE_URL));
await client.connect();
try {
  await client.query("SELECT pg_advisory_lock(67241123)");
  await client.query('CREATE TABLE IF NOT EXISTS public.alphatrace_migrations (name text PRIMARY KEY, checksum text NOT NULL, applied_at timestamptz NOT NULL DEFAULT now())');
  await client.query('REVOKE ALL ON public.alphatrace_migrations FROM PUBLIC');
  const dir = new URL('../migrations/', import.meta.url);
  for (const name of (await readdir(dir)).filter(n => /^\d+.*\.sql$/.test(n)).sort()) {
    const sql = await readFile(new URL(name, dir), 'utf8');
    const checksum = createHash('sha256').update(sql).digest('hex');
    const applied = await client.query('SELECT checksum FROM public.alphatrace_migrations WHERE name=$1', [name]);
    if (applied.rowCount) {
      if (applied.rows[0].checksum !== checksum) throw new Error(`Applied migration changed: ${name}`);
      console.log(`Already applied: ${name}`); continue;
    }
    await client.query('BEGIN');
    try {
      await client.query(sql);
      await client.query('INSERT INTO public.alphatrace_migrations(name, checksum) VALUES ($1,$2)', [name,checksum]);
      await client.query('COMMIT'); console.log(`Applied: ${name}`);
    } catch (error) { await client.query('ROLLBACK'); throw error; }
  }
} finally { await client.end(); }
