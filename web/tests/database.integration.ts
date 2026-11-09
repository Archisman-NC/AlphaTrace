import assert from 'node:assert/strict';
import { randomUUID } from 'node:crypto';
import pg from 'pg';
import { databaseConfig } from '../src/lib/database-config.mjs';
import { asUser,closeDatabase } from '../src/lib/db';
import { getWorkspace,saveWorkspace } from '../src/lib/workspaces';
const alice=randomUUID(),bob=randomUUID();
const admin=new pg.Client(databaseConfig(process.env.MIGRATION_DATABASE_URL));
await admin.connect();
try {
  const created=await saveWorkspace(alice,'Original research');
  const changed=await saveWorkspace(alice,'Revised research');
  assert.equal(created.id,changed.id,'upsert must preserve workspace identity');
  await closeDatabase(); // New pool / physical connection, not an in-memory result.
  assert.equal((await getWorkspace(alice))?.name,'Revised research');
  assert.equal(await getWorkspace(bob),null);
  assert.equal(await asUser(bob,async c=>(await c.query('SELECT * FROM alphatrace.workspaces WHERE id=$1',[created.id])).rowCount),0);
  assert.equal(await asUser(bob,async c=>(await c.query('UPDATE alphatrace.workspaces SET name=$1 WHERE id=$2',['Stolen',created.id])).rowCount),0);
  await assert.rejects(asUser(bob,c=>c.query('INSERT INTO alphatrace.workspaces(owner_id,name) VALUES ($1,$2)',[alice,'Spoofed'])),/row-level security/);
  assert.equal((await getWorkspace(alice))?.name,'Revised research','failed operation must rollback');
  const raw=new pg.Client(databaseConfig(process.env.DATABASE_URL));
  await raw.connect();
  try { assert.equal((await raw.query('SELECT * FROM alphatrace.workspaces')).rowCount,0,'no identity sees no rows'); }
  finally { await raw.end(); }
  const runtimeUrl=process.env.DATABASE_URL;
  await closeDatabase(); process.env.DATABASE_URL=process.env.MIGRATION_DATABASE_URL;
  await assert.rejects(getWorkspace(alice),/Runtime database role/);
  await closeDatabase(); process.env.DATABASE_URL=runtimeUrl;
  console.log('PASS: persistence, idempotent upsert, two-user RLS reads/writes, spoof rejection, rollback, missing identity and privileged-role rejection');
} finally {
  await closeDatabase();
  await admin.query('DELETE FROM alphatrace.workspaces WHERE owner_id=ANY($1::uuid[])',[[alice,bob]]);
  await admin.end();
}
