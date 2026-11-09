import pg, { type PoolClient } from 'pg';
import { UUID } from './access';
import { databaseConfig } from './database-config.mjs';
const globalDb=globalThis as unknown as { alphaPool?: pg.Pool };
function pool() {
  if (!process.env.DATABASE_URL) throw new Error('Database is not configured.');
  if (globalDb.alphaPool) return globalDb.alphaPool;
  const created = new pg.Pool({ ...databaseConfig(process.env.DATABASE_URL),
    max:3, idleTimeoutMillis:10000 });
  created.on('error', () => { console.error('An idle database connection was lost; a new request will reconnect.'); });
  globalDb.alphaPool = created;
  return created;
}
export async function asUser<T>(ownerId:string, operation:(client:PoolClient)=>Promise<T>, options:{readOnlySnapshot?:boolean;accountControl?:boolean;statementTimeoutMs?:number}={}):Promise<T> {
  if (!UUID.test(ownerId)) throw new Error('Invalid identity');
  const client=await pool().connect();
  try {
    // Prevent accidental use of a privileged connection that would defeat RLS.
    const role=await client.query('SELECT current_user AS name, rolsuper, rolbypassrls FROM pg_roles WHERE rolname=current_user');
    if (role.rows[0]?.name !== 'alphatrace_app' || role.rows[0]?.rolsuper || role.rows[0]?.rolbypassrls)
      throw new Error('Runtime database role must be alphatrace_app without RLS bypass.');
    await client.query(options.readOnlySnapshot?'BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY':'BEGIN');
    if(options.statementTimeoutMs)await client.query("SELECT set_config('statement_timeout',$1,true)",[String(Math.min(15000,Math.max(1000,options.statementTimeoutMs)))]);
    await client.query("SELECT set_config('app.user_id',$1,true)",[ownerId]);
    await client.query(options.accountControl?'SELECT pg_advisory_xact_lock(hashtext($1))':'SELECT pg_advisory_xact_lock_shared(hashtext($1))',['account:'+ownerId.toLowerCase()]);
    if(!options.accountControl&&(await client.query('SELECT 1 FROM alphatrace.account_closures WHERE owner_id=$1',[ownerId])).rowCount)throw new AccountClosedError('This account is closed.');
    if(!options.accountControl){
      const status=(await client.query('SELECT alphatrace.beta_access_status() AS status')).rows[0].status;
      if(!['legacy','active'].includes(status))throw new AccessRevokedError('Beta access is not active.');
    }
    const result=await operation(client);
    await client.query('COMMIT'); return result;
  } catch(error) { await client.query('ROLLBACK'); throw error; }
  finally { client.release(); }
}
export class AccessRevokedError extends Error {}
export class AccountClosedError extends Error {}
export async function closeDatabase() { await globalDb.alphaPool?.end(); globalDb.alphaPool=undefined; }
