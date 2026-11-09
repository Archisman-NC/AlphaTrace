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
export async function asUser<T>(ownerId:string, operation:(client:PoolClient)=>Promise<T>, options:{readOnlySnapshot?:boolean}={}):Promise<T> {
  if (!UUID.test(ownerId)) throw new Error('Invalid identity');
  const client=await pool().connect();
  try {
    // Prevent accidental use of a privileged connection that would defeat RLS.
    const role=await client.query('SELECT current_user AS name, rolsuper, rolbypassrls FROM pg_roles WHERE rolname=current_user');
    if (role.rows[0]?.name !== 'alphatrace_app' || role.rows[0]?.rolsuper || role.rows[0]?.rolbypassrls)
      throw new Error('Runtime database role must be alphatrace_app without RLS bypass.');
    await client.query(options.readOnlySnapshot?'BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY':'BEGIN');
    await client.query("SELECT set_config('app.user_id',$1,true)",[ownerId]);
    const result=await operation(client);
    await client.query('COMMIT'); return result;
  } catch(error) { await client.query('ROLLBACK'); throw error; }
  finally { client.release(); }
}
export async function closeDatabase() { await globalDb.alphaPool?.end(); globalDb.alphaPool=undefined; }
