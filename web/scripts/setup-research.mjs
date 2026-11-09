// Explicit operator enrollment; administrative credentials are never uploaded to deployment runtimes.
import { readLocalEnvironment,writeLocalEnvironment } from './lib/local-environment.mjs';
import { randomBytes } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import pg from 'pg';
import { databaseConfig } from '../src/lib/database-config.mjs';

async function main(){
 const operator=process.argv[2];
 if(!/^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(operator??''))throw Error('Pass the invited operator UUID.');
 const input=readLocalEnvironment(),hosted=input,current=input;
 if(!hosted.ALPHATRACE_INVITED_USER_IDS?.split(',').map(s=>s.trim()).includes(operator)||current.DATABASE_URL!==hosted.DATABASE_URL)throw Error('Operator must already be invited on the active hosted configuration.');
 const adminUrl=new URL(hosted.DATABASE_URL);
 adminUrl.username='postgres.'+new URL(input.SUPABASE_URL).hostname.split('.')[0];
 adminUrl.password=input.SUPABASE_DB_PASSWORD;
 Object.assign(process.env,hosted,{MIGRATION_DATABASE_URL:adminUrl.href});
 await import('./migrate.mjs');
 const result=spawnSync(process.execPath,['--import','tsx','tests/research.integration.ts'],{env:process.env,encoding:'utf8',timeout:120000});
 if(result.status!==0)throw Error('Hosted research integration verification failed. No operator or ingestion credential was activated.');
 console.log('Hosted research workflow and RLS verification passed.');
 const db=new pg.Client(databaseConfig(adminUrl.href));await db.connect();
 try{
  const user=await db.query('SELECT id FROM auth.users WHERE id=$1 AND email_confirmed_at IS NOT NULL',[operator]);
  if(!user.rowCount)throw Error('Operator must be a confirmed Auth user.');
  await db.query('INSERT INTO alphatrace.research_operators(user_id) VALUES ($1) ON CONFLICT DO NOTHING',[operator]);
 }finally{await db.end();}
 const token=hosted.ALPHATRACE_INGEST_TOKEN??randomBytes(32).toString('hex');
 Object.assign(current,{ALPHATRACE_INGEST_USER_ID:operator,ALPHATRACE_INGEST_TOKEN:token});
 writeLocalEnvironment(current);
 console.log('Invited operator enabled; draft-only ingestion credential saved privately. No research was published.');
}
main().catch(error=>{console.error(/^(Pass |Operator must |Hosted research )/.test(error.message??'')?error.message:'Research setup failed; credentials were not logged.');process.exitCode=1;});
