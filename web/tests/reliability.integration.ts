import assert from 'node:assert/strict';
import {randomUUID} from 'node:crypto';
import pg from 'pg';
import {databaseConfig} from '../src/lib/database-config.mjs';
import {asUser,closeDatabase} from '../src/lib/db';
import {requestIdentity} from '../src/lib/identity-service';
import {ServiceUnavailableError,serviceBoundary} from '../src/lib/service-errors';
import {getHealth} from '../src/lib/health';
const operator=randomUUID(),reader=randomUUID(),db=new pg.Client(databaseConfig(process.env.MIGRATION_DATABASE_URL));await db.connect();
const savedEnv={SUPABASE_URL:process.env.SUPABASE_URL,SUPABASE_PUBLISHABLE_KEY:process.env.SUPABASE_PUBLISHABLE_KEY,ALPHATRACE_INVITED_USER_IDS:process.env.ALPHATRACE_INVITED_USER_IDS},originalFetch=globalThis.fetch;
try{
 await db.query('INSERT INTO alphatrace.research_operators(user_id) VALUES($1)',[operator]);
 const before=(await db.query('SELECT count(*)::int AS n FROM alphatrace.provider_runs')).rows[0].n;
 const view=await getHealth(operator);assert.ok(view.checkedAt);assert.ok(view.checks.find(c=>c.id==='database'));assert.ok(!JSON.stringify(view).includes(operator));assert.ok(!JSON.stringify(view).includes('owner_id'));assert.equal((await db.query('SELECT count(*)::int AS n FROM alphatrace.provider_runs')).rows[0].n,before,'health checks cannot consume provider requests');
 await assert.rejects(getHealth(reader),/operator/);await assert.rejects(asUser(reader,c=>c.query('SELECT alphatrace.operator_health_snapshot()')),/Operator/);
 await db.query("INSERT INTO alphatrace.account_closures(owner_id,requested_at) VALUES($1,now()-interval '4 hours')",[reader]);assert.equal((await getHealth(operator)).checks.find(c=>c.id==='closures')!.state,'attention');
 Object.assign(process.env,{SUPABASE_URL:'https://synthetic.supabase.co',SUPABASE_PUBLISHABLE_KEY:'sb_publishable_synthetic',ALPHATRACE_INVITED_USER_IDS:operator+','+reader});
 const request=new Request('https://alpha.example.test/api/research',{headers:{Authorization:'Bearer synthetic.valid.token'}});
 globalThis.fetch=async()=>Response.json({id:operator,email:'synthetic@example.test',email_confirmed_at:new Date().toISOString()});assert.equal(await requestIdentity(request),operator);
 globalThis.fetch=async()=>Response.json({id:reader,email_confirmed_at:new Date().toISOString()});assert.equal(await requestIdentity(request),null,'closed account remains unauthenticated');
 globalThis.fetch=async()=>Response.json({msg:'Invalid token'},{status:401});assert.equal(await requestIdentity(request),null);
 globalThis.fetch=async()=>Response.json({msg:'Unavailable'},{status:503});await assert.rejects(requestIdentity(request),ServiceUnavailableError);
 globalThis.fetch=async()=>{throw new TypeError('PRIVATE upstream failure');};await assert.rejects(requestIdentity(request),ServiceUnavailableError);
 // A verified identity with unavailable storage must be 503, not a false expired-login 401.
 globalThis.fetch=async()=>Response.json({id:operator,email_confirmed_at:new Date().toISOString()});await closeDatabase();const oldUrl=process.env.DATABASE_URL;process.env.DATABASE_URL='postgres://alphatrace_app:synthetic@127.0.0.1:1/no_database';try{await assert.rejects(requestIdentity(request),ServiceUnavailableError);}finally{await closeDatabase();process.env.DATABASE_URL=oldUrl;}
 console.log('PASS: operator-only aggregate health, no private identifiers or provider calls, overdue cleanup detection, verified identity, closed/expired sessions, upstream outage and database outage distinction');
}finally{globalThis.fetch=originalFetch;for(const [key,value] of Object.entries(savedEnv)){if(value===undefined)delete process.env[key];else process.env[key]=value;}await closeDatabase();await db.query('DELETE FROM alphatrace.account_closures WHERE owner_id=$1',[reader]);await db.query('DELETE FROM alphatrace.research_operators WHERE user_id=$1',[operator]);await db.end();}
