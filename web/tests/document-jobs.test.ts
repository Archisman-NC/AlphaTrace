import test from 'node:test';
import assert from 'node:assert/strict';
import {randomUUID} from 'node:crypto';
import {documentCommand,documentJobsEnabled} from '../src/lib/document-job-contract';
import {isolatedTestEnvironment,writeLocalEnvironment,readLocalEnvironment} from '../scripts/lib/local-environment.mjs';
import {vercelEnvironment} from '../scripts/lib/vercel-environment';
import {productionEnvironment} from '../scripts/lib/trigger-environment';
import {mkdtempSync,rmSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
const command=()=>({commandId:randomUUID(),kind:'attach',input:{id:randomUUID(),url:'https://www.tcs.com/original',title:'Original',publishedOn:'2026-10-08',kind:'results'},files:[{field:'file',name:'original.pdf',size:10,sha256:'a'.repeat(64)}]});
test('document manifests reject unsupported inputs and oversized or ambiguous files',()=>{
 assert.equal(documentCommand(command()).files.length,1);
 for(const mutate of [(b:any)=>b.files[0].name='../secret.pdf',(b:any)=>b.files[0].name='program.exe',(b:any)=>b.files[0].size=20*1024*1024+1,(b:any)=>b.files.push(b.files[0]),(b:any)=>b.input.ownerId=randomUUID(),(b:any)=>b.files[0].sha256='not-a-digest',(b:any)=>b.kind='publish']){const b=command();mutate(b);assert.throws(()=>documentCommand(b));}
 assert.throws(()=>documentCommand({commandId:randomUUID(),kind:'tcs',input:{originalsConfirmed:'false'},files:[]}));
 assert.throws(()=>documentCommand({...command(),kind:'check',input:{id:randomUUID()}}));
 assert.equal(documentJobsEnabled({VERCEL:'1',NODE_ENV:'production'}),true);
});
test('isolated integration settings cannot inherit a production database',()=>{
 const values={DATABASE_URL:'postgresql://app:private@db.example/postgres',SUPABASE_SECRET_KEY:'private',TEST_DATABASE_URL:'postgresql://app:test@127.0.0.1/test',TEST_MIGRATION_DATABASE_URL:'postgresql://postgres:test@localhost/test'};
 const output=isolatedTestEnvironment(values);assert.equal(output.DATABASE_URL,values.TEST_DATABASE_URL);assert.equal(output.SUPABASE_SECRET_KEY,undefined);
 assert.throws(()=>isolatedTestEnvironment({...values,TEST_DATABASE_URL:values.DATABASE_URL}),/loopback/);
});
test('cloud export excludes administrative/test secrets and rejects development Trigger keys',()=>{
 const owner=randomUUID(),env={DATABASE_URL:'postgresql://alphatrace_app:test@db.example/postgres',DATABASE_SSL:'true',DATABASE_SSL_CA_PEM:'-----BEGIN CERTIFICATE-----\nTEST\n-----END CERTIFICATE-----',ALPHATRACE_INGEST_USER_ID:owner,ALPHATRACE_INVITED_USER_IDS:owner,ALPHATRACE_LOCAL_DEV:'false',SUPABASE_URL:'https://abcdefghijklmnopqrst.supabase.co',SUPABASE_PUBLISHABLE_KEY:'sb_publishable_test',SUPABASE_SECRET_KEY:'sb_secret_test',TRIGGER_SECRET_KEY:'tr_prod_test',TRIGGER_DEV_SECRET_KEY:'tr_dev_test',SUPABASE_DB_PASSWORD:'private',TEST_DATABASE_URL:'private',INDIANAPI_API_KEY:'private'};
 const output=vercelEnvironment(env);assert.equal(output.ALPHATRACE_DOCUMENT_JOBS,'true');for(const key of ['SUPABASE_DB_PASSWORD','TEST_DATABASE_URL','TRIGGER_DEV_SECRET_KEY','INDIANAPI_API_KEY'])assert.equal((output as Record<string,string>)[key],undefined);
 assert.throws(()=>vercelEnvironment({...env,TRIGGER_SECRET_KEY:'tr_dev_test'}),/Production key/);
 assert.throws(()=>vercelEnvironment({...env,DATABASE_URL:'postgresql://postgres:test@db.example/postgres'}));
 const dir=mkdtempSync(join(tmpdir(),'alphatrace-env-test-'));try{writeLocalEnvironment(env,dir);assert.deepEqual(readLocalEnvironment(dir),env);const keys=productionEnvironment(dir).map(v=>v.name);assert.ok(keys.includes('SUPABASE_SECRET_KEY'));for(const key of ['SUPABASE_DB_PASSWORD','TEST_DATABASE_URL','TRIGGER_SECRET_KEY','TRIGGER_DEV_SECRET_KEY'])assert.equal(keys.includes(key),false);}finally{rmSync(dir,{recursive:true,force:true});}
});
