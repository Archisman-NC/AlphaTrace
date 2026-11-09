import test from 'node:test';
import assert from 'node:assert/strict';
import { workerOnline,workerLabel,cloudHeartbeatMs } from '../src/lib/worker-status';
import { monitorState,type MonitorTarget } from '../src/lib/monitor-contract';
import { databaseConfig } from '../src/lib/database-config.mjs';
import { mkdtemp,writeFile,rm,access } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { retrieveSource } from '../scripts/lib/retrieve-source';

test('hourly cloud check-ins remain healthy between runs while local workers expire promptly',()=>{
 const now=Date.parse('2026-11-09T10:00:00Z'),worker_seen_at=new Date(now-90*60_000).toISOString();
 assert.equal(workerOnline({worker_seen_at,worker_mode:'cloud'},now),true);
 assert.equal(workerOnline({worker_seen_at,worker_mode:'local'},now),false);
 assert.equal(workerOnline({worker_seen_at:null,worker_mode:'cloud'},now),false);
 assert.equal(workerLabel([{worker_seen_at:new Date(now-cloudHeartbeatMs-1).toISOString(),worker_mode:'cloud'},{worker_seen_at:new Date(now).toISOString(),worker_mode:'local'}],now),'Cloud scheduler overdue');
});
test('a healthy cloud scheduler does not conceal source failures or expired leases',()=>{
 const now=Date.now();
 const target:MonitorTarget={symbol:'TCS',enabled:true,worker_mode:'cloud',worker_seen_at:new Date(now-60*60_000).toISOString(),next_due_at:null,last_successful_check:null,active_run_id:null,lease_until:null,latest:{id:'test',symbol:'TCS',state:'failed',attempt:1,started_at:new Date(now).toISOString(),finished_at:null,failure_code:'http_403',message:'Access denied',draft_created:false}};
 assert.equal(monitorState(target,now).label,'Source access blocked');
 assert.equal(monitorState({...target,enabled:false},now).label,'Paused');
 assert.equal(monitorState({...target,active_run_id:'test',lease_until:new Date(now-1).toISOString()},now).label,'Run interrupted');
});
test('cloud CA content preserves TLS verification and does not read a laptop-only path',()=>{
 assert.deepEqual(databaseConfig('postgresql://app@db.example.com/db',{DATABASE_SSL:'true',DATABASE_SSL_CA_PEM:'test certificate',DATABASE_SSL_CA_FILE:'/unavailable/cert'}).ssl,{rejectUnauthorized:true,ca:'test certificate'});
});

test('source subprocess receives no application credentials and its temporary cache is removed',async()=>{
 const scripts=await mkdtemp(join(tmpdir(),'alphatrace-reader-test-'));
 const previous=process.env.ALPHATRACE_TEST_PRIVATE;process.env.ALPHATRACE_TEST_PRIVATE='must-not-reach-python';
 try{
  await writeFile(join(scripts,'batch.py'),`import os,json,pathlib,sys
root=pathlib.Path(os.environ['ALPHATRACE_RESEARCH_ROOT'])
assert 'ALPHATRACE_TEST_PRIVATE' not in os.environ
assert 'DATABASE_URL' not in os.environ
assert 'INDIANAPI_API_KEY' not in os.environ
(root/'TCS.json').write_text(json.dumps({'check':{'symbol':'TCS','status':'failed'},'directory':str(root)}))
sys.exit(1)
`);
  const bundle=await retrieveSource('TCS',scripts);
  assert.equal(bundle.check.status,'failed');assert.ok(bundle.directory);
  await assert.rejects(access(bundle.directory));
  await writeFile(join(scripts,'batch.py'),`import os,json,pathlib,sys
(pathlib.Path(os.environ['ALPHATRACE_RESEARCH_ROOT'])/'TCS.json').write_text(json.dumps({'check':{'symbol':'INFY','status':'success'}}))
`);
  assert.equal((await retrieveSource('TCS',scripts)).failureCode,'validation');
 }finally{
  if(previous===undefined)delete process.env.ALPHATRACE_TEST_PRIVATE;else process.env.ALPHATRACE_TEST_PRIVATE=previous;
  await rm(scripts,{recursive:true,force:true});
 }
});
