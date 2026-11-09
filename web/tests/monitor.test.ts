import test from 'node:test';
import assert from 'node:assert/strict';
import { nextMonitorTime,monitorNextAttempt,monitorState,monitorCommand,validateDiscovery,retryPending,monitorGuidance,type MonitorTarget } from '../src/lib/monitor-contract';
import { officialSourceUrl } from '../src/lib/research-contract';

test('source checks use fixed Monday/Thursday 6 pm IST across year boundaries and missed runs',()=>{
 assert.equal(nextMonitorTime(Date.parse('2026-11-05T12:29:59Z')),'2026-11-05T12:30:00.000Z');
 assert.equal(nextMonitorTime(Date.parse('2026-11-05T12:30:00Z')),'2026-11-09T12:30:00.000Z');
 assert.equal(nextMonitorTime(Date.parse('2026-12-31T13:00:00Z')),'2027-01-04T12:30:00.000Z');
 assert.equal(nextMonitorTime(Date.parse('2026-11-10T19:00:00Z')),'2026-11-12T12:30:00.000Z');
});
test('SEC discovery and document URLs are bound to Infosys and one accession',()=>{
 const d={checkedAt:new Date().toISOString(),feedUrl:'https://data.sec.gov/submissions/CIK0001067491.json',feedSha256:'a'.repeat(64),accession:'0001067491-26-000044',filedOn:'2026-10-28',periodEnd:'2026-09-30',url:'https://www.sec.gov/Archives/edgar/data/1067491/000106749126000044/index.htm',candidateCount:2};
 assert.equal(validateDiscovery(d,'INFY')?.accession,d.accession);
 assert.throws(()=>validateDiscovery(d,'TCS'));
 for(const changed of [{candidateCount:301},{feedSha256:'invalid'},{periodEnd:'2026-11-01'},{filedOn:'2026-02-30'},{accession:'0001067491-26-000042'},{url:d.url.replace('/1067491/','/123/')}])assert.throws(()=>validateDiscovery({...d,...changed},'INFY'));
 for(const url of ['https://www.sec.gov/unrelated.htm',d.url.replace('/1067491/','/123/'),d.url+'?redirect=x'])assert.throws(()=>officialSourceUrl(url,'INFY'));
 assert.equal(officialSourceUrl(d.url,'INFY'),d.url);
 assert.throws(()=>officialSourceUrl(d.url,'TCS'));
});
test('transient errors receive only one retry; denied access and validation wait for the next regular run',()=>{
 const now=Date.parse('2026-11-05T13:00:00Z');
 for(const code of ['network','timeout','http_429','http_503','interrupted']){
  assert.deepEqual(monitorNextAttempt(1,code,now),{due:'2026-11-05T13:05:00.000Z',attempt:1});
  assert.deepEqual(monitorNextAttempt(2,code,now),{due:'2026-11-09T12:30:00.000Z',attempt:0});
 }
 for(const code of ['http_403','validation',null])assert.equal(monitorNextAttempt(1,code,now).due,'2026-11-09T12:30:00.000Z');
});
test('a configured schedule is never presented as an online or successful runner without evidence',()=>{
 const now=Date.parse('2026-11-05T12:00:00Z');
 const target:MonitorTarget={symbol:'TCS',enabled:true,next_due_at:'2026-11-05T12:30:00Z',worker_seen_at:null,last_successful_check:null,active_run_id:null,lease_until:null,latest:null};
 assert.equal(monitorState(target,now).label,'Runner offline');
 assert.equal(monitorState({...target,enabled:false},now).label,'Paused');
 assert.equal(monitorState({...target,worker_seen_at:'2026-11-05T11:59:00Z'},now).label,'Awaiting first check');
 assert.equal(monitorState({...target,active_run_id:'id',lease_until:'2026-11-05T11:50:00Z'},now).label,'Run interrupted');
 assert.equal(monitorState({...target,active_run_id:'id',lease_until:'2026-11-05T12:03:00Z'},now).label,'Checking sources');
 assert.throws(()=>monitorCommand({symbol:'WIPRO',action:'resume'}));
 assert.throws(()=>monitorCommand({symbol:'TCS',action:'publish'}));
});
test('retry guidance uses durable retry state, never a nearby date or manual evidence check',()=>{
 const now=Date.parse('2026-11-08T10:00:00Z');
 const target:MonitorTarget={symbol:'TCS',enabled:true,next_due_at:'2026-11-08T10:05:00Z',worker_seen_at:'2026-11-08T10:00:00Z',last_successful_check:'2026-11-08T09:59:00Z',last_automatic_success:null,attempt:0,active_run_id:null,lease_until:null,latest:{id:'id',symbol:'TCS',state:'failed',attempt:1,started_at:'2026-11-08T10:00:00Z',finished_at:'2026-11-08T10:00:00Z',failure_code:'http_403',message:'blocked',draft_created:false}};
 assert.equal(retryPending(target),false);assert.equal(monitorState(target,now).label,'Source access blocked');assert.match(monitorGuidance(target,now),/official-file import/);
 const retry={...target,attempt:1,latest:{...target.latest!,failure_code:'timeout'}};
 assert.equal(retryPending(retry),true);assert.match(monitorGuidance(retry,now),/One automatic retry/);
 assert.equal(retryPending({...retry,enabled:false}),false);assert.match(monitorGuidance({...retry,worker_seen_at:null},now),/offline/);
 assert.equal(retryPending({...retry,attempt:0}),false);assert.equal(retryPending({...retry,active_run_id:'active'}),false);
});
