import {test} from 'node:test';
import assert from 'node:assert/strict';
import {healthView,type HealthSnapshot} from '../src/lib/health-contract';
import {serviceBoundary,ServiceUnavailableError} from '../src/lib/service-errors';
import {apiFetch} from '../src/lib/api-fetch';
const now='2026-11-09T12:00:00Z';
const snapshot=():HealthSnapshot=>({checkedAt:now,official:[],provider:[],quotaUsed:0,documents:{stalled:0,failed:0,cleanupOverdue:0},closures:{pending:0,overdue:0},emails:{failed:0,overdue:0},reviewQueue:{pending:0,olderThanDay:0},coverage:{published:3,withdrawn:0,needsSourceCheck:0}});
const target={symbol:'TCS',enabled:true,next_due_at:'2026-11-09T08:00:00Z',worker_seen_at:now,worker_mode:'cloud' as const,lease_until:null,latest_state:'success'};
test('operational health separates paused checks, quota waits and overdue workers',()=>{
 const s=snapshot();assert.equal(healthView(s,false).checks.find(c=>c.id==='official')!.state,'paused');assert.equal(healthView(s,false).checks.find(c=>c.id==='emails')!.state,'paused');
 s.provider=[target];s.quotaUsed=5;assert.equal(healthView(s,false).checks.find(c=>c.id==='provider')!.state,'waiting');
 s.quotaUsed=4;assert.equal(healthView(s,false).checks.find(c=>c.id==='provider')!.state,'attention');
 s.provider=[{...target,next_due_at:null,worker_seen_at:'2026-11-09T09:31:00Z'}];assert.equal(healthView(s,false).checks.find(c=>c.id==='provider')!.state,'ok');
 s.provider[0].worker_seen_at='2026-11-09T09:29:00Z';assert.equal(healthView(s,false).checks.find(c=>c.id==='provider')!.state,'attention');
 s.provider=[{...target,enabled:false,next_due_at:null,worker_seen_at:null,latest_state:'failed'}];assert.equal(healthView(s,false).checks.find(c=>c.id==='provider')!.state,'paused');
});
test('expired jobs, source failures, old review backlog and disabled email are distinguished',()=>{
 const s=snapshot();s.documents.stalled=1;s.closures.overdue=1;s.reviewQueue.olderThanDay=1;s.coverage.needsSourceCheck=1;s.emails.failed=1;
 const view=healthView(s,true);assert.equal(view.status,'attention');for(const id of ['documents','closures','review','coverage','emails'])assert.equal(view.checks.find(c=>c.id===id)!.state,'attention');
 assert.equal(healthView(s,false).checks.find(c=>c.id==='emails')!.state,'paused');
});
test('service errors return safe references without logging private error details',async()=>{
 const original=console.error,logs:string[]=[];console.error=v=>logs.push(String(v));
 try{const response=await serviceBoundary(async()=>{throw new ServiceUnavailableError('SECRET DATABASE URL');})();assert.equal(response.status,503);assert.equal(response.headers.get('retry-after'),'30');const body=await response.json();assert.ok(body.reference);assert.ok(!JSON.stringify(body).includes('SECRET'));assert.ok(!logs.join('').includes('SECRET'));
 const unexpected=await serviceBoundary(async()=>{throw Error('PRIVATE NOTE');})();assert.equal(unexpected.status,500);assert.ok(!JSON.stringify(await unexpected.json()).includes('PRIVATE'));}finally{console.error=original;}
});
test('network timeout preserves mutation uncertainty and never automatically retries',async()=>{
 const original=globalThis.fetch;let attempts=0;
 try{globalThis.fetch=async()=>{attempts++;throw new DOMException('Timed out','TimeoutError');};await assert.rejects(apiFetch('/api/inbox',{method:'POST',body:'synthetic'}),e=>e instanceof TypeError&&e.message.includes('may have completed'));assert.equal(attempts,1);
 globalThis.fetch=async()=>Response.json({error:'expired'},{status:401});assert.equal((await apiFetch('/api/inbox')).status,401);
 globalThis.fetch=async()=>new Response('{broken JSON',{status:200});const unreadable=await apiFetch('/api/inbox',{method:'POST'});await assert.rejects(unreadable.json(),e=>e instanceof TypeError&&e.message.includes('may have completed'));}finally{globalThis.fetch=original;}
});
