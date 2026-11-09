import { test } from 'node:test';
import assert from 'node:assert/strict';
import { fixture } from './research-fixture';
import { researchChanges,sourceIdentity,draftReadiness } from '../src/lib/research-edit-contract';
import { downloadOfficial,inspectDocument } from '../src/lib/research-source';
test('revision diff identifies value, basis, removed locations and sources',()=>{
 const a=fixture(),b=structuredClone(a);a.comparisons[0].current.worksheet='Sheet1';b.comparisons[0].current.value='42';b.comparisons[0].current.basis='standalone';b.sources.pop();
 const changes=researchChanges(a,b).join('\n');assert.match(changes,/current value/);assert.match(changes,/current basis/);assert.match(changes,/current worksheet/);assert.match(changes,/Removed source/);
});
test('labels cannot change document identity; fingerprints and retrieval times do',()=>{
 const s=fixture().sources[0];assert.equal(sourceIdentity(s),sourceIdentity({...s,title:'Corrected label'}));assert.notEqual(sourceIdentity(s),sourceIdentity({...s,sha256:'f'.repeat(64)}));assert.notEqual(sourceIdentity(s),sourceIdentity({...s,fetchedAt:new Date(Date.parse(s.fetchedAt)+1000).toISOString()}));
});
test('readiness requires exact fresh successful fingerprints',()=>{
 const p=fixture(),check={status:'success',checked_at:new Date().toISOString(),fingerprints:p.sources.map(s=>s.sha256)};
 assert.equal(draftReadiness(p,check).length,0);assert.match(draftReadiness(p,{...check,fingerprints:[]}).join(),/Recheck/);assert.match(draftReadiness(p,{...check,status:'failed'}).join(),/Recheck/);assert.match(draftReadiness(p,check,Date.now()+8*86400000).join(),/Recheck/);
});
test('source retrieval rejects off-domain URLs before sending a request and forbids redirects',async()=>{
 let calls=0;const fake=(async(_url:unknown,init:RequestInit)=>{calls++;assert.equal(init.redirect,'error');return new Response('<html><body>Official quarterly results</body></html>');}) as typeof fetch;
 await assert.rejects(downloadOfficial('https://localhost/private','TCS',fake),/approved official/);assert.equal(calls,0);
 const bytes=await downloadOfficial('https://www.tcs.com/results.html','TCS',fake);assert.equal(calls,1);assert.deepEqual(await inspectDocument(bytes),{mediaType:'text/html',pages:1});
});
test('blocked access pages and arbitrary files are not evidence',async()=>{
 await assert.rejects(inspectDocument(Buffer.from('<html>Access denied</html>')),/blocked/);await assert.rejects(inspectDocument(Buffer.from('not an original')),/original PDF/);
});
