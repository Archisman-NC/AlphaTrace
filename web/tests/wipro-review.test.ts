import {test} from 'node:test';
import assert from 'node:assert/strict';
import {fixture} from './research-fixture';
import {completeWiproDraft,wiproReviewOriginals} from '../scripts/lib/wipro-review';
import {comparisonResult,validateResearch,type Source} from '../src/lib/research-contract';
const sources=():Source[]=>wiproReviewOriginals.map(ref=>({...ref,fetchedAt:'2026-11-08T10:10:20Z',bytes:100,pages:1,mediaType:'text/html',fingerprintKind:'document_bytes',retrieval:'operator_file'}));
const base=()=>({...fixture(),symbol:'WIPRO'});
test('Wipro package preserves owner-profit definition and separates segment margin',()=>{
 const p=completeWiproDraft(base(),sources());
 assert.deepEqual(p.comparisons.map(c=>[c.previous.value,c.current.value]),[['24478.6','24752.7'],['3352','3200.6'],['16.0','16.5']]);
 assert.deepEqual(p.comparisons.map(c=>comparisonResult(c,p.comparison).change),['1.12','-4.52','0.50']);
 assert.equal(comparisonResult(p.comparisons[2],'QoQ').changeUnit,'pp');
 assert.equal(p.comparisons[2].current.scope,'IT services segment');assert.equal(p.comparisons[2].current.standard,'IFRS');assert.equal(p.comparisons[0].current.standard,'Ind AS');
 assert.ok(p.limitations.some(s=>s.includes('4.7%')));
});
test('dated Wipro package refuses unknown originals and later periods',()=>{
 const changed=sources();changed[0].sha256='0'.repeat(64);assert.throws(()=>completeWiproDraft(base(),changed),/exact official/);
 assert.throws(()=>completeWiproDraft({...base(),periodEnd:'2026-12-31'},sources()),/restricted/);
 assert.throws(()=>completeWiproDraft({...base(),symbol:'TCS'},sources()),/restricted/);
});
test('business scope survives validation and blocks segment-to-group comparisons',()=>{
 const p=completeWiproDraft(base(),sources()),c=structuredClone(p.comparisons[2]);
 assert.equal(validateResearch(p).comparisons[2].current.scope,'IT services segment');
 delete c.previous.scope;assert.equal(comparisonResult(c,'QoQ').available,false);assert.match(comparisonResult(c,'QoQ').reason!,/business scope/);
 const legacy=fixture();assert.deepEqual(validateResearch(legacy),legacy);assert.equal(comparisonResult(legacy.comparisons[0],'QoQ').available,true);
});
