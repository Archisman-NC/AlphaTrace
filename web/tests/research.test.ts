import test from 'node:test';
import assert from 'node:assert/strict';
import { fixture,check } from './research-fixture';
import { comparisonResult,validateResearch,publicationGaps,validateCheck,freshness,ResearchInputError,spreadsheetMediaType } from '../src/lib/research-contract';
import { contentHash } from '../src/lib/research';

test('operator workbook evidence requires original bytes and precise cell locations',()=>{
 const p=fixture();p.sources=p.sources.slice(0,1);
 Object.assign(p.sources[0],{mediaType:spreadsheetMediaType,retrieval:'operator_file',kind:'facts',pages:0});
 for(const comparison of p.comparisons)for(const point of [comparison.previous,comparison.current])Object.assign(point,{sourceId:p.sources[0].id,worksheet:'IFRS-PnL,BS-USD',cell:'G4'});
 const valid=validateResearch(p);
 assert.equal(valid.comparisons[0].current.cell,'G4');
 assert.equal(valid.sources[0].retrieval,'operator_file');
 assert.ok(publicationGaps(valid).some(s=>s.includes('official results')));
 const hash=contentHash(valid);valid.sources[0].fetchedAt=new Date().toISOString();assert.equal(contentHash(valid),hash);
 for(const cell of ['', 'A0', 'XFE1','A1048577','A1:A3']){const bad=structuredClone(p);bad.comparisons[0].current.cell=cell;assert.throws(()=>validateResearch(bad),ResearchInputError);}
 const missing=structuredClone(p);delete missing.comparisons[0].current.worksheet;assert.throws(()=>validateResearch(missing));
 const direct=structuredClone(p);direct.sources[0].retrieval='direct';assert.throws(()=>validateResearch(direct),/operator-supplied/);
 const text=structuredClone(p);text.sources[0].fingerprintKind='extracted_text';assert.throws(()=>validateResearch(text),/original document bytes/);
 const page=structuredClone(p);page.comparisons[0].current.page=1;assert.throws(()=>validateResearch(page));
});
test('exact quarter growth, negative change, percentage points and invalid baselines',()=>{
 const p=fixture(),c=p.comparisons[0];
 assert.equal(comparisonResult(c,'QoQ').change,'25.00');
 c.current.value='99.995';assert.equal(comparisonResult(c,'QoQ').change,'-0.01');
 c.previous.value='21.1';c.current.value='21.0';c.previous.unit=c.current.unit='%';
 assert.deepEqual(comparisonResult(c,'QoQ'),{available:true,change:'-0.10',changeUnit:'pp',reason:null});
 c.previous.unit=c.current.unit='USD million';c.previous.value='0';assert.equal(comparisonResult(c,'QoQ').available,false);
 c.previous.value='-3';assert.equal(comparisonResult(c,'QoQ').available,false);
 c.previous.value=null;assert.match(comparisonResult(c,'QoQ').reason!,/missing/);
});
test('basis, units, accounting standard, exceptional items and periods cannot mix',()=>{
 for(const [field,value] of [['basis','standalone'],['standard','Ind AS'],['unit','INR crore'],['treatment','excluding exceptional items'],['periodType','half_year']] as const){const c=fixture().comparisons[0];Object.assign(c.previous,{[field]:value});assert.equal(comparisonResult(c,'QoQ').available,false,field);}
 const c=fixture().comparisons[0];c.previous.periodEnd='2025-09-30';assert.equal(comparisonResult(c,'QoQ').available,false);assert.equal(comparisonResult(c,'YoY').available,true);
 c.previous.periodEnd='2026-03-31';c.current.periodEnd='2026-06-30';assert.equal(comparisonResult(c,'QoQ').available,true);
 c.previous.periodEnd='2025-12-31';c.current.periodEnd='2026-03-31';assert.equal(comparisonResult(c,'QoQ').available,true);
 c.previous.treatment=c.current.treatment='guidance';assert.equal(comparisonResult(c,'QoQ').available,false);
});
test('validation binds evidence to approved URLs, PDF pages and publication dates',()=>{
 assert.equal(validateResearch(fixture()).comparisons.length,3);
 for(const url of ['http://www.tcs.com/a','https://www.tcs.com.evil.test/a','https://user@www.tcs.com/a','https://www.tcs.com/a?redirect=evil','https://127.0.0.1/a']){const p=fixture();p.sources[0].url=url;assert.throws(()=>validateResearch(p),ResearchInputError);}
 const p=fixture();p.sources[0].mediaType='application/pdf';assert.throws(()=>validateResearch(p),/page/);p.comparisons.forEach(c=>c.current.page=1);assert.doesNotThrow(()=>validateResearch(p));
 p.sources[0].publishedOn='2026-09-01';assert.throws(()=>validateResearch(p),/later than its source/);
});
test('bank-specific gate, duplicate metrics, exact values and no source text execution',()=>{
 const p=fixture();p.symbol='HDFCBANK';p.sources.forEach((s,i)=>s.url=`https://www.hdfc.bank.in/test-${i}`);assert.throws(()=>validateResearch(p),/Bank research/);
 const q=fixture();q.comparisons[1].metric='revenue';assert.throws(()=>validateResearch(q),/Duplicate/);
 const r=fixture();r.comparisons[0].current.value='1e6';assert.throws(()=>validateResearch(r),/exact decimal/);
 const s=fixture();s.title='<script>publishEverything()</script>';assert.equal(validateResearch(s).title,s.title); // Plain data; React escapes on rendering.
});
test('publication gaps, source freshness and idempotent draft fingerprint',()=>{
 const p=fixture();assert.deepEqual(publicationGaps(p),[]);
 const hash=contentHash(p);p.sources[0].fetchedAt=new Date(Date.now()+1000).toISOString();assert.equal(contentHash(p),hash);
 p.comparisons[0].current.value='130';assert.notEqual(contentHash(p),hash);
 p.comparisons.pop();assert.equal(publicationGaps(p).length,1);
 assert.equal(freshness(null),'unchecked');assert.equal(freshness({status:'failed',checked_at:new Date().toISOString()}),'failed');
 assert.equal(freshness({status:'success',checked_at:new Date(Date.now()-8*86400000).toISOString()}),'stale');
 const c=check(p);c.fingerprints=[];assert.throws(()=>validateCheck(c),/fingerprints/);
});
