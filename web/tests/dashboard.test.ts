import test from 'node:test';
import assert from 'node:assert/strict';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { coverageStatus,trackedCoverage,evidenceNumber } from '../src/lib/dashboard-view';
import type { CompanyCoverage } from '../src/lib/research-view';
import UpdateEvidence from '../src/components/update-evidence';
import { fixture } from './research-fixture';

const now=Date.parse('2026-11-05T12:00:00Z');
function company(symbol='TCS'):CompanyCoverage {return {symbol,name:symbol,sector:'IT services',status:'published',version:1,periodEnd:'2026-09-30',publishedAt:'2026-11-05T11:00:00Z',sourceChanged:false,lastSuccessfulCheck:'2026-11-02T12:00:00Z',check:{status:'success',checked_at:'2026-11-02T12:00:00Z',message:'Test check',fingerprints:['1'.repeat(64)]}};}

test('coverage never presents a draft, withdrawal or stale/changed source as current checked research',()=>{
 const c=company();
 assert.equal(coverageStatus(c,now).label,'Sources checked');
 assert.deepEqual(coverageStatus({...c,status:'planned'},now),{label:'No published research',tone:'neutral',available:false});
 assert.equal(coverageStatus({...c,status:'withdrawn'},now).available,false);
 assert.equal(coverageStatus({...c,sourceChanged:true},now).label,'Source changed');
 assert.equal(coverageStatus({...c,check:{...c.check!,checked_at:'2026-10-01T12:00:00Z'}},now).label,'Source check overdue');
 assert.equal(coverageStatus({...c,check:{...c.check!,status:'failed'}},now).label,'Source check failed');
 assert.equal(coverageStatus({...c,check:null},now).label,'Sources unchecked');
});

test('tracked coverage deduplicates holdings plus follows and distinguishes unloaded coverage from empty',()=>{
 const portfolio={version:3,entries:[{symbol:'TCS',quantity:'23',followed:true},{symbol:'INFY',quantity:null,followed:true},{symbol:'WIPRO',quantity:null,followed:false}]};
 assert.deepEqual(trackedCoverage(portfolio,[company('TCS'),company('INFY'),company('WIPRO')])?.map(c=>c.symbol),['TCS','INFY']);
 assert.equal(trackedCoverage(portfolio,null),null);
 assert.deepEqual(trackedCoverage({version:0,entries:[]},[company()]),[]);
});

test('evidence display preserves exact decimals and both sources, and refuses misleading growth',()=>{
 assert.equal(evidenceNumber('999999999999999.123456'),'999,999,999,999,999.123456');
 assert.equal(evidenceNumber('-1000.00'),'-1,000.00');
 assert.equal(evidenceNumber(null),'Unavailable');
 const payload=fixture();payload.comparisons=payload.comparisons.slice(0,1);
 payload.comparisons[0].previous.treatment='excluding exceptional items';
 payload.comparisons[0].note='<script>alert(1)</script>';
 const html=renderToStaticMarkup(createElement(UpdateEvidence,{payload}));
 assert.match(html,/0 of 1 metrics are comparable/);
 assert.match(html,/Not comparable: treatment differs/);
 assert.doesNotMatch(html,/25\.00%/);
 assert.match(html,/href="https:\/\/www.tcs.com\/test-results"/);
 assert.match(html,/href="https:\/\/www.tcs.com\/test-release"/);
 assert.match(html,/excluding exceptional items/);
 assert.match(html,/&lt;script&gt;/);
 assert.doesNotMatch(html,/<script>/);
 payload.comparisons[0].previous.treatment='reported';
 payload.comparisons[0].previous.value='21.1';payload.comparisons[0].current.value='21.0';
 payload.comparisons[0].previous.unit=payload.comparisons[0].current.unit='%';
 assert.match(renderToStaticMarkup(createElement(UpdateEvidence,{payload})),/-0\.10 pp/);
});
