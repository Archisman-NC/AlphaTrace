import test from 'node:test';
import assert from 'node:assert/strict';
import { normalizeProvider,providerSemanticHash,detectProviderChanges,meaningfulProviderChange } from '../src/lib/provider-changes';
import type { ProviderSnapshot } from '../src/lib/provider-evaluation';
const normalized=(raw:unknown)=>normalizeProvider({responseText:JSON.stringify(raw),requestedSymbol:'TCS',fetchedAt:'2027-02-01T00:00:00Z'} as ProviderSnapshot);
const a=()=>({'Sales':{'Jun 2026':'1,234.500','Sep 2026':2000},'Operating Profit':{'Jun 2026':100,'Sep 2026':200},'Net Profit':{'Jun 2026':80,'Sep 2026':160}});
test('formatting, key order and unrelated fields do not create financial change alerts',()=>{
 const first=normalized(a()),second=normalized({'EPS':{'Sep 2026':99},'Net Profit':{'Sep 2026':'160.00','Jun 2026':'080'},'Operating Profit':{'Sep 2026':200,'Jun 2026':100},Sales:{'Sep 2026':'2,000.00','Jun 2026':1234.5}});
 assert.equal(providerSemanticHash(first),providerSemanticHash(second));assert.equal(meaningfulProviderChange(detectProviderChanges(first,second)),false);
});
test('new quarters, historic corrections, missing values and removed periods remain distinct',()=>{
 const first=normalized(a()),raw=a();raw.Sales['Jun 2026']='1,234.501';
 const changed=normalized({...raw,Sales:{...raw.Sales,'Dec 2026':3000},'Net Profit':{'Jun 2026':80,'Sep 2026':null,'Dec 2026':180}});
 const diff=detectProviderChanges(first,changed);
 assert.deepEqual(diff.newQuarters,['2026-12-31']);
 assert.ok(diff.changes.some(c=>c.metric==='Sales'&&c.period==='2026-06-30'&&c.kind==='changed'&&c.before==='1234.5'&&c.after==='1234.501'));
 assert.ok(diff.changes.some(c=>c.metric==='Net Profit'&&c.period==='2026-09-30'&&c.kind==='missing'));
 const removed=detectProviderChanges(first,normalized({Sales:{'Jun 2026':1234.5}}));
 assert.deepEqual(removed.removedQuarters,['2026-09-30']);assert.deepEqual(removed.missingMetrics,['Operating Profit','Net Profit']);
});
test('history backfill is not described as a new reporting quarter',()=>{
 const first=normalized(a()),diff=detectProviderChanges(first,normalized({...a(),Sales:{'Mar 2026':999,...a().Sales}}));
 assert.deepEqual(diff.newQuarters,[]);assert.deepEqual(diff.addedHistory,['2026-03-31']);
 const emptyQuarter=detectProviderChanges(first,normalized({...a(),Sales:{...a().Sales,'Dec 2026':null}}));
 assert.deepEqual(emptyQuarter.newQuarters,['2026-12-31']);assert.equal(meaningfulProviderChange(emptyQuarter),true);
});
test('first observation is a baseline and reverse changes are meaningful',()=>{
 const first=normalized(a()),second=normalized({...a(),Sales:{...a().Sales,'Sep 2026':2001}});
 const baseline=detectProviderChanges(null,first);assert.equal(baseline.baseline,true);assert.equal(baseline.changes.length,6);assert.deepEqual(baseline.newQuarters,[]);
 assert.equal(detectProviderChanges(first,second).changes[0].after,'2001');assert.equal(detectProviderChanges(second,first).changes[0].after,'2000');
});
test('canonical decimals preserve tiny numeric values and normalize signed zero',()=>{
 const q=normalized({Sales:{'Jun 2026':1e-7,'Sep 2026':'-0.000'}});
 assert.equal(q[0].values.Sales,'0.0000001');assert.equal(q[1].values.Sales,'0');
});
