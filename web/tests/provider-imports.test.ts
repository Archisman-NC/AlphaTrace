import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { validateProviderSnapshot,evaluateProvider } from '../src/lib/provider-evaluation';
import { providerResearchDraft } from '../src/lib/provider-draft';
import { publicationGaps,officialSourceUrl } from '../src/lib/research-contract';
import { providerFreshness } from '../src/lib/provider-contract';
import { indianStatsUrl } from '../scripts/lib/indianapi';
function snapshot(symbol='TCS',sales=[72275,73188],op=[18556,18815],net=[13420,13934]){
 const responseText=JSON.stringify(Object.fromEntries([['Sales',sales],['Operating Profit',op],['Net Profit',net]].map(([k,v])=>[k,{'Jun 2026':v[0],'Sep 2026':v[1]}])));
 return validateProviderSnapshot({requestedSymbol:symbol,endpoint:indianStatsUrl(symbol),responseText,sha256:createHash('sha256').update(responseText).digest('hex'),fetchedAt:'2026-11-07T18:00:00Z'});
}
test('staging rejects forged hashes, endpoints, future times and incompatible schemas',()=>{
 const s=snapshot();
 for(const changed of [{sha256:'a'.repeat(64)},{endpoint:'https://evil.example/'},{fetchedAt:'2099-01-01T00:00:00Z'},{requestedSymbol:'UNKNOWN'}])assert.throws(()=>validateProviderSnapshot({...s,...changed}));
 assert.throws(()=>validateProviderSnapshot(null));
});
test('TCS segment matching is distinct from reported IFRS profit and approval',()=>{
 const s=snapshot(),e=evaluateProvider(s),p=providerResearchDraft(s);
 assert.equal(e.reconciled,true);assert.equal(e.rows[1].mapping,'definition_differs');
 assert.match(e.rows[1].definition,/segment/);
 assert.equal(p.comparisons[1].previous.value,'13349');assert.equal(p.comparisons[2].previous.value,'16649');
 assert.equal(p.comparisons[2].current.value,'17553');assert.equal(p.comparisons[2].previous.standard,'IFRS');
 assert.match(p.comparisons[2].previous.locator,/cell comment/);assert.equal(p.sources[0].fetchedAt,'2026-11-05T16:10:13.000Z');
 assert.ok(publicationGaps(p).length,'preparation is not publication readiness');
 assert.throws(()=>providerResearchDraft(snapshot('TCS',[72275,1])),/Resolve/);
});
test('Wipro rounded provider values never replace precise official owner-profit observations',()=>{
 const s=snapshot('WIPRO',[24479,24753],[4633,4863],[3356,3200]);
 const p=providerResearchDraft(s);assert.equal(evaluateProvider(s).reconciled,true);
 assert.equal(p.comparisons[0].current.value,'24752.7');assert.equal(p.comparisons[1].current.value,'3200.6');
 assert.equal(p.comparisons.length,2);assert.equal(p.sources[0].publishedOn,'2026-10-15');
 assert.ok(publicationGaps(p).some(g=>g.includes('three')));
 assert.equal(officialSourceUrl(p.sources[0].url,'WIPRO'),p.sources[0].url);
 assert.throws(()=>officialSourceUrl(p.sources[0].url,'INFY'));
 assert.throws(()=>officialSourceUrl(p.sources[0].url.replace('/1123799/','/999999/'),'WIPRO'));
});
test('later periods cannot reuse dated references and retrieval age is not report freshness',()=>{
 const s=snapshot();s.fetchedAt='2027-02-01T00:00:00Z';
 assert.equal(evaluateProvider(s).referenceAvailable,false);assert.throws(()=>providerResearchDraft(s));
 assert.equal(providerFreshness('2026-11-07T18:00:00Z','2026-09-30',Date.parse('2026-11-08T18:00:01Z')),'Refresh due');
 assert.equal(providerFreshness('2027-02-01T18:00:00Z','2026-09-30',Date.parse('2027-02-01T18:01:00Z')),'Previous reporting period');
});
