import { createHash } from 'node:crypto';
import { inspectIndianStats,indianStatsUrl,tcsRevenueCheck,tcsProfitChecks } from '../../scripts/lib/indianapi';
import { officialReferenceChecks } from '../../scripts/lib/indianapi-references';
import { ResearchInputError } from './research-contract';
import { providerSymbols,type ProviderSymbol,type ProviderEvaluation } from './provider-contract';

export type ProviderSnapshot = {requestedSymbol:ProviderSymbol;endpoint:string;responseText:string;sha256:string;fetchedAt:string};
export function validateProviderSnapshot(input:unknown):ProviderSnapshot {
 const s=input as ProviderSnapshot;
 if(!s||!providerSymbols.includes(s.requestedSymbol)||typeof s.responseText!=='string'||Buffer.byteLength(s.responseText)>5*1024*1024
  ||s.endpoint!==indianStatsUrl(s.requestedSymbol)||s.sha256!==createHash('sha256').update(s.responseText).digest('hex')
  ||typeof s.fetchedAt!=='string'||!Number.isFinite(Date.parse(s.fetchedAt))||Date.parse(s.fetchedAt)>Date.now()+60_000)
  throw new ResearchInputError('Provider snapshot failed identity, time or integrity validation.');
 try { inspectIndianStats(JSON.parse(s.responseText),s.requestedSymbol,new Date(s.fetchedAt)); }
 catch {throw new ResearchInputError('Provider response does not contain a supported quarterly series.');}
 return {requestedSymbol:s.requestedSymbol,endpoint:s.endpoint,responseText:s.responseText,sha256:s.sha256,fetchedAt:new Date(s.fetchedAt).toISOString()};
}
export function evaluateProvider(s:ProviderSnapshot):ProviderEvaluation {
 const report=inspectIndianStats(JSON.parse(s.responseText),s.requestedSymbol,new Date(s.fetchedAt));
 const reference=officialReferenceChecks(report);
 if(reference)return {periods:report.periods,sourceUrl:reference.sourceUrl,note:reference.note,referenceAvailable:true,
  ownersProfit:reference.ownersProfitInrCrore,
  reconciled:reference.checks.every(c=>c.results.every(r=>['exact','rounded'].includes(r))),
  rows:reference.checks.map(c=>({label:c.providerLabel,definition:c.officialDefinition,previous:c.observed[0],current:c.observed[1],
   expected:c.expectedInrCrore as [number,number],results:c.results,mapping:c.providerLabel==='Sales'?'revenue':'definition_differs'}))};
 if(s.requestedSymbol==='TCS'&&report.periods.join(',')==='2026-06-30,2026-09-30'){
  const revenue=tcsRevenueCheck(report),profits=tcsProfitChecks(report);
  const definitions=['Consolidated revenue','Total segment results before unallocable expenses','Total profit including non-controlling interests'];
  const expected:[[number,number],[number,number],[number,number]]=[[72275,73188],[18556,18815],[13420,13934]];
  return {periods:report.periods,sourceUrl:revenue.sourceUrl,referenceAvailable:true,ownersProfit:[13349,13884],
   reconciled:revenue.result==='matches_consolidated_revenue'&&profits[0].result==='matches_segment_results_before_unallocable_expenses'&&profits[1].result==='matches_total_profit_including_non_controlling_interests',
   note:'Operating Profit matches total segment results (Ind AS filing page 9), before unallocable expenses. It is not reported operating income. The IFRS workbook has adjusted June headlines; reported amounts are in cell comments. No accounting-basis conversion is inferred.',
   rows:report.rows.map((row,i)=>({label:row.providerLabel,definition:definitions[i],previous:row.previous,current:row.current,expected:expected[i],mapping:i===0?'revenue':'definition_differs',
    results:[row.previous,row.current].map((v,j)=>v===null?'missing':Number(v)===expected[i][j]?'exact':'mismatch')}))};
 }
 return {periods:report.periods,sourceUrl:null,referenceAvailable:false,reconciled:false,ownersProfit:null,
  note:'No checked official reference exists for these quarters. Obtain the originals before mapping these values.',
  rows:report.rows.map(row=>({label:row.providerLabel,definition:'Unverified definition',previous:row.previous,current:row.current,expected:null,results:['unverified','unverified'],mapping:row.providerLabel==='Sales'?'revenue':'definition_differs'}))};
}
