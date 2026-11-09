import type { ResearchPayload,ResearchCheck } from '../src/lib/research-contract';
export function fixture(title='SYNTHETIC TEST — not investment research'):ResearchPayload {
 const sources=([
  {id:'results',title:'Synthetic results fixture',url:'https://www.tcs.com/test-results',kind:'results'},
  {id:'release',title:'Synthetic supporting fixture',url:'https://www.tcs.com/test-release',kind:'release'},
 ] as const).map((s,i)=>({...s,sha256:String(i+1).repeat(64),publishedOn:'2026-10-08',fetchedAt:new Date().toISOString(),mediaType:'text/html' as const,fingerprintKind:'document_bytes' as const,retrieval:'direct' as const,pages:1,bytes:100}));
 const point={unit:'USD million' as const,basis:'consolidated' as const,standard:'IFRS' as const,treatment:'reported' as const,periodType:'quarter' as const,locator:'Synthetic fixture row',page:null};
 return {schemaVersion:1,symbol:'TCS',title,periodEnd:'2026-09-30',comparison:'QoQ',sources,extractorVersion:'synthetic-test-v1',limitations:['SYNTHETIC FIXTURE. These numbers do not describe any company.'],comparisons:(['revenue','operating_profit','net_profit'] as const).map(metric=>({metric,previous:{...point,value:'100',periodEnd:'2026-06-30',sourceId:'release'},current:{...point,value:'125',periodEnd:'2026-09-30',sourceId:'results'},note:'Synthetic comparison'}))};
}
export function check(p:ResearchPayload,time=Date.now()):ResearchCheck{return {symbol:p.symbol,checkedAt:new Date(time).toISOString(),status:'success',message:'SYNTHETIC integration check',fingerprints:p.sources.map(s=>s.sha256)};}
