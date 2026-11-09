import { officialReferenceChecks } from '../../scripts/lib/indianapi-references';
import { inspectIndianStats } from '../../scripts/lib/indianapi';
import { evaluateProvider,type ProviderSnapshot } from './provider-evaluation';
import { validateResearch,ResearchConflictError,spreadsheetMediaType,type EvidencePoint,type Source,type Comparison } from './research-contract';

// These describe the checked original bytes, never the time of an API retrieval.
const originals:Record<string,{publishedOn:string;fetchedAt:string;bytes:number;pages:number;page:number|null;mediaType:'application/pdf'|'text/html'}>={
 INFY:{publishedOn:'2026-10-23',fetchedAt:'2026-11-07T21:27:18.113Z',bytes:168098,pages:1,page:null,mediaType:'text/html'},
 HCLTECH:{publishedOn:'2026-10-12',fetchedAt:'2026-11-07T21:27:24.564Z',bytes:8365727,pages:27,page:2,mediaType:'application/pdf'},
 WIPRO:{publishedOn:'2026-10-15',fetchedAt:'2026-11-07T21:27:21.949Z',bytes:1146948,pages:1,page:null,mediaType:'text/html'},
 HINDUNILVR:{publishedOn:'2026-10-28',fetchedAt:'2026-11-07T21:30:35.469Z',bytes:5723982,pages:35,page:7,mediaType:'application/pdf'},
};
export function providerResearchDraft(snapshot:ProviderSnapshot){
 const evaluation=evaluateProvider(snapshot);
 if(!evaluation.referenceAvailable||!evaluation.reconciled)throw new ResearchConflictError('Resolve missing or mismatched values against official evidence before preparing a draft.');
 let source:Source, page:number|null,standard:'IFRS'|'Ind AS'='Ind AS';
 const report=inspectIndianStats(JSON.parse(snapshot.responseText),snapshot.requestedSymbol,new Date(snapshot.fetchedAt));
 const ref=officialReferenceChecks(report);
 if(snapshot.requestedSymbol==='TCS'){
  standard='IFRS';page=null;
  source={id:'official',title:'TCS Q2 FY27 consolidated IFRS workbook',url:'https://www.tcs.com/content/dam/tcs/pdf/discover-tcs/investor-relations/tcs-data-sheet/TCS-Data-Sheet-Q2FY27.xlsx',
   sha256:'0817bf86f26bdab3ddcf7e6b237e9e24a12501605937e9e8d8c1d626e0e868df',publishedOn:'2026-10-08',fetchedAt:'2026-11-05T16:10:13Z',
   kind:'facts',mediaType:spreadsheetMediaType,fingerprintKind:'document_bytes',retrieval:'operator_file',pages:0,bytes:389352};
 }else{
  const meta=originals[snapshot.requestedSymbol];
  if(!ref||!meta)throw new ResearchConflictError('Original evidence is not registered for this company.');
  page=meta.page;
  source={id:'official',title:`${snapshot.requestedSymbol} consolidated results, September 2026`,url:ref.sourceUrl,sha256:ref.sourceSha256,
   publishedOn:meta.publishedOn,fetchedAt:meta.fetchedAt,kind:'results',mediaType:meta.mediaType,fingerprintKind:'document_bytes',retrieval:'direct',pages:meta.pages,bytes:meta.bytes};
 }
 const point=(value:number,index:number,metric:'revenue'|'net_profit'|'operating_profit'):EvidencePoint=>({
  value:String(value),unit:'INR crore',basis:'consolidated',standard,treatment:'reported',periodEnd:evaluation.periods[index],periodType:'quarter',sourceId:source.id,page,
  locator:snapshot.requestedSymbol==='TCS'?`${metric==='revenue'?'Revenue':metric==='net_profit'?'Owner net income':'Operating income'}; INR millions / 10${index===0&&metric!=='revenue'?'; reported value in cell comment':''}`:
   `${metric==='revenue'?'Revenue from operations':'Profit attributable to owners'}; ${index===0?'June':'September'} quarter${snapshot.requestedSymbol==='WIPRO'?'; INR millions / 10':''}`,
  ...(snapshot.requestedSymbol==='TCS'?{worksheet:'IFRS-PnL,BS-INR',cell:`${index===0?'G':'H'}${metric==='revenue'?4:metric==='net_profit'?33:21}`}:{})});
 const comparisons:Comparison[]=[
  {metric:'revenue',previous:point(evaluation.rows[0].expected![0],0,'revenue'),current:point(evaluation.rows[0].expected![1],1,'revenue'),note:'Values come from the checked official filing; provider rounding is not copied into the draft.'},
  {metric:'net_profit',previous:point(evaluation.ownersProfit![0],0,'net_profit'),current:point(evaluation.ownersProfit![1],1,'net_profit'),note:'Owner-attributable profit from the official source. The API total-profit series is not substituted.'},
 ];
 if(snapshot.requestedSymbol==='TCS')comparisons.push({metric:'operating_profit',previous:point(16649,0,'operating_profit'),current:point(17553,1,'operating_profit'),note:'Reported IFRS operating income. June includes the exceptional legal expense from the cell comment; API segment totals are not used.'});
 return validateResearch({schemaVersion:1,symbol:snapshot.requestedSymbol,title:`${snapshot.requestedSymbol}: official quarterly comparison for review`,periodEnd:evaluation.periods[1],comparison:'QoQ',sources:[source],comparisons,
  limitations:['Prepared from dated official reference evidence. Recheck the original documents and add a supporting source before publication.',
   'API retrieval is not verification of the original filing. No new successful source check or human approval is recorded.',
   ...(snapshot.requestedSymbol==='TCS'?['June reported profit includes exceptional items; the headline workbook figures exclude them.']:['Add a third comparable metric from official evidence. Provider operating profit has not been mapped to reported operating income.'])],extractorVersion:'official-reference-draft-2026-11-08-v1'});
}
