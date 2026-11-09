import type {WorkerMode} from './worker-status';
export const providerSymbols = ['TCS','INFY','HCLTECH','WIPRO','HINDUNILVR'] as const;
export type ProviderSymbol = typeof providerSymbols[number];
export const providerDailyLimit = 5;
export const providerCacheMs = 24 * 60 * 60 * 1000;
export type ProviderComparison = {
 label:string; definition:string; previous:string|null; current:string|null;
 expected:[number,number]|null; results:string[]; mapping:'revenue'|'definition_differs';
};
export type ProviderEvaluation = {
 periods:string[]; rows:ProviderComparison[]; sourceUrl:string|null; note:string;
 referenceAvailable:boolean; reconciled:boolean; ownersProfit:number[]|null;
};
export type ProviderImport = {
 id:string;symbol:ProviderSymbol;sha256:string;fetched_at:string;staged_at:string;last_seen_at:string;
 draft_id:string|null; evaluation:ProviderEvaluation;
};
export type ProviderDetection='not_recorded'|'baseline'|'changed'|'unchanged'|'stale';
export type ProviderRun = {id:string;symbol:ProviderSymbol;started_at:string;finished_at:string|null;state:string;message:string;import_id:string|null;detection_result?:ProviderDetection;review_id?:string|null;review_status?:'pending'|'reviewed'|null;worker_run_id?:string|null};
export type ProviderBatch={id:string;started_at:string;attempts:number;succeeded:number;failed:number;running:number;reviews:number;unchanged:number;unclassified:number};
export type ProviderView = {
 serverNow:string;usage:number;limit:number;nextCreditAt:string|null;
 targets:{symbol:ProviderSymbol;enabled:boolean;next_due_at:string|null;worker_seen_at:string|null;worker_mode?:WorkerMode}[];
 imports:ProviderImport[]; runs:ProviderRun[];latestBatch?:ProviderBatch|null;
};
export function providerRunOutcome(run:ProviderRun){
 if(run.state==='running')return 'Fetching data';
 if(run.state==='failed')return 'Retrieval failed · no change result';
 switch(run.detection_result){
  case 'baseline':return 'Baseline review created';
  case 'changed':return 'Changes sent to review';
  case 'unchanged':return 'No tracked changes';
  case 'stale':return 'Older observation · current baseline retained';
  default:return 'Change result not recorded for this older import';
 }
}
export function providerFreshness(fetchedAt:string,periodEnd:string|undefined,now=Date.now()) {
 const d=new Date(now),lastQuarter=new Date(Date.UTC(d.getUTCFullYear(),Math.floor(d.getUTCMonth()/3)*3,0)).toISOString().slice(0,10);
 if(!periodEnd||periodEnd<lastQuarter)return 'Previous reporting period';
 return now-Date.parse(fetchedAt)>providerCacheMs?'Refresh due':'Retrieved within 24 hours';
}
