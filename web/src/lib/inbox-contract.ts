import { metrics,type Metric,type ResearchPayload } from './research-contract';
import type { Assumption,ThesisRevision } from './thesis-input';
import type { PublishedResearch } from './research-view';

export const inboxFilters=['pending','deferred','reviewed','all'] as const;
export type InboxFilter=typeof inboxFilters[number];
export type ReviewAction='keep'|'dismiss'|'defer'|'revise'|'reopen';
export type ReviewState='unreviewed'|'kept'|'dismissed'|'deferred'|'revised';
export type InboxSnapshot={ruleVersion:'metric-topic-v1';reason:'holding'|'watchlist'|'holding and watchlist';thesis:ThesisRevision|null;matches:{assumption:Assumption;metrics:Metric[]}[]};
export type InboxItem={update_id:string;symbol:string;title:string;publication_version:number;period_end:string;created_at:string;version:number;state:ReviewState;revisit_at:string|null;match_count:number;withdrawn_at:string|null;withdrawal_reason:string|null;newer_update_id:string|null};
export type InboxList={items:InboxItem[];nextCursor:string|null;counts:{pending:number;deferred:number;reviewed:number;all:number}};
export type ReviewEvent={version:number;action:ReviewAction;note:string;revisit_at:string|null;thesis_revision:ThesisRevision|null;created_at:string};
export type InboxDetail={item:InboxItem;snapshot:InboxSnapshot;publication:PublishedResearch;events:ReviewEvent[];hasMoreEvents:boolean;currentThesisVersion:number|null;canRecordRevision:boolean;check:{status:string;checked_at:string;message:string;fingerprints:string[]}|null};
export class InboxInputError extends Error {}
export class InboxConflictError extends Error {}
export const inboxUuid=(v:unknown):string=>{if(typeof v!=='string'||!/^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(v))throw new InboxInputError('Invalid review identifier.');return v.toLowerCase();};
const aliases:Record<Metric,string[]>={
 revenue:['revenue','sales'],operating_profit:['operating profit'],net_profit:['net profit','profit attributable to owners'],
 operating_margin:['operating margin'],deal_tcv:['total contract value','tcv'],employees:['employees','headcount','workforce'],
 operating_cash_flow:['operating cash flow','cash flow from operations'],debt:['debt','borrowings'],
 net_interest_income:['net interest income','nii'],net_interest_margin:['net interest margin','nim'],
 gross_npa:['gross npa','gnpa','gross non performing assets'],net_npa:['net npa','nnpa','net non performing assets'],
};
const normalized=(s:string)=>s.toLowerCase().replace(/[^a-z0-9]+/g,' ').trim();
// Only the user's explicit metric/topic field is matched. No inferred beliefs, thresholds or sentiment.
export function matchAssumptions(assumptions:Assumption[],payload:ResearchPayload):InboxSnapshot['matches']{
 return assumptions.flatMap(assumption=>{
  const topic=' '+normalized(assumption.metric)+' ';
  const matched=payload.comparisons.map(c=>c.metric).filter(metric=>aliases[metric].some(alias=>topic.includes(' '+normalized(alias)+' ')));
  return matched.length?[{assumption,metrics:[...new Set(matched)]}]:[];
 });
}
export const metricLabel=(metric:Metric)=>metrics[metric];
export type ReviewCommand={updateId:string;commandId:string;version:number;action:ReviewAction;note:string;deferUntil:string|null;thesisVersion:number|null};
export function validateReview(input:unknown):ReviewCommand {
 if(!input||typeof input!=='object'||Array.isArray(input))throw new InboxInputError('Invalid review.');
 const b=input as Record<string,unknown>;
 if(!['keep','dismiss','defer','revise','reopen'].includes(String(b.action)))throw new InboxInputError('Choose a review response.');
 if(!Number.isInteger(b.version)||(b.version as number)<0||(b.version as number)>2147483646)throw new InboxInputError('Invalid review version.');
 if(typeof b.note!=='string'||b.note.trim().length<3||b.note.length>2000||/[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]/.test(b.note))throw new InboxInputError('Explain your response in 3–2,000 characters.');
 let deferUntil:string|null=null,thesisVersion:number|null=null;
 if(b.action==='defer'){
  if(typeof b.deferUntil!=='string'||!/^\d{4}-\d{2}-\d{2}$/.test(b.deferUntil)||!Number.isFinite(Date.parse(b.deferUntil))||new Date(b.deferUntil).toISOString().slice(0,10)!==b.deferUntil)throw new InboxInputError('Choose a valid revisit date.');
  deferUntil=b.deferUntil;
 }
 if(b.action==='revise'){
  if(!Number.isInteger(b.thesisVersion)||(b.thesisVersion as number)<1||(b.thesisVersion as number)>2147483647)throw new InboxInputError('Save your revised thesis and link this publication first.');
  thesisVersion=b.thesisVersion as number;
 }
 return {updateId:inboxUuid(b.updateId),commandId:inboxUuid(b.commandId),version:b.version as number,action:b.action as ReviewAction,note:b.note.trim(),deferUntil,thesisVersion};
}
export function revisitTimestamp(date:string,now=Date.now()):string{
 const ms=Date.parse(date+'T09:00:00+05:30');
 if(ms<=now||ms>now+366*86400000)throw new InboxInputError('Choose a future revisit date within one year. Reviews return at 9 am IST.');
 return new Date(ms).toISOString();
}
export function inboxFilter(value:unknown):InboxFilter{if(value===undefined||value===null)return 'pending';if(!inboxFilters.includes(value as InboxFilter))throw new InboxInputError('Invalid inbox filter.');return value as InboxFilter;}
export function inboxCursor(value?:string|null):[string,string]|null{
 if(!value)return null;
 const [date,id,...rest]=value.split('|');
 if(rest.length||!/^\d{4}-\d{2}-\d{2}T/.test(date)||date.length>32||!Number.isFinite(Date.parse(date)))throw new InboxInputError('Invalid inbox page.');
 return [date,inboxUuid(id)];
}
