import { metrics,type Comparison,type Source } from './research-contract';
export const issueCategories={incorrect_value:'Incorrect value',period_or_units:'Wrong quarter or units',source_link:'Source link problem',unclear_explanation:'Unclear explanation'} as const;
export const issueStatuses={open:'Open',in_review:'In review',resolved:'Resolved'} as const;
export const issueOutcomes={corrected:'Published correction',explained:'Evidence explained',withdrawn:'Publication withdrawn'} as const;
export type IssueCategory=keyof typeof issueCategories;
export type IssueStatus=keyof typeof issueStatuses;
export type IssueOutcome=keyof typeof issueOutcomes;
export type IssueContext={symbol:string;publicationVersion:number;title:string;periodEnd:string;comparisonType:string;comparison:Comparison|null;sources:Source[]};
export type ResearchIssue={id:string;publicationId:string;metric:string|null;category:IssueCategory;note:string;context:IssueContext;status:IssueStatus;version:number;outcome:IssueOutcome|null;response:string|null;correctionId:string|null;createdAt:string;updatedAt:string;withdrawnAt:string|null;correctionWithdrawnAt:string|null};
export type IssueEvent={version:number;action:'submitted'|'in_review'|'resolved'|'reopened';note:string;outcome:IssueOutcome|null;correctionId:string|null;createdAt:string};
export type IssueDetail={report:ResearchIssue;events:IssueEvent[];corrections:{id:string;version:number;title:string}[]};
export type IssueView={items:ResearchIssue[];nextCursor:string|null;isOperator:boolean;counts:Record<IssueStatus,number>;detail:IssueDetail|null};
export type IssueQuery={scope?:string;status?:string;cursor?:string;report?:string};
export type IssueCommand={action:'submit';commandId:string;publicationId:string;metric:string|null;category:IssueCategory;note:string}|{action:'in_review'|'resolve'|'reopen';commandId:string;id:string;version:number;note:string;outcome:IssueOutcome|null;correctionId:string|null};
export type IssueResult={issues?:IssueView;detail?:IssueDetail;duplicate?:boolean;error?:string;status?:number};
export class IssueInputError extends Error {}
export class IssueConflictError extends Error {}
export class IssueForbiddenError extends Error {}
const uuid=/^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
export function issueId(v:unknown):string{if(typeof v!=='string'||!uuid.test(v))throw new IssueInputError('Invalid report or publication identifier.');return v.toLowerCase();}
function note(v:unknown,min:number){if(typeof v!=='string'||v.trim().length<min||v.trim().length>2000||/[\u0000-\u0008\u000b\u000c\u000e-\u001f]/.test(v))throw new IssueInputError(`Use ${min}–2,000 characters for the note.`);return v.trim();}
export function validateIssue(input:unknown):IssueCommand{
 if(!input||typeof input!=='object')throw new IssueInputError('Invalid report request.');
 const v=input as Record<string,unknown>,commandId=issueId(v.commandId);
 if(v.action==='submit'){
  if(typeof v.category!=='string'||!Object.hasOwn(issueCategories,v.category))throw new IssueInputError('Choose an issue category.');
  if(v.metric!==null&&(typeof v.metric!=='string'||!Object.hasOwn(metrics,v.metric)))throw new IssueInputError('Choose a metric from this publication.');
  return {action:'submit',commandId,publicationId:issueId(v.publicationId),metric:v.metric as string|null,category:v.category as IssueCategory,note:note(v.note,3)};
 }
 if(!['in_review','resolve','reopen'].includes(String(v.action))||!Number.isSafeInteger(v.version)||(v.version as number)<0)throw new IssueInputError('Reload the report before updating it.');
 const action=v.action as 'in_review'|'resolve'|'reopen',outcome=action==='resolve'?v.outcome:null;
 if(action==='resolve'&&(typeof outcome!=='string'||!Object.hasOwn(issueOutcomes,outcome)))throw new IssueInputError('Choose how this report was resolved.');
 return {action,commandId,id:issueId(v.id),version:v.version as number,note:action==='in_review'?'Review started.':note(v.note,action==='resolve'?10:3),outcome:outcome as IssueOutcome|null,correctionId:outcome==='corrected'?issueId(v.correctionId):null};
}
export function issueQuery(query:IssueQuery){
 const scope=query.scope??'mine',status=query.status??'active';
 if(!['mine','queue'].includes(scope)||!['active','all','open','in_review','resolved'].includes(status))throw new IssueInputError('Invalid report filter.');
 let cursor:[string,string]|null=null;
 if(query.cursor){const parts=query.cursor.split('|');if(parts.length!==2||query.cursor.length>120||!/^\d{4}-\d\d-\d\dT/.test(parts[0])||!Number.isFinite(Date.parse(parts[0])))throw new IssueInputError('Invalid report cursor.');cursor=[parts[0],issueId(parts[1])];}
 return {scope,status,cursor,report:query.report?issueId(query.report):null};
}
