import { universe } from './companies';
export type Assumption={id:string;statement:string;metric:string;baseline:string;direction:'unspecified'|'up'|'down'|'stable';horizon:string;reviewCondition:string};
export type ThesisPayload={thesis:string;risks:string;assumptions:Assumption[];evidenceIds:string[]};
export type ThesisRevision={version:number;payload:ThesisPayload;reason:string;created_at:string};
export type EvidenceReference={id:string;symbol:string;version:number;title:string;withdrawn:boolean};
export type ThesisView={symbol:string;version:number;current:ThesisRevision|null;history:ThesisRevision[];hasMore:boolean;references:EvidenceReference[]};
export class ThesisInputError extends Error {}
export class ThesisConflictError extends Error {}
const uuid=/^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
function obj(v:unknown):Record<string,unknown>{if(!v||typeof v!=='object'||Array.isArray(v))throw new ThesisInputError('Expected an object.');return v as Record<string,unknown>;}
function text(v:unknown,max:number,min=0):string{if(typeof v!=='string'||v.trim().length<min||v.length>max||/[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]/.test(v))throw new ThesisInputError(`Text must contain ${min}–${max} characters without control characters.`);return v.trim();}
export function thesisSymbol(v:unknown):string{if(typeof v!=='string'||!universe.some(c=>c.symbol===v))throw new ThesisInputError('Choose a company from the research universe.');return v;}
export function thesisId(v:unknown):string{if(typeof v!=='string'||!uuid.test(v))throw new ThesisInputError('Invalid identifier.');return v.toLowerCase();}
export function validateThesis(input:unknown){
 const body=obj(input),p=obj(body.payload),symbol=thesisSymbol(body.symbol),commandId=thesisId(body.commandId);
 if(typeof body.version!=='number'||!Number.isSafeInteger(body.version)||body.version<0||body.version>2147483646)throw new ThesisInputError('Invalid revision number.');
 if(!Array.isArray(p.assumptions)||p.assumptions.length>10)throw new ThesisInputError('Save up to 10 explicit assumptions per company.');
 const assumptions=p.assumptions.map(v=>{const a=obj(v);if(!['unspecified','up','down','stable'].includes(String(a.direction)))throw new ThesisInputError('Invalid expected direction.');return {id:thesisId(a.id),statement:text(a.statement,1000,3),metric:text(a.metric,100),baseline:text(a.baseline,200),direction:a.direction as Assumption['direction'],horizon:text(a.horizon,120),reviewCondition:text(a.reviewCondition,1000)};});
 if(new Set(assumptions.map(a=>a.id)).size!==assumptions.length)throw new ThesisInputError('Assumption identifiers must be unique.');
 if(!Array.isArray(p.evidenceIds)||p.evidenceIds.length>5)throw new ThesisInputError('Link up to five published updates.');
 const evidenceIds=p.evidenceIds.map(thesisId).sort();if(new Set(evidenceIds).size!==evidenceIds.length)throw new ThesisInputError('Duplicate evidence links.');
 const payload:ThesisPayload={thesis:text(p.thesis,6000),risks:text(p.risks,3000),assumptions,evidenceIds};
 if(!payload.thesis&&!payload.risks&&!assumptions.length)throw new ThesisInputError('Write your reasoning or add an assumption before saving.');
 if(assumptions.length&&body.confirmed!==true)throw new ThesisInputError('Confirm that these assumptions reflect your own view.');
 return {symbol,commandId,version:body.version,payload,reason:text(body.reason,1000,3)};
}
export const emptyThesis=():ThesisPayload=>({thesis:'',risks:'',assumptions:[],evidenceIds:[]});
