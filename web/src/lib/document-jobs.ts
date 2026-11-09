import {createHash} from 'node:crypto';
import type {PoolClient} from 'pg';
import {asUser} from './db';
import {UUID} from './access';
import {documentCommand,type DocumentCommand,type DocumentJobView} from './document-job-contract';
import {ResearchConflictError,ResearchForbiddenError,ResearchInputError,officialSourceUrl} from './research-contract';
async function operator(c:PoolClient,owner:string){if(!(await c.query('SELECT 1 FROM alphatrace.research_operators WHERE user_id=$1',[owner])).rowCount)throw new ResearchForbiddenError('Document jobs require research operator access.');}
export async function createDocumentJob(owner:string,value:unknown){
 const b=documentCommand(value),hash=createHash('sha256').update(JSON.stringify(b)).digest('hex');
 return asUser(owner,async c=>{
  await operator(c,owner);await c.query("SELECT pg_advisory_xact_lock(hashtext('document-jobs-quota'))");
  const old=(await c.query('SELECT * FROM alphatrace.document_jobs WHERE owner_id=$1 AND command_id=$2',[owner,b.commandId])).rows[0];
  if(old){if(old.input_hash!==hash)throw new ResearchConflictError('This operation ID belongs to different files. Start a new operation.');return old;}
  const count=(await c.query("SELECT count(*)::int AS n FROM alphatrace.document_jobs WHERE created_at>now()-interval '24 hours'")).rows[0].n;
  if(count>=10)throw new ResearchConflictError('The shared limit of 10 document operations per 24 hours has been reached. Try again after earlier operations leave that window.');
  if(b.kind!=='tcs'){
   const draft=(await c.query('SELECT symbol,payload FROM alphatrace.research_drafts WHERE id=$1',[b.input.id])).rows[0];
   if(!draft)throw new ResearchInputError('Choose an existing evidence draft.');
   if(b.kind==='attach'){
    b.input.url=officialSourceUrl(b.input.url,draft.symbol);
    if(b.input.sourceId&&!draft.payload.sources.some((s:{id:string})=>s.id===b.input.sourceId))throw new ResearchInputError('Choose an existing source to replace.');
   }
  }
  return (await c.query('INSERT INTO alphatrace.document_jobs(owner_id,command_id,kind,input,files,input_hash) VALUES($1,$2,$3,$4,$5,$6) RETURNING *',[owner,b.commandId,b.kind,JSON.stringify(b.input),JSON.stringify(b.files),hash])).rows[0];
 });
}
export async function getDocumentJob(owner:string,id:string){
 if(!UUID.test(id))throw new ResearchInputError('Invalid document operation.');
 return asUser(owner,async c=>{await operator(c,owner);
  await c.query("UPDATE alphatrace.document_jobs SET state='expired',finished_at=now(),message='This document operation expired. Start a new operation.' WHERE id=$1 AND owner_id=$2 AND ((state IN ('uploading','queued') AND expires_at<now()) OR (state='running' AND lease_until<now()))",[id,owner]);
  const row=(await c.query('SELECT * FROM alphatrace.document_jobs WHERE id=$1 AND owner_id=$2',[id,owner])).rows[0];if(!row)throw new ResearchInputError('Document operation not found.');return row;
 });
}
export const publicDocumentJob=(r:Record<string,any>):DocumentJobView=>JSON.parse(JSON.stringify({id:r.id,kind:r.kind,state:r.state,message:r.message,result:r.result,expires_at:r.expires_at,created_at:r.created_at}));
export async function queueDocumentJob(owner:string,id:string){
 await getDocumentJob(owner,id);
 return asUser(owner,async c=>{await operator(c,owner);return (await c.query("UPDATE alphatrace.document_jobs SET state='queued',message='Waiting for document processing.' WHERE id=$1 AND owner_id=$2 AND state='uploading' AND expires_at>now() RETURNING id",[id,owner])).rowCount;});
}
export async function claimDocumentJob(owner:string,id:string,runId:string){
 if(!UUID.test(id)||!/^run_[a-zA-Z0-9]{10,80}$/.test(runId))throw new ResearchInputError('Invalid document worker claim.');
 return asUser(owner,async c=>{await operator(c,owner);return (await c.query("UPDATE alphatrace.document_jobs SET state='running',worker_run_id=$3,lease_until=now()+interval '10 minutes',message='Checking document bytes and evidence.' WHERE id=$1 AND owner_id=$2 AND state='queued' AND expires_at>now() RETURNING *",[id,owner,runId])).rows[0] as (DocumentCommand&{id:string;owner_id:string})|undefined;});
}
export async function finishDocumentJob(owner:string,id:string,runId:string,result:unknown,error?:string){
 return asUser(owner,async c=>{await operator(c,owner);return !!(await c.query("UPDATE alphatrace.document_jobs SET state=$4,result=$5,message=$6,finished_at=now() WHERE id=$1 AND owner_id=$2 AND state='running' AND worker_run_id=$3 AND lease_until>now() RETURNING id",[id,owner,runId,error?'failed':'succeeded',error?null:JSON.stringify(result),error??'Document processing complete. Human approval is still required.'])).rowCount;});
}
