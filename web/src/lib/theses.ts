import type { PoolClient } from 'pg';
import { asUser } from './db';
import { validateThesis,thesisSymbol,ThesisInputError,ThesisConflictError,type ThesisView,type ThesisRevision } from './thesis-input';
async function read(c:PoolClient,id:string,symbol:string,before?:number):Promise<ThesisView>{
 const parent=(await c.query('SELECT version FROM alphatrace.private_research WHERE owner_id=$1 AND symbol=$2 FOR SHARE',[id,symbol])).rows[0];
 const current=parent?.version?(await c.query('SELECT version,payload,reason,created_at FROM alphatrace.private_research_revisions WHERE owner_id=$1 AND symbol=$2 AND version=$3',[id,symbol,parent.version])).rows[0]:null;
 const rows=(await c.query('SELECT version,payload,reason,created_at FROM alphatrace.private_research_revisions WHERE owner_id=$1 AND symbol=$2 AND ($3::integer IS NULL OR version<$3) ORDER BY version DESC LIMIT 21',[id,symbol,before??null])).rows;
 const history=rows.slice(0,20) as ThesisRevision[],ids=[...new Set([...(current?.payload.evidenceIds??[]),...history.flatMap(r=>r.payload.evidenceIds)])];
 const references=ids.length?(await c.query("SELECT id,symbol,version,payload->>'title' AS title,withdrawn_at IS NOT NULL AS withdrawn FROM alphatrace.research_updates WHERE symbol=$1 AND id=ANY($2::uuid[])",[symbol,ids])).rows:[];
 return JSON.parse(JSON.stringify({symbol,version:parent?.version??0,current,history,hasMore:rows.length>20,references}));
}
export async function getThesis(id:string,symbol:unknown,before?:number){
 const company=thesisSymbol(symbol);
 if(before!==undefined&&(!Number.isSafeInteger(before)||before<1||before>2147483647))throw new ThesisInputError('Invalid history cursor.');
 return asUser(id,c=>read(c,id,company,before));
}
export async function saveThesis(id:string,input:unknown){
 const command=validateThesis(input),{symbol,version,commandId,payload,reason}=command;
 return asUser(id,async c=>{
  await c.query("INSERT INTO alphatrace.workspaces(owner_id,name) VALUES ($1,'My research') ON CONFLICT(owner_id) DO NOTHING",[id]);
  // Serialize a new company record and retries before reading the current revision.
  await c.query('SELECT pg_advisory_xact_lock(hashtext($1))',[`thesis:${id}:${symbol}`]);
  const retry=(await c.query('SELECT symbol,version,payload=$3::jsonb AS same_payload,reason=$4 AS same_reason FROM alphatrace.private_research_revisions WHERE owner_id=$1 AND command_id=$2',[id,commandId,JSON.stringify(payload),reason])).rows[0];
  if(retry){if(retry.symbol!==symbol||retry.version!==version+1||!retry.same_payload||!retry.same_reason)throw new ThesisConflictError('This save identifier was already used for different content. Reload before retrying.');return read(c,id,symbol);}
  await c.query('INSERT INTO alphatrace.private_research(owner_id,symbol) VALUES ($1,$2) ON CONFLICT DO NOTHING',[id,symbol]);
  const head=(await c.query('SELECT version FROM alphatrace.private_research WHERE owner_id=$1 AND symbol=$2 FOR UPDATE',[id,symbol])).rows[0];
  if(head.version!==version)throw new ThesisConflictError('Your research changed in another tab. Your edits are still here; load the latest saved revision before saving again.');
  if(payload.evidenceIds.length){
   for(const evidenceId of payload.evidenceIds)await c.query('SELECT pg_advisory_xact_lock_shared(hashtext($1))',['publication:'+evidenceId]);
   const evidence=await c.query('SELECT id FROM alphatrace.research_updates WHERE symbol=$1 AND id=ANY($2::uuid[]) AND withdrawn_at IS NULL',[symbol,payload.evidenceIds]);
   if(evidence.rowCount!==payload.evidenceIds.length)throw new ThesisInputError('Evidence must be published, available and for this company. Remove withdrawn or unavailable links before saving.');
  }
  await c.query('INSERT INTO alphatrace.private_research_revisions(owner_id,symbol,version,command_id,payload,reason) VALUES ($1,$2,$3,$4,$5,$6)',[id,symbol,version+1,commandId,JSON.stringify(payload),reason]);
  await c.query('UPDATE alphatrace.private_research SET version=version+1,updated_at=now() WHERE owner_id=$1 AND symbol=$2',[id,symbol]);
  return read(c,id,symbol);
 });
}
