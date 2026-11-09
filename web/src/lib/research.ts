import { createHash } from 'node:crypto';
import type { PoolClient } from 'pg';
import { asUser } from './db';
import { UUID } from './access';
import { researchChanges } from './research-edit-contract';
import { universe } from './companies';
import { evaluateProvider } from './provider-evaluation';
import { validateResearch,validateCheck,publicationGaps,researchSymbol,ResearchInputError,ResearchConflictError,ResearchForbiddenError,type ResearchPayload } from './research-contract';

async function operator(c:PoolClient,id:string){return !!(await c.query('SELECT 1 FROM alphatrace.research_operators WHERE user_id=$1',[id])).rowCount;}
async function requireOperator(c:PoolClient,id:string){if(!await operator(c,id))throw new ResearchForbiddenError('Research review requires operator access.');}
function object(v:unknown):Record<string,unknown>{if(!v||typeof v!=='object'||Array.isArray(v))throw new ResearchInputError('Expected an object.');return v as Record<string,unknown>;}
function identifier(v:unknown):string{if(typeof v!=='string'||!UUID.test(v))throw new ResearchInputError('Invalid research identifier.');return v.toLowerCase();}
function note(v:unknown):string{if(typeof v!=='string'||v.trim().length<10||v.length>1000)throw new ResearchInputError('Write a review note between 10 and 1,000 characters.');return v.trim();}
// Derive supersession without rewriting old drafts or their human review records.
// A correction prepared after the published draft remains actionable.
const supersedingPublication=`SELECT u.id,u.version FROM alphatrace.research_updates u
 JOIN alphatrace.research_drafts published_draft ON published_draft.id=u.draft_id
 WHERE u.symbol=d.symbol AND u.period_end>=(d.payload->>'periodEnd')::date
 AND published_draft.created_at>d.created_at AND d.state<>'published'
 ORDER BY u.version DESC LIMIT 1`;
// Retrieval timestamps do not create a new draft when a document is unchanged.
export function contentHash(p:ResearchPayload):string{return createHash('sha256').update(JSON.stringify({...p,sources:p.sources.map(({fetchedAt,...s})=>s)})).digest('hex');}
export async function ingestResearchWithClient(c:PoolClient,id:string,input:unknown){
 const body=object(input),check=validateCheck(body.check),draft=body.draft===undefined?null:validateResearch(body.draft);
 if(draft&&(check.status!=='success'||check.symbol!==draft.symbol||!draft.sources.every(s=>check.fingerprints.includes(s.sha256))))throw new ResearchInputError('The successful check must cover every draft source.');
  await requireOperator(c,id);
  await c.query('SELECT pg_advisory_xact_lock(hashtext($1))',['research:'+check.symbol]);
  let draftId:string|null=null,created=false;
  if(draft){
   const hash=contentHash(draft);
   const inserted=await c.query('INSERT INTO alphatrace.research_drafts(symbol,content_hash,payload,created_by) VALUES ($1,$2,$3,$4) ON CONFLICT(content_hash) DO NOTHING RETURNING id',[draft.symbol,hash,JSON.stringify(draft),id]);
   created=!!inserted.rowCount;
   draftId=inserted.rows[0]?.id??(await c.query('SELECT id FROM alphatrace.research_drafts WHERE content_hash=$1',[hash])).rows[0].id;
  }
  await c.query('INSERT INTO alphatrace.research_checks(symbol,checked_at,status,message,fingerprints,recorded_by) VALUES ($1,$2,$3,$4,$5,$6) ON CONFLICT(symbol,checked_at) DO NOTHING',[check.symbol,check.checkedAt,check.status,check.message,JSON.stringify(check.fingerprints),id]);
  return {draftId,created};
}
export async function ingestResearch(id:string,input:unknown){
 return asUser(id,c=>ingestResearchWithClient(c,id,input));
}
export async function getResearch(id:string,symbol?:string,admin=false,updateId?:string){
 if(symbol)researchSymbol(symbol);
 if(updateId){identifier(updateId);if(!symbol)throw new ResearchInputError('Choose a company for the selected publication.');}
 return asUser(id,async c=>{
  const isOperator=await operator(c,id);
  if(admin&&!isOperator)throw new ResearchForbiddenError('Research review requires operator access.');
  const latest=(await c.query('SELECT DISTINCT ON(symbol) * FROM alphatrace.research_updates ORDER BY symbol,version DESC')).rows;
  const checks=(await c.query('SELECT DISTINCT ON(symbol) symbol,checked_at,status,message,fingerprints FROM alphatrace.research_checks ORDER BY symbol,checked_at DESC')).rows;
  const successful=(await c.query("SELECT DISTINCT ON(symbol) symbol,checked_at FROM alphatrace.research_checks WHERE status='success' ORDER BY symbol,checked_at DESC")).rows;
  const catalog=universe.map(company=>{
   const update=latest.find(u=>u.symbol===company.symbol),check=checks.find(s=>s.symbol===company.symbol)??null;
   const changed=!!update&&check?.status==='success'&&!update.payload.sources.every((s:{sha256:string})=>check.fingerprints.includes(s.sha256));
   return {...company,status:update?(update.withdrawn_at?'withdrawn':'published'):'planned',version:update?.version??null,periodEnd:update?.payload.periodEnd??null,publishedAt:update?.published_at??null,check,lastSuccessfulCheck:successful.find(s=>s.symbol===company.symbol)?.checked_at??null,sourceChanged:changed};
  });
  const history=symbol?(await c.query('SELECT id,draft_id,symbol,version,period_end,payload,published_at,supersedes,withdrawn_at,withdrawal_reason FROM alphatrace.research_updates WHERE symbol=$1 ORDER BY version DESC LIMIT 20',[symbol])).rows:[];
  if(updateId&&!history.some(u=>u.id===updateId)){
   const exact=(await c.query('SELECT id,draft_id,symbol,version,period_end,payload,published_at,supersedes,withdrawn_at,withdrawal_reason FROM alphatrace.research_updates WHERE id=$1 AND symbol=$2',[updateId,symbol])).rows[0];
   if(!exact)throw new ResearchInputError('The linked publication is unavailable for this company.');
   history.push(exact);
  }
  const drafts=admin?(await c.query(`SELECT d.id,d.payload,d.state,d.created_at,d.review_note,d.reviewed_at,d.published_update_id,
   replacement.id AS superseded_by,replacement.version AS superseded_version, successor.draft_id AS revised_by, revision.parent_id,revision.reason AS revision_reason, parent.payload AS parent_payload
   FROM alphatrace.research_drafts d LEFT JOIN alphatrace.research_revisions successor ON successor.parent_id=d.id LEFT JOIN alphatrace.research_revisions revision ON revision.draft_id=d.id LEFT JOIN alphatrace.research_drafts parent ON parent.id=revision.parent_id LEFT JOIN LATERAL (${supersedingPublication}) replacement ON true
   WHERE ($1::text IS NULL OR d.symbol=$1)
   ORDER BY (successor.draft_id IS NULL AND replacement.id IS NULL AND d.state<>'published') DESC,d.created_at DESC LIMIT 50`,[symbol??null])).rows:[];
  // JSON round-trip makes Postgres dates explicit ISO strings for server actions too.
  return JSON.parse(JSON.stringify({catalog,history,drafts:drafts.map(({parent_payload,...d})=>({...d,changes:parent_payload?researchChanges(parent_payload,d.payload):[]})),isOperator}));
 });
}
export async function reviewResearch(id:string,input:unknown){
 const body=object(input),action=body.action;
 if(!['approve','reject','publish','withdraw'].includes(String(action)))throw new ResearchInputError('Unknown research action.');
 const target=identifier(body.id);
 return asUser(id,async c=>{
  await requireOperator(c,id);
  if(action==='withdraw'){
   const reason=note(body.note);
   await c.query('SELECT pg_advisory_xact_lock(hashtext($1))',['publication:'+target]);
   const result=await c.query('UPDATE alphatrace.research_updates SET withdrawn_at=now(),withdrawal_reason=$1 WHERE id=$2 AND withdrawn_at IS NULL RETURNING id',[reason,target]);
   if(!result.rowCount)throw new ResearchConflictError('This version is already withdrawn or unavailable.');
   return {id:target};
  }
  // One lock order for all review/publish requests avoids deadlocks across drafts.
  const lookup=(await c.query('SELECT symbol FROM alphatrace.research_drafts WHERE id=$1',[target])).rows[0];
  if(!lookup)throw new ResearchInputError('Draft not found.');
  await c.query("SELECT pg_advisory_xact_lock(hashtext('indianapi-quota'))");
  await c.query('SELECT pg_advisory_xact_lock(hashtext($1))',['research:'+lookup.symbol]);
  const draft=(await c.query('SELECT * FROM alphatrace.research_drafts WHERE id=$1 FOR UPDATE',[target])).rows[0];
  if(action==='publish'&&draft.state==='published')return {id:draft.published_update_id};
  if(draft.state==='published')throw new ResearchConflictError('Published evidence is immutable. Submit a corrected draft.');
  const replacement=(await c.query(`SELECT replacement.* FROM alphatrace.research_drafts d
   JOIN LATERAL (${supersedingPublication}) replacement ON true WHERE d.id=$1`,[target])).rows[0];
  if(replacement)throw new ResearchConflictError(`This draft was superseded by published version ${replacement.version}. Submit a new correction instead.`);
  if((await c.query('SELECT 1 FROM alphatrace.research_revisions WHERE parent_id=$1',[target])).rowCount)throw new ResearchConflictError('This draft has a newer correction. Review the latest draft.');
  if(action==='reject'){
   await c.query("UPDATE alphatrace.research_drafts SET state='needs_correction',review_note=$1,reviewed_by=$2,reviewed_at=now() WHERE id=$3",[note(body.note),id,target]);
   return {id:target};
  }
  const payload=validateResearch(draft.payload),gaps=publicationGaps(payload);
  const linked=(await c.query(`WITH RECURSIVE ancestry AS (
   SELECT $1::uuid AS id UNION ALL SELECT r.parent_id FROM alphatrace.research_revisions r JOIN ancestry a ON r.draft_id=a.id
  ) SELECT q.*,h.sequence AS current_sequence,i.endpoint,i.sha256,i.response_text FROM ancestry a
   JOIN alphatrace.provider_review_drafts l ON l.draft_id=a.id JOIN alphatrace.provider_reviews q ON q.id=l.review_id
   JOIN alphatrace.provider_review_heads h ON h.symbol=q.symbol JOIN alphatrace.provider_imports i ON i.id=q.current_import_id`,[target])).rows[0];
  if(linked){
   if(linked.sequence!==linked.current_sequence)throw new ResearchConflictError('A newer provider change exists. Review the latest observation before publication.');
   const evaluation=evaluateProvider({requestedSymbol:linked.symbol,endpoint:linked.endpoint,sha256:linked.sha256,responseText:linked.response_text,fetchedAt:new Date(linked.observed_at).toISOString()});
   if(payload.periodEnd!==evaluation.periods.at(-1))throw new ResearchConflictError('Complete the evidence for the reporting period in the linked provider observation.');
  }
  if(gaps.length)throw new ResearchInputError(gaps.join(' '));
  const check=(await c.query('SELECT * FROM alphatrace.research_checks WHERE symbol=$1 ORDER BY checked_at DESC LIMIT 1',[payload.symbol])).rows[0];
  if(!check||check.status!=='success'||Date.now()-new Date(check.checked_at).getTime()>7*86400000||!payload.sources.every(s=>check.fingerprints.includes(s.sha256)))throw new ResearchConflictError('Recheck these exact sources successfully before approval or publication.');
  if(action==='approve'){
   if(body.evidenceConfirmed!==true||body.basisConfirmed!==true)throw new ResearchInputError('Confirm the evidence and comparison basis before approval.');
   await c.query("UPDATE alphatrace.research_drafts SET state='approved',review_note=$1,reviewed_by=$2,reviewed_at=now() WHERE id=$3",[note(body.note),id,target]);
   return {id:target};
  }
  if(draft.state!=='approved')throw new ResearchConflictError('Approve the evidence before publishing.');
  const previous=(await c.query('SELECT id,version,period_end::text FROM alphatrace.research_updates WHERE symbol=$1 ORDER BY version DESC LIMIT 1',[payload.symbol])).rows[0];
  if(previous&&previous.period_end>payload.periodEnd)throw new ResearchConflictError('An older reporting period cannot replace the latest update.');
  const published=await c.query('INSERT INTO alphatrace.research_updates(draft_id,symbol,version,period_end,payload,published_by,supersedes) VALUES ($1,$2,$3,$4,$5,$6,$7) RETURNING id',[target,payload.symbol,(previous?.version??0)+1,payload.periodEnd,JSON.stringify(payload),id,previous?.id??null]);
  const publishedId=published.rows[0].id;
  await c.query("UPDATE alphatrace.research_drafts SET state='published',published_update_id=$1 WHERE id=$2",[publishedId,target]);
  return {id:publishedId};
 });
}
