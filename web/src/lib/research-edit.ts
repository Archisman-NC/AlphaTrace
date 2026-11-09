import { asUser } from './db';
import { contentHash } from './research';
import { UUID } from './access';
import { validateResearch,ResearchInputError,ResearchConflictError,ResearchForbiddenError } from './research-contract';
import { sourceIdentity,researchChanges } from './research-edit-contract';
export async function requireResearchEditor(owner:string){
 await asUser(owner,async c=>{if(!(await c.query('SELECT 1 FROM alphatrace.research_operators WHERE user_id=$1',[owner])).rowCount)throw new ResearchForbiddenError('Evidence editing requires operator access.');});
}
export async function reviseResearch(owner:string,input:unknown){
 const b=input as {id?:string;payload?:unknown;reason?:string};
 if(!b||typeof b.id!=='string'||!UUID.test(b.id)||typeof b.reason!=='string'||b.reason.trim().length<10||b.reason.length>1000)throw new ResearchInputError('Choose a draft and describe the correction in 10–1,000 characters.');
 const p=validateResearch(b.payload),reason=b.reason.trim();
 return asUser(owner,async c=>{
  if(!(await c.query('SELECT 1 FROM alphatrace.research_operators WHERE user_id=$1',[owner])).rowCount)throw new ResearchForbiddenError('Evidence editing requires operator access.');
  await c.query('SELECT pg_advisory_xact_lock(hashtext($1))',['research:'+p.symbol]);
  const parent=(await c.query('SELECT * FROM alphatrace.research_drafts WHERE id=$1',[b.id])).rows[0];
  if(!parent||parent.symbol!==p.symbol)throw new ResearchInputError('The correction must belong to the same company.');
  const hash=contentHash(p),existing=(await c.query('SELECT d.id,d.content_hash FROM alphatrace.research_revisions r JOIN alphatrace.research_drafts d ON d.id=r.draft_id WHERE r.parent_id=$1',[b.id])).rows[0];
  if(existing){if(existing.content_hash===hash)return {id:existing.id};throw new ResearchConflictError('This version already has a correction. Reload and edit the latest draft.');}
  const latestPublication=(await c.query('SELECT draft_id,period_end::text FROM alphatrace.research_updates WHERE symbol=$1 ORDER BY version DESC LIMIT 1',[p.symbol])).rows[0];
  if(latestPublication&&latestPublication.period_end>p.periodEnd)throw new ResearchConflictError('An older period cannot replace the latest publication.');
  if(parent.state==='published'&&latestPublication?.draft_id!==parent.id)throw new ResearchConflictError('Create the correction from the latest published version.');
  if(parent.state!=='published'&&(await c.query('SELECT 1 FROM alphatrace.research_updates u JOIN alphatrace.research_drafts d ON d.id=u.draft_id WHERE u.symbol=$1 AND u.period_end>=($2)::date AND d.created_at>$3 LIMIT 1',[p.symbol,parent.payload.periodEnd,parent.created_at])).rowCount)throw new ResearchConflictError('This evidence was superseded by a publication. Start from the latest published version.');
  if(!researchChanges(parent.payload,p).length)throw new ResearchInputError('Make an evidence change before saving a revision.');
  const receipts=(await c.query('SELECT source FROM alphatrace.research_attachments WHERE symbol=$1 AND created_by=$2',[p.symbol,owner])).rows;
  const known=[...parent.payload.sources,...receipts.map(r=>r.source)];
  if(p.sources.some(s=>!known.some(k=>sourceIdentity(s)===sourceIdentity(k))))throw new ResearchInputError('Attach each new or changed document through the source form. Retrieval metadata cannot be entered manually.');
  // Deduplicating into an already reviewed row would inherit approval; refuse it.
  if((await c.query('SELECT 1 FROM alphatrace.research_drafts WHERE content_hash=$1',[hash])).rowCount)throw new ResearchConflictError('This exact evidence already exists. Open its existing draft instead.');
  const saved=(await c.query('INSERT INTO alphatrace.research_drafts(symbol,content_hash,payload,created_by) VALUES($1,$2,$3,$4) RETURNING id',[p.symbol,hash,JSON.stringify(p),owner])).rows[0];
  await c.query('INSERT INTO alphatrace.research_revisions(draft_id,parent_id,reason,created_by) VALUES($1,$2,$3,$4)',[saved.id,parent.id,reason,owner]);
  return {id:saved.id};
 });
}
