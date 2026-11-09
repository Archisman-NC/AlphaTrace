import type { PoolClient } from 'pg';
import { asUser } from './db';
import { UUID } from './access';
import { ResearchInputError,ResearchForbiddenError,ResearchConflictError } from './research-contract';
import { normalizeProvider,providerSemanticHash,detectProviderChanges,meaningfulProviderChange } from './provider-changes';
import { evaluateProvider,type ProviderSnapshot } from './provider-evaluation';
import type { ProviderSymbol,ProviderDetection } from './provider-contract';
import type { ImportReviewView,ImportReview } from './provider-review-contract';

const snapshot=(r:Record<string,unknown>,observedAt?:string):ProviderSnapshot=>({requestedSymbol:r.symbol as ProviderSymbol,endpoint:r.endpoint as string,responseText:r.response_text as string,sha256:r.sha256 as string,fetchedAt:observedAt??new Date(r.fetched_at as string).toISOString()});
async function operator(c:PoolClient,owner:string){if(!(await c.query('SELECT 1 FROM alphatrace.research_operators WHERE user_id=$1',[owner])).rowCount)throw new ResearchForbiddenError('Import review requires operator access.');}

/** Called inside the import transaction, while holding the shared provider lock. */
export async function recordProviderObservation(c:PoolClient,id:string,s:ProviderSnapshot):Promise<{created:boolean;outcome:ProviderDetection;reviewId:string|null}>{
 const head=(await c.query('SELECT * FROM alphatrace.provider_review_heads WHERE symbol=$1 FOR UPDATE',[s.requestedSymbol])).rows[0];
 // Historical replay, duplicated completions and out-of-order cache seeds cannot move the head backwards.
 if(head&&new Date(head.observed_at).getTime()>=Date.parse(s.fetchedAt))return {created:false,outcome:'stale',reviewId:null};
 const current=normalizeProvider(s),hash=providerSemanticHash(current);
 let sequence=head?.sequence??0,created=false,reviewId:string|null=null,outcome:ProviderDetection='unchanged';
 if(!head||hash!==head.semantic_hash){
  const prior=head?(await c.query('SELECT * FROM alphatrace.provider_imports WHERE id=$1',[head.import_id])).rows[0]:null;
  const diff=detectProviderChanges(prior?normalizeProvider(snapshot(prior)):null,current);
  if(meaningfulProviderChange(diff)){
   sequence++;created=true;
   reviewId=(await c.query(`INSERT INTO alphatrace.provider_reviews(symbol,sequence,current_import_id,previous_import_id,observed_at,previous_observed_at,diff)
    VALUES($1,$2,$3,$4,$5,$6,$7) RETURNING id`,[s.requestedSymbol,sequence,id,head?.import_id??null,s.fetchedAt,head?.observed_at??null,JSON.stringify(diff)])).rows[0].id;
   outcome=diff.baseline?'baseline':'changed';
  }
 }
 await c.query(`INSERT INTO alphatrace.provider_review_heads(symbol,import_id,observed_at,semantic_hash,sequence) VALUES($1,$2,$3,$4,$5)
  ON CONFLICT(symbol) DO UPDATE SET import_id=excluded.import_id,observed_at=excluded.observed_at,semantic_hash=excluded.semantic_hash,sequence=excluded.sequence`,[s.requestedSymbol,id,s.fetchedAt,hash,sequence]);
 return {created,outcome,reviewId};
}

/** Upgrade existing successful observations without making provider calls or changing freshness. */
export async function backfillProviderReviews(owner:string){
 return asUser(owner,async c=>{
  await operator(c,owner);await c.query("SELECT pg_advisory_xact_lock(hashtext('indianapi-quota'))");
  const rows=(await c.query(`SELECT i.*,r.finished_at AS observed_at FROM alphatrace.provider_runs r JOIN alphatrace.provider_imports i ON i.id=r.import_id
   LEFT JOIN alphatrace.provider_review_heads h ON h.symbol=i.symbol
   WHERE r.state='success' AND (h.observed_at IS NULL OR r.finished_at>h.observed_at)
   ORDER BY r.finished_at,r.started_at,r.id`)).rows;
  let created=0;
  for(const r of rows)if((await recordProviderObservation(c,r.id,snapshot(r,new Date(r.observed_at).toISOString()))).created)created++;
  return {observations:rows.length,created};
 });
}

export async function getProviderReviews(owner:string,input:{status?:string;before?:string;id?:string}={}):Promise<ImportReviewView>{
 const status=input.status??'pending';if(!['pending','reviewed'].includes(status))throw new ResearchInputError('Choose pending or reviewed items.');
 if(input.id&&!UUID.test(input.id))throw new ResearchInputError('Invalid review item.');
 let cursor:{time:string;id:string}|null=null;
 if(input.before){try{cursor=JSON.parse(Buffer.from(input.before,'base64url').toString());}catch{throw new ResearchInputError('Invalid review cursor.');}
  if(!cursor||typeof cursor.time!=='string'||!Number.isFinite(Date.parse(cursor.time))||typeof cursor.id!=='string'||!UUID.test(cursor.id))throw new ResearchInputError('Invalid review cursor.');}
 return asUser(owner,async c=>{
  await operator(c,owner);
  const counts={pending:0,reviewed:0};for(const r of (await c.query('SELECT status,count(*)::int AS n FROM alphatrace.provider_reviews GROUP BY status')).rows)counts[r.status as keyof typeof counts]=r.n;
  const rows=(await c.query(`SELECT q.*,i.endpoint,i.response_text,i.sha256 AS current_sha256,i.fetched_at,p.sha256 AS previous_sha256,l.draft_id,
   (q.sequence=h.sequence) AS is_current, handoff.result AS handoff FROM alphatrace.provider_reviews q
   JOIN alphatrace.provider_imports i ON i.id=q.current_import_id
   LEFT JOIN alphatrace.provider_imports p ON p.id=q.previous_import_id
   JOIN alphatrace.provider_review_heads h ON h.symbol=q.symbol
   LEFT JOIN alphatrace.provider_draft_links l ON l.import_id=q.current_import_id
   LEFT JOIN LATERAL (WITH RECURSIVE chain AS (
    SELECT draft_id,0 AS depth FROM alphatrace.provider_review_drafts WHERE review_id=q.id
    UNION ALL SELECT r.draft_id,chain.depth+1 FROM alphatrace.research_revisions r JOIN chain ON r.parent_id=chain.draft_id
   ) SELECT jsonb_build_object('id',d.id,'state',d.state,'periodEnd',d.payload->>'periodEnd','updateId',d.published_update_id) AS result
     FROM chain JOIN alphatrace.research_drafts d ON d.id=chain.draft_id ORDER BY depth DESC LIMIT 1) handoff ON true
   WHERE ($4::uuid IS NOT NULL AND q.id=$4::uuid) OR ($4::uuid IS NULL AND q.status=$1 AND ($2::timestamptz IS NULL OR (q.observed_at,q.id)<($2::timestamptz,$3::uuid)))
   ORDER BY q.observed_at DESC,q.id DESC LIMIT 21`,[status,cursor?.time??null,cursor?.id??null,input.id??null])).rows;
  const shown=rows.slice(0,20),last=shown.at(-1);
  const items=shown.map(r=>{
   const {response_text,is_current,...item}=r;
   const evaluation=evaluateProvider(snapshot({...r,sha256:r.current_sha256,response_text},new Date(r.observed_at).toISOString()));
   return {...item,isCurrent:is_current,sourceUrl:evaluation.sourceUrl,evaluation} as ImportReview;
  });
  return JSON.parse(JSON.stringify({items,counts,nextCursor:rows.length>20&&last?Buffer.from(JSON.stringify({time:new Date(last.observed_at).toISOString(),id:last.id})).toString('base64url'):null}));
 },{readOnlySnapshot:true});
}
export async function reviewProviderChange(owner:string,input:unknown){
 const b=input as {id?:string;version?:number;action?:string;note?:string};
 if(!b||typeof b.id!=='string'||!UUID.test(b.id)||!Number.isSafeInteger(b.version)||(b.version??0)<1||!['review','reopen'].includes(b.action??'')||typeof b.note!=='string'||b.note.length>500)throw new ResearchInputError('Choose a review item, version, and action; notes can contain up to 500 characters.');
 const note=b.note.trim();
 return asUser(owner,async c=>{
  await operator(c,owner);
  const row=(await c.query('SELECT * FROM alphatrace.provider_reviews WHERE id=$1 FOR UPDATE',[b.id])).rows[0];
  if(!row)throw new ResearchInputError('Review item not found.');
  if(row.version!==b.version)throw new ResearchConflictError('This review changed in another session. Refresh the queue before saving.');
  const status=b.action==='review'?'reviewed':'pending';
  if(row.status===status)throw new ResearchConflictError('This item is already in that state. Refresh the queue.');
  await c.query(`UPDATE alphatrace.provider_reviews SET status=$2,version=version+1,note=$3,
   reviewed_at=CASE WHEN $2='reviewed' THEN now() ELSE NULL END,reviewed_by=CASE WHEN $2='reviewed' THEN $4::uuid ELSE NULL END WHERE id=$1`,[b.id,status,note,owner]);
  await c.query('INSERT INTO alphatrace.provider_review_actions(review_id,version,action,note,actor) VALUES($1,$2,$3,$4,$5)',[b.id,row.version+1,b.action,note,owner]);
  return {message:status==='reviewed'?'Import review recorded. Evidence approval and publication are unchanged.':'Import review reopened.'};
 });
}
