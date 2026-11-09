import type { PoolClient } from 'pg';
import type { WorkerMode } from './worker-status';
import { asUser } from './db';
import { UUID } from './access';
import { contentHash } from './research';
import { ResearchInputError,ResearchForbiddenError,ResearchConflictError } from './research-contract';
import { providerSymbols,providerDailyLimit,type ProviderSymbol,type ProviderView } from './provider-contract';
import { validateProviderSnapshot,evaluateProvider,type ProviderSnapshot } from './provider-evaluation';
import { providerResearchDraft } from './provider-draft';
import { recordProviderObservation } from './provider-reviews';

async function requireOperator(c:PoolClient,id:string){
 if(!(await c.query('SELECT 1 FROM alphatrace.research_operators WHERE user_id=$1',[id])).rowCount)throw new ResearchForbiddenError('Data imports require research operator access.');
}
const lock=(c:PoolClient)=>c.query("SELECT pg_advisory_xact_lock(hashtext('indianapi-quota'))");
const serialize=<T>(v:unknown):T=>JSON.parse(JSON.stringify(v));
const observedImports=`SELECT i.*,greatest(i.fetched_at,coalesce((SELECT max(finished_at) FROM alphatrace.provider_runs r WHERE r.import_id=i.id AND r.state='success'),i.fetched_at)) AS last_seen_at FROM alphatrace.provider_imports i`;
function fromRow(r:Record<string,unknown>):ProviderSnapshot {
 return {requestedSymbol:r.symbol as ProviderSymbol,endpoint:r.endpoint as string,responseText:r.response_text as string,sha256:r.sha256 as string,fetchedAt:new Date(r.fetched_at as string).toISOString()};
}
async function saveSnapshot(c:PoolClient,owner:string,s:ProviderSnapshot){
 const inserted=await c.query(`INSERT INTO alphatrace.provider_imports(symbol,endpoint,response_text,sha256,fetched_at,staged_by)
  VALUES($1,$2,$3,$4,$5,$6) ON CONFLICT(symbol,sha256) DO NOTHING RETURNING id`,[s.requestedSymbol,s.endpoint,s.responseText,s.sha256,s.fetchedAt,owner]);
 return inserted.rows[0]?.id??(await c.query('SELECT id FROM alphatrace.provider_imports WHERE symbol=$1 AND sha256=$2',[s.requestedSymbol,s.sha256])).rows[0].id as string;
}
export async function stageProviderSnapshot(owner:string,input:unknown){
 const s=validateProviderSnapshot(input);
 return asUser(owner,async c=>{
  await requireOperator(c,owner);await lock(c);
  const id=await saveSnapshot(c,owner,s);
  // Seeded snapshots already consumed a request. Preserve its time for the quota.
  const inserted=await c.query(`INSERT INTO alphatrace.provider_runs(symbol,started_at,lease_until,finished_at,state,message,import_id)
   SELECT $1,$2::timestamptz,$2::timestamptz,$2::timestamptz,'success','Earlier evaluation request imported; no new API call.',$3
   WHERE NOT EXISTS(SELECT 1 FROM alphatrace.provider_runs WHERE import_id=$3 AND started_at=$2::timestamptz) RETURNING id`,[s.requestedSymbol,s.fetchedAt,id]);
  if(inserted.rows[0]){
   const review=await recordProviderObservation(c,id,s);
   await c.query('UPDATE alphatrace.provider_runs SET detection_result=$2,review_id=$3 WHERE id=$1',[inserted.rows[0].id,review.outcome,review.reviewId]);
  }
  return {id};
 });
}
export async function getProviderImports(owner:string):Promise<ProviderView>{
 return asUser(owner,async c=>{
  await requireOperator(c,owner);
  const targets=(await c.query('SELECT * FROM alphatrace.provider_targets ORDER BY symbol')).rows;
  const imports=(await c.query(`WITH observed AS (${observedImports}) SELECT i.*,l.draft_id FROM
   (SELECT *,row_number() OVER(PARTITION BY symbol ORDER BY last_seen_at DESC,id DESC) AS rank FROM observed) i
   LEFT JOIN alphatrace.provider_draft_links l ON l.import_id=i.id WHERE i.rank<=5 ORDER BY i.last_seen_at DESC,i.id DESC`)).rows;
  const runs=(await c.query('SELECT r.id,r.symbol,r.started_at,r.finished_at,r.state,r.message,r.import_id,r.detection_result,r.review_id,r.worker_run_id,q.status AS review_status FROM alphatrace.provider_runs r LEFT JOIN alphatrace.provider_reviews q ON q.id=r.review_id ORDER BY r.started_at DESC,r.id DESC LIMIT 20')).rows;
  const latestBatch=(await c.query(`SELECT worker_run_id AS id,min(started_at) AS started_at,count(*)::int AS attempts,
   count(*) FILTER(WHERE state='success')::int AS succeeded,count(*) FILTER(WHERE state='failed')::int AS failed,count(*) FILTER(WHERE state='running')::int AS running,
   count(*) FILTER(WHERE detection_result IN ('baseline','changed'))::int AS reviews,count(*) FILTER(WHERE detection_result='unchanged')::int AS unchanged,
   count(*) FILTER(WHERE state='success' AND detection_result IN ('not_recorded','stale'))::int AS unclassified
   FROM alphatrace.provider_runs WHERE worker_run_id=(SELECT worker_run_id FROM alphatrace.provider_runs WHERE worker_run_id IS NOT NULL ORDER BY started_at DESC,id DESC LIMIT 1)
   GROUP BY worker_run_id`)).rows[0]??null;
  const usage=(await c.query("SELECT count(*)::int AS n,min(started_at)+interval '24 hours' AS next FROM alphatrace.provider_runs WHERE started_at>now()-interval '24 hours'")).rows[0];
  const now=(await c.query('SELECT now() AS time')).rows[0].time;
  return serialize({serverNow:now,usage:usage.n,limit:providerDailyLimit,nextCreditAt:usage.n>=providerDailyLimit?usage.next:null,targets,runs,latestBatch,
   imports:imports.map(r=>({id:r.id,symbol:r.symbol,sha256:r.sha256,fetched_at:r.fetched_at,staged_at:r.staged_at,last_seen_at:r.last_seen_at,draft_id:r.draft_id,evaluation:evaluateProvider(fromRow(r))}))});
 },{readOnlySnapshot:true});
}
export async function providerCommand(owner:string,input:unknown){
 const b=input as {action?:string;symbol?:ProviderSymbol;id?:string};
 if(!b||!['refresh','resume','pause','prepare'].includes(b.action??''))throw new ResearchInputError('Choose a supported import action.');
 if(b.action!=='prepare'&&!providerSymbols.includes(b.symbol!))throw new ResearchInputError('Choose one of the five pilot companies.');
 if(b.action==='prepare'&&(typeof b.id!=='string'||!UUID.test(b.id)))throw new ResearchInputError('Choose a saved import.');
 return asUser(owner,async c=>{
  await requireOperator(c,owner);await lock(c);
  if(b.action==='prepare'){
   const row=(await c.query('SELECT * FROM alphatrace.provider_imports WHERE id=$1',[b.id])).rows[0];
   if(!row)throw new ResearchInputError('Import not found.');
   const linked=(await c.query('SELECT draft_id FROM alphatrace.provider_draft_links WHERE import_id=$1',[b.id])).rows[0];
   if(linked)return {message:'This import already has a research draft.',draftId:linked.draft_id};
   const latest=(await c.query(`WITH observed AS (${observedImports}) SELECT id FROM observed WHERE symbol=$1 ORDER BY last_seen_at DESC,id DESC LIMIT 1`,[row.symbol])).rows[0];
   if(latest.id!==row.id)throw new ResearchConflictError('A newer import exists. Review it before preparing a draft.');
   const payload=providerResearchDraft(fromRow(row)),hash=contentHash(payload);
   await c.query('SELECT pg_advisory_xact_lock(hashtext($1))',['research:'+row.symbol]);
   const inserted=await c.query('INSERT INTO alphatrace.research_drafts(symbol,content_hash,payload,created_by) VALUES($1,$2,$3,$4) ON CONFLICT(content_hash) DO NOTHING RETURNING id',[row.symbol,hash,JSON.stringify(payload),owner]);
   const draftId=inserted.rows[0]?.id??(await c.query('SELECT id FROM alphatrace.research_drafts WHERE content_hash=$1',[hash])).rows[0].id;
   await c.query('INSERT INTO alphatrace.provider_draft_links(import_id,draft_id,created_by) VALUES($1,$2,$3)',[b.id,draftId,owner]);
   return {message:'Official-reference draft prepared. Add the remaining evidence in the review workflow before publication.',draftId};
  }
  if(b.action==='pause'){
   await c.query('UPDATE alphatrace.provider_targets SET enabled=false,next_due_at=NULL WHERE symbol=$1',[b.symbol]);
   return {message:'Daily refresh paused. A request already running may finish.'};
  }
  const recent=(await c.query(`SELECT max(t) AS time FROM (
   SELECT fetched_at AS t FROM alphatrace.provider_imports WHERE symbol=$1 UNION ALL
   SELECT finished_at FROM alphatrace.provider_runs WHERE symbol=$1 AND state='success') x`,[b.symbol])).rows[0].time;
  const due=recent?new Date(Math.max(Date.now(),new Date(recent).getTime()+86400000)):new Date();
  await c.query(`UPDATE alphatrace.provider_targets SET enabled=CASE WHEN $2 THEN true ELSE enabled END,
   next_due_at=CASE WHEN next_due_at IS NOT NULL THEN least(next_due_at,$3::timestamptz) ELSE $3::timestamptz END WHERE symbol=$1`,[b.symbol,b.action==='resume',due]);
  return {message:due.getTime()>Date.now()+1000?'Cached retrieval retained. Refresh queued after its 24-hour window.':'Refresh queued for the next worker pass, within the five-call daily cap.'};
 });
}
export type ProviderClaim={id:string;symbol:ProviderSymbol;started_at:string};
export async function claimProviderRun(owner:string,mode:WorkerMode='local',workerRunId?:string):Promise<ProviderClaim|null>{
 if(workerRunId&&(mode!=='cloud'||!/^run_[a-zA-Z0-9]{10,80}$/.test(workerRunId)))throw new ResearchInputError('Invalid cloud run identifier.');
 return asUser(owner,async c=>{
  await requireOperator(c,owner);await lock(c);
  await c.query('UPDATE alphatrace.provider_targets SET worker_seen_at=now(),worker_mode=$1',[mode]);
  const expired=(await c.query("UPDATE alphatrace.provider_runs SET state='failed',finished_at=now(),message='Worker stopped or request lease expired. No successful retrieval recorded.' WHERE state='running' AND lease_until<=now() RETURNING symbol,started_at")).rows;
  for(const r of expired)await c.query("UPDATE alphatrace.provider_targets SET next_due_at=CASE WHEN enabled THEN $2::timestamptz+interval '24 hours' ELSE NULL END WHERE symbol=$1",[r.symbol,r.started_at]);
  const usage=(await c.query("SELECT count(*)::int AS n FROM alphatrace.provider_runs WHERE started_at>now()-interval '24 hours'")).rows[0].n;
  if(usage>=providerDailyLimit)return null;
  const target=(await c.query(`SELECT * FROM alphatrace.provider_targets t WHERE next_due_at<=now()
   AND NOT EXISTS(SELECT 1 FROM alphatrace.provider_runs r WHERE r.symbol=t.symbol AND r.state='running')
   AND NOT EXISTS(SELECT 1 FROM alphatrace.provider_runs r WHERE r.symbol=t.symbol AND r.started_at>now()-interval '24 hours')
   AND NOT EXISTS(SELECT 1 FROM alphatrace.provider_imports i WHERE i.symbol=t.symbol AND i.fetched_at>now()-interval '24 hours')
   ORDER BY next_due_at,symbol LIMIT 1 FOR UPDATE`)).rows[0];
  if(!target)return null;
  const run=(await c.query('INSERT INTO alphatrace.provider_runs(symbol,worker_run_id) VALUES($1,$2) RETURNING id,symbol,started_at',[target.symbol,workerRunId??null])).rows[0];
  await c.query('UPDATE alphatrace.provider_targets SET next_due_at=NULL WHERE symbol=$1',[target.symbol]);
  return serialize(run);
 });
}
export async function finishProviderRun(owner:string,claim:ProviderClaim,input:unknown,errorCode?:string){
 const snapshot=input===null?null:validateProviderSnapshot(input);
 if(snapshot&&(snapshot.requestedSymbol!==claim.symbol||Date.parse(snapshot.fetchedAt)<Date.parse(claim.started_at)))throw new ResearchInputError('A live run requires a new response for its claimed company.');
 return asUser(owner,async c=>{
  await requireOperator(c,owner);await lock(c);
  const run=(await c.query("SELECT * FROM alphatrace.provider_runs WHERE id=$1 AND symbol=$2 AND state='running' AND lease_until>now() FOR UPDATE",[claim.id,claim.symbol])).rows[0];
  if(!run)return {accepted:false};
  if(snapshot&&Date.parse(snapshot.fetchedAt)<new Date(run.started_at).getTime())throw new ResearchInputError('Cached evidence cannot finish a new live run.');
  const id=snapshot?await saveSnapshot(c,owner,snapshot):null;
  const review=snapshot?await recordProviderObservation(c,id,snapshot):{created:false,outcome:'not_recorded',reviewId:null};
  const message=snapshot?'Quarterly response saved privately. Original-source freshness and publication status are unchanged.':
   errorCode==='http_429'?'Provider quota reached. No automatic retry.':errorCode==='http_401'||errorCode==='http_403'?'Provider denied access. Check the key or endpoint entitlement.':'Retrieval failed. No fresh data was recorded; retry is bounded by the daily limit.';
  await c.query('UPDATE alphatrace.provider_runs SET state=$2,finished_at=now(),message=$3,import_id=$4,detection_result=$5,review_id=$6 WHERE id=$1',[claim.id,snapshot?'success':'failed',message,id,review.outcome,review.reviewId]);
  await c.query("UPDATE alphatrace.provider_targets SET next_due_at=CASE WHEN enabled THEN now()+interval '24 hours' ELSE next_due_at END WHERE symbol=$1",[claim.symbol]);
  return {accepted:true,id,reviewCreated:review.created,detectionResult:review.outcome,reviewId:review.reviewId};
 });
}
