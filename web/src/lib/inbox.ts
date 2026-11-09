import type { PoolClient } from 'pg';
import { asUser } from './db';
import { inboxUuid,inboxFilter,inboxCursor,validateReview,revisitTimestamp,matchAssumptions,InboxInputError,InboxConflictError,type InboxFilter,type InboxList,type InboxDetail,type InboxSnapshot } from './inbox-contract';
import type { ThesisRevision } from './thesis-input';

const serial=<T>(value:T):T=>JSON.parse(JSON.stringify(value));
const itemColumns=`i.update_id,u.symbol,u.payload->>'title' AS title,u.version AS publication_version,u.period_end::text,
 to_char(i.created_at AT TIME ZONE 'UTC','YYYY-MM-DD"T"HH24:MI:SS.US"Z"') AS created_at,
 i.version,i.state,i.revisit_at,jsonb_array_length(i.snapshot->'matches') AS match_count,u.withdrawn_at,u.withdrawal_reason,
 (SELECT n.id FROM alphatrace.research_updates n WHERE n.symbol=u.symbol AND n.version>u.version ORDER BY n.version DESC LIMIT 1) AS newer_update_id`;
const pending="(i.state='unreviewed' OR (i.state='deferred' AND i.revisit_at<=now()))";
const deferred="(i.state='deferred' AND i.revisit_at>now())";
const reviewed="i.state IN ('kept','dismissed','revised')";
const filterSql:Record<InboxFilter,string>={pending,deferred,reviewed,all:'true'};

/** Each completed insertion is the durable discovery cursor; late commits cannot be skipped. */
export async function syncInbox(owner:string){
 return asUser(owner,async c=>{
  await c.query("INSERT INTO alphatrace.workspaces(owner_id,name) VALUES ($1,'My research') ON CONFLICT DO NOTHING",[owner]);
  await c.query('SELECT pg_advisory_xact_lock(hashtext($1))',['inbox:'+owner]);
  const rows=(await c.query(`SELECT u.id,u.payload,p.quantity IS NOT NULL AS holding,p.followed,
   r.version AS thesis_version,r.payload AS thesis_payload,r.reason AS thesis_reason,r.created_at AS thesis_created_at
   FROM alphatrace.research_updates u
   JOIN alphatrace.portfolio_entries p ON p.owner_id=$1 AND p.symbol=u.symbol
   LEFT JOIN alphatrace.private_research h ON h.owner_id=$1 AND h.symbol=u.symbol
   LEFT JOIN alphatrace.private_research_revisions r ON r.owner_id=h.owner_id AND r.symbol=h.symbol AND r.version=h.version
   WHERE u.withdrawn_at IS NULL AND (p.followed OR p.quantity IS NOT NULL)
    AND NOT EXISTS(SELECT 1 FROM alphatrace.review_items i WHERE i.owner_id=$1 AND i.update_id=u.id)
   ORDER BY u.published_at,u.id LIMIT 51`,[owner])).rows;
  for(const row of rows.slice(0,50)){
   const thesis:ThesisRevision|null=row.thesis_version?{version:row.thesis_version,payload:row.thesis_payload,reason:row.thesis_reason,created_at:row.thesis_created_at}:null;
   const snapshot:InboxSnapshot={ruleVersion:'metric-topic-v1',reason:row.holding?(row.followed?'holding and watchlist':'holding'):'watchlist',thesis,matches:matchAssumptions(thesis?.payload.assumptions??[],row.payload)};
   await c.query('INSERT INTO alphatrace.review_items(owner_id,update_id,snapshot) VALUES ($1,$2,$3) ON CONFLICT DO NOTHING',[owner,row.id,JSON.stringify(snapshot)]);
  }
  return {added:Math.min(rows.length,50),hasMore:rows.length>50};
 });
}
export async function getInbox(owner:string,filter?:unknown,cursor?:string|null):Promise<InboxList>{
 const f=inboxFilter(filter),before=inboxCursor(cursor);
 return asUser(owner,async c=>{
  const rows=(await c.query(`SELECT ${itemColumns} FROM alphatrace.review_items i JOIN alphatrace.research_updates u ON u.id=i.update_id
   WHERE i.owner_id=$1 AND ${filterSql[f]} AND ($2::timestamptz IS NULL OR (i.created_at,i.update_id)<($2::timestamptz,$3::uuid))
   ORDER BY i.created_at DESC,i.update_id DESC LIMIT 26`,[owner,before?.[0]??null,before?.[1]??null])).rows;
  const counts=(await c.query(`SELECT count(*) FILTER(WHERE ${pending})::int AS pending,count(*) FILTER(WHERE ${deferred})::int AS deferred,
   count(*) FILTER(WHERE ${reviewed})::int AS reviewed,count(*)::int AS all FROM alphatrace.review_items i WHERE owner_id=$1`,[owner])).rows[0];
  const items=rows.slice(0,25),last=items.at(-1);
  return serial({items,nextCursor:rows.length>25?`${last.created_at}|${last.update_id}`:null,counts});
 });
}
async function currentThesis(c:PoolClient,owner:string,symbol:string):Promise<ThesisRevision|null>{
 const head=(await c.query('SELECT version FROM alphatrace.private_research WHERE owner_id=$1 AND symbol=$2 FOR SHARE',[owner,symbol])).rows[0];
 return head?.version?(await c.query('SELECT version,payload,reason,created_at FROM alphatrace.private_research_revisions WHERE owner_id=$1 AND symbol=$2 AND version=$3',[owner,symbol,head.version])).rows[0]:null;
}
async function detail(c:PoolClient,owner:string,updateId:string,before?:number):Promise<InboxDetail>{
 const row=(await c.query(`SELECT ${itemColumns},i.snapshot FROM alphatrace.review_items i JOIN alphatrace.research_updates u ON u.id=i.update_id
  WHERE i.owner_id=$1 AND i.update_id=$2 FOR SHARE OF i`,[owner,updateId])).rows[0];
 if(!row)throw new InboxInputError('This review is unavailable in your inbox.');
 const {snapshot,...item}=row;
 const publication=(await c.query('SELECT id,symbol,version,payload,published_at,withdrawn_at,withdrawal_reason,supersedes FROM alphatrace.research_updates WHERE id=$1',[updateId])).rows[0];
 const events=(await c.query('SELECT version,action,note,revisit_at,thesis_revision,created_at FROM alphatrace.review_events WHERE owner_id=$1 AND update_id=$2 AND ($3::integer IS NULL OR version<$3) ORDER BY version DESC LIMIT 21',[owner,updateId,before??null])).rows;
 const current=await currentThesis(c,owner,row.symbol);
 const previous=(await c.query("SELECT max((thesis_revision->>'version')::int) AS version FROM alphatrace.review_events WHERE owner_id=$1 AND update_id=$2 AND action='revise'",[owner,updateId])).rows[0]?.version??0;
 const canRecordRevision=!!current&&!publication.withdrawn_at&&current.version>Math.max(snapshot.thesis?.version??0,previous)&&current.payload.evidenceIds.includes(updateId);
 const check=(await c.query('SELECT status,checked_at,message,fingerprints FROM alphatrace.research_checks WHERE symbol=$1 ORDER BY checked_at DESC LIMIT 1',[row.symbol])).rows[0]??null;
 return serial({item,snapshot,publication,events:events.slice(0,20),hasMoreEvents:events.length>20,currentThesisVersion:current?.version??null,canRecordRevision,check});
}
export async function getInboxDetail(owner:string,update:unknown,before?:number){
 const id=inboxUuid(update);
 if(before!==undefined&&(!Number.isInteger(before)||before<1||before>2147483647))throw new InboxInputError('Invalid history page.');
 return asUser(owner,c=>detail(c,owner,id,before));
}
export async function reviewInbox(owner:string,input:unknown){
 const command=validateReview(input);
 return asUser(owner,async c=>{
  await c.query('SELECT pg_advisory_xact_lock(hashtext($1))',['inbox:'+owner]);
  const retry=(await c.query('SELECT command=$3::jsonb AS matches FROM alphatrace.review_events WHERE owner_id=$1 AND command_id=$2',[owner,command.commandId,JSON.stringify(command)])).rows[0];
  if(retry){if(!retry.matches)throw new InboxConflictError('This save identifier was already used for another response. Reload before saving.');return detail(c,owner,command.updateId);}
  const item=(await c.query('SELECT * FROM alphatrace.review_items WHERE owner_id=$1 AND update_id=$2 FOR UPDATE',[owner,command.updateId])).rows[0];
  if(!item)throw new InboxInputError('This review is unavailable in your inbox.');
  if(item.version!==command.version)throw new InboxConflictError('This review changed in another tab. Your note is still here; reload the saved response before retrying.');
  // Row locks also require the operator-only UPDATE policy. Readers instead
  // share the advisory lock used by withdrawal, without gaining write access.
  await c.query('SELECT pg_advisory_xact_lock_shared(hashtext($1))',['publication:'+command.updateId]);
  const publication=(await c.query('SELECT symbol,withdrawn_at FROM alphatrace.research_updates WHERE id=$1',[command.updateId])).rows[0];
  if(publication.withdrawn_at&&['keep','revise'].includes(command.action))throw new InboxInputError('This publication was withdrawn. Read the withdrawal reason before recording a different response.');
  let thesis:ThesisRevision|null=null;
  if(command.action==='revise'){
   thesis=await currentThesis(c,owner,publication.symbol);
   const previous=(await c.query("SELECT max((thesis_revision->>'version')::int) AS version FROM alphatrace.review_events WHERE owner_id=$1 AND update_id=$2 AND action='revise'",[owner,command.updateId])).rows[0]?.version??0;
   if(!thesis||thesis.version!==command.thesisVersion||thesis.version<=Math.max(item.snapshot.thesis?.version??0,previous)||!thesis.payload.evidenceIds.includes(command.updateId))throw new InboxConflictError('Save a new thesis revision linked to this exact publication, then reload this review. Your current thesis is never changed by recording a response.');
  }
  const revisit=command.deferUntil?revisitTimestamp(command.deferUntil):null;
  await c.query('INSERT INTO alphatrace.review_events(owner_id,update_id,version,command_id,action,note,revisit_at,thesis_revision,command) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)',[owner,command.updateId,item.version+1,command.commandId,command.action,command.note,revisit,thesis?JSON.stringify(thesis):null,JSON.stringify(command)]);
  const states={keep:'kept',dismiss:'dismissed',defer:'deferred',revise:'revised',reopen:'unreviewed'};
  await c.query('UPDATE alphatrace.review_items SET version=version+1,state=$3,revisit_at=$4,updated_at=now() WHERE owner_id=$1 AND update_id=$2',[owner,command.updateId,states[command.action],revisit]);
  return detail(c,owner,command.updateId);
 });
}
