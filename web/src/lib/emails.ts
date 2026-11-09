import { createHash,randomBytes,randomUUID } from 'node:crypto';
import type { PoolClient } from 'pg';
import { asUser } from './db';
import { PortfolioConflictError,PortfolioInputError } from './portfolio-input';
import { digestPayload,emailsAvailable,validateEmailPreferences,type DigestItem,type EmailConfig,type EmailPayload,type EmailView } from './email-contract';
import type { SendOutcome } from './resend';

const serial=<T>(value:T):T=>JSON.parse(JSON.stringify(value));
const prefColumns='version,enabled,reminders,enabled_since AS "enabledSince",suppressed';
async function settings(c:PoolClient,owner:string,available:boolean):Promise<EmailView>{
 const preferences=(await c.query(`SELECT ${prefColumns} FROM alphatrace.email_preferences WHERE owner_id=$1`,[owner])).rows[0]??{version:0,enabled:false,reminders:true,enabledSince:null,suppressed:false};
 const deliveries=(await c.query(`SELECT id,digest_day::text AS day,state,message,created_at AS "createdAt",accepted_at AS "acceptedAt",delivered_at AS "deliveredAt",bounced_at AS "bouncedAt",complained_at AS "complainedAt"
  FROM alphatrace.email_deliveries WHERE owner_id=$1 ORDER BY created_at DESC LIMIT 10`,[owner])).rows;
 return serial({preferences,deliveries,available});
}
export async function getEmailSettings(owner:string){return asUser(owner,c=>settings(c,owner,emailsAvailable(process.env)));}
export async function saveEmailPreferences(owner:string,input:unknown,available=emailsAvailable(process.env)){
 const command=validateEmailPreferences(input);
 if(command.enabled&&!available)throw new PortfolioInputError('Email delivery is not available yet. You can still use your review inbox.');
 return asUser(owner,async c=>{
  await c.query("INSERT INTO alphatrace.workspaces(owner_id,name) VALUES($1,'My research') ON CONFLICT DO NOTHING",[owner]);
  await c.query('SELECT id FROM alphatrace.workspaces WHERE owner_id=$1 FOR UPDATE',[owner]);
  await c.query('INSERT INTO alphatrace.email_preferences(owner_id) VALUES($1) ON CONFLICT DO NOTHING',[owner]);
  const current=(await c.query('SELECT * FROM alphatrace.email_preferences WHERE owner_id=$1 FOR UPDATE',[owner])).rows[0];
  if(current.version!==command.version)throw new PortfolioConflictError('Email settings changed in another tab. Reload before saving.');
  if(command.enabled&&current.suppressed)throw new PortfolioInputError('Email delivery was stopped after a bounce or complaint. Contact the beta operator before enabling it again.');
  await c.query(`UPDATE alphatrace.email_preferences SET enabled=$2,reminders=$3,version=version+1,updated_at=now(),
   enabled_since=CASE WHEN $2 AND NOT enabled THEN now() ELSE enabled_since END WHERE owner_id=$1`,[owner,command.enabled,command.reminders]);
  await c.query("UPDATE alphatrace.email_deliveries SET state='cancelled',message='Email settings changed before delivery.' WHERE owner_id=$1 AND state IN ('pending','retry','sending')",[owner]);
  return settings(c,owner,available);
 });
}
async function eligibleItems(c:PoolClient,owner:string,since:string,reminders:boolean):Promise<DigestItem[]>{
 const rows=(await c.query(`SELECT u.id AS "updateId",u.symbol,u.payload->>'title' AS title,u.period_end::text AS "periodEnd",i.state,i.version,i.revisit_at,
  CASE WHEN i.state='deferred' THEN 'reminder:'||u.id::text||':'||i.version::text ELSE 'publication:'||u.id::text END AS key
  FROM alphatrace.research_updates u JOIN alphatrace.portfolio_entries p ON p.owner_id=$1 AND p.symbol=u.symbol
  LEFT JOIN alphatrace.review_items i ON i.owner_id=$1 AND i.update_id=u.id
  WHERE u.withdrawn_at IS NULL AND NOT EXISTS(SELECT 1 FROM alphatrace.research_updates n WHERE n.symbol=u.symbol AND n.version>u.version)
  AND (((i.state IS NULL OR i.state='unreviewed') AND u.published_at>=$2)
   OR ($3 AND i.state='deferred' AND i.revisit_at<=now()))
  AND NOT EXISTS(SELECT 1 FROM alphatrace.email_delivery_items e WHERE e.owner_id=$1 AND e.event_key=CASE WHEN i.state='deferred' THEN 'reminder:'||u.id::text||':'||i.version::text ELSE 'publication:'||u.id::text END)
  ORDER BY u.published_at,u.id LIMIT 50`,[owner,since,reminders])).rows;
 return rows.map(r=>({key:r.key,updateId:r.updateId,symbol:r.symbol,title:r.title,periodEnd:r.periodEnd,kind:r.state==='deferred'?'reminder':'publication',reviewVersion:r.state==='deferred'?r.version:null}));
}
/** Queue a frozen payload first, before any external send. Empty days create no email. */
export async function prepareEmailDigest(owner:string,recipient:string,config:EmailConfig):Promise<string|null>{
 return asUser(owner,async c=>{
  const pref=(await c.query('SELECT * FROM alphatrace.email_preferences WHERE owner_id=$1 FOR UPDATE',[owner])).rows[0];
  if(!pref?.enabled||pref.suppressed)return null;
  const existing=(await c.query("SELECT id FROM alphatrace.email_deliveries WHERE owner_id=$1 AND (digest_day=(now() AT TIME ZONE 'Asia/Kolkata')::date OR (accepted_at AT TIME ZONE 'Asia/Kolkata')::date=(now() AT TIME ZONE 'Asia/Kolkata')::date)",[owner])).rows[0];
  if(existing)return null;
  const items=await eligibleItems(c,owner,pref.enabled_since,pref.reminders);if(!items.length)return null;
  const id=randomUUID(),token=randomBytes(32).toString('base64url'),hash=createHash('sha256').update(token).digest('hex');
  const payload=digestPayload(config,recipient,items,token);
  await c.query(`INSERT INTO alphatrace.email_deliveries(id,owner_id,digest_day,preference_version,recipient,payload,items,unsubscribe_hash)
   VALUES($1,$2,(now() AT TIME ZONE 'Asia/Kolkata')::date,$3,$4,$5,$6,$7)`,[id,owner,pref.version,recipient,JSON.stringify(payload),JSON.stringify(items),hash]);
  for(const item of items)await c.query('INSERT INTO alphatrace.email_delivery_items(owner_id,delivery_id,event_key) VALUES($1,$2,$3)',[owner,id,item.key]);
  return id;
 });
}
async function stillRelevant(c:PoolClient,owner:string,items:DigestItem[]){
 const rows=(await c.query(`SELECT u.id,i.state,i.version,i.revisit_at,EXISTS(SELECT 1 FROM alphatrace.portfolio_entries p WHERE p.owner_id=$1 AND p.symbol=u.symbol) AS tracked,
  u.withdrawn_at,EXISTS(SELECT 1 FROM alphatrace.research_updates n WHERE n.symbol=u.symbol AND n.version>u.version) AS superseded
  FROM alphatrace.research_updates u LEFT JOIN alphatrace.review_items i ON i.owner_id=$1 AND i.update_id=u.id WHERE u.id=ANY($2::uuid[])`,[owner,items.map(i=>i.updateId)])).rows;
 return items.every(item=>{const row=rows.find(r=>r.id===item.updateId);return row?.tracked&&!row.withdrawn_at&&!row.superseded&&(item.kind==='reminder'?row.state==='deferred'&&row.version===item.reviewVersion&&new Date(row.revisit_at).getTime()<=Date.now():!row.state||row.state==='unreviewed');});
}
/** The first attempt time is committed separately, so a crash cannot restart the
 * provider's 24-hour idempotency window or generate a different retry payload. */
export async function deliverEmailDigest(owner:string,id:string,recipient:string|null,send:(payload:EmailPayload,id:string)=>Promise<SendOutcome>):Promise<string>{
 const lease=randomUUID();
 const claimed=await asUser(owner,async c=>{
  const pref=(await c.query('SELECT * FROM alphatrace.email_preferences WHERE owner_id=$1 FOR UPDATE',[owner])).rows[0];
  const row=(await c.query("SELECT * FROM alphatrace.email_deliveries WHERE owner_id=$1 AND id=$2 FOR UPDATE",[owner,id])).rows[0];
  if(!row||!['pending','retry','sending'].includes(row.state)||row.lease_until&&Date.parse(row.lease_until)>Date.now()||Date.parse(row.next_attempt_at)>Date.now())return false;
  if(!pref?.enabled||pref.suppressed||pref.version!==row.preference_version||!recipient||recipient!==row.recipient||!await stillRelevant(c,owner,row.items)){
   await c.query("UPDATE alphatrace.email_deliveries SET state='cancelled',message='Account, preferences or research changed before delivery.' WHERE id=$1",[id]);return false;
  }
  if(row.first_attempt_at&&Date.now()-Date.parse(row.first_attempt_at)>23*60*60*1000||row.attempts>=5){
   await c.query("UPDATE alphatrace.email_deliveries SET state='uncertain',message='Delivery could not be confirmed. Automatic retries stopped to prevent duplicates.' WHERE id=$1",[id]);return false;
  }
  const other=(await c.query("SELECT id FROM alphatrace.email_deliveries WHERE owner_id=$1 AND id<>$2 AND (accepted_at AT TIME ZONE 'Asia/Kolkata')::date=(now() AT TIME ZONE 'Asia/Kolkata')::date LIMIT 1",[owner,id])).rowCount;
  if(other)return false;
  await c.query("UPDATE alphatrace.email_deliveries SET state='sending',attempts=attempts+1,first_attempt_at=COALESCE(first_attempt_at,now()),lease_id=$2,lease_until=now()+interval '90 seconds' WHERE id=$1",[id,lease]);return true;
 });
 if(!claimed)return 'skipped';
 return asUser(owner,async c=>{
  // Serialize dispatch with opt-out. An already in-flight message cannot be recalled.
  const pref=(await c.query('SELECT * FROM alphatrace.email_preferences WHERE owner_id=$1 FOR UPDATE',[owner])).rows[0];
  const row=(await c.query("SELECT * FROM alphatrace.email_deliveries WHERE owner_id=$1 AND id=$2 AND lease_id=$3 AND state='sending' FOR UPDATE",[owner,id,lease])).rows[0];
  if(!row)return 'skipped';
  if(!pref?.enabled||pref.suppressed||pref.version!==row.preference_version||!await stillRelevant(c,owner,row.items)){
   await c.query("UPDATE alphatrace.email_deliveries SET state='cancelled',message='Preferences or research changed before dispatch.' WHERE id=$1",[id]);return 'cancelled';
  }
  // Publication withdrawal uses this advisory lock too. A send already in flight
  // may finish first; a committed withdrawal is never included in a new send.
  for(const update of [...new Set((row.items as DigestItem[]).map(i=>i.updateId))].sort())await c.query('SELECT pg_advisory_xact_lock_shared(hashtext($1))',['publication:'+update]);
  if(!await stillRelevant(c,owner,row.items)){await c.query("UPDATE alphatrace.email_deliveries SET state='cancelled',message='Research changed before dispatch.' WHERE id=$1",[id]);return 'cancelled';}
  const result=await send(row.payload,id);
  if(result.state==='accepted')await c.query("UPDATE alphatrace.email_deliveries SET state='accepted',provider_id=$2,accepted_at=now(),lease_id=NULL,lease_until=NULL,message='Accepted by Resend; delivery confirmation is pending.' WHERE id=$1",[id,result.providerId]);
  else await c.query("UPDATE alphatrace.email_deliveries SET state=$2,message=$3,lease_id=NULL,lease_until=NULL,next_attempt_at=now()+interval '15 minutes' WHERE id=$1",[id,result.state,result.message]);
  return result.state;
 });
}
export async function pendingEmailDigests(owner:string):Promise<string[]>{return asUser(owner,async c=>(await c.query("SELECT id FROM alphatrace.email_deliveries WHERE owner_id=$1 AND state IN ('pending','retry','sending') AND next_attempt_at<=now() ORDER BY created_at LIMIT 2",[owner])).rows.map(r=>r.id));}

// These two capability functions reveal no user data and cannot subscribe anyone.
const capabilityIdentity='00000000-0000-4000-8000-000000000000';
export async function unsubscribeEmail(token:unknown){
 if(typeof token!=='string'||!/^[A-Za-z0-9_-]{43}$/.test(token))throw new PortfolioInputError('This unsubscribe link is invalid. Open email preferences after signing in instead.');
 const hash=createHash('sha256').update(token).digest('hex');
 await asUser(capabilityIdentity,c=>c.query('SELECT alphatrace.unsubscribe_email($1)',[hash]),{accountControl:true});
}
export async function recordEmailEvent(id:string,providerId:string,type:string,at:string){return asUser(capabilityIdentity,async c=>(await c.query('SELECT alphatrace.record_email_event($1,$2,$3,$4) AS recorded',[id,providerId,type,at])).rows[0].recorded as boolean,{accountControl:true});}
