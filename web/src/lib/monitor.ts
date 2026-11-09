import type { PoolClient } from 'pg';
import type { WorkerMode } from './worker-status';
import { asUser } from './db';
import { ingestResearchWithClient } from './research';
import { ResearchConflictError,ResearchForbiddenError,ResearchInputError,validateCheck,validateResearch } from './research-contract';
import { monitorCommand,monitorLeaseMs,monitorNextAttempt,validateDiscovery,type MonitorSymbol,type MonitorView } from './monitor-contract';

async function operator(c:PoolClient,id:string) {
 if(!(await c.query('SELECT 1 FROM alphatrace.research_operators WHERE user_id=$1',[id])).rowCount)throw new ResearchForbiddenError('Source monitoring controls require operator access.');
}
const serialize=<T>(value:unknown):T=>JSON.parse(JSON.stringify(value));
const publicRun='id,symbol,started_at,finished_at,state,attempt,message,failure_code,draft_created,discovery';
export async function getMonitor(owner:string):Promise<MonitorView> {
 return asUser(owner,async c=>{
  const isOperator=!!(await c.query('SELECT 1 FROM alphatrace.research_operators WHERE user_id=$1',[owner])).rowCount;
  const targets=(await c.query("SELECT symbol,enabled,next_due_at,worker_seen_at,worker_mode,active_run_id,lease_until,attempt,(SELECT max(finished_at) FROM alphatrace.monitor_runs r WHERE r.symbol=t.symbol AND r.state='success') AS last_automatic_success,(SELECT max(checked_at) FROM alphatrace.research_checks c WHERE c.symbol=t.symbol AND c.status='success') AS last_successful_check,(SELECT discovery FROM alphatrace.monitor_runs r WHERE r.symbol=t.symbol AND r.discovery IS NOT NULL ORDER BY r.started_at DESC,r.id DESC LIMIT 1) AS discovery FROM alphatrace.monitor_targets t ORDER BY symbol")).rows;
  const latest=(await c.query(`SELECT DISTINCT ON(symbol) ${publicRun} FROM alphatrace.monitor_runs ORDER BY symbol,started_at DESC,id DESC`)).rows;
  const runs=(await c.query(`SELECT ${publicRun} FROM alphatrace.monitor_runs ORDER BY started_at DESC,id DESC LIMIT 20`)).rows;
  const serverNow=(await c.query('SELECT now() AS time')).rows[0].time;
  return serialize({isOperator,serverNow,targets:targets.map(t=>({...t,latest:latest.find(r=>r.symbol===t.symbol)??null})),runs});
 });
}
export async function controlMonitor(owner:string,input:unknown) {
 const command=monitorCommand(input);
 return asUser(owner,async c=>{
  await operator(c,owner);
  const target=(await c.query('SELECT * FROM alphatrace.monitor_targets WHERE symbol=$1 FOR UPDATE',[command.symbol])).rows[0];
  if(!target)throw new ResearchInputError('Monitoring is not configured for this company.');
  if(command.action==='pause')await c.query('UPDATE alphatrace.monitor_targets SET enabled=false,next_due_at=NULL WHERE symbol=$1',[command.symbol]);
  else {
   if(target.active_run_id)throw new ResearchConflictError('A source check is already running. Wait for it to finish.');
   if(command.action==='check'&&!target.enabled)throw new ResearchConflictError('Resume this monitor before requesting a check.');
   // Duplicate queued commands must not change the run or reset retry bounds.
   if(target.enabled&&target.next_due_at&&new Date(target.next_due_at).getTime()<=Date.now())return {queued:true};
   if(command.action==='check'){
    const recent=(await c.query("SELECT 1 FROM alphatrace.monitor_runs WHERE symbol=$1 AND started_at>now()-interval '5 minutes' LIMIT 1",[command.symbol])).rowCount;
    if(recent)throw new ResearchConflictError('A check ran recently. Wait five minutes before checking again.');
   }
   await c.query('UPDATE alphatrace.monitor_targets SET enabled=true,next_due_at=now(),attempt=0 WHERE symbol=$1',[command.symbol]);
  }
  return {queued:command.action!=='pause'};
 });
}
export async function monitorHeartbeat(owner:string,mode:WorkerMode='local') {
 return asUser(owner,async c=>{await operator(c,owner);await c.query('UPDATE alphatrace.monitor_targets SET worker_seen_at=now(),worker_mode=$1',[mode]);});
}
export type MonitorClaim={id:string;symbol:MonitorSymbol;attempt:number};
export async function claimMonitorRun(owner:string):Promise<MonitorClaim|null> {
 return asUser(owner,async c=>{
  await operator(c,owner);
  // Recover abandoned leases even if an operator paused that target mid-run.
  const expired=(await c.query('SELECT * FROM alphatrace.monitor_targets WHERE active_run_id IS NOT NULL AND lease_until<=now() FOR UPDATE SKIP LOCKED')).rows;
  for(const target of expired){
   const message='The source-check process stopped before completion. No fresh evidence was confirmed.';
   await c.query("UPDATE alphatrace.monitor_runs SET state='failed',finished_at=now(),failure_code='interrupted',message=$2 WHERE id=$1 AND state='running'",[target.active_run_id,message]);
   await ingestResearchWithClient(c,owner,{check:{symbol:target.symbol,status:'failed',checkedAt:new Date().toISOString(),message,fingerprints:[]}});
   const next=monitorNextAttempt(target.attempt,'interrupted');
   await c.query('UPDATE alphatrace.monitor_targets SET active_run_id=NULL,lease_until=NULL,next_due_at=CASE WHEN enabled THEN $2::timestamptz ELSE NULL END,attempt=$3 WHERE symbol=$1',[target.symbol,next.due,next.attempt]);
  }
  const target=(await c.query('SELECT * FROM alphatrace.monitor_targets WHERE enabled AND active_run_id IS NULL AND next_due_at<=now() ORDER BY next_due_at,symbol FOR UPDATE SKIP LOCKED LIMIT 1')).rows[0];
  if(!target)return null;
  const run=(await c.query('INSERT INTO alphatrace.monitor_runs(symbol,attempt) VALUES($1,$2) RETURNING id,symbol,attempt',[target.symbol,target.attempt+1])).rows[0];
  await c.query("UPDATE alphatrace.monitor_targets SET active_run_id=$2,lease_until=now()+$3*interval '1 millisecond',attempt=$4 WHERE symbol=$1",[target.symbol,run.id,monitorLeaseMs,run.attempt]);
  return run;
 });
}

export async function finishMonitorRun(owner:string,claim:MonitorClaim,bundle:unknown,failureCode:string|null=null) {
 const body=bundle as {check?:unknown;draft?:unknown;discovery?:unknown};
 const check=validateCheck(body?.check),draft=body?.draft===undefined?undefined:validateResearch(body.draft);
 const discovery=validateDiscovery(body?.discovery,claim.symbol);
 if(check.symbol!==claim.symbol||draft&&draft.symbol!==claim.symbol)throw new ResearchInputError('Source check does not match the claimed company.');
 if(check.status==='success'&&(!draft||draft.sources.some(s=>s.retrieval!=='direct'||s.fingerprintKind!=='document_bytes')))throw new ResearchInputError('Scheduled checks require fresh direct source retrieval.');
 return asUser(owner,async c=>{
  await operator(c,owner);
  const target=(await c.query('SELECT *,lease_until>now() AS valid_lease FROM alphatrace.monitor_targets WHERE symbol=$1 FOR UPDATE',[claim.symbol])).rows[0];
  if(!target||target.active_run_id!==claim.id||!target.valid_lease)return {accepted:false};
  const run=(await c.query('SELECT attempt,started_at FROM alphatrace.monitor_runs WHERE id=$1 AND state=\'running\'',[claim.id])).rows[0];
  if(!run)return {accepted:false};
  if(discovery&&Date.parse(discovery.checkedAt)<new Date(run.started_at).getTime())throw new ResearchInputError('A scheduled run cannot replay an older discovery.');
  if(discovery&&draft&&(draft.periodEnd!==discovery.periodEnd||draft.sources.some(s=>!s.url.startsWith(`https://www.sec.gov/Archives/edgar/data/1067491/${discovery.accession.replaceAll('-','')}/`))))throw new ResearchInputError('Draft sources must match the discovered SEC filing.');
  if(check.status==='success'&&(Date.parse(check.checkedAt)<new Date(run.started_at).getTime()||draft!.sources.some(s=>Date.parse(s.fetchedAt)<new Date(run.started_at).getTime())))throw new ResearchInputError('A scheduled run cannot replay an older source check.');
  const result=await ingestResearchWithClient(c,owner,{check,...(draft?{draft}:{})});
  const code=check.status==='failed'?(failureCode??'validation'):null;
  const next=monitorNextAttempt(run.attempt,code);
  await c.query('UPDATE alphatrace.monitor_runs SET state=$2,finished_at=now(),message=$3,failure_code=$4,draft_id=$5,draft_created=$6,discovery=$7::jsonb WHERE id=$1',[claim.id,check.status,check.message,code,result.draftId,result.created,discovery?JSON.stringify(discovery):null]);
  await c.query('UPDATE alphatrace.monitor_targets SET active_run_id=NULL,lease_until=NULL,next_due_at=CASE WHEN enabled THEN $2::timestamptz ELSE NULL END,attempt=$3 WHERE symbol=$1',[claim.symbol,next.due,next.attempt]);
  return {accepted:true,created:result.created};
 });
}
