import {workerOnline,type WorkerMode} from './worker-status';
import { ResearchInputError,officialSourceUrl } from './research-contract';
export const monitorSymbols=['TCS','INFY'] as const;
export type MonitorSymbol=typeof monitorSymbols[number];
export const monitorCadence='Monday & Thursday · 6 pm IST';
export const monitorLeaseMs=5*60_000;
export const monitorHeartbeatMs=3*60_000;
export const retryableMonitorCodes=['network','timeout','http_429','http_500','http_502','http_503','http_504','interrupted'];
export type MonitorDiscovery={checkedAt:string;feedUrl:string;feedSha256:string;accession:string;filedOn:string;periodEnd:string;url:string;candidateCount:number};
export type MonitorRun={id:string;symbol:MonitorSymbol;started_at:string;finished_at:string|null;state:'running'|'success'|'failed';attempt:number;message:string;failure_code:string|null;draft_created:boolean;discovery?:MonitorDiscovery|null};
export type MonitorTarget={symbol:MonitorSymbol;enabled:boolean;next_due_at:string|null;worker_seen_at:string|null;worker_mode?:WorkerMode;last_successful_check:string|null;last_automatic_success?:string|null;attempt?:number;active_run_id:string|null;lease_until:string|null;latest:MonitorRun|null;discovery?:MonitorDiscovery|null};
export type MonitorView={isOperator:boolean;serverNow:string;targets:MonitorTarget[];runs:MonitorRun[]};

export function validateDiscovery(input:unknown,symbol:MonitorSymbol):MonitorDiscovery|null {
 if(input===undefined||input===null)return null;
 const d=input as MonitorDiscovery;
 const date=(v:unknown)=>typeof v==='string'&&/^\d{4}-\d{2}-\d{2}$/.test(v)&&Number.isFinite(Date.parse(v))&&new Date(v).toISOString().slice(0,10)===v;
 if(symbol!=='INFY'||d.feedUrl!=='https://data.sec.gov/submissions/CIK0001067491.json'||typeof d.feedSha256!=='string'||!/^[a-f0-9]{64}$/.test(d.feedSha256)||typeof d.accession!=='string'||!/^\d{10}-\d{2}-\d{6}$/.test(d.accession)||!date(d.filedOn)||!date(d.periodEnd)||d.periodEnd>d.filedOn||d.filedOn>new Date().toISOString().slice(0,10)||!Number.isInteger(d.candidateCount)||d.candidateCount<1||d.candidateCount>300||typeof d.checkedAt!=='string'||!/^\d{4}-\d{2}-\d{2}T/.test(d.checkedAt)||!Number.isFinite(Date.parse(d.checkedAt))||Date.parse(d.checkedAt)>Date.now()+300000)throw new ResearchInputError('Invalid SEC filing discovery.');
 const url=officialSourceUrl(d.url,symbol);
 if(!url.startsWith(`https://www.sec.gov/Archives/edgar/data/1067491/${d.accession.replaceAll('-','')}/`))throw new ResearchInputError('Discovery URL does not match the SEC accession.');
 return {checkedAt:new Date(d.checkedAt).toISOString(),feedUrl:d.feedUrl,feedSha256:d.feedSha256,accession:d.accession,filedOn:d.filedOn,periodEnd:d.periodEnd,url,candidateCount:d.candidateCount};
}

/** Next fixed Mon/Thu 18:00 Asia/Kolkata. IST has no daylight-saving transition. */
export function nextMonitorTime(now=Date.now()):string {
 const day=new Date(now);day.setUTCHours(12,30,0,0);
 for(let n=0;n<8;n++){
  if(day.getTime()>now&&[1,4].includes(day.getUTCDay()))return day.toISOString();
  day.setUTCDate(day.getUTCDate()+1);
 }
 throw new Error('Unable to calculate monitor schedule.');
}
export function monitorNextAttempt(attempt:number,failureCode:string|null,now=Date.now()):{due:string;attempt:number} {
 return attempt<2&&failureCode&&retryableMonitorCodes.includes(failureCode)?{due:new Date(now+5*60_000).toISOString(),attempt}:{due:nextMonitorTime(now),attempt:0};
}
export function monitorCommand(input:unknown):{symbol:MonitorSymbol;action:'pause'|'resume'|'check'} {
 const b=input as Record<string,unknown>|null;
 if(!b||!monitorSymbols.includes(b.symbol as MonitorSymbol)||!['pause','resume','check'].includes(String(b.action)))throw new ResearchInputError('Choose a supported company and monitoring action.');
 return {symbol:b.symbol as MonitorSymbol,action:b.action as 'pause'|'resume'|'check'};
}
export function monitorState(target:MonitorTarget,now=Date.now()):{label:string;tone:string} {
 if(target.active_run_id&&Date.parse(target.lease_until!)<=now)return {label:'Run interrupted',tone:'amber'};
 if(target.active_run_id)return {label:'Checking sources',tone:'lavender'};
 if(!target.enabled)return {label:'Paused',tone:'neutral'};
 if(!workerOnline(target,now,monitorHeartbeatMs))return {label:target.worker_mode==='cloud'?'Cloud scheduler overdue':'Runner offline',tone:'amber'};
 if(target.latest?.state==='failed')return {label:['http_401','http_403'].includes(target.latest.failure_code??'')?'Source access blocked':'Check failed',tone:'amber'};
 if(target.next_due_at&&Date.parse(target.next_due_at)<=now)return {label:'Queued',tone:'lavender'};
 return {label:target.latest?.state==='success'?'Scheduled':'Awaiting first check',tone:'neutral'};
}

export function retryPending(target:MonitorTarget):boolean {
 return target.enabled&&!target.active_run_id&&!!target.next_due_at&&target.attempt===1&&target.latest?.state==='failed'&&retryableMonitorCodes.includes(target.latest.failure_code??'');
}
export function monitorGuidance(target:MonitorTarget,now=Date.now()):string {
 const offline=!workerOnline(target,now,monitorHeartbeatMs);
 if(target.active_run_id)return Date.parse(target.lease_until!)<=now?'This run was interrupted. The runner will recover its expired lease and schedule a bounded retry when it returns.':'A check is in progress. Validated changes will be saved as an unpublished draft.';
 const failure=target.latest?.state==='failed'?target.latest.failure_code:null;
 const action=failure==='http_403'||failure==='http_401'?(target.symbol==='TCS'?'The source refused automated access. Use the official-file import below to prepare evidence for review.':'The source refused automated access. An operator must check official source availability.'):
 failure==='dependency'?'The source reader could not start. An operator must restore its required tools before retrying.':
 failure==='validation'?'The downloaded evidence could not be validated. An operator must inspect the originals and extraction rules; no fresh evidence was confirmed.':
 retryPending(target)?'One automatic retry is queued. If it also fails, checks return to the regular schedule.':
 failure?'The latest check failed. No fresh evidence was confirmed; the next regular attempt is shown above.':
 'Validated changes create drafts for human review. An unchanged document does not create another draft.';
 return (!target.enabled?'Monitoring is paused. Resume it to schedule checks. ':offline?'The runner is offline or overdue. Queued work will start when it returns. ':'')+action;
}
