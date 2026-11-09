if(process.env.ALPHATRACE_CLOUD_MONITORING==='true')throw Error('Cloud monitoring is enabled. Use the cloud task; local workers are disabled to avoid duplicate scheduling.');
/** Local service: no model, no snapshots, no publication privileges in its workflow. */
import { spawn,type ChildProcess } from 'node:child_process';
import { mkdtemp,readFile,stat,rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { closeDatabase } from '../src/lib/db';
import { UUID } from '../src/lib/access';
import { claimMonitorRun,finishMonitorRun,monitorHeartbeat,controlMonitor,type MonitorClaim } from '../src/lib/monitor';
import { monitorSymbols } from '../src/lib/monitor-contract';

const args=new Set(process.argv.slice(2));
if([...args].some(a=>!['--watch','--once','--enable'].includes(a))||args.has('--watch')===args.has('--once'))throw Error('Choose --watch or --once; optionally --enable for initial activation.');
const owner=process.env.ALPHATRACE_INGEST_USER_ID;
if(!owner||!UUID.test(owner))throw Error('Configure the enrolled ALPHATRACE_INGEST_USER_ID before starting monitoring.');
let stopped=false,child:ChildProcess|undefined,wake:(()=>void)|undefined;
const stop=()=>{stopped=true;child?.kill('SIGTERM');wake?.();};
process.on('SIGTERM',stop);process.on('SIGINT',stop);
const root=fileURLToPath(new URL('../../',import.meta.url));
const fail=(symbol:string,code:string,message:string)=>({failureCode:code,check:{symbol,checkedAt:new Date().toISOString(),status:'failed',message,fingerprints:[]}});
async function retrieve(claim:MonitorClaim) {
 const directory=await mkdtemp(join(tmpdir(),'alphatrace-monitor-'));
 try {
  const result=await new Promise<{code:number|null;timeout:boolean;failed:boolean}>(resolve=>{
   let timeout=false;
   const processChild=spawn('python3',['scripts/research/batch.py','--symbol',claim.symbol,'--output-dir',directory],{cwd:root,env:{PATH:process.env.PATH,HOME:process.env.HOME,LANG:'en_US.UTF-8',NODE_ENV:process.env.NODE_ENV},stdio:'ignore'});
   child=processChild;
   const timer=setTimeout(()=>{timeout=true;child?.kill('SIGKILL');},180_000);
   processChild.once('error',()=>{clearTimeout(timer);resolve({code:null,timeout:false,failed:true});});
   processChild.once('close',code=>{clearTimeout(timer);child=undefined;resolve({code,timeout,failed:false});});
  });
  if(stopped)return fail(claim.symbol,'interrupted','The local runner stopped during this source check. No fresh evidence was confirmed.');
  if(result.timeout)return fail(claim.symbol,'timeout','The source check exceeded its three-minute limit. No fresh evidence was confirmed.');
  if(result.failed)return fail(claim.symbol,'dependency','The Python source-check process could not start. Operator attention is required.');
  const path=join(directory,claim.symbol+'.json');
  if((await stat(path)).size>131072)throw Error('Oversized source bundle');
  const bundle=JSON.parse(await readFile(path,'utf8'));
  if(!bundle.check||!['success','failed'].includes(bundle.check.status)||result.code!==0&&bundle.check.status==='success')throw Error('Invalid source output');
  return bundle;
 } catch {return fail(claim.symbol,'validation','Source output could not be validated. Manual source review is required.');}
 finally {await rm(directory,{recursive:true,force:true});}
}
try {
 if(args.has('--enable'))for(const symbol of monitorSymbols)await controlMonitor(owner,{symbol,action:'resume'});
 console.log('Source monitor started. Infosys SEC discovery and configured TCS documents; drafts require human publication review.');
 let lastHeartbeat=0;
 do {
  try {
   if(Date.now()-lastHeartbeat>=60_000){await monitorHeartbeat(owner);lastHeartbeat=Date.now();}
   const claim=await claimMonitorRun(owner);
   if(claim){
    console.log(`${claim.symbol}: source check started (attempt ${claim.attempt}).`);
    const bundle=await retrieve(claim);
    try {
     const result=await finishMonitorRun(owner,claim,bundle,bundle.failureCode??null);
     console.log(`${claim.symbol}: ${result.accepted?bundle.check.status:'expired run ignored'}${result.created?'; unpublished draft created':''}.`);
    } catch {
     // Validation failures are recorded once. Database outages leave the lease
     // for durable recovery instead of falsely reporting successful ingestion.
     await finishMonitorRun(owner,claim,fail(claim.symbol,'validation','Source evidence failed validation. No draft was published.'),'validation');
     console.error(`${claim.symbol}: source evidence rejected; review required.`);
    }
    if(!stopped)continue;
   }
  } catch {
   console.error('Monitor storage is unavailable. Work remains queued or leased for recovery.');
   if(args.has('--once')){process.exitCode=1;break;}
  }
  if(args.has('--once')||stopped)break;
  await new Promise<void>(resolve=>{const timer=setTimeout(()=>{wake=undefined;resolve();},30_000);wake=()=>{clearTimeout(timer);wake=undefined;resolve();};});
 } while(!stopped);
} finally {await closeDatabase();}
