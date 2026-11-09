if(process.env.ALPHATRACE_CLOUD_MONITORING==='true')throw Error('Cloud monitoring is enabled. Use the cloud task; local workers are disabled to avoid duplicate scheduling.');
import { readFile } from 'node:fs/promises';
import { indianApiSymbols,fetchIndianStats,IndianApiError } from './lib/indianapi';
import { stageProviderSnapshot,claimProviderRun,finishProviderRun } from '../src/lib/provider-imports';
import { closeDatabase } from '../src/lib/db';
import { UUID } from '../src/lib/access';

const args=process.argv.slice(2),mode=args[0];
if(args.length!==1||!['--seed-cache','--once','--watch'].includes(mode))throw Error('Choose --seed-cache, --once or --watch.');
const owner=process.env.ALPHATRACE_INGEST_USER_ID;
if(!owner||!UUID.test(owner))throw Error('Configure an enrolled ALPHATRACE_INGEST_USER_ID.');
let stopped=false,wake:(()=>void)|undefined;
process.on('SIGTERM',()=>{stopped=true;wake?.();});process.on('SIGINT',()=>{stopped=true;wake?.();});
try{
 if(mode==='--seed-cache'){
  for(const symbol of indianApiSymbols){
   const snapshot=JSON.parse(await readFile(new URL(`../.local/indianapi/${symbol}.json`,import.meta.url),'utf8'));
   await stageProviderSnapshot(owner,snapshot);console.log(`${symbol}: cached evaluation staged; no API request.`);
  }
 }else{
  const key=process.env.INDIANAPI_API_KEY?.trim();
  if(!key)throw Error('Configure the private INDIANAPI_API_KEY before starting refreshes.');
  console.log('IndianAPI worker: private staging only; five attempts per rolling 24 hours across workers.');
  do{
   try{
    const claim=await claimProviderRun(owner);
    if(claim){
     try{const snapshot=await fetchIndianStats(claim.symbol,key);await finishProviderRun(owner,claim,snapshot);console.log(`${claim.symbol}: response staged.`);}
     catch(e){await finishProviderRun(owner,claim,null,e instanceof IndianApiError?e.code:'validation');console.error(`${claim.symbol}: retrieval failed; no automatic retry.`);}
     if(!stopped)continue;
    }
   }catch{console.error('Private import storage unavailable. Queued work and leases will recover on reconnection.');if(mode==='--once'){process.exitCode=1;break;}}
   if(mode==='--once'||stopped)break;
   await new Promise<void>(resolve=>{const t=setTimeout(()=>{wake=undefined;resolve();},30_000);wake=()=>{clearTimeout(t);wake=undefined;resolve();};});
  }while(!stopped);
 }
}catch{console.error('IndianAPI worker stopped. Check private configuration and cached snapshots; credentials and raw responses are not logged.');process.exitCode=1;}
finally{await closeDatabase();}
