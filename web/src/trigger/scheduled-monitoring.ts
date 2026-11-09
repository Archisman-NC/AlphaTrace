import { logger,schedules } from '@trigger.dev/sdk';
import { join } from 'node:path';
import { UUID } from '../lib/access';
import { closeDatabase } from '../lib/db';
import { claimMonitorRun,finishMonitorRun,monitorHeartbeat } from '../lib/monitor';
import { claimProviderRun,finishProviderRun } from '../lib/provider-imports';
import { fetchIndianStats,IndianApiError } from '../../scripts/lib/indianapi';
import { checkSourceTools,retrieveSource,sourceFailure } from '../../scripts/lib/retrieve-source';

/** A bounded pass. Database leases, due dates and the shared quota govern all work. */
export const scheduledMonitoring=schedules.task({
 id:'alphatrace-scheduled-monitoring',
 // UTC :30 is the top of each IST hour. Avoid server timezone-alias differences.
 cron:{pattern:'30 * * * *',timezone:'UTC',environments:['PRODUCTION'],window:'60m'},
 queue:{concurrencyLimit:1},
 maxDuration:600,
 retry:{maxAttempts:1},
 run:async(_payload,{ctx})=>{
  const owner=process.env.ALPHATRACE_INGEST_USER_ID,key=process.env.INDIANAPI_API_KEY;
  if(!owner||!UUID.test(owner)||!key)throw Error('Cloud monitoring configuration is incomplete.');
  const results:{symbol:string;kind:string;state:string;detectionResult?:string;reviewId?:string|null}[]=[];
  let infrastructureFailure=false;
  try {
   try {
    const directory=join(process.cwd(),'research');
    await checkSourceTools(directory);
    await monitorHeartbeat(owner,'cloud');
    for(let i=0;i<2;i++){
     const claim=await claimMonitorRun(owner);if(!claim)break;
     const bundle=await retrieveSource(claim.symbol,directory);
     let result;
     try{result=await finishMonitorRun(owner,claim,bundle,bundle.failureCode??null);}
     catch{result=await finishMonitorRun(owner,claim,sourceFailure(claim.symbol,'validation','Source evidence failed validation. No draft was published.'),'validation');bundle.check.status='failed';}
     results.push({kind:'official-source',symbol:claim.symbol,state:result.accepted?bundle.check.status:'lease-expired'});
     await monitorHeartbeat(owner,'cloud');
    }
   }catch{infrastructureFailure=true;logger.error('Source worker unavailable. Check dependencies and database connectivity; leases recover on the next pass.');}
   try {
    for(let i=0;i<5;i++){
     const claim=await claimProviderRun(owner,'cloud',ctx.run.id);if(!claim)break;
     let snapshot=null,code:string|undefined;
     try{snapshot=await fetchIndianStats(claim.symbol,key);}catch(e){code=e instanceof IndianApiError?e.code:'validation';}
     const result=await finishProviderRun(owner,claim,snapshot,code);
     results.push({kind:'private-import',symbol:claim.symbol,state:result.accepted?(snapshot?'success':'failed'):'lease-expired',detectionResult:result.detectionResult,reviewId:result.reviewId});
    }
   }catch{infrastructureFailure=true;logger.error('Import storage unavailable. Existing quota and leases remain in force.');}
   logger.info('Monitoring pass finished.',{results});
   if(infrastructureFailure)throw Error('Monitoring infrastructure needs attention. See the worker status; credentials and responses are not logged.');
   return {checkedAt:new Date().toISOString(),results,publication:'human-review-required'};
  }finally{await closeDatabase();}
 }
});
