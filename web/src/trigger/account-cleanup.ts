import {schedules} from '@trigger.dev/sdk';
import {pendingClosures,finishClosure} from '../lib/accounts';
import {removeAccountLogin} from '../lib/account-auth';
import {closeDatabase} from '../lib/db';
export const accountCleanup=schedules.task({id:'alphatrace-account-cleanup',queue:{concurrencyLimit:1},cron:{pattern:'15 * * * *',timezone:'UTC',environments:['PRODUCTION'],window:'60m'},maxDuration:300,retry:{maxAttempts:1},run:async()=>{
 const worker=process.env.ALPHATRACE_INGEST_USER_ID;if(!worker)throw Error('Account cleanup identity is missing');
 let completed=0,failed=0;
 try{for(const owner of await pendingClosures(worker)){try{if(await finishClosure(owner,removeAccountLogin))completed++;}catch{failed++;}}if(failed)throw Error(`Account cleanup requires attention: ${failed} pending retries.`);return {completed};}finally{await closeDatabase();}
}});
