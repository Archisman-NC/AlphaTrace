import { schedules } from '@trigger.dev/sdk';
import { closeDatabase,AccountClosedError } from '../lib/db';
import { emailConfig,emailsAvailable } from '../lib/email-contract';
import { getEmailSettings,prepareEmailDigest,pendingEmailDigests,deliverEmailDigest } from '../lib/emails';
import { verifiedEmailRecipient } from '../lib/email-recipient';
import { invitedUser } from '../lib/invites';
import { sendResend } from '../lib/resend';

export const personalUpdateEmails=schedules.task({
 id:'alphatrace-personal-update-emails',queue:{concurrencyLimit:1},
 cron:{pattern:'30 * * * *',timezone:'UTC',environments:['PRODUCTION'],window:'60m'},
 retry:{maxAttempts:1},maxDuration:300,
 run:async()=>{
  if(!emailsAvailable(process.env))return {state:'disabled'};
  const config=emailConfig(process.env),users=[...new Set((process.env.ALPHATRACE_INVITED_USER_IDS??'').split(',').map(s=>s.trim()))];
  if(!users.length||users.some(id=>!invitedUser(id,process.env.ALPHATRACE_INVITED_USER_IDS)))throw Error('Email pilot allowlist is not configured.');
  const hour=Number(new Intl.DateTimeFormat('en-GB',{timeZone:'Asia/Kolkata',hour:'2-digit',hourCycle:'h23'}).format(new Date()));
  let accepted=0,failed=0;
  try{
   for(const owner of users){
    try{
     const settings=await getEmailSettings(owner);if(!settings.preferences.enabled||settings.preferences.suppressed)continue;
     const recipient=await verifiedEmailRecipient(owner);
     for(const id of await pendingEmailDigests(owner)){const state=await deliverEmailDigest(owner,id,recipient,(payload,key)=>sendResend(payload,config.apiKey,key));if(state==='accepted')accepted++;if(state==='failed'||state==='retry')failed++;}
     if(hour>=9&&recipient){
      const id=await prepareEmailDigest(owner,recipient,config);
      if(id){const verified=await verifiedEmailRecipient(owner);const state=await deliverEmailDigest(owner,id,verified,(payload,key)=>sendResend(payload,config.apiKey,key));if(state==='accepted')accepted++;if(state==='failed'||state==='retry')failed++;}
     }
    }catch(e){if(!(e instanceof AccountClosedError))failed++;}
   }
   // Never log recipients, unsubscribe capabilities, payloads or credentials.
   if(failed)throw Error(`Email pass needs attention: ${accepted} accepted; ${failed} failed or pending retry. Check delivery records.`);
   return {state:'complete',accepted};
  }finally{await closeDatabase();}
 }
});
