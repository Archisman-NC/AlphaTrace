import type { EmailPayload } from './email-contract';
export type SendOutcome={state:'accepted';providerId:string}|{state:'retry'|'failed';message:string};
/** Fixed provider origin, redacted failures, bounded request, durable caller-owned key. */
export async function sendResend(payload:EmailPayload,key:string,id:string,request:typeof fetch=fetch):Promise<SendOutcome>{
 try{
  const response=await request('https://api.resend.com/emails',{method:'POST',redirect:'error',signal:AbortSignal.timeout(12_000),headers:{Authorization:`Bearer ${key}`,'Content-Type':'application/json','Idempotency-Key':`alphatrace-digest/${id}`},body:JSON.stringify(payload)});
  if(response.ok){const data=await response.json();if(typeof data.id==='string'&&data.id.length<200)return {state:'accepted',providerId:data.id};return {state:'retry',message:'Provider acceptance could not be confirmed.'};}
  if(response.status===429||response.status>=500)return {state:'retry',message:'Email provider temporarily unavailable; a retry is scheduled.'};
  return {state:'failed',message:'Email provider rejected the request. Check sender configuration and provider logs.'};
 }catch{return {state:'retry',message:'Provider acceptance could not be confirmed; retrying with the same delivery identifier.'};}
}
