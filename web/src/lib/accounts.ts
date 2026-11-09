import {asUser} from './db';
import {AccountConflictError,type AccountState} from './account-contract';
export async function accountState(owner:string):Promise<AccountState>{
 return asUser(owner,async c=>{const closure=(await c.query('SELECT completed_at FROM alphatrace.account_closures WHERE owner_id=$1',[owner])).rows[0];const blocked=(await c.query('SELECT alphatrace.account_deletion_blocked() AS blocked')).rows[0].blocked;return {closed:!!closure,completed:!!closure?.completed_at,blocked:blocked||owner.toLowerCase()===process.env.ALPHATRACE_INGEST_USER_ID?.toLowerCase()};},{accountControl:true});
}
export async function requestClosure(owner:string){
 return asUser(owner,async c=>{
  if(owner.toLowerCase()===process.env.ALPHATRACE_INGEST_USER_ID?.toLowerCase()||(await c.query('SELECT alphatrace.account_deletion_blocked() AS blocked')).rows[0].blocked)throw new AccountConflictError('Transfer operator responsibilities and finish document cleanup before deleting this account.');
  await c.query('SELECT alphatrace.request_account_closure()');
 },{accountControl:true});
}
export async function finishClosure(owner:string,removeLogin:(id:string)=>Promise<void>){
 const claimed=await asUser(owner,async c=>(await c.query('SELECT alphatrace.claim_account_closure() AS claimed')).rows[0].claimed,{accountControl:true});
 if(!claimed)return (await accountState(owner)).completed;
 // A failure leaves a durable, blocked account that the hourly worker retries.
 await removeLogin(owner);
 await asUser(owner,c=>c.query('SELECT alphatrace.complete_account_closure()'),{accountControl:true});
 return true;
}
export async function pendingClosures(worker:string){return asUser(worker,async c=>(await c.query('SELECT owner_id FROM alphatrace.pending_account_closures()')).rows.map(r=>r.owner_id as string),{accountControl:true});}
