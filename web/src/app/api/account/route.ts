import {serviceBoundary} from '@/lib/service-errors';
import {requestIdentity} from '@/lib/auth';
import {accountState,requestClosure,finishClosure} from '@/lib/accounts';
import {deletionInput,AccountInputError,AccountConflictError} from '@/lib/account-contract';
import {reauthenticate,removeAccountLogin} from '@/lib/account-auth';
import {boundedJson,RequestBodyError} from '@/lib/http-input';
export const runtime='nodejs';export const dynamic='force-dynamic';
const json=(body:unknown,status=200)=>Response.json(body,{status,headers:{'Cache-Control':'private, no-store'}});
async function GETHandler(request:Request){const owner=await requestIdentity(request,{allowClosed:true});if(!owner)return json({error:'Your session expired. Sign in again.'},401);try{return json({account:await accountState(owner)});}catch{return json({error:'Account settings could not be loaded.'},503);}}
async function DELETEHandler(request:Request){
 const owner=await requestIdentity(request,{allowClosed:true});if(!owner)return json({error:'Your session expired. Sign in again.'},401);
 try{
  const {password}=deletionInput(await boundedJson(request,4096));
  const state=await accountState(owner);if(state.closed)return json({closed:true,completed:state.completed});
  if(state.blocked)throw new AccountConflictError('Transfer operator responsibilities and finish document cleanup before deleting this account.');
  if(!process.env.SUPABASE_SECRET_KEY)throw Error('Deletion unavailable');
  const client=await reauthenticate(owner,request.headers.get('authorization')!.slice(7),password);
  try{await requestClosure(owner);}finally{await client.auth.signOut({scope:'local'}).catch(()=>{});}
  let completed=false;try{completed=await finishClosure(owner,removeAccountLogin);}catch{/* durable retry; never expose provider credentials */}
  return json({closed:true,completed},completed?200:202);
 }catch(e){if(e instanceof AccountInputError)return json({error:e.message},400);if(e instanceof AccountConflictError)return json({error:e.message},409);if(e instanceof RequestBodyError)return json({error:e.message},e.status);return json({error:'Deletion could not be confirmed. Check account status before trying again.'},503);}
}

export const GET=serviceBoundary(GETHandler);
export const DELETE=serviceBoundary(DELETEHandler);
