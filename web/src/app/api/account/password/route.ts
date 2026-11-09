import {serviceBoundary} from '@/lib/service-errors';
import {requestIdentity} from '@/lib/auth';
import {passwordInput,AccountInputError} from '@/lib/account-contract';
import {reauthenticate} from '@/lib/account-auth';
import {boundedJson,RequestBodyError} from '@/lib/http-input';
export const runtime='nodejs';export const dynamic='force-dynamic';
const json=(body:unknown,status=200)=>Response.json(body,{status,headers:{'Cache-Control':'private, no-store'}});
async function POSTHandler(request:Request){
 const owner=await requestIdentity(request);if(!owner)return json({error:'Your session expired. Sign in again.'},401);
 try{
  const input=passwordInput(await boundedJson(request,4096));
  const client=await reauthenticate(owner,request.headers.get('authorization')!.slice(7),input.currentPassword);
  try{const {error}=await client.auth.updateUser({password:input.password});if(error)return json({error:'The password change was not confirmed. Your provider may require additional verification; try the password recovery flow.'},400);await client.auth.signOut({scope:'global'});return json({changed:true});}
  finally{await client.auth.signOut({scope:'local'}).catch(()=>{});}
 }catch(e){if(e instanceof AccountInputError)return json({error:e.message},400);if(e instanceof RequestBodyError)return json({error:e.message},e.status);return json({error:'The password change could not be confirmed. Try signing in with your new password before repeating it.'},503);}
}

export const POST=serviceBoundary(POSTHandler);
