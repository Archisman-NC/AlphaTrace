import {serviceBoundary} from '@/lib/service-errors';
import {acceptInvitation,inspectInvitation} from '@/lib/beta-people';
import {BetaConflictError,BetaInvitationError} from '@/lib/beta-contract';
import {boundedJson,RequestBodyError} from '@/lib/http-input';
export const runtime='nodejs';export const dynamic='force-dynamic';
const json=(body:unknown,status=200)=>Response.json(body,{status,headers:{'Cache-Control':'private, no-store','Referrer-Policy':'no-referrer'}});
async function POSTHandler(request:Request){
 // Capabilities are carried in POST bodies, never URL query strings or logs.
 if(request.headers.get('origin')!==new URL(request.url).origin)return json({error:'Open your invitation on this website to continue.'},403);
 try{const input=await boundedJson(request,4096) as {action?:string;token?:unknown};
 if(input.action==='inspect')return json({invitation:await inspectInvitation(input.token)});
 if(input.action!=='accept')return json({error:'Choose an invitation action.'},400);
 return json(await acceptInvitation(input,request.headers.get('authorization')?.match(/^Bearer (.+)$/)?.[1]));
 }catch(e){if(e instanceof BetaInvitationError)return json({error:e.message},410);if(e instanceof BetaConflictError)return json({error:e.message},409);if(e instanceof RequestBodyError)return json({error:e.message},e.status);return json({error:'Activation could not be confirmed. Try signing in with the password you chose. If needed, reopen this invitation and retry with that same password.'},503);}
}
export const POST=serviceBoundary(POSTHandler);
