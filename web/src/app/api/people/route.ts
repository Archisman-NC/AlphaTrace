import {serviceBoundary} from '@/lib/service-errors';
import {requestIdentity} from '@/lib/auth';
import {getPeople,managePeople} from '@/lib/beta-people';
import {BetaInputError,BetaConflictError,BetaForbiddenError,BetaInvitationError} from '@/lib/beta-contract';
import {boundedJson,RequestBodyError} from '@/lib/http-input';
export const runtime='nodejs';export const dynamic='force-dynamic';
const json=(body:unknown,status=200)=>Response.json(body,{status,headers:{'Cache-Control':'private, no-store'}});
async function handler(request:Request){
 const owner=await requestIdentity(request);if(!owner)return json({error:'Sign in with an active beta account.'},401);
 try{return json({people:request.method==='POST'?await managePeople(owner,await boundedJson(request,4096)):await getPeople(owner)});}
 catch(e){if(e instanceof BetaForbiddenError)return json({error:e.message},403);if(e instanceof BetaConflictError)return json({error:e.message},409);if(e instanceof BetaInputError||e instanceof BetaInvitationError)return json({error:e.message},400);if(e instanceof RequestBodyError)return json({error:e.message},e.status);return json({error:'Access changes could not be confirmed. Refresh People before repeating a change.'},503);}
}
export const GET=serviceBoundary(handler);export const POST=serviceBoundary(handler);
