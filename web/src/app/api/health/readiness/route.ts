import {requestIdentity} from '@/lib/auth';
import {serviceBoundary} from '@/lib/service-errors';
import {getHealth} from '@/lib/health';
import {ResearchForbiddenError} from '@/lib/research-contract';
export const runtime='nodejs';export const dynamic='force-dynamic';
const json=(body:unknown,status=200)=>Response.json(body,{status,headers:{'Cache-Control':'private, no-store'}});
export const GET=serviceBoundary(async(request:Request)=>{
 const owner=await requestIdentity(request);if(!owner)return json({error:'Sign in with an invited operator account.'},401);
 try{return json({health:await getHealth(owner)});}catch(e){if(e instanceof ResearchForbiddenError)return json({error:e.message},403);return json({error:'Operational status could not be read. Database availability and worker state are unconfirmed.'},503);}
});
