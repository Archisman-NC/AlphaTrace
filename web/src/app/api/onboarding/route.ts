import {serviceBoundary} from '@/lib/service-errors';
import { requestIdentity } from '@/lib/auth';
import { getOnboarding,saveOnboarding } from '@/lib/onboarding';
import { PortfolioConflictError,PortfolioInputError } from '@/lib/portfolio-input';
import { boundedJson,RequestBodyError } from '@/lib/http-input';
export const runtime='nodejs';
export const dynamic='force-dynamic';
const json=(body:unknown,status=200)=>Response.json(body,{status,headers:{'Cache-Control':'no-store'}});
async function GETHandler(request:Request){
 const id=await requestIdentity(request);if(!id)return json({error:'Sign in to set up your workspace.'},401);
 try{return json({onboarding:await getOnboarding(id)});}catch{return json({error:'Setup could not be loaded. Please retry.'},503);}
}
async function POSTHandler(request:Request){
 const id=await requestIdentity(request);if(!id)return json({error:'Sign in to set up your workspace.'},401);
 try{return json(await saveOnboarding(id,await boundedJson(request)));}catch(e){
  if(e instanceof RequestBodyError)return json({error:e.message},e.status);
  if(e instanceof PortfolioInputError)return json({error:e.message},400);
  if(e instanceof PortfolioConflictError)return json({error:e.message},409);
  return json({error:'The save could not be confirmed. Reload setup before retrying.'},503);
 }
}

export const GET=serviceBoundary(GETHandler);
export const POST=serviceBoundary(POSTHandler);
