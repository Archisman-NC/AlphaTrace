import { requestIdentity } from '@/lib/auth';
import { getEmailSettings,saveEmailPreferences } from '@/lib/emails';
import { boundedJson,RequestBodyError } from '@/lib/http-input';
import { PortfolioConflictError,PortfolioInputError } from '@/lib/portfolio-input';
export const runtime='nodejs';
export const dynamic='force-dynamic';
const json=(body:unknown,status=200)=>Response.json(body,{status,headers:{'Cache-Control':'no-store'}});
export async function GET(request:Request){
 const owner=await requestIdentity(request);if(!owner)return json({error:'Sign in to manage email preferences.'},401);
 try{return json({email:await getEmailSettings(owner)});}catch{return json({error:'Email settings could not be loaded. Please retry.'},503);}
}
export async function POST(request:Request){
 const owner=await requestIdentity(request);if(!owner)return json({error:'Sign in to manage email preferences.'},401);
 try{return json({email:await saveEmailPreferences(owner,await boundedJson(request,2048))});}catch(e){
  if(e instanceof RequestBodyError)return json({error:e.message},e.status);
  if(e instanceof PortfolioInputError)return json({error:e.message},400);
  if(e instanceof PortfolioConflictError)return json({error:e.message},409);
  return json({error:'The save could not be confirmed. Reload email settings before retrying.'},503);
 }
}
