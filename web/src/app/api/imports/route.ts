import {serviceBoundary} from '@/lib/service-errors';
import { requestIdentity } from '@/lib/auth';
import { getProviderImports,providerCommand } from '@/lib/provider-imports';
import { boundedJson } from '@/lib/http-input';
import { researchJson as json,researchError } from '@/lib/research-response';
export const runtime='nodejs';
export const dynamic='force-dynamic';
async function GETHandler(request:Request){
 const owner=await requestIdentity(request);if(!owner)return json({error:'Sign in with an invited operator account.'},401);
 try{return json({imports:await getProviderImports(owner)});}catch(e){const r=researchError(e);return json(r,r.status);}
}
async function POSTHandler(request:Request){
 const owner=await requestIdentity(request);if(!owner)return json({error:'Sign in with an invited operator account.'},401);
 try{const result=await providerCommand(owner,await boundedJson(request,2048));return json({...result,imports:await getProviderImports(owner)});}
 catch(e){const r=researchError(e);return json(r,r.status);}
}

export const GET=serviceBoundary(GETHandler);
export const POST=serviceBoundary(POSTHandler);
