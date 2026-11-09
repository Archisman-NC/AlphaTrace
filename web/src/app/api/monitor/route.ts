import {serviceBoundary} from '@/lib/service-errors';
import { requestIdentity } from '@/lib/auth';
import { getMonitor,controlMonitor } from '@/lib/monitor';
import { boundedJson } from '@/lib/http-input';
import { researchJson as json,researchError } from '@/lib/research-response';
export const runtime='nodejs';
export const dynamic='force-dynamic';
async function GETHandler(request:Request){
 const owner=await requestIdentity(request);if(!owner)return json({error:'Sign in with an invited account.'},401);
 try{return json({monitor:await getMonitor(owner)});}catch(e){const result=researchError(e);return json(result,result.status);}
}
async function POSTHandler(request:Request){
 const owner=await requestIdentity(request);if(!owner)return json({error:'Sign in with an invited account.'},401);
 try{await controlMonitor(owner,await boundedJson(request,2048));return json({monitor:await getMonitor(owner)});}catch(e){const result=researchError(e);return json(result,result.status);}
}

export const GET=serviceBoundary(GETHandler);
export const POST=serviceBoundary(POSTHandler);
