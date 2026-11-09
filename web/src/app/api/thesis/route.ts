import {serviceBoundary} from '@/lib/service-errors';
import { requestIdentity } from '@/lib/auth';
import { getThesis,saveThesis } from '@/lib/theses';
import { boundedJson } from '@/lib/http-input';
import { thesisError } from '@/lib/thesis-response';
export const runtime='nodejs';
export const dynamic='force-dynamic';
const json=(body:unknown,status=200)=>Response.json(body,{status,headers:{'Cache-Control':'no-store'}});
async function GETHandler(request:Request){
 const id=await requestIdentity(request);if(!id)return json({error:'Sign in with an invited account.'},401);
 const query=new URL(request.url).searchParams;
 try{return json({thesis:await getThesis(id,query.get('symbol'),query.has('before')?Number(query.get('before')):undefined)});}
 catch(e){const result=thesisError(e);return json(result,result.status);}
}
async function POSTHandler(request:Request){
 const id=await requestIdentity(request);if(!id)return json({error:'Sign in with an invited account.'},401);
 try{return json({thesis:await saveThesis(id,await boundedJson(request,131072))});}
 catch(e){const result=thesisError(e);return json(result,result.status);}
}

export const GET=serviceBoundary(GETHandler);
export const POST=serviceBoundary(POSTHandler);
