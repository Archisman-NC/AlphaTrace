import {serviceBoundary} from '@/lib/service-errors';
import { requestIdentity } from '@/lib/auth';
import { getProviderReviews,reviewProviderChange } from '@/lib/provider-reviews';
import { prepareProviderReview } from '@/lib/provider-handoff';
import { boundedJson } from '@/lib/http-input';
import { researchJson as json,researchError } from '@/lib/research-response';
export const runtime='nodejs';
export const dynamic='force-dynamic';
async function GETHandler(request:Request){
 const owner=await requestIdentity(request);if(!owner)return json({error:'Sign in with an invited operator account.'},401);
 try{const url=new URL(request.url);return json({reviews:await getProviderReviews(owner,{status:url.searchParams.get('status')??undefined,before:url.searchParams.get('before')??undefined,id:url.searchParams.get('id')??undefined})});}
 catch(e){const r=researchError(e);return json(r,r.status);}
}
async function POSTHandler(request:Request){
 const owner=await requestIdentity(request);if(!owner)return json({error:'Sign in with an invited operator account.'},401);
 try{const body=await boundedJson(request,4096) as {action?:string};return json(await (body?.action==='prepare'?prepareProviderReview(owner,body):reviewProviderChange(owner,body)));}
 catch(e){const r=researchError(e);return json(r,r.status);}
}

export const GET=serviceBoundary(GETHandler);
export const POST=serviceBoundary(POSTHandler);
