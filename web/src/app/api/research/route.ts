import { requestIdentity } from '@/lib/auth';
import { getResearch,reviewResearch } from '@/lib/research';
import { boundedJson } from '@/lib/http-input';
import { researchJson as json,researchError } from '@/lib/research-response';
export const runtime='nodejs';
export const dynamic='force-dynamic';
export async function GET(request:Request){
 const id=await requestIdentity(request);if(!id)return json({error:'Sign in with an invited account.'},401);
 const url=new URL(request.url);
 try{return json(await getResearch(id,url.searchParams.get('symbol')||undefined,url.searchParams.get('admin')==='1',url.searchParams.get('update')||undefined));}
 catch(e){const result=researchError(e);return json(result,result.status);}
}
export async function POST(request:Request){
 const id=await requestIdentity(request);if(!id)return json({error:'Sign in with an invited account.'},401);
 try{return json(await reviewResearch(id,await boundedJson(request)));}
 catch(e){const result=researchError(e);return json(result,result.status);}
}
