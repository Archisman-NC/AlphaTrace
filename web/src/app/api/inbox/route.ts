import { requestIdentity } from '@/lib/auth';
import { getInbox,getInboxDetail,syncInbox,reviewInbox } from '@/lib/inbox';
import { boundedJson } from '@/lib/http-input';
import { inboxError } from '@/lib/inbox-response';
export const dynamic='force-dynamic';
export const runtime='nodejs';
const json=(body:unknown,status=200)=>Response.json(body,{status,headers:{'Cache-Control':'no-store'}});
export async function GET(request:Request){
 const owner=await requestIdentity(request);if(!owner)return json({error:'Sign in with an invited account.'},401);
 const q=new URL(request.url).searchParams;
 try{return json(q.has('update')?{detail:await getInboxDetail(owner,q.get('update'),q.has('before')?Number(q.get('before')):undefined)}:{inbox:await getInbox(owner,q.get('filter'),q.get('cursor'))});}
 catch(e){const r=inboxError(e);return json(r,r.status);}
}
export async function POST(request:Request){
 const owner=await requestIdentity(request);if(!owner)return json({error:'Sign in with an invited account.'},401);
 try{const body=await boundedJson(request,16384) as {action?:string};return json(body?.action==='sync'?{sync:await syncInbox(owner)}:{detail:await reviewInbox(owner,body)});}
 catch(e){const r=inboxError(e);return json(r,r.status);}
}
