import { requestIdentity } from '@/lib/auth';
import { getIssues,saveIssue } from '@/lib/issues';
import { boundedJson } from '@/lib/http-input';
import { issueJson as json,issueError } from '@/lib/issue-response';
export const runtime='nodejs';
export const dynamic='force-dynamic';
export async function GET(request:Request){
 const owner=await requestIdentity(request);if(!owner)return json({error:'Sign in with an invited account to view reports.'},401);
 const params=new URL(request.url).searchParams;
 try{return json({issues:await getIssues(owner,{scope:params.get('scope')??undefined,status:params.get('status')??undefined,cursor:params.get('cursor')??undefined,report:params.get('report')??undefined})});}catch(e){const r=issueError(e);return json(r,r.status);}
}
export async function POST(request:Request){
 const owner=await requestIdentity(request);if(!owner)return json({error:'Sign in with an invited account to report an issue.'},401);
 try{return json(await saveIssue(owner,await boundedJson(request,8192)));}catch(e){const r=issueError(e);return json(r,r.status);}
}
