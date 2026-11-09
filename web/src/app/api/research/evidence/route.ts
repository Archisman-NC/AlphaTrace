import { requestIdentity } from '@/lib/auth';
import { localAccessAllowed,UUID } from '@/lib/access';
import { boundedJson } from '@/lib/http-input';
import { boundedManualForm } from '@/lib/manual-tcs';
import { requireResearchEditor,reviseResearch } from '@/lib/research-edit';
import { attachResearchSource,checkResearchSources } from '@/lib/research-source';
import { researchJson as json,researchError } from '@/lib/research-response';
import { ResearchInputError } from '@/lib/research-contract';
import {documentJobsEnabled} from '@/lib/document-job-contract';
export const runtime='nodejs';
export const dynamic='force-dynamic';
export async function POST(request:Request){
 let owner=await requestIdentity(request);
 if(!owner&&localAccessAllowed(process.env,request.headers.get('host'))){try{const id=process.env.ALPHATRACE_DEV_USER_ID;if(id&&UUID.test(id)&&new URL(request.headers.get('origin')??'').host===request.headers.get('host'))owner=id;}catch{}}
 if(!owner)return json({error:'Sign in with an invited operator account.'},401);
 try{
  await requireResearchEditor(owner);
  if(request.headers.get('content-type')?.startsWith('multipart/form-data;')){
   if(documentJobsEnabled())return json({error:'Upload documents directly to private storage using the document workflow.'},409);
   const form=await boundedManualForm(request),file=form.get('file');
   if(!(file instanceof File)||!file.size)throw new ResearchInputError('Select an original PDF or HTML file.');
   return json(await attachResearchSource(owner,Object.fromEntries([...form.entries()].filter(([key])=>key!=='file')),file));
  }
  const body=await boundedJson(request,131072) as {action?:string;id?:string};
  if(body?.action==='revise')return json(await reviseResearch(owner,body));
  if(documentJobsEnabled())return json({error:'Use the background document workflow for attachments and source checks.'},409);
  if(body?.action==='attach')return json(await attachResearchSource(owner,body));
  if(body?.action==='check')return json(await checkResearchSources(owner,body.id));
  throw new ResearchInputError('Unknown evidence action.');
 }catch(e){const result=researchError(e);return json(result,result.status);}
}
