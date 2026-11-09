import {tasks} from '@trigger.dev/sdk';
import {requestIdentity} from '@/lib/auth';
import {localAccessAllowed,UUID} from '@/lib/access';
import {boundedJson} from '@/lib/http-input';
import {createDocumentJob,getDocumentJob,publicDocumentJob,queueDocumentJob} from '@/lib/document-jobs';
import {documentJobsEnabled} from '@/lib/document-job-contract';
import {documentStorage,uploadTickets} from '@/lib/document-storage';
import {requireResearchEditor} from '@/lib/research-edit';
import {researchJson as json,researchError} from '@/lib/research-response';
import {ResearchInputError,ResearchConflictError} from '@/lib/research-contract';
export const runtime='nodejs';export const dynamic='force-dynamic';export const maxDuration=60;
async function identity(request:Request){
 const owner=await requestIdentity(request);if(owner)return owner;
 if(localAccessAllowed(process.env,request.headers.get('host'))){const id=process.env.ALPHATRACE_DEV_USER_ID;try{if(id&&UUID.test(id)&&new URL(request.headers.get('origin')??'').host===request.headers.get('host'))return id;}catch{}}
 return null;
}
export async function POST(request:Request){
 const owner=await identity(request);if(!owner)return json({error:'Sign in with an invited operator account.'},401);
 try{
  await requireResearchEditor(owner);
  const body=await boundedJson(request,20_000) as {action?:string;id?:string;command?:unknown};
  if(body?.action==='configuration')return json({enabled:documentJobsEnabled()});
  if(!documentJobsEnabled())throw new ResearchInputError('Cloud document processing is not enabled.');
  if(body.action==='status')return json({job:publicDocumentJob(await getDocumentJob(owner,body.id??''))});
  // Never dispatch from a preview to the production task environment.
  if(!process.env.TRIGGER_SECRET_KEY||process.env.VERCEL_ENV==='preview'||(process.env.VERCEL_ENV==='production'&&!process.env.TRIGGER_SECRET_KEY.startsWith('tr_prod_')))throw new ResearchInputError('Document worker is not configured for this deployment.');
  documentStorage();
  if(body.action==='create'){
   const job=await createDocumentJob(owner,body.command);
   if(job.state==='uploading'&&new Date(job.expires_at).getTime()<=Date.now())throw new ResearchConflictError('This upload window expired. Start a new operation.');
   return json({job:publicDocumentJob(job),uploads:job.state==='uploading'?await uploadTickets(owner,job.id,job.files):[]});
  }
  if(body.action==='submit'){
   const job=await getDocumentJob(owner,body.id??'');
   if(['succeeded','failed','expired','running'].includes(job.state))return json({job:publicDocumentJob(job)});
   await queueDocumentJob(owner,job.id);
   try{await tasks.trigger('alphatrace-process-document',{jobId:job.id,ownerId:owner},{idempotencyKey:`evidence-job:${job.id}`,idempotencyKeyTTL:'24h',ttl:'1h'});}
   catch{throw new ResearchInputError('Could not confirm worker dispatch. Resume this operation to retry safely.');}
   return json({job:publicDocumentJob(await getDocumentJob(owner,job.id))});
  }
  throw new ResearchInputError('Unknown document operation.');
 }catch(e){const r=researchError(e);return json(r,r.status);}
}
