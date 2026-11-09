import {serviceBoundary} from '@/lib/service-errors';
import { requestIdentity } from '@/lib/auth';
import { localAccessAllowed,UUID } from '@/lib/access';
import { requireManualOperator,boundedManualForm,importTcsFiles } from '@/lib/manual-tcs';
import { researchJson as json,researchError } from '@/lib/research-response';
import {documentJobsEnabled} from '@/lib/document-job-contract';
export const runtime='nodejs';
export const dynamic='force-dynamic';
async function POSTHandler(request:Request){
 let owner=await requestIdentity(request);
 // Same-origin development bypass follows the existing server-action boundary.
 if(!owner&&localAccessAllowed(process.env,request.headers.get('host'))){
  try{const id=process.env.ALPHATRACE_DEV_USER_ID;if(id&&UUID.test(id)&&new URL(request.headers.get('origin')??'').host===request.headers.get('host'))owner=id;}catch{}
 }
 if(!owner)return json({error:'Sign in with an invited operator account.'},401);
 try{await requireManualOperator(owner);if(documentJobsEnabled())return json({error:'Use the private upload and background document workflow.'},409);return json({import:await importTcsFiles(owner,await boundedManualForm(request))});}
 catch(e){const result=researchError(e);return json(result,result.status);}
}

export const POST=serviceBoundary(POSTHandler);
