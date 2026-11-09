import {task,schedules} from '@trigger.dev/sdk';
import {join} from 'node:path';
import {processDocumentJob} from '../lib/document-processor';
import {asUser,closeDatabase} from '../lib/db';
import {documentStorage} from '../lib/document-storage';
import {documentObjectPath,type DocumentFile} from '../lib/document-job-contract';
import {UUID} from '../lib/access';
export const processEvidenceDocument=task({
 id:'alphatrace-process-document',queue:{concurrencyLimit:1},retry:{maxAttempts:1},maxDuration:300,
 run:async(payload:{jobId:string;ownerId:string},{ctx})=>{
  if(!UUID.test(payload.jobId)||!UUID.test(payload.ownerId))throw Error('Invalid document task identity.');
  process.env.ALPHATRACE_RESEARCH_DIR=join(process.cwd(),'research');
  try{return await processDocumentJob(payload.ownerId,payload.jobId,ctx.run.id);}finally{await closeDatabase();}
 }
});
// Signed upload URLs last two hours. Cleanup waits four hours from creation,
// beyond the final permitted ticket issuance, to prevent late uploads after deletion.
export const cleanEvidenceUploads=schedules.task({
 id:'alphatrace-clean-document-uploads',queue:{concurrencyLimit:1},cron:{pattern:'15 * * * *',timezone:'UTC',environments:['PRODUCTION'],window:'60m'},retry:{maxAttempts:1},maxDuration:120,
 run:async()=>{
  const owner=process.env.ALPHATRACE_INGEST_USER_ID;if(!owner||!UUID.test(owner))throw Error('Document cleanup operator is not configured.');
  try{
   const rows=await asUser(owner,async c=>(await c.query("SELECT id,owner_id,files FROM alphatrace.document_jobs WHERE cleaned_at IS NULL AND cleanup_after<now() ORDER BY cleanup_after LIMIT 30")).rows);
   if(!rows.length)return {cleaned:0};
   const storage=documentStorage();let cleaned=0;
   for(const row of rows){
    const paths=(row.files as DocumentFile[]).map(f=>documentObjectPath(row.owner_id,row.id,f.field));
    if(paths.length){const {error}=await storage.remove(paths);if(error)throw Error('Private upload cleanup failed. Retry on the next scheduled pass.');}
    await asUser(owner,c=>c.query("UPDATE alphatrace.document_jobs SET cleaned_at=now(),state=CASE WHEN state IN ('uploading','queued','running') THEN 'expired' ELSE state END,message=CASE WHEN state IN ('uploading','queued','running') THEN 'Operation expired; temporary uploads removed.' ELSE message END WHERE id=$1",[row.id]));cleaned++;
   }
   return {cleaned};
  }finally{await closeDatabase();}
 }
});
