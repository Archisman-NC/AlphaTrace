import {createHash} from 'node:crypto';
import {claimDocumentJob,finishDocumentJob} from './document-jobs';
import {readDocumentObject} from './document-storage';
import {attachResearchSource,checkResearchSources} from './research-source';
import {importTcsFiles} from './manual-tcs';
import {ResearchInputError} from './research-contract';
export async function processDocumentJob(owner:string,id:string,runId:string,download=readDocumentObject){
 const job=await claimDocumentJob(owner,id,runId);if(!job)return {claimed:false};
 try{
  const files=new Map<string,File>();
  for(const file of job.files){
   const bytes=await download(owner,id,file);
   if(bytes.length!==file.size||createHash('sha256').update(bytes).digest('hex')!==file.sha256)throw new ResearchInputError('Document bytes do not match the selected original. Upload the original again.');
   files.set(file.field,new File([new Uint8Array(bytes)],file.name));
  }
  let result:unknown;
  if(job.kind==='attach')result=await attachResearchSource(owner,job.input,files.get('file'));
  else if(job.kind==='check')result=await checkResearchSources(owner,job.input.id);
  else {const form=new FormData();for(const [key,value] of Object.entries(job.input))form.set(key,value);for(const [key,file] of files)form.set(key,file);result={import:await importTcsFiles(owner,form)};}
  return {claimed:true,succeeded:await finishDocumentJob(owner,id,runId,result)};
 }catch(e){
  const message=e instanceof ResearchInputError?e.message:'Document processing failed. No approval or publication was performed. Start a new operation after checking the service.';
  await finishDocumentJob(owner,id,runId,null,message);
  return {claimed:true,succeeded:false};
 }
}
