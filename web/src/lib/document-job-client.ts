'use client';
import type {DocumentCommand,DocumentJobView} from './document-job-contract';
type Pending={id:string;kind:DocumentCommand['kind'];signature:string;commandId:string};
type Progress=(message:string)=>void;
const key=(scope:string)=>'alphatrace-document:'+scope;
export function pendingDocument(scope:string):Pending|null{try{return JSON.parse(sessionStorage.getItem(key(scope))??'null');}catch{return null;}}
export function forgetDocument(scope:string){sessionStorage.removeItem(key(scope));}
async function call(body:object,token?:string){
 const response=await fetch('/api/document-jobs',{method:'POST',headers:{'Content-Type':'application/json',...(token?{Authorization:`Bearer ${token}`}:{})},body:JSON.stringify(body),signal:AbortSignal.timeout(60_000)});
 const value=await response.json();if(!response.ok||value.error)throw Error(value.error??'Document service is unavailable.');return value;
}
async function waitForDocument(scope:string,pending:Pending,token:string|undefined,progress:Progress){
 let {job}:{job:DocumentJobView}=await call({action:'status',id:pending.id},token);
 if(job.state==='uploading')throw Error('Upload is incomplete. Select the same files and retry, or discard this pending operation.');
 if(job.state==='queued')({job}=await call({action:'submit',id:pending.id},token));
 const deadline=Date.now()+10*60_000;
 while(true){
  progress(job.message??'Waiting for document processing…');
  if(job.state==='succeeded'){forgetDocument(scope);return {kind:pending.kind,result:job.result??{}};}
  if(job.state==='failed'||job.state==='expired'){forgetDocument(scope);throw Error(job.message??'Document operation failed.');}
  if(Date.now()>deadline)throw Error('This operation is still pending. Resume it to check its result.');
  await new Promise(resolve=>setTimeout(resolve,2500));
  ({job}=await call({action:'status',id:pending.id},token));
 }
}
export async function resumeDocument(scope:string,token:string|undefined,progress:Progress){
 const pending=pendingDocument(scope);if(!pending)throw Error('No pending document operation in this tab.');
 return waitForDocument(scope,pending,token,progress);
}
export async function cloudDocumentRequest(scope:string,kind:DocumentCommand['kind'],input:Record<string,string>,files:{field:string;file:File}[],token:string|undefined,progress:Progress){
 const config=await call({action:'configuration'},token);if(!config.enabled)return null;
 progress('Preparing private document upload…');
 const digest=async(bytes:ArrayBuffer)=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))).map(n=>n.toString(16).padStart(2,'0')).join('');
 const manifest=[];
 for(const f of files)manifest.push({field:f.field,name:f.file.name,size:f.file.size,sha256:await digest(await f.file.arrayBuffer())});
 const signature=await digest(new TextEncoder().encode(JSON.stringify({kind,input,files:manifest})).buffer);
 const prior=pendingDocument(scope);
 if(prior&&prior.signature!==signature)throw Error('Resume or discard the pending document operation before selecting different inputs.');
 const commandId=prior?.commandId??crypto.randomUUID();
 const {job,uploads}=await call({action:'create',command:{commandId,kind,input,files:manifest}},token);
 const pending={id:job.id,kind,signature,commandId};sessionStorage.setItem(key(scope),JSON.stringify(pending));
 if(job.state==='uploading'){
  for(const ticket of uploads){
   const file=files.find(f=>f.field===ticket.field)?.file;if(!file)throw Error('An expected upload is missing.');
   progress(`Uploading ${file.name} privately…`);
   const response=await fetch(ticket.url,{method:'PUT',headers:{'Content-Type':'application/octet-stream','x-upsert':'false'},body:file,signal:AbortSignal.timeout(120_000)});
   if(!response.ok){let duplicate=response.status===409;try{const body=await response.json();duplicate ||=String(body.statusCode)==='409'||body.error==='Duplicate';}catch{}if(!duplicate)throw Error('Upload failed. Select the same files and retry; completed uploads are retained temporarily.');}
  }
  await call({action:'submit',id:job.id},token);
 }
 return (await waitForDocument(scope,pending,token,progress)).result;
}
