import {createClient} from '@supabase/supabase-js';
import {documentBucket,documentObjectPath,type DocumentFile} from './document-job-contract';
import {ResearchInputError} from './research-contract';
import {manualFileLimit} from './manual-tcs-contract';
export function documentStorage(){
 const url=process.env.SUPABASE_URL,key=process.env.SUPABASE_SECRET_KEY;
 if(!url||!key||new URL(url).protocol!=='https:')throw new ResearchInputError('Cloud document storage is not configured yet.');
 return createClient(url,key,{auth:{persistSession:false,autoRefreshToken:false}}).storage.from(documentBucket);
}
export async function uploadTickets(owner:string,id:string,files:DocumentFile[]){
 const storage=documentStorage(),tickets=[];
 for(const file of files){
  const {data,error}=await storage.createSignedUploadUrl(documentObjectPath(owner,id,file.field),{upsert:false});
  if(error||!data)throw new ResearchInputError('Could not prepare private document uploads. Check the storage configuration.');
  tickets.push({field:file.field,url:data.signedUrl});
 }
 return tickets;
}
export async function readDocumentObject(owner:string,id:string,file:DocumentFile){
 const {data,error}=await documentStorage().createSignedUrl(documentObjectPath(owner,id,file.field),60);
 if(error||!data)throw new ResearchInputError('An uploaded document is missing or expired. Start a new operation.');
 // Only the configured Supabase origin may receive the signed object request.
 const url=new URL(data.signedUrl),expected=new URL(process.env.SUPABASE_URL!);
 if(url.origin!==expected.origin)throw new ResearchInputError('Unexpected document storage origin.');
 const response=await fetch(url,{redirect:'error',signal:AbortSignal.timeout(60_000)});
 if(!response.ok||!response.body)throw new ResearchInputError('Could not read the private uploaded document.');
 const reader=response.body.getReader(),parts:Uint8Array[]=[];let size=0;
 try{while(true){const part=await reader.read();if(part.done)break;size+=part.value.length;if(size>manualFileLimit||size>file.size)throw new ResearchInputError('Uploaded document exceeds its declared size.');parts.push(part.value);}}
 finally{await reader.cancel();reader.releaseLock();}
 if(size!==file.size)throw new ResearchInputError('The document upload is incomplete.');
 return Buffer.concat(parts);
}
