import { createHash } from 'node:crypto';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { mkdtemp,writeFile,rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { asUser } from './db';
import { requireResearchEditor } from './research-edit';
import { ingestResearch } from './research';
import { sourceIdentity } from './research-edit-contract';
import { UUID } from './access';
import { officialSourceUrl,validateResearch,ResearchInputError,type Source,type ResearchPayload } from './research-contract';
const maxBytes=20*1024*1024;
export async function downloadOfficial(url:string,symbol:string,fetcher:typeof fetch=fetch){
 const safe=officialSourceUrl(url,symbol);
 try{
  const response=await fetcher(safe,{redirect:'error',cache:'no-store',signal:AbortSignal.timeout(20000),headers:{Accept:'application/pdf,text/html','User-Agent':'AlphaTrace evidence review'}});
  if(!response.ok||!response.body)throw Error();
  const reader=response.body.getReader(),chunks:Uint8Array[]=[];let size=0;
  try{while(true){const part=await reader.read();if(part.done)break;size+=part.value.length;if(size>maxBytes)throw Error();chunks.push(part.value);}}finally{await reader.cancel();reader.releaseLock();}
  return Buffer.concat(chunks);
 }catch{throw new ResearchInputError('Could not retrieve this official document. Download the original in your browser and attach the PDF or HTML file instead.');}
}
export async function inspectDocument(bytes:Buffer):Promise<{mediaType:'application/pdf'|'text/html';pages:number}>{
 if(!bytes.length||bytes.length>maxBytes)throw new ResearchInputError('Choose a document between 1 byte and 20 MB.');
 if(bytes.subarray(0,5).toString()==='%PDF-'){
  const dir=await mkdtemp(join(tmpdir(),'alphatrace-evidence-'));
  try{const file=join(dir,'source.pdf');await writeFile(file,bytes,{mode:0o600});
   const {stdout}=await promisify(execFile)('pdfinfo',[file],{timeout:15000,maxBuffer:16384,env:{PATH:process.env.PATH,NODE_ENV:process.env.NODE_ENV}});
   const pages=Number(stdout.match(/^Pages:\s+(\d+)/m)?.[1]);if(!Number.isInteger(pages)||pages<1||pages>100)throw Error();return {mediaType:'application/pdf',pages};
  }catch{throw new ResearchInputError('Could not inspect this PDF. Use an unencrypted official PDF of up to 100 pages; the server requires pdfinfo.');}finally{await rm(dir,{recursive:true,force:true});}
 }
 const html=bytes.toString('utf8');
 if(!/<(?:!doctype html|html)[\s>]/i.test(html)||/captcha|access denied|just a moment|request blocked/i.test(html.slice(0,12000)))throw new ResearchInputError('Choose an original PDF or complete HTML page, not a blocked-access response.');
 return {mediaType:'text/html',pages:1};
}
async function draftFor(owner:string,id:unknown):Promise<ResearchPayload>{
 await requireResearchEditor(owner);if(typeof id!=='string'||!UUID.test(id))throw new ResearchInputError('Choose a saved draft.');
 return asUser(owner,async c=>{const row=(await c.query('SELECT payload FROM alphatrace.research_drafts WHERE id=$1',[id])).rows[0];if(!row)throw new ResearchInputError('Draft not found.');return row.payload;});
}
export async function attachResearchSource(owner:string,input:unknown,file?:File){
 const b=input as {id?:string;sourceId?:string;url?:string;title?:string;kind?:string;publishedOn?:string};
 const p=await draftFor(owner,b?.id),url=officialSourceUrl(b.url,p.symbol);
 if(file&&file.size>maxBytes)throw new ResearchInputError('The document exceeds 20 MB.');
 const bytes=file?Buffer.from(await file.arrayBuffer()):await downloadOfficial(url,p.symbol),format=await inspectDocument(bytes);
 const source={id:b.sourceId||'source_'+createHash('sha256').update(url).digest('hex').slice(0,12),title:b.title,url,kind:b.kind,publishedOn:b.publishedOn,
  sha256:createHash('sha256').update(bytes).digest('hex'),fetchedAt:new Date().toISOString(),...format,fingerprintKind:'document_bytes',retrieval:file?'operator_file':'direct',bytes:bytes.length} as Source;
 const verified=validateResearch({...p,sources:[...p.sources.filter(s=>s.id!==source.id),source]}).sources.find(s=>s.id===source.id)!;
 await asUser(owner,c=>c.query('INSERT INTO alphatrace.research_attachments(symbol,source,created_by) VALUES($1,$2,$3)',[p.symbol,JSON.stringify(verified),owner]));
 const original=p.sources.find(s=>s.id===verified.id&&sourceIdentity({...s,fetchedAt:verified.fetchedAt})===sourceIdentity(verified));
 return {source:original??verified};
}
export async function checkResearchSources(owner:string,id:unknown){
 const p=await draftFor(owner,id);let message='Exact source fingerprints checked. Values and accounting basis still require human review.';
 try{
  for(const s of p.sources){
   if(s.fingerprintKind!=='document_bytes')throw new ResearchInputError('Replace text snapshots with original PDF or HTML attachments before checking.');
   if(s.retrieval==='direct'){
    const bytes=await downloadOfficial(s.url,p.symbol);
    if(createHash('sha256').update(bytes).digest('hex')!==s.sha256)throw new ResearchInputError('An official document changed. Attach the new original and reconcile its values in a new draft.');
   }else{
    const receipt=await asUser(owner,async c=>(await c.query("SELECT 1 FROM alphatrace.research_attachments WHERE created_by=$1 AND symbol=$2 AND source->>'sha256'=$3 AND source->>'url'=$4 AND created_at>now()-interval '7 days'",[owner,p.symbol,s.sha256,s.url])).rowCount);
    if(!receipt)throw new ResearchInputError('Reattach operator-supplied originals before checking. Existing workbook imports can be checked through Monitoring’s original-file workflow.');
   }
  }
 }catch(e){message=e instanceof ResearchInputError?e.message:'Source verification failed. No successful check recorded.';
  await ingestResearch(owner,{check:{symbol:p.symbol,checkedAt:new Date().toISOString(),status:'failed',fingerprints:[],message}});throw new ResearchInputError(message);
 }
 await ingestResearch(owner,{check:{symbol:p.symbol,checkedAt:new Date().toISOString(),status:'success',fingerprints:p.sources.map(s=>s.sha256),message}});
 return {message};
}
