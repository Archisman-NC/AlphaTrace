import { execFile } from 'node:child_process';
import { mkdtemp,writeFile,readFile,rm,stat } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { resolve,join } from 'node:path';
import { promisify } from 'node:util';
import { asUser } from './db';
import { ingestResearch } from './research';
import { ResearchForbiddenError,ResearchInputError,validateResearch,validateCheck } from './research-contract';
import { RequestBodyError } from './http-input';
import { manualRequestLimit,validateManualFiles } from './manual-tcs-contract';

export async function requireManualOperator(owner:string){
 await asUser(owner,async c=>{if(!(await c.query('SELECT 1 FROM alphatrace.research_operators WHERE user_id=$1',[owner])).rowCount)throw new ResearchForbiddenError('Official-file imports require operator access.');});
}
export async function boundedManualForm(request:Request):Promise<FormData>{
 const contentType=request.headers.get('content-type')??'';
 if(!contentType.startsWith('multipart/form-data;'))throw new RequestBodyError('Choose the three original files in the import form.',415);
 if(Number(request.headers.get('content-length'))>manualRequestLimit)throw new RequestBodyError('The combined upload exceeds 60 MB.',413);
 const reader=request.body?.getReader();if(!reader)throw new RequestBodyError('Files are required.',400);
 const chunks:Uint8Array[]=[];let total=0;
 try{
  while(true){const part=await reader.read();if(part.done)break;total+=part.value.byteLength;if(total>manualRequestLimit){await reader.cancel();throw new RequestBodyError('The combined upload exceeds 60 MB.',413);}chunks.push(part.value);}
  return await new Response(Buffer.concat(chunks),{headers:{'Content-Type':contentType}}).formData();
 }catch(e){if(e instanceof RequestBodyError)throw e;throw new RequestBodyError('Could not read these files. Select them again.',400);}
 finally{reader.releaseLock();}
}
// Per-process bound: never run multiple large PDF/workbook extractions concurrently.
let importing=false;
export async function importTcsFiles(owner:string,form:FormData){
 await requireManualOperator(owner);
 const files=validateManualFiles(form);
 if(importing)throw new RequestBodyError('Another import is running. Wait for it to finish, then retry.',429);
 importing=true;let directory:string|undefined;
 try{
  directory=await mkdtemp(join(tmpdir(),'alphatrace-tcs-'));
  const paths:Record<string,string>={};
  for(const entry of files){
   const bytes=Buffer.from(await entry.file.arrayBuffer());
   if(entry.extension==='.pdf'?!bytes.subarray(0,5).equals(Buffer.from('%PDF-')):!bytes.subarray(0,4).equals(Buffer.from([0x50,0x4b,0x03,0x04])))throw new ResearchInputError('A selected file is not an original PDF or Excel workbook.');
   paths[entry.key]=join(directory,entry.key+entry.extension);await writeFile(paths[entry.key],bytes,{mode:0o600});
  }
  const researchDirectory=process.env.ALPHATRACE_RESEARCH_DIR||resolve(process.cwd(),'../scripts/research');
  try{
   await promisify(execFile)('python3',[join(researchDirectory,'tcs_workbook.py'),paths.workbook,'--retrieved-at',files[0].retrievedAt,'--q1-results',paths.q1,'--q1-retrieved-at',files[1].retrievedAt,'--q2-results',paths.q2,'--q2-retrieved-at',files[2].retrievedAt,'--output-dir',directory],{
    cwd:researchDirectory,env:{PATH:process.env.PATH,LANG:'en_US.UTF-8',NODE_ENV:process.env.NODE_ENV,ALPHATRACE_RESEARCH_ROOT:directory},timeout:120000,killSignal:'SIGKILL',maxBuffer:32768,
   });
  }catch{
   throw new ResearchInputError('The files could not be reconciled. Use the linked Q1/Q2 FY27 USD originals and actual download times. If those are correct, the operator must check the importer and its Python/Poppler installation. Nothing was submitted.');
  }
  const output=join(directory,'TCS-reconciled.json');if((await stat(output)).size>131072)throw new ResearchInputError('Prepared evidence exceeds the import limit.');
  const bundle=JSON.parse(await readFile(output,'utf8'));
  const draft=validateResearch(bundle.draft),check=validateCheck(bundle.check);
  if(draft.symbol!=='TCS'||draft.sources.length!==3||draft.sources.some(s=>s.retrieval!=='operator_file'))throw new ResearchInputError('Unexpected evidence returned by the importer.');
  const result=await ingestResearch(owner,{draft,check});
  const saved=await asUser(owner,async c=>(await c.query('SELECT state,published_update_id FROM alphatrace.research_drafts WHERE id=$1',[result.draftId])).rows[0]);
  return {...result,state:saved.state,publicationId:saved.published_update_id};
 }finally{importing=false;if(directory)await rm(directory,{recursive:true,force:true});}
}
