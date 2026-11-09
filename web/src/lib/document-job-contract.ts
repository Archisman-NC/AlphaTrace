import {manualDownloadTime,manualFileLimit,tcsImportFiles} from './manual-tcs-contract';
import {ResearchInputError} from './research-contract';
export const documentBucket='alphatrace-evidence-jobs';
export type DocumentFile={field:string;name:string;size:number;sha256:string};
export type DocumentCommand={commandId:string;kind:'attach'|'check'|'tcs';input:Record<string,string>;files:DocumentFile[]};
const uuid=/^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
export function documentCommand(value:unknown):DocumentCommand{
 const b=value as DocumentCommand;
 const fail=()=>{throw new ResearchInputError('Choose a supported document operation, valid metadata and files up to 20 MB each.');};
 if(!b||!uuid.test(b.commandId??'')||!['attach','check','tcs'].includes(b.kind)||!b.input||typeof b.input!=='object'||Array.isArray(b.input)||!Array.isArray(b.files)||b.files.length>3)fail();
 const input:Record<string,string>={};
 const allowed=b.kind==='tcs'?['originalsConfirmed','workbookTime','q1Time','q2Time']:b.kind==='check'?['id']:['id','sourceId','url','title','publishedOn','kind'];
 for(const [k,v] of Object.entries(b.input)){if(!allowed.includes(k)||typeof v!=='string'||v.length>1500||/[\u0000-\u001f]/.test(v))fail();input[k]=v;}
 if(b.kind!=='tcs'&&!uuid.test(input.id??''))fail();
 const files=b.files.map(f=>{
  if(!f||typeof f.field!=='string'||typeof f.name!=='string'||f.name.length>120||!/^[-\w .()]+$/.test(f.name)||!Number.isSafeInteger(f.size)||f.size<1||f.size>manualFileLimit||!/^[a-f0-9]{64}$/.test(f.sha256))fail();
  return {field:f.field,name:f.name,size:f.size,sha256:f.sha256};
 });
 if(new Set(files.map(f=>f.field)).size!==files.length)fail();
 if(b.kind==='check'&&files.length)fail();
 if(b.kind==='attach'){
  if(files.length>1||files.some(f=>f.field!=='file'||!(/\.(pdf|html?)$/i.test(f.name))))fail();
  if(!input.url||!input.title?.trim()||input.title.length>160||!['results','release','facts'].includes(input.kind)||!/^\d{4}-\d{2}-\d{2}$/.test(input.publishedOn??''))fail();
  if(input.sourceId&&!/^[a-z0-9_-]{1,50}$/.test(input.sourceId))fail();
 }
 if(b.kind==='tcs'){
  if(input.originalsConfirmed!=='true'||files.length!==3)fail();
  for(const spec of tcsImportFiles){if(!files.find(f=>f.field===spec.key&&f.name.toLowerCase().endsWith(spec.extension)))fail();input[spec.key+'Time']=manualDownloadTime(input[spec.key+'Time']);}
 }
 return {commandId:b.commandId,kind:b.kind,input,files:files.sort((a,b)=>a.field.localeCompare(b.field))};
}
export const documentObjectPath=(owner:string,id:string,field:string)=>`${owner}/${id}/${field}`;
export function documentJobsEnabled(env:NodeJS.ProcessEnv=process.env){return env.ALPHATRACE_DOCUMENT_JOBS==='true'||env.VERCEL==='1';}
export type DocumentJobView={id:string;kind:DocumentCommand['kind'];state:string;message:string|null;result:Record<string,unknown>|null;expires_at:string;created_at:string};
