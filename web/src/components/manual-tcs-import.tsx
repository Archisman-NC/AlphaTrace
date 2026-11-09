'use client';
import {apiFetch} from '@/lib/api-fetch';

import { useEffect,useState,type FormEvent } from 'react';
import {cloudDocumentRequest,resumeDocument,pendingDocument,forgetDocument} from '@/lib/document-job-client';
import { tcsImportFiles,manualFileLimit } from '@/lib/manual-tcs-contract';

export default function ManualTcsImport({token,onImported}:{token?:string;onImported:()=>void}){
 const [busy,setBusy]=useState(false),[error,setError]=useState('');
 const scope='manual-tcs';
 const [progress,setProgress]=useState(''),[pending,setPending]=useState(false);
 useEffect(()=>{setPending(!!pendingDocument(scope));},[busy]);
 async function resume(){setBusy(true);setError('');try{const done=await resumeDocument(scope,token,setProgress);setResult(done.result.import as {created:boolean;state:string;publicationId:string|null});onImported();}catch(e){setError((e as Error).message);}finally{setBusy(false);}}
 const [result,setResult]=useState<{created:boolean;state:string;publicationId:string|null}|null>(null);
 async function submit(event:FormEvent<HTMLFormElement>){
  event.preventDefault();if(busy)return;setError('');setResult(null);
  const data=new FormData(event.currentTarget);
  for(const spec of tcsImportFiles){
   const file=data.get(spec.key);if(!(file instanceof File)||file.size>manualFileLimit||!file.size){setError('Select all three original files, each no larger than 20 MB.');return;}
   data.set(spec.key+'Time',String(data.get(spec.key+'Time'))+'+05:30');
  }
  setBusy(true);
  try{
   const input=Object.fromEntries([...data.entries()].filter((entry):entry is [string,string]=>typeof entry[1]==='string'));
   const cloud=await cloudDocumentRequest(scope,'tcs',input,tcsImportFiles.map(spec=>({field:spec.key,file:data.get(spec.key) as File})),token,setProgress);
   if(cloud){setResult(cloud.import as {created:boolean;state:string;publicationId:string|null});onImported();return;}
   const response=await apiFetch('/api/monitor/manual-tcs',{method:'POST',headers:token?{Authorization:`Bearer ${token}`}:{},body:data});
   const body=await response.json();if(!response.ok||!body.import)throw Error(body.error??'The import could not be confirmed. Retry the same files; identical evidence will not create another draft.');
   setResult(body.import);onImported();
  }catch(e){setError(e instanceof TypeError?'Connection lost. Retry the same files to confirm the result; identical evidence will not create another draft.':(e as Error).message);}
  finally{setBusy(false);}
 }
 return <details className="manual-import"><summary>Import official TCS files</summary><div className="manual-import-body">
  <h4>Prepare TCS Q2 FY27 evidence</h4>
  <p>Use this when automated access is blocked. This importer supports the quarter ended 30 September 2026 and its June comparison only.</p>
  <ol><li>Download the three unchanged originals from the official links below.</li><li>Select each file and enter when you actually downloaded it, in India time. Downloads must be within seven days.</li><li>Validate the files to prepare an unpublished draft, then inspect it in the review desk.</li></ol>
  <p className="fine-print">A manual import does not count as a successful automated check or approve any evidence. Reusing a file preserves its actual download time.</p>
  <form onSubmit={submit}><fieldset disabled={busy}>
   {tcsImportFiles.map(spec=><div className="manual-import-file" key={spec.key}><label htmlFor={`manual-${spec.key}`}>{spec.label}</label><a href={spec.url} target="_blank" rel="noreferrer">Download official original ↗</a><input id={`manual-${spec.key}`} name={spec.key} type="file" accept={spec.extension} required aria-describedby="manual-file-limit"/><label htmlFor={`manual-${spec.key}-time`}>Actual download time · IST</label><input id={`manual-${spec.key}-time`} name={spec.key+'Time'} type="datetime-local" step="1" required/></div>)}
   <p id="manual-file-limit" className="fine-print">One .xlsx workbook and two .pdf statements. Maximum 20 MB per file. Only these official financial documents are needed.</p>
   <label className="research-consent"><input type="checkbox" name="originalsConfirmed" value="true" required/><span>These are unchanged originals from the linked official sources, and the download times are accurate.</span></label>
   <button className="primary" type="submit">{busy?'Validating originals…':'Validate files & prepare draft'}</button>
  </fieldset></form>
  {busy&&<p role="status">{progress||"Checking workbook cells, PDF statements, periods and accounting basis. This can take up to two minutes."}</p>}
  {pending&&!busy&&<div className="account-alert"><button onClick={()=>void resume()}>Resume document operation</button><button onClick={()=>{forgetDocument(scope);setPending(false);}}>Discard pending result</button><small>Discarding does not cancel work already running.</small></div>}
  {error&&<p className="account-alert error" role="alert">{error}</p>}
  {result&&<div className="account-alert success" role="status"><p>{result.created?'An unpublished draft is ready for your review.':result.state==='published'?'These documents match an already published version. No duplicate draft was created.':'This evidence already has a draft. No duplicate was created.'}</p><a href={result.publicationId?`/research?symbol=TCS&update=${result.publicationId}`:'/research?symbol=TCS&admin=1'}>{result.publicationId?'Open published evidence':'Open review desk'} →</a></div>}
 </div></details>;
}
