'use client';
import { useEffect,useId,useRef,useState } from 'react';
import { metrics,type ResearchPayload } from '@/lib/research-contract';
import { issueCategories,type IssueCategory,type IssueCommand } from '@/lib/issue-contract';
import { requestIssues } from '@/lib/issue-client';
export default function IssueReportButton({publicationId,version,payload,metric=null,local,token}:{publicationId:string;version:number;payload:ResearchPayload;metric?:string|null;local:boolean;token?:string}){
 const [open,setOpen]=useState(false),[category,setCategory]=useState<IssueCategory>('incorrect_value'),[note,setNote]=useState(''),[busy,setBusy]=useState(false),[error,setError]=useState(''),[uncertain,setUncertain]=useState(false),[saved,setSaved]=useState<{id:string;duplicate:boolean}|null>(null);
 const dialog=useRef<HTMLDialogElement>(null),command=useRef<IssueCommand|null>(null),alive=useRef(true),lock=useRef(false),id=useId();
 const label=metric?metrics[metric as keyof typeof metrics]:'Entire publication';
 useEffect(()=>{alive.current=true;return()=>{alive.current=false;};},[]);
 useEffect(()=>{if(open)dialog.current?.showModal();else dialog.current?.close();},[open]);
 async function submit(){
  if(lock.current)return;lock.current=true;setBusy(true);setError('');let rejected=false;
  command.current??={action:'submit',commandId:crypto.randomUUID(),publicationId,metric,category,note:note.trim()};
  try{const result=await requestIssues(local,token,{},command.current);if(!alive.current)return;if(!result.detail){rejected=!!result.status&&result.status<500;throw Error(result.error||'The save could not be confirmed.');}setSaved({id:result.detail.report.id,duplicate:!!result.duplicate});setUncertain(false);}
  catch(e){if(alive.current){setError(e instanceof Error?e.message:'Could not submit the report.');setUncertain(!rejected);if(rejected)command.current=null;}}
  finally{lock.current=false;if(alive.current)setBusy(false);}
 }
 return <><button className="text-button issue-report-trigger" aria-label={`Report an issue: ${label}`} onClick={()=>setOpen(true)}>Report an issue</button><dialog ref={dialog} className="issue-dialog" aria-labelledby={id+'-title'} onCancel={e=>{if(busy)e.preventDefault();else setOpen(false);}} onClose={()=>setOpen(false)}>
  <div className="issue-dialog-top"><span className="eyebrow">HELP KEEP THE EVIDENCE ACCURATE</span><button aria-label="Close report" disabled={busy} onClick={()=>setOpen(false)}>×</button></div><h2 id={id+'-title'}>{saved?'Your report is recorded.':'What needs a closer look?'}</h2><p className="issue-attached">{payload.symbol} · Version {version} · {label}<br/>Quarter ended {payload.periodEnd}</p>
  {saved?<><p role="status">{saved.duplicate?'You already have an active report for this issue. Your new note was not added; the original report is linked below.':'The exact publication and evidence context were attached. You can follow the operator’s response in your report history.'}</p><a className="button-link primary" href={`/issues?report=${saved.id}`}>View report & status →</a></>:<form onSubmit={e=>{e.preventDefault();void submit();}}>
   <label htmlFor={id+'-category'}>Issue type</label><select id={id+'-category'} value={category} disabled={busy||uncertain} onChange={e=>{setCategory(e.target.value as IssueCategory);command.current=null;}}>{Object.entries(issueCategories).map(([key,label])=><option key={key} value={key}>{label}</option>)}</select>
   <label htmlFor={id+'-note'}>What did you notice?</label><textarea id={id+'-note'} value={note} maxLength={2000} disabled={busy||uncertain} required aria-describedby={id+'-hint'} onChange={e=>{setNote(e.target.value);command.current=null;}} placeholder="Tell us which figure, quarter, source or explanation looks wrong."/><p id={id+'-hint'} className="fine-print">3–2,000 characters. Visible to you and research operators. This reports a concern; it does not change the published figures.</p>
   {error&&<p className="account-alert error" role="alert">{error}{uncertain?' Your original request is preserved. Retry to confirm whether it was saved.':''}</p>}
   <div className="issue-form-actions"><button type="button" disabled={busy} onClick={()=>setOpen(false)}>Close</button><button className="primary" disabled={busy||note.trim().length<3}>{busy?'Submitting…':uncertain?'Retry same report':'Submit report'}</button></div>
  </form>}
 </dialog></>;
}
