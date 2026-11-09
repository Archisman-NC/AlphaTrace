'use client';
import {apiFetch} from '@/lib/api-fetch';

import { useCallback,useEffect,useRef,useState } from 'react';
import { localImportReviews } from '@/app/imports/actions';
import { importReviewTitle,type ImportReview,type ImportReviewStatus,type ImportReviewView } from '@/lib/provider-review-contract';

const time=(s:string|null)=>s?new Date(s).toLocaleString('en-IN',{dateStyle:'medium',timeStyle:'short',timeZone:'Asia/Kolkata'})+' IST':'No previous observation';
const value=(s:string|null)=>s===null?'Missing':s; // Preserve exact provider decimals.
type Reply={reviews?:ImportReviewView;error?:string;message?:string};
export default function ProviderReviewQueue({local,token,onInspect}:{local:boolean;token?:string;onInspect:(symbol:string)=>void}){
 const [focusId,setFocusId]=useState<string|undefined>();
 const loadedLink=useRef<string|undefined>(undefined);
 useEffect(()=>{const id=new URLSearchParams(window.location.search).get('review');if(id)setFocusId(id);},[]);
 const [data,setData]=useState<ImportReviewView|null>(null),[status,setStatus]=useState<ImportReviewStatus>('pending');
 const [cursors,setCursors]=useState<string[]>([]),[selected,setSelected]=useState<ImportReview|null>(null);
 const [note,setNote]=useState(''),[busy,setBusy]=useState(false),[error,setError]=useState(''),[message,setMessage]=useState('');
 const seq=useRef(0),working=useRef(false),saving=useRef(false),tokenRef=useRef(token);tokenRef.current=token;
 const before=cursors.at(-1),dirty=note!==(selected?.note??'');
 const load=useCallback(async()=>{
  if(working.current||saving.current)return;
  const generation=++seq.current;working.current=true;setBusy(true);
  try{
   const query={status,before,id:focusId};const params=new URLSearchParams({status,...(before?{before}:{}),...(focusId?{id:focusId}:{})});
   const reply:Reply=local?await localImportReviews(query):await apiFetch('/api/imports/reviews?'+params,{cache:'no-store',headers:{Authorization:`Bearer ${tokenRef.current}`},signal:AbortSignal.timeout(20_000)}).then(r=>r.json());
   if(generation!==seq.current)return;
   if(reply.error||!reply.reviews)throw Error(reply.error??'Review queue unavailable.');
   if(focusId&&!reply.reviews.items.length)throw Error('This review item was not found.');
   setData(reply.reviews);setError('');
   if(focusId&&loadedLink.current!==focusId){const item=reply.reviews.items[0];setSelected(item);setNote(item.note);loadedLink.current=focusId;}
  }catch(e){if(generation===seq.current)setError(e instanceof Error?e.message:'Review queue unavailable.');}
  finally{if(generation===seq.current){working.current=false;setBusy(false);}}
 },[local,status,before,focusId]);
 useEffect(()=>{
  working.current=false;void load();const timer=setInterval(()=>{if(document.visibilityState==='visible')void load();},30_000);
  return()=>{seq.current++;clearInterval(timer);};
 },[load]);
 function choose(item:ImportReview){setSelected(item);setNote(item.note);setMessage('');}
 function changePage(next:string[]){setFocusId(undefined);loadedLink.current=undefined;setCursors(next);setSelected(null);setNote('');setData(null);setMessage('');}
 async function prepare(){
  if(!selected||busy||dirty)return;
  saving.current=true;setBusy(true);setError('');
  try{
   const command={action:'prepare',id:selected.id};
   const reply=local?await localImportReviews({},command):await apiFetch('/api/imports/reviews',{method:'POST',headers:{Authorization:`Bearer ${tokenRef.current}`,'Content-Type':'application/json'},body:JSON.stringify(command),signal:AbortSignal.timeout(20_000)}).then(r=>r.json());
   if(reply.error)throw Error(reply.error);
   window.location.assign(`/research?admin=1&symbol=${selected.symbol}&importReview=${selected.id}`);
  }catch(e){setError(e instanceof Error?e.message:'Could not prepare evidence review.');}
  finally{saving.current=false;setBusy(false);}
 }
 async function save(){
  if(!selected||working.current||saving.current)return;
  saving.current=true;setBusy(true);setError('');setMessage('');
  try{
   const command={id:selected.id,version:selected.version,action:selected.status==='pending'?'review':'reopen',note};
   const reply:Reply=local?await localImportReviews({},command):await apiFetch('/api/imports/reviews',{method:'POST',headers:{Authorization:`Bearer ${tokenRef.current}`,'Content-Type':'application/json'},body:JSON.stringify(command),signal:AbortSignal.timeout(20_000)}).then(r=>r.json());
   if(reply.error)throw Error(reply.error);
   setMessage(reply.message??'Review saved.');setSelected(null);setNote('');loadedLink.current=undefined;
   saving.current=false;await load();
  }catch(e){setError(e instanceof Error?e.message:'Could not save review.');}
  finally{saving.current=false;setBusy(false);}
 }
 const updated=selected?data?.items.find(i=>i.id===selected.id):undefined;
 return <section id="import-review" className="import-review" aria-labelledby="import-review-title">
  <div className="import-review-heading"><div><span className="eyebrow">CHANGES TO INVESTIGATE</span><h2 id="import-review-title">Import review queue</h2><p>New quarters, revised values and missing data across the five pilot companies.</p></div><button disabled={busy} onClick={()=>void load()}>Refresh queue</button></div>
  <div className="import-review-tabs" aria-label="Review status">{(['pending','reviewed'] as const).map(s=><button key={s} aria-pressed={status===s} disabled={busy||dirty} onClick={()=>{setStatus(s);changePage([]);}}>{s==='pending'?'Needs review':'Reviewed'} <span>{data?.counts[s]??'—'}</span></button>)}</div>
  {focusId&&<div className="account-alert"><span>Showing the linked observation, including completed reviews.</span><button disabled={busy||dirty} onClick={()=>changePage([])}>View full queue</button></div>}
  {error&&<p className="account-alert error" role="alert">{error} Your unsaved note is retained.</p>}
  {message&&<p className="account-alert success" role="status">{message}</p>}
  <div className="import-review-layout"><div className="panel import-review-list">
   {!data?<p className="import-panel-body" role="status">{error?'Queue could not be loaded.':'Loading changes…'}</p>:!data.items.length?<div className="import-panel-body"><h3>{status==='pending'?'No pending import reviews':'No reviewed items yet'}</h3><p>{status==='pending'?'Unchanged responses do not create more work. Failed requests are shown in Import attempts below.':'Record an import review to keep a history here.'}</p></div>:data.items.map(item=><button key={item.id} className={selected?.id===item.id?'selected':''} disabled={saving.current||dirty} onClick={()=>choose(item)}><span><strong>{item.symbol}</strong><span className={`badge ${item.diff.baseline?'neutral':'amber'}`}>{item.diff.baseline?'Baseline':'Change'}</span></span><strong>{importReviewTitle(item.diff)}</strong><small>Observed {time(item.observed_at)}</small>{!item.isCurrent&&<small>Later observations available</small>}</button>)}
   <div className="import-review-pagination"><button disabled={busy||dirty||!cursors.length} onClick={()=>changePage(cursors.slice(0,-1))}>Newer</button><button disabled={busy||dirty||!data?.nextCursor} onClick={()=>changePage([...cursors,data!.nextCursor!])}>Older</button></div>
  </div><div className="panel import-review-detail">
   {!selected?<div className="import-panel-body"><h3>Select an observation</h3><p>Compare saved provider values, investigate the originals, and record that you reviewed the import.</p><p className="fine-print">This queue tracks Sales, Operating Profit and Net Profit across reported quarters. Other provider fields are outside this detector’s scope.</p></div>:<>
    <div className="panel-heading"><h3>{selected.symbol} · {importReviewTitle(selected.diff)}</h3><span className="badge amber">Official evidence still required</span></div>
    <div className="import-panel-body"><p>{selected.diff.baseline?'This is the first saved baseline. The latest two available quarters are shown; this is not a newly announced event.':'These are differences between provider observations. A revised value is not a confirmed company restatement.'}</p>
     <dl className="import-review-times"><div><dt>Previous observation</dt><dd>{time(selected.previous_observed_at)}</dd></div><div><dt>This observation</dt><dd>{time(selected.observed_at)}</dd></div></dl>
     {!!selected.diff.newQuarters.length&&<p>New quarters: {selected.diff.newQuarters.join(', ')}.</p>}
     {!!selected.diff.addedHistory.length&&<p>Older history added: {selected.diff.addedHistory.join(', ')}.</p>}
     {!!selected.diff.removedQuarters.length&&<p className="monitor-failure">Quarters absent from this response: {selected.diff.removedQuarters.join(', ')}. The provider may have shortened its history; investigate before treating this as a financial change.</p>}
     {!!selected.diff.missingMetrics.length&&<p className="monitor-failure">No values supplied for: {selected.diff.missingMetrics.join(', ')}.</p>}
     {!selected.isCurrent&&<p className="monitor-failure">A later change exists. Reviewing this item will not resolve the newer observation.</p>}
     <p className="fine-print">Values below retain the provider’s labels and scale. Units, standalone/consolidated basis, period dates and definitions must be checked against official evidence.</p>
    </div>
    <div className="import-table-scroll"><table className="import-table"><caption className="sr-only">Exact values in the two provider observations</caption><thead><tr><th>Quarter end</th><th>Provider metric</th><th>Before</th><th>After</th><th>Finding</th></tr></thead><tbody>{selected.diff.changes.map(c=><tr key={c.period+c.metric}><th scope="row">{c.period}</th><td>{c.metric}</td><td>{selected.diff.baseline?'No baseline':value(c.before)}</td><td>{value(c.after)}</td><td>{c.kind==='changed'?'Revised value':c.kind==='missing'?'Now missing':c.kind==='added'?'Value added':'Baseline'}</td></tr>)}</tbody></table></div>
    <div className="import-panel-body">
     <details><summary>Saved source and snapshot identifiers</summary><p>Provider endpoint: <a href={selected.endpoint} target="_blank" rel="noreferrer">IndianAPI quarterly results ↗</a></p><p className="fine-print">Current SHA-256: {selected.current_sha256}<br/>Previous SHA-256: {selected.previous_sha256??'None'}<br/>Queue entry created: {time(selected.detected_at)}</p></details>
     <h3>Definitions to check</h3><p>{selected.evaluation.note}</p><ul>{selected.evaluation.rows.map(row=><li key={row.label}><strong>{row.label}:</strong> {row.definition}{row.mapping==='definition_differs'?' — do not substitute this for the reported metric.':''}</li>)}</ul>
     <div className="import-review-links">{selected.sourceUrl&&<a href={selected.sourceUrl} target="_blank" rel="noreferrer">Open dated official reference ↗</a>}<button disabled={busy||dirty||(!selected.isCurrent&&!selected.handoff)} onClick={()=>void prepare()}>{selected.handoff?'Continue linked evidence review →':'Prepare evidence review →'}</button><button onClick={()=>onInspect(selected.symbol)}>Open latest company imports ↓</button></div>
     {selected.handoff&&<p>Linked evidence: {selected.handoff.state.replace('_',' ')} · reporting period {selected.handoff.periodEnd}. {selected.handoff.updateId&&<a href={`/research?symbol=${selected.symbol}&update=${selected.handoff.updateId}`}>Read publication →</a>}</p>}
     <p className="fine-print">Preparing evidence keeps this exact observation attached to a separate draft with blank values. Complete the official-source values before approval. Save or discard your review note before continuing.</p>
     <p className="fine-print">Check company identity, reporting period, units, accounting basis and metric definitions. Existing reference links may concern older periods. Recording an import review does not approve evidence, prepare a draft or publish to users.</p>
     {selected.status==='reviewed'&&<p>Reviewed {time(selected.reviewed_at)}. {selected.note||'No note recorded.'}</p>}
     {updated&&updated.version!==selected.version&&<p className="monitor-failure">This item changed in another session. Clear your draft note and select the item again to load its current state.</p>}
     <label htmlFor="import-review-note">Review note <small>Optional · up to 500 characters</small></label><textarea id="import-review-note" maxLength={500} disabled={saving.current} value={note} onChange={e=>setNote(e.target.value)}/>
     <div className="import-review-actions"><button className="primary" disabled={busy} onClick={()=>void save()}>{selected.status==='pending'?'Mark import reviewed':'Reopen review'}</button>{dirty&&<button disabled={busy} onClick={()=>setNote(selected.note)}>Discard note changes</button>}</div>
    </div>
   </>}
  </div></div>
 </section>;
}
