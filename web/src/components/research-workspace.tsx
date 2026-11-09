'use client';
import { useEffect,useRef,useState } from 'react';
import IssueReportButton from './issue-report-button';
import ResearchEditor from './research-editor';
import ImportEvidenceContext from './import-evidence-context';
import { localImportReviews } from '@/app/imports/actions';
import type { ImportReview } from '@/lib/provider-review-contract';
import { draftReadiness } from '@/lib/research-edit-contract';
import WorkspaceHeader from './workspace-header';
import PrivateThesis from '@/components/private-thesis';
import type { Session } from '@supabase/supabase-js';
import { browserAuth } from '@/lib/browser-auth';
import { localResearch } from '@/app/research/actions';
import { universe } from '@/lib/companies';
import { metrics,comparisonResult,freshness,publicationGaps,spreadsheetMediaType,type ResearchPayload,type EvidencePoint } from '@/lib/research-contract';
import type { ResearchView,ResearchDraft } from '@/lib/research-view';

const date=(s:string)=>new Date(s).toLocaleDateString('en-IN',{day:'numeric',month:'short',year:'numeric',timeZone:'UTC'});
const activityDate=(s:string)=>new Date(s).toLocaleDateString('en-IN',{day:'numeric',month:'short',year:'numeric',timeZone:'Asia/Kolkata'});
const stamp=(s:string)=>new Date(s).toLocaleString('en-IN',{day:'numeric',month:'short',hour:'2-digit',minute:'2-digit',timeZone:'Asia/Kolkata'})+' IST';
function Evidence({point,payload}:{point:EvidencePoint;payload:ResearchPayload}){
 const source=payload.sources.find(s=>s.id===point.sourceId)!;
 const parts=point.value?.split('.'),value=parts?parts[0].replace(/\B(?=(\d{3})+(?!\d))/g,',')+(parts[1]!==undefined?'.'+parts[1]:''):'Unavailable';
 return <div className="evidence-value"><strong>{value}</strong><small>{point.unit} · {date(point.periodEnd)}</small><a href={source.url+(point.page?`#page=${point.page}`:'')} target="_blank" rel="noreferrer">{point.worksheet?`${point.worksheet}!${point.cell}`:point.page?`Page ${point.page}`:'Source section'} ↗</a><small>{point.locator}</small><small>{point.scope&&`${point.scope} · `}{point.basis} · {point.standard} · {point.treatment}</small></div>;
}
function EvidenceSheet({payload,report}:{payload:ResearchPayload;report?:{publicationId:string;version:number;local:boolean;token?:string}}){
 return <>
  {report&&<IssueReportButton key={report.publicationId} {...report} payload={payload}/>}
  <div className="research-table-wrap"><table className="research-table"><thead><tr><th scope="col">Metric</th><th scope="col">Previous period</th><th scope="col">Current period</th><th scope="col">Change · {payload.comparison}</th></tr></thead><tbody>{payload.comparisons.map(c=>{const result=comparisonResult(c,payload.comparison);return <tr key={c.metric}><th scope="row">{metrics[c.metric]}{c.note&&<small>{c.note}</small>}{report&&<IssueReportButton key={report.publicationId+c.metric} {...report} payload={payload} metric={c.metric}/>}</th><td><Evidence point={c.previous} payload={payload}/></td><td><Evidence point={c.current} payload={payload}/></td><td>{result.available?<strong className="research-change">{!result.change!.startsWith('-')&&result.change!=='0.00'?'+':''}{result.change}{result.changeUnit==='pp'?' pp':'%'}</strong>:<><span className="badge amber">Not comparable</span><small>{result.reason}</small></>}</td></tr>;})}</tbody></table></div>
  <section className="research-limitations"><span className="eyebrow">READ WITH THE NUMBERS</span><h3>What this comparison doesn’t establish</h3><ul>{payload.limitations.map((s,i)=><li key={i}>{s}</li>)}</ul></section>
  <section className="research-sources"><div className="panel-heading"><h2>Evidence trail</h2><span>{payload.sources.length} official sources</span></div>{payload.sources.map(s=><div className="research-source" key={s.id}><div><a href={s.url} target="_blank" rel="noreferrer">{s.title} ↗</a><small>Published {date(s.publishedOn)} · Retrieved {stamp(s.fetchedAt)}</small><small>{s.retrieval==='operator_file'?'Operator-supplied file · automatic access not verified':s.retrieval==='operator_snapshot'?'Operator-supplied text snapshot':'Direct source retrieval'} · {s.mediaType===spreadsheetMediaType?'Excel workbook':s.mediaType==='application/pdf'?`${s.pages} PDF pages`:'Official web page'}</small></div><details><summary>Source fingerprint</summary><small>SHA-256 of {s.fingerprintKind==='extracted_text'?'retrieved text; not the original document bytes':'original document bytes'}</small><code>{s.sha256}</code></details></div>)}</section>
 </>;
}
function ReviewControls({draft,busy,onAction,check,editing}:{editing:boolean;check:ResearchView['catalog'][number]['check'];draft:ResearchDraft;busy:boolean;onAction:(command:unknown)=>Promise<void>}){
 const [note,setNote]=useState(''),[evidence,setEvidence]=useState(false),[basis,setBasis]=useState(false);
 const gaps=draftReadiness(draft.payload,check),noteLength=note.trim().length;
 const approvalRequirements=[
  ...(!evidence||!basis?['Complete both review confirmations.']:[]),
  ...(noteLength<10?[`Add ${10-noteLength} more character${10-noteLength===1?'':'s'} to the review note.`]:[]),
  ...(gaps.length?['Resolve the evidence issues listed above.']:[]),
 ];
 return <section className="review-controls"><div><span className="eyebrow">OPERATOR CHECKPOINT</span><h2>{draft.state==='approved'?'Approved. Ready for publication.':'Check the originals before approval.'}</h2><p>Approval records your review of these exact values, periods and footnotes.</p></div>{draft.review_note&&<p className="review-previous">Previous review: {draft.review_note}</p>}{gaps.length>0&&<div className="account-alert error">{gaps.join(' ')}</div>}
 {draft.state==='approved'?<><p>Publishing makes this version available to all invited users. A correction creates a new version.</p><button className="primary" disabled={busy||gaps.length>0} onClick={()=>void onAction({action:'publish',id:draft.id})}>Publish reviewed update →</button></>:<><label className="research-consent"><input type="checkbox" checked={evidence} disabled={busy} onChange={e=>setEvidence(e.target.checked)}/><span>I checked every value and source location against the original documents.</span></label><label className="research-consent"><input type="checkbox" checked={basis} disabled={busy} onChange={e=>setBasis(e.target.checked)}/><span>I checked units, quarter dates, accounting basis and exceptional-item footnotes.</span></label><label htmlFor="review-note">Review note</label><p id="review-note-help" className="fine-print">Describe what you checked and any unresolved limitations so the review can be traced later. Minimum 10 characters, not words; leading and trailing spaces don’t count.</p><textarea id="review-note" aria-describedby="review-note-help review-note-count" maxLength={1000} value={note} onChange={e=>setNote(e.target.value)} placeholder="Record what you checked and any remaining limitations. Explain changed evidence when correcting an earlier publication."/><p id="review-note-count" className="fine-print">{noteLength} / 1,000 characters · 10 minimum</p><p id="approval-requirements" className="fine-print" role="status">{editing?'Save or discard evidence edits before approval.':busy?'Saving review…':approvalRequirements.length?`Before approval: ${approvalRequirements.join(' ')}`:'Ready to approve. Approval records your confirmation of the checks above.'}</p><div className="research-actions"><button className="primary" aria-describedby="approval-requirements" disabled={busy||approvalRequirements.length>0} onClick={()=>void onAction({action:'approve',id:draft.id,note,evidenceConfirmed:evidence,basisConfirmed:basis})}>Approve evidence</button><button aria-describedby="review-note-help review-note-count" disabled={busy||noteLength<10} onClick={()=>void onAction({action:'reject',id:draft.id,note})}>Request correction</button></div></>}
 </section>;
}
export default function ResearchWorkspace({local,authConfig,initialSymbol,initialAdmin,initialUpdate,returnReview,initialImportReview}:{local:boolean;authConfig:{url:string;key:string}|null;initialSymbol:string;initialAdmin:boolean;initialUpdate?:string;returnReview?:string;initialImportReview?:string}){
 const [session,setSession]=useState<Session|null>(null),[ready,setReady]=useState(local||!authConfig);
 const [data,setData]=useState<ResearchView|null>(null),[symbol,setSymbol]=useState(universe.some(c=>c.symbol===initialSymbol)?initialSymbol:'TCS'),[admin,setAdmin]=useState(initialAdmin);
 const [search,setSearch]=useState(''),[filter,setFilter]=useState('all'),[selectedDraft,setSelectedDraft]=useState(''),[version,setVersion]=useState(initialUpdate??''),[showSuperseded,setShowSuperseded]=useState(false);
 const [loading,setLoading]=useState(true),[busy,setBusy]=useState(false),[error,setError]=useState(''),[message,setMessage]=useState(''),[reload,setReload]=useState(0),[withdrawNote,setWithdrawNote]=useState('');
 const [requestedUpdate,setRequestedUpdate]=useState(initialUpdate);
 const [importReview,setImportReview]=useState(initialImportReview),[importContext,setImportContext]=useState<ImportReview|null>(null);
 const resolvedHandoff=useRef<string|undefined>(undefined);
 const [notesDirty,setNotesDirty]=useState(false),[evidenceDirty,setEvidenceDirty]=useState(false);
 const unsaved=notesDirty||evidenceDirty;
 const generation=useRef(0),authUrl=authConfig?.url,authKey=authConfig?.key;
 useEffect(()=>{
  if(local||!authUrl||!authKey)return;
  let active=true;
  try{
   const client=browserAuth(authUrl,authKey);
   const {data:{subscription}}=client.auth.onAuthStateChange((_event,next)=>{if(active){setSession(next);setReady(true);}});
   client.auth.getSession().then(({data})=>{if(active){setSession(data.session);setReady(true);}}).catch(()=>{if(active){setReady(true);setError('Could not restore your session. Sign in again.');}});
   return()=>{active=false;subscription.unsubscribe();};
  }catch{setReady(true);setError('Browser session storage is unavailable.');}
 },[local,authUrl,authKey]);
 useEffect(()=>{
  const current=++generation.current;setData(null);setError('');setLoading(true);
  if(!ready||(!local&&!session)){setLoading(false);return;}
  async function load(){try{
   const result=local?await localResearch(symbol,admin,undefined,requestedUpdate):await fetch(`/api/research?symbol=${encodeURIComponent(symbol)}${admin?'&admin=1':''}${requestedUpdate?'&update='+encodeURIComponent(requestedUpdate):''}`,{headers:{Authorization:`Bearer ${session!.access_token}`},cache:'no-store'}).then(r=>r.json());
   if(current!==generation.current)return;
   if(result.error)throw Error(result.error);
   if(importReview&&result.isOperator){
    const context=local?await localImportReviews({id:importReview}):await fetch(`/api/imports/reviews?id=${importReview}`,{headers:{Authorization:`Bearer ${session!.access_token}`},cache:'no-store'}).then(r=>r.json());
    if(current!==generation.current)return;
    const item=context.reviews?.items[0];
    if(context.error||!item||item.symbol!==symbol)throw Error(context.error??'The linked import observation is unavailable for this company.');
    setImportContext(item);
    if(item.handoff&&resolvedHandoff.current!==item.handoff.id){
     resolvedHandoff.current=item.handoff.id;
     if(item.handoff.updateId){setAdmin(false);setVersion(item.handoff.updateId);setRequestedUpdate(item.handoff.updateId);}
     else {setAdmin(true);setShowSuperseded(false);setSelectedDraft(item.handoff.id);}
    }
   }
   setData(result);
  }catch(e){if(current===generation.current)setError(e instanceof Error?e.message:'Research could not be loaded.');}finally{if(current===generation.current)setLoading(false);}}
  void load();return()=>{generation.current++;};
 },[local,ready,session?.access_token,symbol,admin,reload,requestedUpdate,importReview]);
 async function act(command:unknown){
  if(busy)return;const current=generation.current;setBusy(true);setError('');setMessage('');
  try{
   const result=local?await localResearch(undefined,false,command):await fetch('/api/research',{method:'POST',headers:{Authorization:`Bearer ${session!.access_token}`,'Content-Type':'application/json'},body:JSON.stringify(command)}).then(r=>r.json());
   if(current!==generation.current)return;
   if(result.error)throw Error(result.error);
   if((command as {action?:string}).action==='publish'){
    setAdmin(false);setVersion(result.id);setRequestedUpdate(result.id);setSelectedDraft('');setShowSuperseded(false);
    window.history.replaceState(null,'',`/research?symbol=${encodeURIComponent(symbol)}&update=${result.id}${importReview?"&importReview="+importReview:""}`);
    setMessage('Published. This version is now available to invited users.');
   }else setMessage('Saved. The research record has been updated.');
   setWithdrawNote('');setReload(n=>n+1);
  }catch(e){if(current===generation.current)setError(e instanceof Error?e.message:'Could not confirm this change. Reload before retrying.');}finally{setBusy(false);}
 }
 const company=data?.catalog.find(c=>c.symbol===symbol),companyDrafts=data?.drafts.filter(d=>d.payload.symbol===symbol&&d.state!=='published')??[];
 const supersededDrafts=companyDrafts.filter(d=>!!d.superseded_by||!!d.revised_by),pendingDrafts=companyDrafts.filter(d=>!d.superseded_by&&!d.revised_by);
 const drafts=showSuperseded?supersededDrafts:pendingDrafts;
 const draft=selectedDraft?drafts.find(d=>d.id===selectedDraft):drafts[0],published=data?.history.find(u=>u.id===version)??data?.history[0];
 const payload=admin?draft?.payload:published?.payload;
 const sourceState=freshness(company?.check??null),covered=data?.catalog.filter(c=>c.status==='published').length??0;
 function mode(next:boolean){if(unsaved){setError('Save or discard your private research edits before switching views.');return;}setAdmin(next);setVersion('');setSelectedDraft('');setShowSuperseded(false);setMessage('');window.history.replaceState(null,'',`/research?symbol=${encodeURIComponent(symbol)}${next?'&admin=1':''}`);}
 function choose(next:string){if(busy)return;if(unsaved){setError('Save or discard your private research edits before switching companies.');return;}setRequestedUpdate(undefined);setImportReview(undefined);setImportContext(null);resolvedHandoff.current=undefined;setSymbol(next);setSelectedDraft('');setShowSuperseded(false);setVersion('');setMessage('');setWithdrawNote('');window.history.replaceState(null,'',`/research?symbol=${encodeURIComponent(next)}${admin?'&admin=1':''}`);}
 return <div className="private-shell research-shell"><WorkspaceHeader active="research" local={local}/><main id="workspace-content" className="private-content research-content">
  {returnReview&&<a className="review-back-link" href={`/inbox?filter=all&update=${returnReview}`}>← Return to your review</a>}
  <div className="page-heading"><div><span className="eyebrow">OFFICIAL SOURCES. VISIBLE REASONING.</span><h1>{admin?'Research review desk':'The evidence behind the change.'}</h1><p>{admin?'Inspect source values, resolve differences, then publish a version users can trace.':'Quarterly comparisons with the source, period and limitations kept in view.'}</p></div>{data&&<button disabled={loading||busy||unsaved} onClick={()=>setReload(n=>n+1)}>↻ Refresh</button>}</div>
  {error&&<div role="alert" className="account-alert error">{error}<button disabled={busy||unsaved} onClick={()=>setReload(n=>n+1)}>Reload</button>{admin&&<button onClick={()=>mode(false)}>Return to research</button>}</div>}{message&&<div role="status" className="account-alert success">{message}</div>}
  {!ready?<div className="panel empty" role="status">Restoring your session…</div>:!local&&!session?<section className="panel empty"><span className="eyebrow">YOUR INVITATION OPENS THE WORKSPACE</span><h2>Sign in to read the research.</h2><p>Use your existing beta account. Research is shared; your portfolio stays private.</p><a className="button-link primary" href="/workspace">Sign in →</a></section>:<>
   <div className="research-toolbar"><nav className="tabs" aria-label="Research mode"><button className={!admin?'selected':''} disabled={busy} onClick={()=>mode(false)}>Published research</button>{data?.isOperator&&<button className={admin?'selected':''} disabled={busy} onClick={()=>mode(true)}>Review desk</button>}</nav><span>{data?admin?`Awaiting review: ${data.drafts.filter(d=>d.state!=='published'&&!d.superseded_by&&!d.revised_by).length}`:`${covered} of 25 companies with published research`:'Loading coverage…'}</span></div>
   <div className="research-layout"><aside className="research-roster panel"><div className="research-roster-header"><label htmlFor="research-search">Company universe</label><input id="research-search" type="search" placeholder="Search companies" value={search} onChange={e=>setSearch(e.target.value)}/><label className="sr-only" htmlFor="coverage-filter">Filter coverage</label><select id="coverage-filter" value={filter} onChange={e=>setFilter(e.target.value)}><option value="all">All companies</option><option value="published">Published</option><option value="planned">Planned</option><option value="withdrawn">Withdrawn</option></select></div><div className="research-company-list">{(data?.catalog??universe.map(c=>({...c,status:'planned',periodEnd:null}))).filter(c=>(filter==='all'||c.status===filter)&&`${c.name} ${c.symbol}`.toLowerCase().includes(search.toLowerCase())).map(c=><button className={c.symbol===symbol?'active':''} key={c.symbol} disabled={busy} onClick={()=>choose(c.symbol)}><span><strong>{c.symbol}</strong><small>{c.name}</small></span><span className={`badge ${c.status==='published'?'green':''}`}>{data?(admin&&data.drafts.some(d=>d.payload.symbol===c.symbol&&d.state!=='published'&&!d.superseded_by&&!d.revised_by)?'Draft ready':c.status==='published'?'Published':c.status==='withdrawn'?'Withdrawn':'Planned'):'…'}</span></button>)}</div></aside>
    <div className="research-main-column">{data?.isOperator&&importContext&&<ImportEvidenceContext item={importContext}/ >}{!admin&&<PrivateThesis key={`${local?'local':session?.user.id}:${symbol}`} symbol={symbol} local={local} token={session?.access_token} publication={published} onDirty={setNotesDirty}/>}<article className="panel research-reader" aria-busy={loading}>{loading?<div className="empty" role="status">Loading evidence…</div>:!data?<div className="empty"><h2>Research is unavailable</h2><p>Your saved portfolio is separate. Retry when research storage is available.</p></div>:<>
     <header className="research-reader-heading"><span className="eyebrow">{symbol} · {company?.sector}</span><h2>{company?.name}</h2><div className="research-reader-meta"><span className={`badge ${sourceState==='checked'&&!company?.sourceChanged?'green':'amber'}`}>{company?.sourceChanged?'Source changed':sourceState==='checked'?'Sources checked':sourceState==='failed'?'Source check failed':sourceState==='stale'?'Source check overdue':'Sources not checked'}</span>{company?.check&&<small>{stamp(company.check.checked_at)}</small>}</div>{company?.check&&(company.check.status==='failed'?<p className="fine-print">{company.check.message}</p>:<details className="fine-print"><summary>Source-check log · {stamp(company.check.checked_at)}</summary><p>Recorded at source-check time. Review and publication status may have changed since this entry.</p><p>{company.check.message}</p></details>)}{sourceState==='failed'&&company?.lastSuccessfulCheck&&<p className="fine-print">Last successful source check: {stamp(company.lastSuccessfulCheck)}</p>}{company?.sourceChanged&&<p className="amber-text">The latest source fingerprints differ from this publication. A new review is needed.</p>}</header>
     {admin&&!importReview&&supersededDrafts.length>0&&<div className="research-actions"><button disabled={busy||unsaved} aria-pressed={showSuperseded} onClick={()=>{setShowSuperseded(!showSuperseded);setSelectedDraft('');}}>{showSuperseded?'Back to pending drafts':`View superseded drafts (${supersededDrafts.length})`}</button><span className="fine-print">Earlier drafts are retained for history, not counted as awaiting review.</span></div>}
     {admin&&drafts.length>0&&<div className="research-version"><label htmlFor="draft-version">{showSuperseded?'Superseded drafts':'Pending drafts'}</label><select id="draft-version" disabled={busy||unsaved||!!importReview} value={draft?.id??''} onChange={e=>setSelectedDraft(e.target.value)}>{drafts.map(d=><option key={d.id} value={d.id}>{activityDate(d.created_at)} · {d.revised_by?'revised':d.superseded_by?'superseded':d.state.replace('_',' ')} · {d.id.slice(0,8)}</option>)}</select></div>}
     {!admin&&!!data.history.length&&<div className="research-version"><label htmlFor="published-version">Publication history · latest 20</label><select id="published-version" disabled={busy||unsaved} value={published?.id??''} onChange={e=>setVersion(e.target.value)}>{data.history.map((u,i)=><option key={u.id} value={u.id}>Version {u.version}{i===0?' · latest':''} · {activityDate(u.published_at)}{u.withdrawn_at?' · withdrawn':''}</option>)}</select></div>}
     {!payload?<div className="empty research-empty"><span className="empty-symbol">◎</span><h2>{admin?'No pending draft for this company.':'Coverage is being prepared.'}</h2><p>{admin?(company?.status==='published'?'The latest reviewed version is published. New evidence will appear here when it is ready for review.':'Run the local source batch to submit evidence for review.'):'There is no reviewed publication for this company yet. Being in the 25-company universe does not mean research is available.'}</p>{admin&&data.history.length>0&&<button onClick={()=>mode(false)}>View published research →</button>}{!admin&&data.isOperator&&<button onClick={()=>mode(true)}>Open review desk →</button>}</div>:<>
      <div className="research-update-heading"><span className={`badge ${admin?'amber':'lavender'}`}>{admin?(draft?.superseded_by?'Superseded draft':draft?.state.replace('_',' ')):`Version ${published?.version}`}</span><h3>{payload.title}</h3><p>Quarter ended {date(payload.periodEnd)} · {payload.comparison} · {admin?(draft?.superseded_by?'Historical draft · superseded':'Unpublished evidence'):'Published '+activityDate(published!.published_at)}</p></div>
      {!admin&&published?.withdrawn_at&&<div className="account-alert error">Withdrawn {activityDate(published.withdrawn_at)}: {published.withdrawal_reason}. Retained for the audit trail; do not use as current research.</div>}
      {!admin&&published&&published.id!==data.history[0]?.id&&<div className="account-alert">You are reading an older version. Select the latest version above for the current publication.</div>}
      {admin&&draft?.superseded_by&&<div className="account-alert">Superseded by published version {draft.superseded_version}. This draft and its original review record are retained for history and cannot be approved or published. <a href={`/research?symbol=${symbol}&update=${draft.superseded_by}`}>Read the published version →</a>{draft.review_note&&<p>Original review note: {draft.review_note}</p>}</div>}
      {!admin&&<p className="fine-print">The evidence and notes below are preserved exactly as reviewed. References to a proposed publication describe its preparation; this version has been published.</p>}
      {admin&&draft?.revised_by&&<div className="account-alert">A newer correction replaces this draft. Its original evidence and review are retained. <button onClick={()=>{setShowSuperseded(false);setSelectedDraft(draft.revised_by!);}}>Open correction</button></div>}
      {admin&&draft?.parent_id&&<details className="revision-history"><summary>Revision history · {draft.changes?.length??0} changes</summary><p>{draft.revision_reason}</p><p>Based on draft {draft.parent_id.slice(0,8)}. Approval was not carried forward.</p><ul>{draft.changes?.map((c,i)=><li key={i}>{c}</li>)}</ul></details>}
      <EvidenceSheet payload={payload} report={!admin&&published?{publicationId:published.id,version:published.version,local,token:session?.access_token}:undefined}/>
      {data.isOperator&&((admin&&draft&&!draft.superseded_by&&!draft.revised_by)||(!admin&&published&&published.id===data.history[0]?.id))&&<ResearchEditor key={admin?draft!.id:published!.draft_id} id={admin?draft!.id:published!.draft_id} payload={payload} token={session?.access_token} check={company?.check??null} onDirty={setEvidenceDirty} onChecked={()=>{setEvidenceDirty(false);setMessage('Exact source check completed. Human approval is still required.');setReload(n=>n+1);}} onSaved={id=>{setEvidenceDirty(false);setAdmin(true);setShowSuperseded(false);setSelectedDraft(id);setMessage('New unapproved revision saved. Earlier evidence and publications are preserved.');setReload(n=>n+1);window.history.replaceState(null,'',`/research?symbol=${symbol}&admin=1${importReview?"&importReview="+importReview:""}`);}}/>}
      {admin&&draft&&!draft.superseded_by&&!draft.revised_by&&<ReviewControls key={draft.id+draft.state} draft={draft} editing={evidenceDirty} check={company?.check??null} busy={busy||evidenceDirty} onAction={act}/>}
      {!admin&&published&&data.isOperator&&!published.withdrawn_at&&<details className="research-withdraw"><summary>Withdraw this version</summary><p>Use this if the published evidence is unreliable. The original remains visible in the audit trail, marked withdrawn.</p><label htmlFor="withdraw-reason">Reason for withdrawal</label><textarea id="withdraw-reason" maxLength={1000} value={withdrawNote} onChange={e=>setWithdrawNote(e.target.value)}/><button disabled={busy||withdrawNote.trim().length<10} onClick={()=>void act({action:'withdraw',id:published.id,note:withdrawNote})}>Withdraw version {published.version}</button></details>}
     </>}
    </>}</article></div>
   </div>
  </>}
 </main><footer className="app-footer"><span>AlphaTrace · Evidence first. Decisions remembered.</span><span>Reported results · No live prices or trade recommendations</span></footer></div>;
}
