'use client';
import { useCallback,useEffect,useRef,useState } from 'react';
import type { Session } from '@supabase/supabase-js';
import WorkspaceHeader from './workspace-header';
import UpdateEvidence from './update-evidence';
import JournalExport from './journal-export';
import AutoUpdateStatus,{useAutoUpdates} from './auto-update-status';
import {inboxHasChanges} from '@/lib/auto-refresh';
import { browserAuth } from '@/lib/browser-auth';
import { localInbox } from '@/app/inbox/actions';
import { inboxFilters,metricLabel,type InboxFilter,type InboxList,type InboxDetail,type ReviewAction,type ReviewCommand } from '@/lib/inbox-contract';
import { freshness } from '@/lib/research-contract';

type Query={filter?:string;cursor?:string;update?:string;before?:number};
type Result={inbox?:InboxList;detail?:InboxDetail;sync?:{added:number;hasMore:boolean};error?:string;status?:number};
type Request=(query?:Query,command?:unknown)=>Promise<Result>;
const date=(value:string)=>new Date(value).toLocaleDateString('en-IN',{day:'numeric',month:'short',year:'numeric',timeZone:'Asia/Kolkata'});
const stamp=(value:string)=>new Date(value).toLocaleString('en-IN',{day:'numeric',month:'short',year:'numeric',hour:'2-digit',minute:'2-digit',timeZone:'Asia/Kolkata'})+' IST';
const actionLabel={keep:'Kept my view',dismiss:'Dismissed update',defer:'Deferred review',revise:'Revised my thesis',reopen:'Reopened review'};
const filterLabel={pending:'To review',deferred:'Later',reviewed:'Reviewed',all:'All updates'};
const stateLabel={unreviewed:'To review',kept:'View kept',dismissed:'Dismissed',deferred:'Deferred',revised:'Thesis revised'};

export default function InboxWorkspace({local,authConfig,initialUpdate,initialFilter}:{local:boolean;authConfig:{url:string;key:string}|null;initialUpdate?:string;initialFilter:InboxFilter}){
 const [session,setSession]=useState<Session|null>(null),[ready,setReady]=useState(local||!authConfig),[error,setError]=useState('');
 const url=authConfig?.url,key=authConfig?.key;
 useEffect(()=>{
  if(local||!url||!key)return;
  let active=true;
  try{
   const client=browserAuth(url,key);
   const {data:{subscription}}=client.auth.onAuthStateChange((_event,next)=>{if(active){setSession(next);setReady(true);}});
   client.auth.getSession().then(({data})=>{if(active){setSession(data.session);setReady(true);}}).catch(()=>{if(active){setError('Could not restore your session. Sign in again.');setReady(true);}});
   return()=>{active=false;subscription.unsubscribe();};
  }catch{setError('Browser session storage is unavailable.');setReady(true);}
 },[local,url,key]);
 return <div className="private-shell"><WorkspaceHeader active="inbox" local={local}/><main id="workspace-content" className="private-content inbox-content">
  <div className="page-heading"><div><span className="eyebrow">EVIDENCE → REASONING → RESPONSE</span><h1>Review inbox<span className="heading-dot">.</span></h1><p>Understand the change. Keep a record of what it means to you.</p></div><a className="text-button" href="/workspace">← Overview</a></div>
  {error&&<p className="account-alert error" role="alert">{error}</p>}
  {!ready?<p role="status">Restoring your session…</p>:!local&&!session?<section className="panel reading"><h2>Your reviews are private.</h2><p>Sign in with your invited account to read updates and save your responses.</p><a className="button-link primary" href="/workspace">Sign in →</a></section>:<InboxSession key={local?'local':session!.user.id} local={local} token={session?.access_token} initialUpdate={initialUpdate} initialFilter={initialFilter}/>}
 </main><footer className="app-footer"><span>AlphaTrace · Evidence first. Decisions remembered.</span><span>Topic matching · No automatic investment conclusions</span></footer></div>;
}

function InboxSession({local,token,initialUpdate,initialFilter}:{local:boolean;token?:string;initialUpdate?:string;initialFilter:InboxFilter}){
 const tokenRef=useRef(token);tokenRef.current=token;
 const active=useRef(true),generation=useRef(0);
 const [list,setList]=useState<InboxList|null>(null),[filter,setFilter]=useState<InboxFilter>(initialFilter),[selected,setSelected]=useState(initialUpdate??'');
 const [loading,setLoading]=useState(true),[error,setError]=useState(''),[syncMessage,setSyncMessage]=useState(''),[moreSync,setMoreSync]=useState(false),[dirty,setDirty]=useState(false),[detailReload,setDetailReload]=useState(0);
 const [checkedAt,setCheckedAt]=useState<string|null>(null),[updatesWaiting,setUpdatesWaiting]=useState(false);
 const request=useCallback<Request>(async(query={},command)=>{
  if(local)return localInbox(query,command);
  const params=new URLSearchParams();for(const [key,value] of Object.entries(query))if(value!==undefined)params.set(key,String(value));
  const response=await fetch('/api/inbox?'+params.toString(),{method:command?'POST':'GET',cache:'no-store',...(!command||(command as {action?:string}).action==='sync'?{signal:AbortSignal.timeout(20_000)}:{}),headers:{Authorization:`Bearer ${tokenRef.current}`,...(command?{'Content-Type':'application/json'}:{})},...(command?{body:JSON.stringify(command)}:{})});
  return {...await response.json(),status:response.status};
 },[local]);
 async function load(nextFilter:InboxFilter=filter,sync=false,cursor?:string){
  const run=++generation.current;setLoading(true);setError('');
  const current=()=>active.current&&run===generation.current;
  try{
   const result=await request({filter:nextFilter,cursor});if(!current())return;
   if(result.error||!result.inbox)throw Error(result.error??'Could not load reviews.');
   setList(previous=>cursor&&previous?{...result.inbox!,items:[...previous.items,...result.inbox!.items]}:result.inbox!);
   if(!cursor&&!sync)setSelected(previous=>previous||result.inbox!.items[0]?.update_id||'');
   if(sync){
    const updated=await request({}, {action:'sync'});if(!current())return;
    if(updated.error||!updated.sync)throw Error(updated.error??'Saved reviews are available, but new updates could not be checked.');
    setMoreSync(updated.sync.hasMore);setSyncMessage(`${updated.sync.added} new review items added.${updated.sync.hasMore?' More publications are waiting; check the next batch.':''}`);
    const refreshed=await request({filter:nextFilter});if(!current())return;
    if(refreshed.error||!refreshed.inbox)throw Error(refreshed.error??'Reload the inbox to see new items.');
    setList(refreshed.inbox);setSelected(previous=>previous||refreshed.inbox!.items[0]?.update_id||'');setDetailReload(n=>n+1);setCheckedAt(new Date().toISOString());setUpdatesWaiting(false);
   }
  }catch(e){if(current())setError((e as Error).message);}finally{if(current())setLoading(false);}
 }
 useEffect(()=>{active.current=true;void load(filter,true);return()=>{active.current=false;generation.current++;};
 // A token refresh must not reset a draft or repeat the initial catch-up.
 // eslint-disable-next-line react-hooks/exhaustive-deps
 },[request]);
 const automatic=useAutoUpdates(async()=>{
  const run=generation.current;
  const synced=await request({}, {action:'sync'});
  if(synced.error||!synced.sync)return false;
  const result=await request({filter});
  if(result.error||!result.inbox)return false;
  if(!active.current||run!==generation.current)return null;
  setMoreSync(synced.sync.hasMore);
  // Never replace the current evidence, response, selection or paginated list.
  // The user applies the fresh snapshot when ready, including during editing.
  const changed=inboxHasChanges(list,result.inbox);
  setUpdatesWaiting(changed||synced.sync.hasMore);
  setCheckedAt(new Date().toISOString());
  return true;
 },loading);
 function pick(id:string){if(dirty)return;setSelected(id);window.history.replaceState(null,'',`/inbox?filter=${filter}&update=${id}`);}
 return <>
  <div className="inbox-summary"><div><strong>{list?.counts.pending??'—'}</strong><span>awaiting your review</span></div><div><strong>{list?.counts.deferred??'—'}</strong><span>set aside for later</span></div><div><strong>{list?.counts.reviewed??'—'}</strong><span>marked reviewed</span></div><button disabled={loading||dirty} onClick={()=>void load(filter,true)}>{loading?'Checking…':moreSync?'Check next batch':'Check for updates'}</button></div>
  <AutoUpdateStatus state={automatic} checkedAt={checkedAt} failed={!!error}/>
  {updatesWaiting&&<div className="inbox-update-notice" role="status"><div><strong>New or changed reviews are ready.</strong><p>{dirty?'Your unsaved response is safe. Save or discard it to load the updated inbox.':'Load the latest queue, including new publications, corrections and reviews now due. Your open review stays selected.'}</p></div><button className="primary" disabled={loading||dirty} onClick={()=>void load(filter,true)}>Load updates</button></div>}
  <p className="fine-print">Automatic checks keep your open review in place. Each check adds up to 50 published versions; existing reviews remain after you unfollow a company.</p>
  {error&&<div className="account-alert error" role="alert">{error}<button disabled={loading||dirty} onClick={()=>void load(filter,true)}>Retry</button></div>}
  {syncMessage&&<p className="fine-print" role="status">{syncMessage}</p>}
  {dirty&&<p className="amber-text" role="status">Save or discard your response before changing reviews.</p>}
  <nav className="tabs" aria-label="Review filters">{inboxFilters.map(f=><button key={f} disabled={loading||dirty} className={filter===f?'selected':''} aria-pressed={filter===f} onClick={()=>{setFilter(f);setSelected('');window.history.replaceState(null,'',`/inbox?filter=${f}`);void load(f);}}>{filterLabel[f]} {list?.counts[f]??''}</button>)}</nav>
  <div className="inbox-layout"><aside className="panel inbox-list" aria-label="Published updates">
   {!list&&loading?<p role="status" className="inbox-empty">Loading your reviews…</p>:!list?<div className="inbox-empty"><h2>Inbox unavailable</h2><p>Retry to read your saved reviews.</p></div>:list.items.length===0?<div className="inbox-empty"><span className="eyebrow">{list.counts.all?'NOTHING IN THIS VIEW':'WAITING FOR REVIEWED RESEARCH'}</span><h2>{list.counts.all?'No reviews here.':'Your first update will appear here.'}</h2><p>{list.counts.all?'Choose another filter to see saved responses or deferred items.':'Follow a company or add a holding. Its published research will appear here, even without personal assumptions. No items does not establish that nothing has changed.'}</p><a href="/workspace?view=companies" className="text-button">Manage companies →</a><a href="/research" className="text-button">Check research coverage →</a></div>:list.items.map(item=><button className={`inbox-row ${selected===item.update_id?'active':''}`} key={item.update_id} aria-current={selected===item.update_id?'true':undefined} disabled={dirty||loading} onClick={()=>pick(item.update_id)}><span className="inbox-row-meta"><strong>{item.symbol}</strong><small>v{item.publication_version} · {date(item.period_end)}</small></span><span className="inbox-row-title">{item.title}</span><span className="inbox-row-meta"><span className={`badge ${item.state==='unreviewed'?'lavender':''}`}>{item.state==='deferred'&&Date.parse(item.revisit_at!)<=Date.now()?'Due for review':stateLabel[item.state]}</span><small>{item.match_count?`${item.match_count} topic ${item.match_count===1?"match":"matches"}`:'Company update'}</small></span>{item.withdrawn_at&&<span className="amber-text">Publication withdrawn</span>}{item.newer_update_id&&<small>A newer version is available</small>}</button>)}
   {list?.nextCursor&&<button className="inbox-more" disabled={loading||dirty} onClick={()=>void load(filter,false,list.nextCursor!)}>Load more reviews</button>}
  </aside><section className="inbox-detail-column">{selected?<ReviewDetail key={selected} updateId={selected} request={request} reloadKey={detailReload} onDirty={setDirty} onSaved={()=>void load(filter)}/>:<div className="panel inbox-empty inbox-start"><span className="eyebrow">A RECORD OF YOUR THINKING</span><h2>Select an update to investigate.</h2><p>Read the source, compare it with your recorded assumptions, then decide whether your view needs to change.</p><ol><li>Inspect the published evidence.</li><li>Review matching topics in your own wording.</li><li>Save your response and its reason.</li></ol><p className="fine-print">Matching identifies a shared topic. It does not evaluate your baseline, expected direction, time horizon or review condition.</p></div>}</section></div>
  <JournalExport local={local} token={token}/>
 </>;
}

function ReviewDetail({updateId,request,reloadKey,onDirty,onSaved}:{updateId:string;request:Request;reloadKey:number;onDirty:(v:boolean)=>void;onSaved:()=>void}){
 const [data,setData]=useState<InboxDetail|null>(null),[loading,setLoading]=useState(true),[busy,setBusy]=useState(false),[error,setError]=useState(''),[message,setMessage]=useState('');
 const [note,setNote]=useState(''),[action,setAction]=useState<ReviewAction>('keep'),[deferUntil,setDeferUntil]=useState(''),[dirty,setDirty]=useState(false),[uncertain,setUncertain]=useState(false),[conflict,setConflict]=useState(false);
 const alive=useRef(true),pending=useRef<ReviewCommand|null>(null);
 const clear=()=>{setNote('');setDeferUntil('');setAction('keep');setDirty(false);setUncertain(false);setConflict(false);pending.current=null;};
 useEffect(()=>{
  alive.current=true;let active=true;setLoading(true);setError('');
  request({update:updateId}).then(result=>{if(!active)return;if(result.error||!result.detail)throw Error(result.error??'Could not load this review.');setData(result.detail);}).catch(e=>{if(active)setError(e.message);}).finally(()=>{if(active)setLoading(false);});
  return()=>{active=false;alive.current=false;};
 },[updateId,request,reloadKey]);
 useEffect(()=>()=>onDirty(false),[onDirty]);
 useEffect(()=>{onDirty(dirty||busy);const warn=(event:BeforeUnloadEvent)=>{event.preventDefault();event.returnValue='';};if(dirty||busy)window.addEventListener('beforeunload',warn);return()=>window.removeEventListener('beforeunload',warn);},[dirty,busy,onDirty]);
 function changed(){setDirty(true);setMessage('');pending.current=null;}
 async function reload(){setBusy(true);try{const result=await request({update:updateId});if(!alive.current)return;if(result.error||!result.detail)throw Error(result.error??'Could not reload.');setData(result.detail);clear();setError('');}catch(e){if(alive.current)setError((e as Error).message);}finally{if(alive.current)setBusy(false);}}
 async function save(){
  if(!data||busy||conflict)return;setBusy(true);setError('');setMessage('');
  pending.current??={updateId,commandId:crypto.randomUUID(),version:data.item.version,action,note,deferUntil:action==='defer'?deferUntil:null,thesisVersion:action==='revise'?data.currentThesisVersion:null};
  try{const result=await request({},pending.current);if(!alive.current)return;
   if(result.error||!result.detail){if(result.status===409){setConflict(true);pending.current=null;}else if(!result.status||result.status>=500)setUncertain(true);else pending.current=null;throw Error(result.error??'Save not confirmed.');}
   setData(result.detail);clear();setMessage('Response saved. Your previous reasoning and the exact evidence version are preserved.');onSaved();
  }catch(e){if(alive.current){if(e instanceof TypeError)setUncertain(true);setError((e as Error).message);}}finally{if(alive.current)setBusy(false);}
 }
 async function older(){if(!data)return;setBusy(true);try{const result=await request({update:updateId,before:data.events.at(-1)?.version});if(!alive.current)return;if(result.error||!result.detail)throw Error(result.error??'Could not load history.');setData({...data,events:[...data.events,...result.detail.events],hasMoreEvents:result.detail.hasMoreEvents});}catch(e){if(alive.current)setError((e as Error).message);}finally{if(alive.current)setBusy(false);}}
 const locked=loading||busy||uncertain||conflict;
 if(loading&&!data)return <div className="panel inbox-empty" role="status">Loading the evidence and your review history…</div>;
 const publication=data?.publication,sourceState=data?freshness(data.check):'unchecked';
 const changedSource=!!data&&data.check?.status==='success'&&!data.publication.payload.sources.every(s=>data.check!.fingerprints.includes(s.sha256));
 const researchLink=data?`/research?symbol=${data.item.symbol}&update=${updateId}&review=${updateId}`:'#';
 return <article className="panel inbox-detail" aria-busy={busy||loading}>
  {error&&<div className="account-alert error" role="alert">{error}<button disabled={busy} onClick={()=>void reload()}>{dirty?'Discard edits and reload':'Reload review'}</button>{uncertain&&<button disabled={busy} onClick={()=>void save()}>Retry the same response</button>}</div>}
  {message&&<div className="account-alert success" role="status">{message}</div>}
  {data&&publication&&<><header className="inbox-detail-heading"><div className="review-version-line"><span className={`badge ${data.item.state==='unreviewed'?'lavender':''}`}>{stateLabel[data.item.state]}</span><span>{data.events.length?`Last response: ${stamp(data.events[0].created_at)}`:'No response yet'}</span></div><span className="eyebrow">{data.item.symbol} / PUBLICATION {publication.version}</span><h2>{publication.payload.title}</h2><p>Quarter ended {date(publication.payload.periodEnd)} · Published {date(publication.published_at)}</p><a href={researchLink}>Company research & thesis ↗</a></header><nav className="review-jump-nav" aria-label="This review"><a href="#review-evidence">01 Evidence</a><a href="#review-thesis">02 Your reasoning</a><a href="#review-response">03 Respond</a><a href="#review-history">History ↓</a></nav>
   {publication.withdrawn_at&&<div className="account-alert error">Withdrawn: {publication.withdrawal_reason}. This version is retained for history and must not be treated as current evidence.</div>}
   {data.item.newer_update_id&&<div className="account-alert">This version has been superseded. <a href={`/research?symbol=${data.item.symbol}&update=${data.item.newer_update_id}`}>Read the newer publication ↗</a>. Responses here remain attached to version {publication.version}.</div>}
   {(sourceState!=='checked'||changedSource)&&<div className="account-alert">{changedSource?'Source contents have changed since this publication.':sourceState==='failed'?'The latest source check failed.':sourceState==='stale'?'Source checking is overdue.':'Sources have not been checked.'} {data.check?.message} <a href={researchLink}>Inspect source status ↗</a></div>}
   <section id="review-evidence" className="inbox-section evidence-review"><UpdateEvidence payload={publication.payload}/></section>
   <section id="review-thesis" className="inbox-section"><span className="eyebrow">02 / YOUR REASONING</span><h3>Why this reached you</h3><p>Included because this company was in your {data.snapshot.reason} when this item was created on {date(data.item.created_at)}.</p>{data.snapshot.matches.length?<><h3>Topics in your recorded assumptions</h3><p className="fine-print">Matched only against your metric/topic field at inbox creation, using a fixed alias list. This does not establish support, a contradiction or a breached threshold.</p>{data.snapshot.matches.map(match=><div className="inbox-assumption" key={match.assumption.id}><blockquote>{match.assumption.statement}</blockquote><p>Recorded topic: <strong>{match.assumption.metric}</strong> · Matched: {match.metrics.map(metricLabel).join(', ')}</p><small>From your thesis revision {data.snapshot.thesis?.version}</small><details><summary>Original assumption details</summary><p>Baseline: {match.assumption.baseline||'Not specified'}<br/>Expected direction: {match.assumption.direction}<br/>Horizon: {match.assumption.horizon||'Not specified'}<br/>Revisit when: {match.assumption.reviewCondition||'Not specified'}</p></details></div>)}</>:<p>No explicit metric/topic match was found. This is a general company update; it may still matter to you.</p>}{data.snapshot.thesis&&<details className="inbox-original"><summary>Your original research at inbox creation · revision {data.snapshot.thesis.version}</summary><p>{data.snapshot.thesis.payload.thesis||'No investment case recorded.'}</p><p>{data.snapshot.thesis.payload.risks||'No risks recorded.'}</p>{data.snapshot.thesis.payload.assumptions.map(a=><p key={a.id}>{a.statement}</p>)}</details>}{data.currentThesisVersion!==data.snapshot.thesis?.version&&data.currentThesisVersion!==null&&<p className="fine-print">Your thesis is now at revision {data.currentThesisVersion}. The original match above remains unchanged.</p>}</section>
   <section id="review-response" className="inbox-section inbox-response"><span className="eyebrow">03 / YOUR RESPONSE</span><h3>{stateLabel[data.item.state]}</h3>{data.item.revisit_at&&<p>Revisit {stamp(data.item.revisit_at)}. It returns to “To review” when due; no email is sent.</p>}
    <fieldset disabled={locked}><label htmlFor="review-action">How do you want to respond?</label><select id="review-action" value={action} onChange={e=>{setAction(e.target.value as ReviewAction);changed();}}><option value="keep" disabled={!!publication.withdrawn_at}>Keep my view</option><option value="dismiss">Dismiss this update</option><option value="defer">Revisit later</option><option value="revise" disabled={!!publication.withdrawn_at}>Record a revised thesis</option><option value="reopen">Reopen for review</option></select>
    {action==='defer'&&<><label htmlFor="review-date">Revisit date · 9 am IST</label><input id="review-date" type="date" value={deferUntil} onChange={e=>{setDeferUntil(e.target.value);changed();}}/></>}
    {action==='revise'&&<div className="inbox-revise-guide"><p>{data.canRecordRevision?`Thesis revision ${data.currentThesisVersion} links this publication and is ready to record.`:'First save a new thesis revision and link this exact publication under “Evidence I considered”. Return here using “Return to review”, then record your response.'}</p><a href={researchLink}>Open my thesis and this evidence →</a></div>}
    <label htmlFor="review-reason">Why? Preserve your reasoning.</label><textarea id="review-reason" maxLength={2000} value={note} onChange={e=>{setNote(e.target.value);changed();}} placeholder="What does this update change—or leave unchanged—in your view?"/></fieldset>
    <div className="research-actions"><button className="primary" disabled={locked||note.trim().length<3||(action==='defer'&&!deferUntil)||(action==='revise'&&!data.canRecordRevision)||(!!publication.withdrawn_at&&['keep','revise'].includes(action))} onClick={()=>void save()}>{busy?'Saving…':'Save response'}</button>{dirty&&!uncertain&&<button disabled={busy} onClick={()=>{clear();setError('');}}>Discard response</button>}</div><p className="fine-print">Keeping or dismissing an update never edits your thesis. A revised response records an existing, explicitly saved thesis revision.</p>
   </section>
   <section id="review-history" className="inbox-section"><span className="eyebrow">YOUR DECISION TRAIL</span><h3>{data.events.length?'Saved responses':'No response recorded yet'}</h3>{data.events.map(event=><div className="inbox-event" key={event.version}><strong>{actionLabel[event.action]}</strong><small>{stamp(event.created_at)} · response {event.version} · publication v{publication.version}</small><p>{event.note}</p>{event.revisit_at&&<small>Revisit scheduled for {stamp(event.revisit_at)}</small>}{event.thesis_revision&&<details><summary>Saved thesis revision {event.thesis_revision.version}</summary><p>{event.thesis_revision.payload.thesis}</p><p>{event.thesis_revision.payload.risks}</p>{event.thesis_revision.payload.assumptions.map(a=><p key={a.id}>{a.statement}<br/>Topic: {a.metric||'Not specified'}<br/>Baseline: {a.baseline||'Not specified'}<br/>Direction: {a.direction}<br/>Horizon: {a.horizon||'Not specified'}<br/>Review condition: {a.reviewCondition||'Not specified'}</p>)}</details>}</div>)}{data.hasMoreEvents&&<button disabled={locked} onClick={()=>void older()}>Load older responses</button>}</section>
  </>}
 </article>;
}
