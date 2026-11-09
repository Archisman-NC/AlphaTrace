'use client';
import { useCallback,useEffect,useRef,useState } from 'react';
import { localInbox } from '@/app/inbox/actions';
import type { InboxList,InboxItem } from '@/lib/inbox-contract';
import type { Portfolio } from '@/lib/portfolio-input';
import type { CompanyCoverage } from '@/lib/research-view';
import { coverageStatus,trackedCoverage,displayDate } from '@/lib/dashboard-view';
import SourceMonitor from './source-monitor';
import JournalExport from './journal-export';
import AutoUpdateStatus,{useAutoUpdates} from './auto-update-status';

type Reviews={pending:InboxList;deferred:InboxList;all:InboxList};
const reviewLink=(item:InboxItem)=>`/inbox?filter=all&update=${item.update_id}`;
const labels={unreviewed:'To review',kept:'View kept',dismissed:'Dismissed',deferred:'Revisit later',revised:'Thesis revised'};

export default function WorkspaceOverview({portfolio,catalog,local,token,coverageError,onRefreshCoverage}:{portfolio:Portfolio;catalog:CompanyCoverage[]|null;local:boolean;token?:string;coverageError:string;onRefreshCoverage:()=>void}) {
 const [reviews,setReviews]=useState<Reviews|null>(null),[loading,setLoading]=useState(true),[error,setError]=useState(''),[syncError,setSyncError]=useState(''),[hasMore,setHasMore]=useState(false),[checkedAt,setCheckedAt]=useState<string|null>(null);
 const tokenRef=useRef(token);tokenRef.current=token;
 const generation=useRef(0);
 const load=useCallback(async(background=false)=>{
  const run=++generation.current;if(!background){setLoading(true);setError('');setSyncError('');}
  let complete=true;
  const current=()=>run===generation.current;
  async function request(filter?:string) {
   if(local)return filter?localInbox({filter}):localInbox({},{action:'sync'});
   const response=await fetch('/api/inbox'+(filter?'?filter='+filter:''),{method:filter?'GET':'POST',cache:'no-store',signal:AbortSignal.timeout(20_000),headers:{Authorization:`Bearer ${tokenRef.current}`,...(!filter?{'Content-Type':'application/json'}:{})},...(!filter?{body:JSON.stringify({action:'sync'})}:{})});
   return response.json();
  }
  try {
   try {
    const result=await request();if(!current())return null;
    if(result.error||!result.sync)throw Error('New publications could not be checked. Saved reviews are shown below.');
    setHasMore(result.sync.hasMore);setCheckedAt(new Date().toISOString());setSyncError('');
   } catch {if(!current())return null;complete=false;setSyncError('New publications could not be checked. Saved reviews are shown below.');}
   const results=await Promise.all(['pending','deferred','all'].map(async filter=>{
    const result=await request(filter);if(result.error||!result.inbox)throw Error('Reviews could not be loaded. Retry to see what needs attention.');return result.inbox as InboxList;
   }));
   if(!current())return null;
   setReviews({pending:results[0],deferred:results[1],all:results[2]});setError('');return complete;
  } catch(e) {if(current())setError((e as Error).message);return false;}
  finally {if(current()&&!background)setLoading(false);}
 },[local]);
 useEffect(()=>{void load();return()=>{generation.current++;};},[load]);
 const automatic=useAutoUpdates(()=>load(true),loading);
 const tracked=trackedCoverage(portfolio,catalog),trackedCount=new Set(portfolio.entries.filter(e=>e.quantity!==null||e.followed).map(e=>e.symbol)).size;
 const published=tracked?.filter(c=>c.status==='published').length??null;
 const gaps=tracked?.filter(c=>c.status!=='published')??[];
 const counts=reviews?.all.counts;
 const clear=!!counts&&!counts.pending&&!loading&&!error&&!syncError&&!hasMore;
 const latest=reviews?.all.items.slice(0,4)??[];
 const pending=reviews?.pending.items.slice(0,4)??[];
 function rows(items:InboxItem[]) {return <div className="overview-updates">{items.map(item=><a href={reviewLink(item)} className="overview-update" key={item.update_id}>
  <span className="company-avatar">{item.symbol.slice(0,2)}</span><span className="overview-update-copy"><span className="overview-update-meta">{item.symbol}<span>v{item.publication_version} · Quarter ended {displayDate(item.period_end)}</span></span><strong>{item.title}</strong><small>{item.match_count?`${item.match_count} recorded assumption topic${item.match_count===1?'':'s'} matched`:'Company update · no matched assumption'}{item.withdrawn_at?' · Withdrawn':item.newer_update_id?' · Newer version available':''}</small></span><span className="overview-update-end"><span className={`badge ${item.state==='unreviewed'?'lavender':''}`}>{item.state==='deferred'&&Date.parse(item.revisit_at!)<=Date.now()?'Due for review':labels[item.state]}</span><span aria-hidden="true">↗</span></span>
 </a>)}</div>;}
 return <>
  <div className="overview-heading"><div><span className="eyebrow">YOUR RESEARCH, IN FOCUS</span><h1>Overview<span className="heading-dot">.</span></h1><p>What needs attention. What you know. What’s still missing.</p></div><button disabled={loading} onClick={()=>{onRefreshCoverage();void load();}}>{loading?'Refreshing…':'Refresh workspace'}<span aria-hidden="true">↻</span></button></div>
  <AutoUpdateStatus state={automatic} checkedAt={checkedAt} failed={!!error||!!syncError}/>
  {(error||syncError)&&<div className="account-alert error" role="alert"><span>{error||syncError}{reviews&&error?' Displaying the last loaded snapshot.':''}</span><button disabled={loading} onClick={()=>void load()}>Retry</button></div>}
  {hasMore&&!syncError&&<div className="account-alert"><span>More publications are waiting. Refresh to load the next batch before treating this overview as complete.</span></div>}
  <div className="overview-stats" aria-label="Workspace summary">
   <a href="/inbox?filter=pending"><span>Needs review <span aria-hidden="true">↗</span></span><strong>{counts?.pending??'—'}</strong><small>Includes deferred reviews now due</small></a>
   <a href="/workspace?view=companies"><span>Companies tracked <span aria-hidden="true">↗</span></span><strong>{trackedCount.toString().padStart(2,'0')}</strong><small>Holdings and watchlist combined</small></a>
   <a href="/research"><span>Published coverage <span aria-hidden="true">↗</span></span><strong>{published??'—'}<em> / {trackedCount}</em></strong><small>Of the companies you track</small></a>
   <a href="/inbox?filter=reviewed"><span>Reviewed updates <span aria-hidden="true">↗</span></span><strong>{counts?.reviewed??'—'}</strong><small>Your responses and evidence, preserved</small></a>
  </div>
  <div className="dashboard-grid"><div className="dashboard-main">
   <section className={`attention-brief ${clear?'is-clear':''}`} aria-labelledby="attention-title">
    <div className="attention-mark" aria-hidden="true">{loading?'·':clear?'✓':'↗'}</div><div><span className="eyebrow">NEXT IN YOUR WORKSPACE</span><h2 id="attention-title">{loading?'Checking your review queue…':error?'Your review queue is unavailable':syncError||hasMore?'Your update check is incomplete':!trackedCount?'Choose your first company':counts?.pending?`${counts.pending} update${counts.pending===1?'':'s'} ready for your review`:'Your saved review queue is clear.'}</h2><p>{!trackedCount?'Add a holding or follow a company to bring its published research into your workspace.':counts?.pending?'Compare the figures, revisit your original reasoning, and record what this changes for you.':gaps.length?`${gaps.length} of your tracked companies ${gaps.length===1?'does':'do'} not have current published research. An empty queue does not mean nothing has changed.`:'Your recorded responses are below. Source checks and new publications may still be pending.'}</p>{pending[0]?<a className="button-link primary" href={reviewLink(pending[0])}>Review {pending[0].symbol} update →</a>:!trackedCount?<a className="button-link primary" href="/workspace?view=companies">Add companies →</a>:<a className="text-button" href="/inbox?filter=all">Open review history →</a>}</div>
   </section>
   <section className="panel overview-feed" aria-labelledby="updates-title"><div className="panel-heading"><h2 id="updates-title">{pending.length?'Needs your attention':'Your latest updates'}</h2><a href={pending.length?'/inbox':'/inbox?filter=all'}>View inbox ↗</a></div>
    {!reviews&&loading?<div className="dashboard-loading" role="status"><span/><span/><span/><p>Loading your published updates…</p></div>:error&&!reviews?<div className="dashboard-empty"><h3>Updates couldn’t be loaded.</h3><p>Retry above to read your saved review queue.</p></div>:pending.length?rows(pending):latest.length?rows(latest):<div className="dashboard-empty"><h3>No published updates in your inbox yet.</h3><p>Published research for companies you hold or follow will appear here. Check the coverage list to see what is available.</p><a href="/research">Explore research coverage →</a></div>}
    <div className="overview-feed-footer"><span>{checkedAt?`Inbox refreshed ${new Date(checkedAt).toLocaleTimeString('en-IN',{hour:'2-digit',minute:'2-digit',timeZone:'Asia/Kolkata'})} IST`:'Inbox has not been refreshed'}</span><span>Checks for published updates, not new source documents</span></div>
   </section>
   {!!reviews?.deferred.items.length&&<section className="panel overview-later"><div className="panel-heading"><h2>Set aside for later</h2><a href="/inbox?filter=deferred">All {counts?.deferred} ↗</a></div>{[...reviews.deferred.items].sort((a,b)=>Date.parse(a.revisit_at!)-Date.parse(b.revisit_at!)).slice(0,3).map(item=><a key={item.update_id} href={reviewLink(item)}><strong>{item.symbol}</strong><span>{item.title}</span><small>Revisit {displayDate(item.revisit_at!)}</small></a>)}</section>}
  </div><aside className="dashboard-context">
   <section className="panel coverage-panel" aria-labelledby="coverage-title"><div className="panel-heading"><h2 id="coverage-title">Your companies</h2><a href="/workspace?view=companies">Manage ↗</a></div><div className="coverage-intro"><strong>{published??'—'} <span>of {trackedCount} with published research</span></strong><p>Coverage and source checks are separate from your review status.</p></div>
    {coverageError?<div className="dashboard-empty" role="alert"><p>{coverageError}</p><button onClick={onRefreshCoverage}>Retry coverage</button></div>:!tracked?<p className="dashboard-empty" role="status">Loading research coverage…</p>:!tracked.length?<div className="dashboard-empty"><p>Your companies will appear here when you add them.</p></div>:<div className="coverage-company-list">{tracked.map(c=>{const status=coverageStatus(c),entry=portfolio.entries.find(e=>e.symbol===c.symbol)!;return <a href={`/research?symbol=${c.symbol}`} key={c.symbol}><div><strong>{c.name}</strong><small>{c.symbol} · {entry.quantity!==null?'Holding':'Watchlist'}{entry.quantity!==null&&entry.followed?' + watchlist':''}</small></div><span className={`badge ${status.tone}`}>{status.label}</span><small>{c.status==='published'?`Quarter ended ${displayDate(c.periodEnd!)} · v${c.version}`:'Awaiting a reviewed publication'}{c.check?` · Last check ${displayDate(c.check.checked_at)}`:''}</small></a>;})}</div>}
    <div className="coverage-footnote"><span className="eyebrow">THE RESEARCH UNIVERSE</span><p>{catalog?catalog.filter(c=>c.status==='published').length:'—'} of 25 companies published. Check monitoring for the latest source-check status.</p><a href="/research">See all coverage →</a></div>
   </section>
   <SourceMonitor local={local} token={token} compact/>
   <section className="overview-note"><span className="eyebrow">A RECORD, NOT A RECOMMENDATION</span><p>AlphaTrace keeps evidence beside your reasoning. A topic match is a prompt to investigate, not a verdict on your investment.</p></section>
  </aside></div>
  <JournalExport local={local} token={token}/>
 </>;
}
