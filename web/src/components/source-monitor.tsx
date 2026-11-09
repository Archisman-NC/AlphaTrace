'use client';
import { useCallback,useEffect,useRef,useState } from 'react';
import { localMonitor } from '@/app/monitoring/actions';
import { workerOnline,workerLabel } from '@/lib/worker-status';
import ManualTcsImport from './manual-tcs-import';
import ProviderImports from './provider-imports';
import { monitorCadence,monitorHeartbeatMs,monitorState,monitorGuidance,retryPending,type MonitorView,type MonitorSymbol } from '@/lib/monitor-contract';

const time=(value:string|null)=>value?new Date(value).toLocaleString('en-IN',{day:'numeric',month:'short',hour:'2-digit',minute:'2-digit',timeZone:'Asia/Kolkata'})+' IST':'Not yet';
export default function SourceMonitor({local,token,compact=false}:{local:boolean;token?:string;compact?:boolean}) {
 const [data,setData]=useState<MonitorView|null>(null),[error,setError]=useState(''),[message,setMessage]=useState(''),[busy,setBusy]=useState(false),[loading,setLoading]=useState(true);
 const tokenRef=useRef(token);tokenRef.current=token;
 const sequence=useRef(0),working=useRef(false);
 const request=useCallback(async(command?:{symbol:MonitorSymbol;action:string})=>{
  const run=++sequence.current;if(!command)setMessage('');if(command){working.current=true;setBusy(true);}setError('');
  try {
   const result=local?await localMonitor(command):await fetch('/api/monitor',{method:command?'POST':'GET',cache:'no-store',headers:{Authorization:`Bearer ${tokenRef.current}`,...(command?{'Content-Type':'application/json'}:{})},...(command?{body:JSON.stringify(command)}:{})}).then(r=>r.json());
   if(run!==sequence.current)return;
   if(result.error||!result.monitor)throw Error(result.error??'Monitoring status is unavailable.');
   setData(result.monitor);setLoadedAt(Date.now());
   if(command)setMessage(command.action==='pause'?`${command.symbol} paused. Any check already running may finish.`:`${command.symbol} queued. The next available worker pass will pick it up.`);
  }catch(e){if(run===sequence.current)setError((e as Error).message);}
  finally{if(run===sequence.current){setBusy(false);setLoading(false);working.current=false;}}
 },[local]);
 useEffect(()=>{
  void request();const timer=setInterval(()=>{if(document.visibilityState==='visible'&&!working.current)void request();},30_000);
  return()=>{sequence.current++;clearInterval(timer);};
 },[request]);
 const [loadedAt,setLoadedAt]=useState(Date.now());
 const now=data?Date.parse(data.serverNow)+Math.max(0,Date.now()-loadedAt):Date.now();
 const online=!!data?.targets.some(t=>workerOnline(t,now,monitorHeartbeatMs));
 const cloud=!!data?.targets.some(t=>t.worker_mode==='cloud');
 const label=workerLabel(data?.targets??[],now,monitorHeartbeatMs);
 const enabled=data?.targets.filter(t=>t.enabled).length??0;
 const failed=data?.targets.filter(t=>t.latest?.state==='failed').length??0;
 if(compact)return <section className="panel monitor-compact" aria-label="Source monitoring"><div className="panel-heading"><h2>Official evidence checks</h2><a href="/monitoring">Details ↗</a></div><div className="monitor-compact-body">{error?<p role="alert">Status unavailable. <a href="/monitoring">Check monitoring →</a></p>:!data?<p role="status">Loading monitor status…</p>:<><span className={`badge ${!online&&enabled?'amber':'neutral'}`}>{!enabled?'Not enabled':label}</span><p>{enabled} of 25 companies scheduled · {failed?`${failed} latest check${failed===1?'':'s'} failed`:'Infosys discovery · TCS pinned sources'}</p><small>{monitorCadence}</small></>}</div></section>;
 return <>
  {data?.isOperator&&<ProviderImports local={local} token={token}/>}
  <details className="official-monitor-details" open={data?.isOperator?undefined:true}><summary>Supporting official-source checks and history</summary>
  <div className="overview-heading"><div><span className="eyebrow">KNOW WHAT WAS CHECKED</span><h1>Official-source checks<span className="heading-dot">.</span></h1><p>Supporting evidence checks are separate from IndianAPI change monitoring.</p></div><button disabled={busy||loading} onClick={()=>void request()}>Refresh status ↻</button></div>
  {error&&<div className="account-alert error" role="alert"><span>{error}{data?' Showing the last loaded status.':''}</span><button disabled={busy} onClick={()=>void request()}>Retry</button></div>}
  {message&&<div className="account-alert success" role="status">{message}</div>}
  {!data?(!error&&<div className="panel dashboard-loading" role="status"><span/><span/><p>Loading source-check history…</p></div>):<>
   {data.isOperator&&<div className="import-entry"><div><strong>Private data imports</strong><p>Review IndianAPI responses against official filings for five pilot companies.</p></div><a className="button-link" href="/imports">Open imports ↗</a></div>}
   <div className="monitor-service"><div><span className={`monitor-indicator ${online?'online':''}`} aria-hidden="true"/><strong>{label}</strong></div><span>{monitorCadence}</span><span>{enabled} of 25 companies scheduled</span></div>
   <div className="monitor-scope"><h2>{cloud?'Checks run in the cloud.':'Checks run on this computer.'}</h2><p>{cloud?'An hourly cloud dispatcher picks up due work within its 60-minute scheduling window. The dates below show when work becomes eligible, not an exact execution time. Your computer can be off.':'The runner must stay open and the computer must be awake and connected. Missed checks run after it returns.'} Infosys checks the official SEC feed for recent results filings and validates the latest available package. TCS checks pinned company documents and does not discover new releases. These adapters cover two companies; provider monitoring for five pilot companies is shown separately above.</p><p>Changed, validated evidence creates a draft. A person must check the originals and approve publication before it reaches users. Failed checks never refresh successful-source dates.</p></div>
   <section className="panel monitor-targets" aria-labelledby="monitor-targets-title"><div className="panel-heading"><h2 id="monitor-targets-title">Configured sources</h2><span>Infosys SEC feed · TCS pinned results</span></div>{data.targets.map(target=>{const state=monitorState(target,now);return <article key={target.symbol} className="monitor-target"><div className="monitor-target-heading"><div><span className="eyebrow">{target.symbol}</span><h3>{target.symbol==='TCS'?'Tata Consultancy Services':'Infosys'}</h3></div><span className={`badge ${state.tone}`}>{state.label}</span></div><dl><div><dt>Last attempt</dt><dd>{time(target.latest?.started_at??null)}</dd></div><div><dt>Last evidence verification</dt><dd>{time(target.last_successful_check)}<small>May include operator-supplied files</small></dd></div><div><dt>Last successful automated check</dt><dd>{time(target.last_automatic_success??null)}<small>Direct retrieval only</small></dd></div><div><dt>{retryPending(target)?'Retry eligible from':'Next eligible check'}</dt><dd>{target.enabled?time(target.next_due_at):'Paused'}{target.active_run_id&&<small>A check is already running.</small>}</dd></div></dl><p className={target.latest?.state==='failed'?'monitor-failure':''}>{target.latest?.message??'No source check has run yet.'}</p>{target.discovery&&<div className="monitor-discovery"><span className="eyebrow">LATEST DISCOVERED RESULTS FILING</span><p><strong>SEC report date {target.discovery.periodEnd}</strong> · Filed {target.discovery.filedOn}</p><p><a href={target.discovery.url} target="_blank" rel="noreferrer">Open official SEC filing ↗</a> <small>Accession {target.discovery.accession}</small></p><small>Feed checked {time(target.discovery.checkedAt)}. Discovery alone does not confirm document validation or publication.</small></div>}{target.latest?.draft_created&&<p className="fine-print">This attempt prepared a draft. Check the review desk for its current publication status.</p>}<p className="monitor-guidance">{monitorGuidance(target,now)}</p><div className="monitor-actions"><a href={`/research?symbol=${target.symbol}`}>Research & source status ↗</a>{data.isOperator&&<><a href={`/research?admin=1&symbol=${target.symbol}`}>Review desk ↗</a><button disabled={busy||!!target.active_run_id||!target.enabled} onClick={()=>void request({symbol:target.symbol,action:'check'})}>Check now</button><button disabled={busy||(!target.enabled&&!!target.active_run_id)} onClick={()=>void request({symbol:target.symbol,action:target.enabled?'pause':'resume'})}>{target.enabled?'Pause':'Resume'}</button></>}</div>{data.isOperator&&target.symbol==='TCS'&&<ManualTcsImport token={token} onImported={()=>void request()}/>}</article>;})}</section>
   <section className="panel monitor-history" aria-labelledby="monitor-history-title"><div className="panel-heading"><h2 id="monitor-history-title">Recent source checks</h2><span>Latest 20 attempts</span></div>{!data.runs.length?<div className="dashboard-empty"><h3>No checks recorded yet.</h3><p>{data.isOperator?'Resume a company to queue its first check.':'Monitoring has not started. Research already published remains available.'}</p></div>:<ol>{data.runs.map(run=><li key={run.id}><div className="monitor-history-meta"><strong>{run.symbol}</strong><span className={`badge ${run.state==='failed'?'amber':run.state==='running'?'lavender':''}`}>{run.state==='running'?'Started':run.state==='failed'?'Failed':run.draft_created?'Draft prepared':'Check passed'}</span><small>{time(run.started_at)} · Attempt {run.attempt}</small></div><p>{run.message}</p>{run.finished_at&&<small>Finished {time(run.finished_at)}</small>}</li>)}</ol>}</section>
   <p className="fine-print">Status refreshes every 30 seconds while this page is visible. Infosys discovery covers recent SEC results filings, which can lag company announcements. TCS checks do not discover new releases. Monitoring sends no holdings, notes or assumptions to company websites.</p>
  </>}
  </details>
 </>;
}
