'use client';
import { useEffect,useRef,useState,type ReactNode } from 'react';
import { localOnboarding } from '@/app/workspace/onboarding-actions';
import { onboardingTopics,rankFirstReviews,type OnboardingResult,type OnboardingView,type OnboardingTopic,type OnboardingCommand } from '@/lib/onboarding-contract';
import { previewHoldings,readHoldingsCsv,type CsvTable,type ImportRow,type Portfolio } from '@/lib/portfolio-input';
import { universe } from '@/lib/companies';
import type { CompanyCoverage } from '@/lib/research-view';

type Props={local:boolean;token?:string;portfolio:Portfolio;catalog:CompanyCoverage[]|null;requested:boolean;onPortfolio:(p:Portfolio)=>void;children:ReactNode};
export default function GuidedOnboarding(props:Props){
 const {local,token,portfolio,requested}=props;
 const [data,setData]=useState<OnboardingView|null>(null),[error,setError]=useState(''),[closed,setClosed]=useState(false),[retry,setRetry]=useState(0);
 useEffect(()=>{
  let active=true;setError('');
  async function load(){try{
   const result:OnboardingResult=local?await localOnboarding():await fetch('/api/onboarding',{headers:{Authorization:`Bearer ${token}`},cache:'no-store',signal:AbortSignal.timeout(20_000)}).then(r=>r.json());
   if(!result.onboarding)throw Error(result.error||'Setup could not be loaded.');
   if(active)setData(result.onboarding);
  }catch(e){if(active)setError(e instanceof Error?e.message:'Setup could not be loaded.');}}
  void load();return()=>{active=false;};
 },[local,token,retry]);
 const show=!closed&&(requested||data?.state?.status==='active'||(!data?.state&&!portfolio.entries.length));
 if(!data&&!error&&show)return <section className="panel reading" aria-busy="true"><span className="eyebrow">YOUR FIRST RESEARCH WORKSPACE</span><h1>Preparing your setup…</h1><p role="status">Checking saved progress and available research.</p></section>;
 if(error)return <><div className="account-alert error" role="alert"><span>{error} Your portfolio is still available.</span><button onClick={()=>setRetry(n=>n+1)}>Retry setup</button></div>{props.children}</>;
 if(!show||!data)return <><div className="setup-entry"><a className="text-button" href="/workspace?setup=1">{data?.state?.status==='completed'?'Edit guided setup':'Guided setup'} ↗</a></div>{props.children}</>;
 return <SetupWizard {...props} initial={data} onSaved={setData} onClose={()=>{setClosed(true);window.history.replaceState(null,'','/workspace');}}/>;
}

function SetupWizard({local,token,portfolio,catalog,initial,onPortfolio,onClose,onSaved}:Props&{initial:OnboardingView;onSaved:(data:OnboardingView)=>void;onClose:()=>void}){
 const [data,setData]=useState(initial),[step,setStep]=useState(initial.state?.status==='active'?initial.state.step:'companies');
 const [selected,setSelected]=useState<string[]>(portfolio.entries.filter(e=>e.followed).map(e=>e.symbol));
 const [topics,setTopics]=useState<OnboardingTopic[]>(initial.state?.topics??[]),[search,setSearch]=useState('');
 const [csv,setCsv]=useState<CsvTable|null>(null),[columns,setColumns]=useState([0,1]),[excluded,setExcluded]=useState<number[]>([]),[confirmed,setConfirmed]=useState(false);
 const [busy,setBusy]=useState(false),[error,setError]=useState(''),[uncertain,setUncertain]=useState(false);
 const mounted=useRef(true),lock=useRef(false),heading=useRef<HTMLHeadingElement>(null),uploadGeneration=useRef(0);
 useEffect(()=>{mounted.current=true;return()=>{mounted.current=false;uploadGeneration.current++;};},[]);
 useEffect(()=>{heading.current?.focus();},[step]);
 async function save(command:OnboardingCommand){
  if(lock.current||uncertain)return false;lock.current=true;setBusy(true);setError('');
  let rejected=false;
  try{
   let result:OnboardingResult;
   if(local)result=await localOnboarding(command);
   else {const response=await fetch('/api/onboarding',{method:'POST',headers:{Authorization:`Bearer ${token}`,'Content-Type':'application/json'},body:JSON.stringify(command),signal:AbortSignal.timeout(25_000)});result={...await response.json(),status:response.status};}
   if(!mounted.current)return false;
   if(!result.onboarding||!result.portfolio){rejected=!!result.status&&result.status<500&&result.status!==409;throw Error(result.error||'The save could not be confirmed. Reload setup.');}
   setData(result.onboarding);onSaved(result.onboarding);onPortfolio(result.portfolio);return true;
  }catch(e){if(mounted.current){setError(e instanceof Error?e.message:'Could not save setup.');if(!rejected)setUncertain(true);}return false;}
  finally{lock.current=false;if(mounted.current)setBusy(false);}
 }
 const version=data.state?.version??0,locked=busy||uncertain;
 const existing=new Set(portfolio.entries.filter(e=>e.followed).map(e=>e.symbol));
 let preview:ImportRow[]=[],mappingError='';
 if(csv)try{preview=previewHoldings(csv,columns[0],columns[1]);}catch(e){mappingError=(e as Error).message;}
 const accepted=preview.filter(r=>!r.error&&!excluded.includes(r.row));
 const companies=[...universe].sort((a,b)=>Number(catalog?.find(c=>c.symbol===b.symbol)?.status==='published')-Number(catalog?.find(c=>c.symbol===a.symbol)?.status==='published'));
 const filtered=companies.filter(c=>`${c.name} ${c.symbol}`.toLowerCase().includes(search.toLowerCase()));
 const ready=rankFirstReviews(data.reviews,topics),first=ready[0];
 const stepIndex=['companies','topics','review'].indexOf(step);
 async function finish(href?:string){if(await save({action:'complete',version})){if(href)window.location.assign(href);else onClose();}}
 return <section className="onboarding" aria-label="Guided workspace setup">
  <div className="onboarding-top"><span className="eyebrow">MAKE ALPHATRACE YOURS</span><button className="text-button" disabled={locked} onClick={async()=>{if(await save({action:'skip',version}))onClose();}}>Set up later</button></div>
  <ol className="onboarding-progress" aria-label="Setup progress">{['Your companies','Your focus','First review'].map((label,i)=><li key={label} aria-current={i===stepIndex?'step':undefined} className={i<=stepIndex?'reached':''}><span>{i<stepIndex?'✓':`0${i+1}`}</span>{label}</li>)}</ol>
  <div className="onboarding-layout"><div className="onboarding-main">
   <span className="eyebrow">STEP {stepIndex+1} OF 3</span>
   <h1 ref={heading} tabIndex={-1}>{step==='companies'?'Start with the companies you care about.':step==='topics'?'What would you like to understand?':'Your workspace is ready for its first review.'}</h1>
   <p className="onboarding-lead">{step==='companies'?'Follow a company or bring your holdings. We’ll help you compare published evidence and keep a record of your response.':step==='topics'?'Pick any topics to prioritize your first comparison, or continue with no preference. These are interests, not an investment thesis.':'Read what changed, inspect the original evidence, then decide what it means to you.'}</p>
   {error&&<div className="account-alert error" role="alert"><span>{error}</span>{uncertain&&<button onClick={()=>window.location.reload()}>Reload setup</button>}</div>}
   {step==='companies'&&<>
    <div className="onboarding-search"><label htmlFor="setup-search">Choose from the research universe</label><input id="setup-search" value={search} onChange={e=>setSearch(e.target.value)} placeholder="Search company or symbol"/><span>{selected.length} / 20 followed</span></div>
    {!catalog&&<p role="status">Published coverage is unavailable right now. You can still choose companies; research availability will be checked at the final step.</p>}
    <div className="onboarding-companies">{filtered.map(c=>{
     const coverage=catalog?.find(row=>row.symbol===c.symbol),monitor=data.monitoring.find(row=>row.symbol===c.symbol),checked=selected.includes(c.symbol);
     return <label key={c.symbol} className={checked?'selected':''}><input type="checkbox" checked={checked} disabled={locked||existing.has(c.symbol)||(!checked&&selected.length>=20)} onChange={e=>setSelected(prev=>e.target.checked?[...prev,c.symbol]:prev.filter(s=>s!==c.symbol))}/><span className="onboarding-company-name"><strong>{c.name}</strong><small>{c.symbol} · {c.sector}{existing.has(c.symbol)?' · Already followed':''}</small></span><span className="onboarding-coverage"><span className={`badge ${coverage?.status==='published'?'green':''}`}>{!catalog?'Coverage unavailable':coverage?.status==='published'?'Published research':coverage?.status==='withdrawn'?'Research withdrawn':'Coverage planned'}</span><small>{monitor?.enabled?'Monitoring enabled':monitor?'Monitoring paused':'Monitoring not configured'}</small></span></label>;
    })}{!filtered.length&&<p className="empty">No matching company in the 25-company universe.</p>}</div>
    <details className="onboarding-import"><summary>Have a broker CSV? Import your holdings</summary><p>Map columns and check quantities before saving. Only supported companies can be imported. Your CSV stays in this browser.</p>
     <label htmlFor="setup-csv">Holdings CSV · up to 1 MB, 100 rows</label><input id="setup-csv" type="file" accept=".csv,text/csv" disabled={locked} onChange={async e=>{
      const file=e.target.files?.[0],run=++uploadGeneration.current;e.target.value='';setCsv(null);setConfirmed(false);setExcluded([]);setError('');if(!file)return;
      try{if(file.size>1024*1024)throw Error('CSV exceeds the 1 MB limit.');const parsed=readHoldingsCsv(await file.text());if(!mounted.current||run!==uploadGeneration.current)return;const names=parsed.headers.map(h=>h.toLowerCase());setColumns([Math.max(0,names.findIndex(n=>['symbol','tradingsymbol','ticker','trading symbol'].includes(n))),Math.max(0,names.findIndex(n=>['quantity','qty','holding quantity'].includes(n)))]);setCsv(parsed);}catch(e){if(mounted.current&&run===uploadGeneration.current)setError((e as Error).message);}
     }}/>
     {csv&&<><div className="column-mapping">{['Symbol','Quantity'].map((label,i)=><label key={label}>{label} column<select value={columns[i]} disabled={locked} onChange={e=>{setColumns(prev=>prev.map((n,j)=>i===j?Number(e.target.value):n));setConfirmed(false);setExcluded([]);}}>{csv.headers.map((h,j)=><option key={h} value={j}>{h}</option>)}</select></label>)}</div>
      {mappingError?<p role="alert" className="error">{mappingError}</p>:<><div className="setup-csv-preview">{preview.map(row=><label key={row.row}><input type="checkbox" aria-label={`Import row ${row.row} ${row.symbol}`} disabled={locked||!!row.error} checked={!row.error&&!excluded.includes(row.row)} onChange={e=>{setExcluded(prev=>e.target.checked?prev.filter(n=>n!==row.row):[...prev,row.row]);setConfirmed(false);}}/><strong>{row.symbol||'Missing symbol'}</strong><span>{row.quantity||'—'}</span><small>{row.error||(portfolio.entries.some(e=>e.symbol===row.symbol&&e.quantity!==null)?'Replaces saved quantity':'New holding')}</small></label>)}</div><p>{accepted.length} included · {preview.length-accepted.length} excluded</p><label className="import-consent"><input type="checkbox" checked={confirmed} disabled={locked} onChange={e=>setConfirmed(e.target.checked)}/><span>I checked these quantities and excluded rows. Included quantities replace existing holdings; other companies are kept.</span></label></>}
      <button disabled={locked} onClick={()=>{uploadGeneration.current++;setCsv(null);setConfirmed(false);}}>Remove CSV</button>
     </>}
    </details>
    <div className="onboarding-actions"><span>Saved when you continue. You can edit companies later.</span><button className="primary" disabled={locked||(!selected.length&&!accepted.length&&!portfolio.entries.length)||!!(csv&&(!confirmed||!accepted.length||mappingError))} onClick={async()=>{if(await save({action:'companies',version,portfolioVersion:portfolio.version,symbols:selected,holdings:accepted.map(({symbol,quantity})=>({symbol,quantity})),importConfirmed:confirmed})){setCsv(null);setStep('topics');}}}>{busy?'Saving…':'Save companies & continue →'}</button></div>
   </>}
   {step==='topics'&&<><div className="onboarding-topics">{Object.entries(onboardingTopics).map(([key,label])=><label key={key} className={topics.includes(key as OnboardingTopic)?'selected':''}><input type="checkbox" disabled={locked} checked={topics.includes(key as OnboardingTopic)} onChange={e=>setTopics(prev=>e.target.checked?[...prev,key as OnboardingTopic]:prev.filter(t=>t!==key))}/><span><strong>{label}</strong><small>{key==='revenue'?'How the business is growing':key==='operating_margin'?'How much operating profit it retains':key==='operating_cash_flow'?'How operations translate into cash':'How reported borrowing changes'}</small></span></label>)}</div><p className="fine-print">Topic availability depends on published evidence. All company updates remain visible in your inbox.</p><div className="onboarding-actions"><button disabled={locked} onClick={()=>setStep('companies')}>← Companies</button><button className="primary" disabled={locked} onClick={async()=>{if(await save({action:'topics',version,topics}))setStep('review');}}>{busy?'Saving…':topics.length?'Save focus & continue →':'Continue without preferences →'}</button></div></>}
   {step==='review'&&<>
    <div className="onboarding-summary"><span><strong>{portfolio.entries.length}</strong> companies tracked</span><span><strong>{ready.length}</strong> published comparisons available</span></div>
    {first?<div className="onboarding-first"><span className="eyebrow">START WITH {first.symbol}</span><h2>{first.title}</h2><p>Quarter ended {first.periodEnd} · Human-reviewed publication</p><p>{topics.some(t=>first.metrics.includes(t))?'Includes one or more of your chosen topics.':'A published company comparison to get you started.'}</p><button className="primary" disabled={locked} onClick={()=>void finish(`/inbox?filter=all&update=${encodeURIComponent(first.id)}`)}>{busy?'Saving…':`Open ${first.symbol} review →`}</button></div>:<div className="onboarding-first"><h2>No published comparison for your companies yet.</h2><p>Your companies are saved. Monitoring and publication are separate: a source check does not mean a reviewed update is ready.</p><button disabled={locked} onClick={()=>setStep('companies')}>Choose another company</button></div>}
    <p className="fine-print">{portfolio.entries.length-ready.length>0?`${portfolio.entries.length-ready.length} of your companies have no current published comparison. `:''}Opening a review never marks it as reviewed. You’ll choose your own response in the inbox.</p>
    <div className="onboarding-actions"><button disabled={locked} onClick={()=>setStep('topics')}>← Research topics</button><button disabled={locked} onClick={()=>void finish()}>{busy?'Saving…':'Finish & open workspace'}</button></div>
   </>}
  </div><aside className="onboarding-aside"><span className="eyebrow">A RESEARCH HABIT, IN THREE MOVES</span><ol><li><strong>Follow your companies</strong><p>Keep holdings and watchlist in one private workspace.</p></li><li><strong>Check the evidence</strong><p>See the prior and current figures, quarter dates and linked originals.</p></li><li><strong>Remember your reasoning</strong><p>Keep, revise, dismiss or defer an update with a note you can revisit.</p></li></ol><div className="onboarding-expectation"><strong>Know what’s covered.</strong><p>Published research is ready to read. Monitoring enabled means scheduled source checks are configured; it does not guarantee fresh data. Planned coverage has no current published comparison.</p></div><p className="fine-print">Private by account. No automatic trades or investment conclusions.</p></aside></div>
 </section>;
}
