'use client';
import {apiFetch} from '@/lib/api-fetch';

import { useEffect,useRef,useState,type FormEvent } from 'react';
import { type Session,type SupabaseClient } from '@supabase/supabase-js';
import JournalExport from './journal-export';
import WorkspaceHeader from './workspace-header';
import WorkspaceOverview from './workspace-overview';
import GuidedOnboarding from './guided-onboarding';
import { browserAuth } from '@/lib/browser-auth';
import { universe } from '@/lib/companies';
import { previewHoldings,readHoldingsCsv,validQuantity,type CsvTable,type Portfolio,type PortfolioCommand,type ImportRow } from '@/lib/portfolio-input';
import { localPortfolio } from '@/app/workspace/actions';
import { localResearch } from '@/app/research/actions';
import type { CompanyCoverage } from '@/lib/research-view';

export default function AccountWorkspace({local,authConfig,companiesView=false,setupRequested=false}:{setupRequested?:boolean;local:boolean;authConfig:{url:string;key:string}|null;companiesView?:boolean}){
 const auth=useRef<SupabaseClient|null>(null);
 const [session,setSession]=useState<Session|null>(null);
 const [authReady,setAuthReady]=useState(local||!authConfig);
 const [recovery,setRecovery]=useState(false),[email,setEmail]=useState(''),[password,setPassword]=useState('');
 const [portfolio,setPortfolio]=useState<Portfolio|null>(null),[loading,setLoading]=useState(false),[busy,setBusy]=useState(false);
 const [error,setError]=useState(''),[message,setMessage]=useState(''),[conflict,setConflict]=useState(false),[reloadKey,setReloadKey]=useState(0);
 const [view,setView]=useState<'holdings'|'watchlist'|'import'>('holdings');
 const [symbol,setSymbol]=useState('INFY'),[quantity,setQuantity]=useState(''),[search,setSearch]=useState('');
 const [quantityError,setQuantityError]=useState('');
 const [csv,setCsv]=useState<CsvTable|null>(null),[symbolColumn,setSymbolColumn]=useState(0),[quantityColumn,setQuantityColumn]=useState(1);
 const [excluded,setExcluded]=useState<number[]>([]),[importApproved,setImportApproved]=useState(false);
 const [removeSymbol,setRemoveSymbol]=useState<string|null>(null);
 const [coverage,setCoverage]=useState<CompanyCoverage[]|null>(null),[coverageError,setCoverageError]=useState('');
 const requestGeneration=useRef(0);
 const authUrl=authConfig?.url,authKey=authConfig?.key;
 useEffect(()=>{
  if(!authUrl||!authKey||local)return;
  let client:SupabaseClient;
  try{client=browserAuth(authUrl,authKey);}
  catch{setAuthReady(true);setError('Browser session storage is unavailable. Enable it and reload to sign in.');return;}
  auth.current=client;
  const params=new URLSearchParams(window.location.search);
  if(params.get('recovery')==='1')setRecovery(true);
  if(params.get('passwordChanged')==='1')setMessage('Sign in again with your new password.');
  if(params.get('accountClosed')==='1')setMessage('Your account closure was recorded. This browser session has been cleared.');
  let active=true;
  const {data:{subscription}}=client.auth.onAuthStateChange((event,next)=>{
   if(!active)return;
   setSession(next);setAuthReady(true);
   if(event==='PASSWORD_RECOVERY')setRecovery(true);
   if(event==='SIGNED_OUT'){requestGeneration.current++;setPortfolio(null);setPassword('');setMessage('Signed out.');}
  });
  client.auth.getSession().then(({data,error})=>{if(!active)return;setSession(data.session);setAuthReady(true);if(error)setError('Your session could not be restored. Please sign in again.');}).catch(()=>{if(active){setAuthReady(true);setError('Your session could not be restored. Please reload.');}});
  return()=>{active=false;subscription.unsubscribe();auth.current=null;};
 },[authUrl,authKey,local]);
 useEffect(()=>{setPortfolio(null);setCsv(null);setQuantity('');setQuantityError('');setRemoveSymbol(null);setConflict(false);setMessage('');},[session?.user.id]);
 useEffect(()=>{
  if(!authReady||(!local&&!session)||recovery)return;
  const generation=++requestGeneration.current;
  setLoading(true);setError('');
  async function load(){
   try{
    let result;
    if(local)result=await localPortfolio();
    else{
     const response=await apiFetch('/api/portfolio',{headers:{Authorization:`Bearer ${session!.access_token}`},cache:'no-store'});
     result=await response.json();
    }
    if(generation!==requestGeneration.current)return;
    if(result.error||!result.portfolio){setPortfolio(null);throw Error(result.error||'Portfolio could not be loaded.');}
    setPortfolio(result.portfolio);setConflict(false);setImportApproved(false);
   }catch(e){if(generation===requestGeneration.current)setError(e instanceof Error?e.message:'Portfolio could not be loaded.');}
   finally{if(generation===requestGeneration.current)setLoading(false);}
  }
  void load();return()=>{requestGeneration.current++;};
 },[authReady,local,session?.access_token,recovery,reloadKey]);
 useEffect(()=>{
  setCoverage(null);setCoverageError('');if(!authReady||(!local&&!session)||recovery)return;
  let active=true;
  async function loadCoverage(){try{
   const result=local?await localResearch():await apiFetch('/api/research',{headers:{Authorization:`Bearer ${session!.access_token}`},cache:'no-store'}).then(r=>r.json());
   if(!Array.isArray(result.catalog))throw Error('Research coverage is unavailable.');
   if(active)setCoverage(result.catalog);
  }catch{if(active)setCoverageError('Research coverage could not be loaded.');}}
  void loadCoverage();return()=>{active=false;};
 },[authReady,local,session?.access_token,recovery,reloadKey]);
 async function mutate(command:PortfolioCommand):Promise<boolean>{
  if(!portfolio||busy||loading||conflict)return false;
  const generation=requestGeneration.current;
  setBusy(true);setError('');setMessage('');
  try{
   const input={version:portfolio.version,command};let result,status=200;
   if(local){result=await localPortfolio(input);status=result.status??200;}
   else{
    if(!session)throw Error('Sign in again to save.');
    const response=await apiFetch('/api/portfolio',{method:'POST',headers:{Authorization:`Bearer ${session.access_token}`,'Content-Type':'application/json'},body:JSON.stringify(input)});
    status=response.status;result=await response.json();
   }
   if(generation!==requestGeneration.current)return false;
   if(result.error||!result.portfolio){if(status===409||status>=500)setConflict(true);throw Error(result.error||'The save could not be confirmed. Reload before trying again.');}
   setPortfolio(result.portfolio);setMessage('Saved to your private portfolio.');return true;
  }catch(e){if(generation===requestGeneration.current){setError(e instanceof Error?e.message:'The save could not be confirmed.');if(e instanceof TypeError)setConflict(true);}return false;}
  finally{setBusy(false);}
 }
 async function signIn(event:FormEvent){
  event.preventDefault();if(!auth.current)return;
  setBusy(true);setError('');setMessage('');
  try{const {error}=await auth.current.auth.signInWithPassword({email:email.trim(),password});if(error)throw Error('Sign-in failed. Check your email and password.');setPassword('');}
  catch(e){setError(e instanceof Error?e.message:'Sign-in is unavailable.');}
  finally{setBusy(false);}
 }
 async function recover(){
  if(!auth.current||!email.trim())return;
  setBusy(true);setError('');setMessage('');
  try{
   const {error}=await auth.current.auth.resetPasswordForEmail(email.trim(),{redirectTo:window.location.origin+'/workspace?recovery=1'});
   if(error)throw Error('Recovery could not be requested. Please try again later.');
   setMessage('If that address has an account, you’ll receive a recovery email.');
  }catch(e){setError((e as Error).message);}finally{setBusy(false);}
 }
 async function updatePassword(event:FormEvent){
  event.preventDefault();if(!auth.current||!session)return;
  setBusy(true);setError('');
  try{
   const {error}=await auth.current.auth.updateUser({password});
   if(error)throw Error('Password could not be updated. The recovery link may have expired, or your password may not meet the configured requirements.');
   setPassword('');setRecovery(false);window.history.replaceState(null,'','/workspace');setMessage('Password updated.');
  }catch(e){setError((e as Error).message);}finally{setBusy(false);}
 }
 async function signOut(){
  if(!auth.current)return;
  requestGeneration.current++;setPortfolio(null);setBusy(true);setError('');
  try{const {error}=await auth.current.auth.signOut({scope:'local'});if(error)throw error;setSession(null);setRecovery(false);setCsv(null);}
  catch{setError('Sign-out could not be completed. Please retry.');}finally{setBusy(false);}
 }
 function changeQuantity(next:string){
  if(!/^(?:[0-9]{0,12}|[0-9]{1,12}\.[0-9]{0,6})$/.test(next)){
   setQuantityError('Input not accepted. Use digits and one decimal point, without commas or signs. Maximum 12 whole digits and 6 decimal places. The field is unchanged; edit or clear it before saving.');
   return;
  }
  setQuantity(next);setQuantityError('');
 }
 async function saveHolding(event:FormEvent){
  event.preventDefault();
  if(quantityError||busy||loading||conflict)return;
  try{const exact=validQuantity(quantity);if(await mutate({type:'holding',symbol,quantity:exact}))setQuantity('');}
  catch(e){setError((e as Error).message);}
 }
 let preview:ImportRow[]=[],mappingError='';
 if(csv){try{preview=previewHoldings(csv,symbolColumn,quantityColumn);}catch(e){mappingError=(e as Error).message;}}
 // Duplicate rows are excluded until resolved in the CSV. Other valid rows can be explicitly imported.
 const accepted=preview.filter(row=>!row.error&&!excluded.includes(row.row));
 const entries=portfolio?.entries??[],holdings=entries.filter(e=>e.quantity!==null),following=entries.filter(e=>e.followed);
 let quantityReady=false;
 try{validQuantity(quantity);quantityReady=true;}catch{/* Incomplete and zero quantities cannot be saved. */}
 const locked=busy||loading||conflict;
 const authPanel=!local&&(!session||recovery);
 return <div className="private-shell">
  <WorkspaceHeader active={companiesView?'companies':'overview'} local={local}>{session&&<button disabled={busy} onClick={signOut}>Sign out</button>}</WorkspaceHeader>
  <main id="workspace-content" className={`private-content ${!companiesView&&!authPanel?'overview-content':''}`}>
   {(authPanel||companiesView)&&<div className="page-heading"><div><div className="eyebrow">YOUR COMPANIES. YOUR CONTEXT.</div><h1>{authPanel?'A home for your research.':'My companies'}</h1><p>{local?'Your holdings and watchlist, saved in your local workspace.':session?'Your holdings and watchlist are tied to your invited account.':'Sign in with the email address used for your beta invitation.'}</p></div>{!authPanel&&portfolio&&<button disabled={busy||loading} onClick={()=>setReloadKey(k=>k+1)}>{loading?'Loading…':'Reload companies'}</button>}</div>}
   {error&&<div className="account-alert error" role="alert">{error}{conflict&&<button disabled={busy||loading} onClick={()=>setReloadKey(k=>k+1)}>Reload latest data</button>}</div>}
   {message&&<div className="account-alert success" role="status">{message}</div>}
   {!authReady?<div className="panel empty" role="status">Restoring your session…</div>:authPanel?<div className="auth-layout"><section className="panel reading auth-card"><span className="badge lavender">Invite-only access</span><h2>{recovery?'Choose a new password':'Welcome back'}</h2>{!authConfig?<><p>Account sign-in is not configured on this installation yet. The sample demo is available without an account.</p><a className="button-link primary" href="/demo">Explore the sample workspace ↗</a></>:recovery?<>{session?<form onSubmit={updatePassword}><label htmlFor="new-password">New password</label><input id="new-password" type="password" autoComplete="new-password" minLength={12} maxLength={128} required value={password} onChange={e=>setPassword(e.target.value)}/><p className="fine-print">Use at least 12 characters. Your project’s password rules also apply.</p><button className="primary wide" disabled={busy}>Update password</button></form>:<p>The recovery link is missing or expired. Request a new email from the sign-in screen.</p>}<button className="text-button" disabled={busy} onClick={()=>{setRecovery(false);window.history.replaceState(null,'','/workspace');}}>Return to sign-in</button></>:<form onSubmit={signIn}><label htmlFor="email">Email address</label><input id="email" type="email" autoComplete="email" required maxLength={254} value={email} onChange={e=>setEmail(e.target.value)} placeholder="you@example.com"/><label htmlFor="password">Password</label><input id="password" type="password" autoComplete="current-password" required maxLength={128} value={password} onChange={e=>setPassword(e.target.value)}/><button className="primary wide" disabled={busy}>{busy?'Please wait…':'Sign in →'}</button><button type="button" className="text-button" disabled={busy||!email.trim()} onClick={recover}>Forgot your password?</button></form>}</section><section className="auth-context"><span className="eyebrow">A SMALL, DELIBERATE BETA</span><h2>Start with what you own.<br/>Keep the reasoning close.</h2><p>Import holdings, choose the companies you want to follow, and build a private record of your research.</p><div className="coverage-line"><span>Initial research universe</span><strong>25 companies</strong></div><div className="coverage-line"><span>Watchlist capacity</span><strong>20 companies</strong></div><div className="coverage-line"><span>Research publication</span><span className="badge">Reviewed before publication</span></div></section></div>:<>
    {local&&<div className="local-mode-note">Local development identity · No account sign-in is being simulated. Production disables this access.</div>}
    {!portfolio&&!error?<div className="panel empty" role="status"><h2>Loading your portfolio…</h2><p>Reading your saved companies from the database.</p></div>:!portfolio?<div className="panel empty"><h2>Your workspace couldn’t be loaded</h2><p>{local?'Your local storage is unavailable. Try again once connected.':'Access requires an invited, verified account and a configured database.'}</p><button onClick={()=>setReloadKey(k=>k+1)}>Try again</button></div>:!companiesView?<GuidedOnboarding key={local?'local':session?.user.id} local={local} token={session?.access_token} portfolio={portfolio} catalog={coverage} requested={setupRequested} onPortfolio={setPortfolio}><WorkspaceOverview key={local?'local':session!.user.id} portfolio={portfolio} catalog={coverage} coverageError={coverageError} local={local} token={session?.access_token} onRefreshCoverage={()=>setReloadKey(k=>k+1)}/></GuidedOnboarding>:<>
     <div className="stats private-stats"><div><span>Holdings recorded</span><strong>{holdings.length}</strong><p>Quantities you entered or imported</p></div><div><span>Companies followed</span><strong>{following.length}<small>/ 20</small></strong><p>Independent of your holdings</p></div><div><span>Research coverage</span><strong>{coverage?coverage.filter(c=>c.status==='published').length:'—'}<small>/ 25 published</small></strong><p><a href="/research">{coverage?'Open source-backed research →':'Check research availability →'}</a></p></div></div>
     <nav className="tabs" aria-label="Portfolio sections">{(['holdings','watchlist','import'] as const).map(v=><button className={view===v?'selected':''} key={v} onClick={()=>{setView(v);setRemoveSymbol(null);}}>{v==='holdings'?'Holdings':v==='watchlist'?'Watchlist':'Import CSV'}</button>)}</nav>
     {view==='holdings'&&<div className="portfolio-grid"><section className="panel"><div className="panel-heading"><h2>Your holdings</h2><span>Absolute quantities · no valuations</span></div>{holdings.length?<div className="private-table"><div className="private-table-head"><span>Company</span><span>Quantity</span><span>Actions</span></div>{holdings.map(entry=><div className="private-table-row" key={entry.symbol}><span><strong>{universe.find(c=>c.symbol===entry.symbol)?.name}</strong><small>{entry.symbol}</small></span><span className="quantity">{entry.quantity?validQuantity(entry.quantity):'—'}</span><div><button disabled={locked} onClick={()=>{setSymbol(entry.symbol);setQuantity(entry.quantity!);setQuantityError('');document.getElementById('holding-quantity')?.focus();}}>Edit</button><button disabled={locked} aria-label={`Remove ${entry.symbol} holding`} onClick={()=>setRemoveSymbol(entry.symbol)}>Remove</button></div>{removeSymbol===entry.symbol&&<div className="remove-confirm"><span>Remove this quantity? Watchlist membership is kept.</span><button disabled={locked} onClick={async()=>{if(await mutate({type:'remove_holding',symbol:entry.symbol}))setRemoveSymbol(null);}}>Confirm removal</button><button disabled={busy} onClick={()=>setRemoveSymbol(null)}>Cancel</button></div>}</div>)}</div>:<div className="empty"><span className="empty-symbol">＋</span><h2>Add your first holding</h2><p>Enter a company and quantity, or import a CSV from your broker.</p><button onClick={()=>setView('import')}>Import holdings ↗</button></div>}</section><section className="panel reading holding-editor"><span className="eyebrow">MANUAL ENTRY</span><h2>{holdings.some(h=>h.symbol===symbol)?'Update a holding':'Add a holding'}</h2><p>Enter the total quantity you currently hold. Saving replaces the previous quantity.</p><form onSubmit={saveHolding}><label htmlFor="holding-company">Company</label><select id="holding-company" disabled={locked} value={symbol} onChange={e=>{setSymbol(e.target.value);setQuantity('');setQuantityError('');}}>{universe.map(c=><option key={c.symbol} value={c.symbol}>{c.name} · {c.symbol}</option>)}</select><label htmlFor="holding-quantity">Total quantity</label><input id="holding-quantity" disabled={locked} value={quantity} inputMode="decimal" autoComplete="off" required aria-invalid={!!quantityError} aria-describedby={quantityError?"holding-quantity-hint holding-quantity-error":"holding-quantity-hint"} placeholder="e.g. 25" onChange={e=>changeQuantity(e.target.value)}/>{quantityError&&<p id="holding-quantity-error" className="error" role="alert">{quantityError}</p>}<button className="primary wide" disabled={locked||!quantityReady||!!quantityError}>{busy?'Saving…':'Save holding'}</button><p id="holding-quantity-hint" className="fine-print">Positive numbers only. Up to 12 whole digits and 6 decimal places.</p></form><p className="fine-print">No broker connection. This records holdings; it never places a trade.</p></section></div>}
     {view==='watchlist'&&<section className="panel"><div className="panel-heading"><h2>Choose the companies you follow</h2><span>{following.length} / 20</span></div><div className="watchlist-search"><label htmlFor="company-search" className="sr-only">Search research universe</label><input id="company-search" placeholder="Search by name or symbol" value={search} onChange={e=>setSearch(e.target.value)}/></div><div className="private-watchlist">{universe.filter(c=>`${c.name} ${c.symbol}`.toLowerCase().includes(search.toLowerCase())).map(c=>{const followed=entries.some(e=>e.symbol===c.symbol&&e.followed);return <div key={c.symbol}><span className="company-avatar">{c.symbol.slice(0,2)}</span><span><strong>{c.name}</strong><small>{c.symbol} · {c.sector}</small></span><a className="badge" href={`/research?symbol=${encodeURIComponent(c.symbol)}`}>{coverage?(coverage.find(item=>item.symbol===c.symbol)?.status==='published'?'Read research ↗':coverage.find(item=>item.symbol===c.symbol)?.status==='withdrawn'?'Withdrawn ↗':'Coverage planned ↗'):'Check coverage ↗'}</a><button disabled={locked||(!followed&&following.length>=20)} aria-label={`${followed?'Unfollow':'Follow'} ${c.name}`} onClick={()=>void mutate({type:'follow',symbol:c.symbol,followed:!followed})}>{followed?'✓ Following':'+ Follow'}</button></div>;})}{!universe.some(c=>`${c.name} ${c.symbol}`.toLowerCase().includes(search.toLowerCase()))&&<div className="empty"><p>No matching company in the initial universe.</p></div>}</div></section>}
     {view==='import'&&<section className="panel reading"><span className="eyebrow">BROKER CSV → PRIVATE HOLDINGS</span><h2>Import with a clear preview</h2><p>Map your symbol and quantity columns, inspect each row, then confirm. Imports replace quantities for included symbols. They never add duplicate holdings or remove other companies.</p><label className="upload-area" htmlFor="portfolio-csv"><span>↑</span><strong>Choose a holdings CSV</strong><small>Up to 1 MB · 100 rows · source file stays in your browser</small></label><input id="portfolio-csv" type="file" accept=".csv,text/csv" disabled={locked} onChange={async e=>{const file=e.target.files?.[0];setError('');setMessage('');setCsv(null);setExcluded([]);setImportApproved(false);if(!file)return;try{if(file.size>1024*1024)throw Error('CSV exceeds the 1 MB limit.');const parsed=readHoldingsCsv(await file.text());setCsv(parsed);const names=parsed.headers.map(s=>s.toLowerCase());const si=names.findIndex(s=>['symbol','tradingsymbol','trading symbol','ticker'].includes(s)),qi=names.findIndex(s=>['quantity','qty','holding quantity'].includes(s));setSymbolColumn(si>=0?si:0);setQuantityColumn(qi>=0?qi:Math.min(1,names.length-1));}catch(e){setError((e as Error).message);}finally{e.target.value='';}}}/>{csv&&<><div className="column-mapping"><div><label htmlFor="symbol-column">Symbol column</label><select id="symbol-column" value={symbolColumn} disabled={locked} onChange={e=>{setSymbolColumn(Number(e.target.value));setImportApproved(false);setExcluded([]);}}>{csv.headers.map((h,i)=><option key={i} value={i}>{h}</option>)}</select></div><div><label htmlFor="quantity-column">Quantity column</label><select id="quantity-column" value={quantityColumn} disabled={locked} onChange={e=>{setQuantityColumn(Number(e.target.value));setImportApproved(false);setExcluded([]);}}>{csv.headers.map((h,i)=><option key={i} value={i}>{h}</option>)}</select></div></div>{mappingError?<p className="error">{mappingError}</p>:<><div className="import-preview private-import">{preview.map(row=><label key={row.row}><input type="checkbox" aria-label={`Include row ${row.row} ${row.symbol}`} disabled={locked||!!row.error} checked={!row.error&&!excluded.includes(row.row)} onChange={e=>{setExcluded(prev=>e.target.checked?prev.filter(n=>n!==row.row):[...prev,row.row]);setImportApproved(false);}}/><span><strong>{row.symbol||'Missing symbol'}</strong><small>Row {row.row}</small></span><span>{row.quantity||'—'}</span><span className={`badge ${row.error?'amber':'green'}`}>{row.error||(holdings.some(h=>h.symbol===row.symbol)?'Replace existing quantity':'New holding')}</span></label>)}</div><p>{accepted.length} rows selected · {preview.length-accepted.length} excluded. Unsupported companies and invalid or duplicate rows are not imported.</p><label className="import-consent"><input type="checkbox" disabled={locked} checked={importApproved} onChange={e=>setImportApproved(e.target.checked)}/><span>Save these absolute quantities to my private portfolio. Existing quantities for included companies will be replaced.</span></label><button className="primary" disabled={locked||!accepted.length||!importApproved} onClick={async()=>{if(await mutate({type:'import',holdings:accepted.map(({symbol,quantity})=>({symbol,quantity}))})){setCsv(null);setImportApproved(false);setView('holdings');setMessage(`${accepted.length} holdings saved. Reimporting the same quantities will not add to them.`);}}}>{busy?'Importing…':`Import ${accepted.length} holdings`}</button></>}</>}</section>}
    </>}
   </>}
   {authReady&&!authPanel&&companiesView&&<JournalExport key={local?'local':session?.user.id} local={local} token={session?.access_token}/>}
  </main><footer className="app-footer"><span>AlphaTrace · Evidence first. Decisions remembered.</span><span>Private holdings · No live prices or trading</span></footer>
 </div>;
}
