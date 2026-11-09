'use client';
import {useEffect,useRef,useState,type FormEvent} from 'react';
import type {Session} from '@supabase/supabase-js';
import {browserAuth} from '@/lib/browser-auth';
import {apiFetch} from '@/lib/api-fetch';
import {betaToken,type BetaInvitation} from '@/lib/beta-contract';
import {recoveryPassword} from '@/lib/password-recovery';
export default function JoinBeta({authConfig}:{authConfig:{url:string;key:string}|null}){
 const token=useRef(''),lock=useRef(false),[invitation,setInvitation]=useState<BetaInvitation|null>(null),[loading,setLoading]=useState(true),[busy,setBusy]=useState(false),[error,setError]=useState(''),[password,setPassword]=useState(''),[confirmation,setConfirmation]=useState(''),[existing,setExisting]=useState(false),[session,setSession]=useState<Session|null>(null),[activated,setActivated]=useState(false);
 const auth=()=>authConfig?browserAuth(authConfig.url,authConfig.key):null;
 // A second invitation in the same tab is a fragment-only navigation. Restart
 // inspection with that capability rather than retaining the previous result.
 useEffect(()=>{const changed=()=>{if(window.location.hash)window.location.reload();};window.addEventListener('hashchange',changed);return()=>window.removeEventListener('hashchange',changed);},[]);
 useEffect(()=>{
  let active=true;
  // Keep the bearer invitation in memory only. It never enters server URLs or browser storage.
  if(!token.current)token.current=window.location.hash.slice(1);
  window.history.replaceState(null,'','/join');
  void (async()=>{try{
   betaToken(token.current);if(!authConfig)throw Error('Account activation is not configured. Contact the beta operator.');
   const r=await apiFetch('/api/beta/accept',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action:'inspect',token:token.current}),signal:AbortSignal.timeout(20000)}),b=await r.json();
   if(!r.ok||!b.invitation)throw Error(b.error||'Invitation unavailable.');
   const current=await auth()!.auth.getSession();if(!active)return;setInvitation(b.invitation);setExisting(b.invitation.existing);setSession(current.data.session);
  }catch(e){if(active)setError(e instanceof Error?e.message:'Invitation unavailable.');}finally{if(active)setLoading(false);}})();
  return()=>{active=false;};
 },[authConfig?.url,authConfig?.key]);
 const wrongAccount=!!session&&session.user.email?.toLowerCase()!==invitation?.email;
 async function accept(event:FormEvent){
  event.preventDefault();if(lock.current||!invitation||wrongAccount||!authConfig)return;
  if(!session&&!existing)try{recoveryPassword(password,confirmation);}catch(e){setError((e as Error).message);return;}
  lock.current=true;setBusy(true);setError('');
  try{
   let current=session;
   if(!current&&existing){const {data,error}=await auth()!.auth.signInWithPassword({email:invitation.email,password});if(error||!data.session)throw Error('Sign-in failed. Use the password for the invited email, or recover it from the sign-in screen.');current=data.session;setSession(current);}
   const r=await apiFetch('/api/beta/accept',{method:'POST',headers:{'Content-Type':'application/json',...(current?{Authorization:`Bearer ${current.access_token}`}:{})},body:JSON.stringify({action:'accept',token:token.current,...(!current?{password,confirmPassword:confirmation}:{})}),signal:AbortSignal.timeout(45000)}),b=await r.json();
   if(!r.ok||!b.activated)throw Error(b.error||'Activation could not be confirmed. Try signing in with the password you chose.');
   setActivated(true);token.current='';
   if(!current){const {data,error}=await auth()!.auth.signInWithPassword({email:invitation.email,password});if(error||!data.session)throw Error('Your beta access is active. Sign in with the password you chose to continue.');}
   window.location.assign('/workspace');
  }catch(e){setError(e instanceof TypeError?'The result could not be confirmed. Try signing in with the password you chose. If activation is still needed, reopen the invitation and use that same password.':e instanceof Error?e.message:'Activation could not be confirmed.');}
  finally{setPassword('');setConfirmation('');lock.current=false;setBusy(false);}
 }
 return <main className="join-shell"><a className="private-brand" href="/workspace"><span className="brand-mark">αT</span>AlphaTrace</a><section className="panel reading join-card"><span className="eyebrow">YOUR PRIVATE RESEARCH WORKSPACE</span><h1>{activated?'Your access is active.':'You’re invited.'}</h1>{loading?<p role="status">Checking your invitation…</p>:<>{error&&<p className="account-alert error" role="alert">{error}</p>}{activated?<a className="button-link primary" href="/workspace">Continue to sign-in →</a>:!invitation?<><p>Reopen the original invitation link. It may have expired, been replaced or already been used.</p><a href="/workspace">Already activated? Sign in →</a></>:<><p className="join-email">{invitation.email}</p><p>Follow companies, compare published evidence and keep a private record of your reasoning.</p><p className="fine-print">Expires {new Date(invitation.expiresAt).toLocaleString('en-IN',{timeZone:'Asia/Kolkata',dateStyle:'medium',timeStyle:'short'})} IST. Opening this page does not activate your account.</p>
  {wrongAccount?<><p className="account-alert">You’re signed in with a different email address. Sign out of this browser session to accept this invitation.</p><button disabled={busy} onClick={async()=>{setBusy(true);try{const {error}=await auth()!.auth.signOut({scope:'local'});if(error)throw error;setSession(null);setError('');}catch{setError('Sign-out failed. Check your connection and try again.');}finally{setBusy(false);}}}>Sign out to continue</button></>:<form onSubmit={accept}><fieldset disabled={busy}>{session?<p>Accept the invitation for your signed-in account. Your existing password will stay the same.</p>:<><label htmlFor="join-password">{existing?'Your existing password':'Choose a password'}</label><input id="join-password" type="password" autoComplete={existing?'current-password':'new-password'} required minLength={existing?undefined:12} maxLength={128} value={password} onChange={e=>setPassword(e.target.value)}/>{!existing&&<><label htmlFor="join-confirm">Confirm password</label><input id="join-confirm" type="password" autoComplete="new-password" minLength={12} maxLength={128} required value={confirmation} onChange={e=>setConfirmation(e.target.value)}/><p className="fine-print">Use 12–128 characters. Your password is never shared with the operator.</p></>}</>}<button className="primary wide" disabled={busy}>{busy?'Activating…':session||existing?'Accept invitation':'Create my account'}</button></fieldset></form>}
  {!session&&!invitation.existing&&<button className="text-button" disabled={busy} onClick={()=>{setExisting(v=>!v);setPassword('');setConfirmation('');setError('');}}>{existing?'I need a new account':'I already have an account'}</button>}
  <p className="fine-print">Your holdings and research notes are private. The operator can see whether you completed setup and saved a first response. Emails are optional and off by default.</p><a href="/workspace">Go to sign-in →</a>
 </>}</>}</section></main>;
}
