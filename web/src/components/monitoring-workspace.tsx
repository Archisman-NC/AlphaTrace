'use client';
import { useEffect,useState } from 'react';
import type { Session } from '@supabase/supabase-js';
import { browserAuth } from '@/lib/browser-auth';
import WorkspaceHeader from './workspace-header';
import SourceMonitor from './source-monitor';
import ProviderImports from './provider-imports';
export default function MonitoringWorkspace({local,authConfig,imports=false}:{local:boolean;authConfig:{url:string;key:string}|null;imports?:boolean}) {
 const [session,setSession]=useState<Session|null>(null),[ready,setReady]=useState(local||!authConfig),[error,setError]=useState('');
 const url=authConfig?.url,key=authConfig?.key;
 useEffect(()=>{
  if(local||!url||!key)return;let active=true;
  try {
   const client=browserAuth(url,key);
   const {data:{subscription}}=client.auth.onAuthStateChange((_event,next)=>{if(active){setSession(next);setReady(true);}});
   client.auth.getSession().then(({data})=>{if(active){setSession(data.session);setReady(true);}}).catch(()=>{if(active){setError('Could not restore your session. Sign in again.');setReady(true);}});
   return()=>{active=false;subscription.unsubscribe();};
  }catch{setError('Browser session storage is unavailable.');setReady(true);}
 },[local,url,key]);
 return <div className="private-shell"><WorkspaceHeader active="monitoring" local={local}/><main id="workspace-content" className="private-content monitoring-content">{error&&<p role="alert" className="account-alert error">{error}</p>}{!ready?<p role="status">Restoring your session…</p>:!local&&!session?<section className="panel reading"><h1>{imports?'Review private data imports.':'See how your sources are monitored.'}</h1><p>Sign in with your invited account{imports?' and operator access':''} to continue.</p><a className="button-link primary" href="/workspace">Sign in →</a></section>:imports?<ProviderImports key={local?'local':session!.user.id} local={local} token={session?.access_token}/>:<SourceMonitor key={local?'local':session!.user.id} local={local} token={session?.access_token}/>}</main><footer className="app-footer"><span>AlphaTrace · Evidence first. Decisions remembered.</span><span>Configured sources · Human-reviewed publication</span></footer></div>;
}
