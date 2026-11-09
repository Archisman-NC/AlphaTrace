'use client';
import {useEffect,useState} from 'react';
export default function ConnectionNotice(){
 const [offline,setOffline]=useState(false),[notice,setNotice]=useState<'session'|'service'|'connection'|null>(null);
 useEffect(()=>{
  const connectivity=()=>setOffline(!navigator.onLine),status=(event:Event)=>{const value=(event as CustomEvent).detail;if(['session','service','connection'].includes(value))setNotice(old=>old==='session'?'session':value);};
  connectivity();window.addEventListener('online',connectivity);window.addEventListener('offline',connectivity);window.addEventListener('alphatrace:request-status',status);
  return()=>{window.removeEventListener('online',connectivity);window.removeEventListener('offline',connectivity);window.removeEventListener('alphatrace:request-status',status);};
 },[]);
 if(!offline&&!notice)return null;
 return <aside className="connection-notice" role="alert"><div><strong>{offline?'You appear to be offline.':notice==='session'?'Sign-in is required to continue.':notice==='service'?'A service request failed.':'A connection was interrupted.'}</strong><p>{offline?'Edits cannot be confirmed while offline. Keep this page open and check the result when your connection returns.':notice==='session'?'If you have unsaved text, copy it before leaving this page. Signing in may reload your workspace.':'Keep any unfinished text. A submitted change may have completed; use its status or retry controls before starting another.'}</p>{!offline&&notice==='session'&&<a href="/workspace">Open sign-in →</a>}</div>{!offline&&<button onClick={()=>setNotice(null)} aria-label="Dismiss connection notice">Dismiss</button>}</aside>;
}
