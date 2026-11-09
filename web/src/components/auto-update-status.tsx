'use client';
import {useEffect,useRef,useState} from 'react';
import {createAutoRefresh,type RefreshState} from '@/lib/auto-refresh';

export function useAutoUpdates(refresh:()=>Promise<boolean|null>,paused=false){
 const latest=useRef({refresh,paused});latest.current={refresh,paused};
 const [state,setState]=useState<RefreshState>({checking:false,failed:false,checkedAt:null});
 const [online,setOnline]=useState(true);
 useEffect(()=>{
  const loop=createAutoRefresh({run:()=>latest.current.refresh(),available:()=>!latest.current.paused&&document.visibilityState==='visible'&&navigator.onLine,changed:setState});
  const wake=()=>{setOnline(navigator.onLine);void loop.wake();};
  setOnline(navigator.onLine);
  window.addEventListener('online',wake);window.addEventListener('offline',wake);
  window.addEventListener('focus',wake);document.addEventListener('visibilitychange',wake);
  return()=>{loop.stop();window.removeEventListener('online',wake);window.removeEventListener('offline',wake);window.removeEventListener('focus',wake);document.removeEventListener('visibilitychange',wake);};
 },[]);
 return {...state,online};
}

export default function AutoUpdateStatus({state,checkedAt,failed=false}:{state:RefreshState&{online:boolean};checkedAt?:string|null;failed?:boolean}){
 const lastTime=Math.max(state.checkedAt??0,checkedAt?Date.parse(checkedAt):0);
 const last=lastTime?new Date(lastTime):null;
 const delayed=state.failed||failed;
 return <div className={`auto-update-status ${!state.online||delayed?'is-delayed':''}`} role="status">
  <span className="auto-update-dot" aria-hidden="true"/>
  <strong>{!state.online?'Offline · showing saved data':state.checking?'Checking for published updates…':delayed?'Updates delayed · retrying automatically':'Auto updates · every minute'}</strong>
  <span>{last?`Last successful check ${last.toLocaleTimeString('en-IN',{hour:'2-digit',minute:'2-digit',second:'2-digit',timeZone:'Asia/Kolkata'})} IST`:'Waiting for a successful check'}</span>
  <small>While this page is visible · Published research, not live market prices</small>
 </div>;
}
