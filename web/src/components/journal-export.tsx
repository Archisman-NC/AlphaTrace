'use client';
import {apiFetch} from '@/lib/api-fetch';

import { useEffect,useId,useRef,useState } from 'react';
import { localJournal } from '@/app/workspace/export-actions';
import type { JournalFormat } from '@/lib/journal-format';
export default function JournalExport({local,token}:{local:boolean;token?:string}){
 const formatId=useId();
 const [format,setFormat]=useState<JournalFormat>('md'),[busy,setBusy]=useState(false),[error,setError]=useState(''),[message,setMessage]=useState('');
 const alive=useRef(true);
 useEffect(()=>{alive.current=true;return()=>{alive.current=false;};},[]);
 async function download(){
  if(busy)return;setBusy(true);setError('');setMessage('');
  try{
   let blob:Blob,filename:string;
   if(local){const result=await localJournal(format);if(!result.file)throw Error(result.error);blob=new Blob([result.file.content],{type:result.file.mediaType});filename=result.file.filename;}
   else{
    const response=await apiFetch('/api/export?format='+format,{cache:'no-store',headers:{Authorization:`Bearer ${token}`}});
    if(!response.ok){const body=await response.json();throw Error(body.error??'Export failed. Please try again.');}
    blob=await response.blob();filename=response.headers.get('Content-Disposition')?.match(/filename="([a-z0-9.-]+)"/)?.[1]??`alphatrace-journal.${format}`;
   }
   if(!alive.current)return;
   const url=URL.createObjectURL(blob),anchor=document.createElement('a');anchor.href=url;anchor.download=filename;document.body.appendChild(anchor);anchor.click();anchor.remove();setTimeout(()=>URL.revokeObjectURL(url),30000);
   setMessage('Your journal is ready. Check your browser downloads.');
  }catch(e){if(alive.current)setError(e instanceof Error?e.message:'Export failed. Please try again.');}
  finally{if(alive.current)setBusy(false);}
 }
 return <section className="panel journal-export" aria-label="Export research journal"><div><span className="eyebrow">YOUR RESEARCH, TO KEEP</span><h2>Take your journal with you.</h2><p>Saved holdings, every thesis revision, review decisions and linked sources. Includes your private notes; save edits before exporting.</p></div><div className="journal-export-controls"><div><label htmlFor={formatId}>File format</label><select id={formatId} value={format} disabled={busy} onChange={e=>setFormat(e.target.value as JournalFormat)}><option value="md">Markdown · readable journal</option><option value="json">JSON · structured data</option></select></div><button disabled={busy} onClick={()=>void download()}>{busy?'Preparing journal…':'Download journal ↓'}</button></div>{error&&<p className="account-alert error" role="alert">{error}</p>}{message&&<p className="fine-print" role="status">{message}</p>}</section>;
}
