'use server';
import { headers } from 'next/headers';
import { localAccessAllowed,UUID } from '@/lib/access';
import { exportJournal } from '@/lib/journal';
import { journalFile,journalFormat,JournalLimitError } from '@/lib/journal-format';
export async function localJournal(format:unknown){
 const h=await headers(),id=process.env.ALPHATRACE_DEV_USER_ID;
 if(!localAccessAllowed(process.env,h.get('host'))||!id||!UUID.test(id))return {error:'Local access is disabled.'};
 try{if(!h.get('origin')||new URL(h.get('origin')!).host!==h.get('host'))return {error:'Invalid request origin.'};}catch{return {error:'Invalid request origin.'};}
 let selected;try{selected=journalFormat(format);}catch{return {error:'Choose JSON or Markdown.'};}
 try{return {file:journalFile(await exportJournal(id),selected)};}
 catch(e){return {error:e instanceof JournalLimitError?e.message:'Your journal could not be exported. Please try again.'};}
}
