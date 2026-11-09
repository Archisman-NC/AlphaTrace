'use server';
import { headers } from 'next/headers';
import { localAccessAllowed,UUID } from '@/lib/access';
import { getInbox,getInboxDetail,syncInbox,reviewInbox } from '@/lib/inbox';
import { inboxError } from '@/lib/inbox-response';
export async function localInbox(query:{filter?:string;cursor?:string;update?:string;before?:number}={},command?:unknown){
 const h=await headers(),id=process.env.ALPHATRACE_DEV_USER_ID;
 if(!localAccessAllowed(process.env,h.get('host'))||!id||!UUID.test(id))return {error:'Local access is disabled.',status:401};
 try{if(!h.get('origin')||new URL(h.get('origin')!).host!==h.get('host'))return {error:'Invalid request origin.',status:403};}catch{return {error:'Invalid request origin.',status:403};}
 try{
  if(command!==undefined)return (command as {action?:string})?.action==='sync'?{sync:await syncInbox(id)}:{detail:await reviewInbox(id,command)};
  return query.update?{detail:await getInboxDetail(id,query.update,query.before)}:{inbox:await getInbox(id,query.filter,query.cursor)};
 }catch(e){return inboxError(e);}
}
