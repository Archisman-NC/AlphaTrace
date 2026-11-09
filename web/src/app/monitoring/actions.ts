'use server';
import { headers } from 'next/headers';
import { localAccessAllowed,UUID } from '@/lib/access';
import { getMonitor,controlMonitor } from '@/lib/monitor';
import { researchError } from '@/lib/research-response';
export async function localMonitor(command?:unknown){
 const h=await headers(),id=process.env.ALPHATRACE_DEV_USER_ID;
 if(!localAccessAllowed(process.env,h.get('host'))||!id||!UUID.test(id))return {error:'Local access is disabled.',status:401};
 try{if(!h.get('origin')||new URL(h.get('origin')!).host!==h.get('host'))return {error:'Invalid request origin.',status:403};}catch{return {error:'Invalid request origin.',status:403};}
 try{if(command!==undefined)await controlMonitor(id,command);return {monitor:await getMonitor(id)};}catch(e){return researchError(e);}
}
