'use server';
import { headers } from 'next/headers';
import { localAccessAllowed,UUID } from '@/lib/access';
import { getResearch,reviewResearch } from '@/lib/research';
import { researchError } from '@/lib/research-response';
export async function localResearch(symbol?:string,admin=false,command?:unknown,updateId?:string){
 const h=await headers(),id=process.env.ALPHATRACE_DEV_USER_ID;
 if(!localAccessAllowed(process.env,h.get('host'))||!id||!UUID.test(id))return {error:'Local access is disabled.',status:401};
 try{if(!h.get('origin')||new URL(h.get('origin')!).host!==h.get('host'))return {error:'Invalid request origin.',status:403};}
 catch{return {error:'Invalid request origin.',status:403};}
 try{return command===undefined?await getResearch(id,symbol,admin,updateId):await reviewResearch(id,command);}
 catch(e){return researchError(e);}
}
