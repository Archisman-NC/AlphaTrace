'use server';
import { headers } from 'next/headers';
import { localAccessAllowed,UUID } from '@/lib/access';
import { getThesis,saveThesis } from '@/lib/theses';
import { thesisError } from '@/lib/thesis-response';
export async function localThesis(symbol:string,command?:unknown,before?:number){
 const h=await headers(),id=process.env.ALPHATRACE_DEV_USER_ID;
 if(!localAccessAllowed(process.env,h.get('host'))||!id||!UUID.test(id))return {error:'Local access is disabled.',status:401};
 try{if(!h.get('origin')||new URL(h.get('origin')!).host!==h.get('host'))return {error:'Invalid request origin.',status:403};}catch{return {error:'Invalid request origin.',status:403};}
 try{return {thesis:command===undefined?await getThesis(id,symbol,before):await saveThesis(id,command)};}catch(e){return thesisError(e);}
}
