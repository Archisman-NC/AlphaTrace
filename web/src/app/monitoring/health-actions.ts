'use server';
import {headers} from 'next/headers';
import {localAccessAllowed,UUID} from '@/lib/access';
import {getHealth} from '@/lib/health';
import {ResearchForbiddenError} from '@/lib/research-contract';
export async function localHealth(){
 const h=await headers(),id=process.env.ALPHATRACE_DEV_USER_ID;
 if(!localAccessAllowed(process.env,h.get('host'))||!id||!UUID.test(id))return {error:'Local access is disabled.',status:401};
 try{if(!h.get('origin')||new URL(h.get('origin')!).host!==h.get('host'))return {error:'Invalid request origin.',status:403};}catch{return {error:'Invalid request origin.',status:403};}
 try{return {health:await getHealth(id),status:200};}catch(e){return {error:e instanceof ResearchForbiddenError?e.message:'Operational status is unavailable. Database and worker state are unconfirmed.',status:e instanceof ResearchForbiddenError?403:503};}
}
