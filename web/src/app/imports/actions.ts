'use server';
import { headers } from 'next/headers';
import { localAccessAllowed,UUID } from '@/lib/access';
import { getProviderImports,providerCommand } from '@/lib/provider-imports';
import { researchError } from '@/lib/research-response';
import { getProviderReviews,reviewProviderChange } from '@/lib/provider-reviews';
import { prepareProviderReview } from '@/lib/provider-handoff';
export async function localImports(command?:unknown){
 const h=await headers(),id=process.env.ALPHATRACE_DEV_USER_ID;
 if(!localAccessAllowed(process.env,h.get('host'))||!id||!UUID.test(id))return {error:'Local access is disabled.',status:401};
 try{if(!h.get('origin')||new URL(h.get('origin')!).host!==h.get('host'))return {error:'Invalid request origin.',status:403};}catch{return {error:'Invalid request origin.',status:403};}
 try{const result=command===undefined?{}:await providerCommand(id,command);return {...result,imports:await getProviderImports(id)};}catch(e){return researchError(e);}
}
export async function localImportReviews(query:{status?:string;before?:string;id?:string},command?:unknown){
 const h=await headers(),id=process.env.ALPHATRACE_DEV_USER_ID;
 if(!localAccessAllowed(process.env,h.get('host'))||!id||!UUID.test(id))return {error:'Local access is disabled.',status:401};
 try{if(!h.get('origin')||new URL(h.get('origin')!).host!==h.get('host'))return {error:'Invalid request origin.',status:403};}catch{return {error:'Invalid request origin.',status:403};}
 try{if(command!==undefined)return (command as {action?:string})?.action==='prepare'?await prepareProviderReview(id,command):await reviewProviderChange(id,command);return {reviews:await getProviderReviews(id,query)};}catch(e){return researchError(e);}
}
