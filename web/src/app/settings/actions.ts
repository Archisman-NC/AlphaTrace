'use server';
import { headers } from 'next/headers';
import { localAccessAllowed,UUID } from '@/lib/access';
import { getEmailSettings,saveEmailPreferences } from '@/lib/emails';
import type { EmailSettingsResult } from '@/lib/email-contract';
import { PortfolioConflictError,PortfolioInputError } from '@/lib/portfolio-input';
export async function localEmailSettings(input?:unknown):Promise<EmailSettingsResult>{
 const h=await headers(),id=process.env.ALPHATRACE_DEV_USER_ID;
 if(!localAccessAllowed(process.env,h.get('host'))||!id||!UUID.test(id))return {error:'Local access is disabled.',status:401};
 try{if(!h.get('origin')||new URL(h.get('origin')!).host!==h.get('host'))return {error:'Invalid request origin.',status:403};}catch{return {error:'Invalid request origin.',status:403};}
 try{return {email:input===undefined?await getEmailSettings(id):await saveEmailPreferences(id,input)};}catch(e){
  if(e instanceof PortfolioInputError)return {error:e.message,status:400};
  if(e instanceof PortfolioConflictError)return {error:e.message,status:409};
  return {error:'Email settings are unavailable. Reload before retrying.',status:503};
 }
}
