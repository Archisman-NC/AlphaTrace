'use server';
import { headers } from 'next/headers';
import { localAccessAllowed,UUID } from '@/lib/access';
import { getOnboarding,saveOnboarding } from '@/lib/onboarding';
import { PortfolioConflictError,PortfolioInputError } from '@/lib/portfolio-input';
import type { OnboardingResult } from '@/lib/onboarding-contract';
export async function localOnboarding(input?:unknown):Promise<OnboardingResult>{
 const h=await headers(),id=process.env.ALPHATRACE_DEV_USER_ID;
 if(!localAccessAllowed(process.env,h.get('host'))||!id||!UUID.test(id))return {error:'Local access is disabled.',status:401};
 try{if(!h.get('origin')||new URL(h.get('origin')!).host!==h.get('host'))return {error:'Invalid request origin.',status:403};}catch{return {error:'Invalid request origin.',status:403};}
 try{return input===undefined?{onboarding:await getOnboarding(id)}:await saveOnboarding(id,input);}catch(e){
  if(e instanceof PortfolioInputError)return {error:e.message,status:400};
  if(e instanceof PortfolioConflictError)return {error:e.message,status:409};
  return {error:'Setup storage is unavailable. Reload before retrying.',status:503};
 }
}
