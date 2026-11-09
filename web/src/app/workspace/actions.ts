'use server';
import { headers } from 'next/headers';
import { localAccessAllowed,UUID } from '@/lib/access';
import { getPortfolio,updatePortfolio } from '@/lib/portfolios';
import { PortfolioInputError,PortfolioConflictError,type Portfolio } from '@/lib/portfolio-input';
export type PortfolioResult={portfolio?:Portfolio;error?:string;status?:number};
export async function localPortfolio(input?:unknown):Promise<PortfolioResult> {
 const h=await headers(),id=process.env.ALPHATRACE_DEV_USER_ID;
 if(!localAccessAllowed(process.env,h.get('host'))||!id||!UUID.test(id))return {error:'Local access is disabled.',status:401};
 try{if(!h.get('origin')||new URL(h.get('origin')!).host!==h.get('host'))return {error:'Invalid request origin.',status:403};}
 catch{return {error:'Invalid request origin.',status:403};}
 try{return {portfolio:input===undefined?await getPortfolio(id):await updatePortfolio(id,input)};}
 catch(e){
  if(e instanceof PortfolioInputError)return {error:e.message,status:400};
  if(e instanceof PortfolioConflictError)return {error:e.message,status:409};
  return {error:'Portfolio storage is unavailable. Start PostgreSQL and apply migrations, then reload.',status:503};
 }
}
