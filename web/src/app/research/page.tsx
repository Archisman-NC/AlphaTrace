import { headers } from 'next/headers';
import { localAccessAllowed,UUID } from '@/lib/access';
import { publicAuthConfig } from '@/lib/invites';
import ResearchWorkspace from '@/components/research-workspace';
export const dynamic='force-dynamic';
export default async function ResearchPage({searchParams}:{searchParams:Promise<{symbol?:string;admin?:string;update?:string;review?:string;importReview?:string}>}){
 const h=await headers(),id=process.env.ALPHATRACE_DEV_USER_ID,params=await searchParams;
 const local=localAccessAllowed(process.env,h.get('host'))&&!!id&&UUID.test(id);
 return <ResearchWorkspace local={local} initialImportReview={params.importReview&&UUID.test(params.importReview)?params.importReview:undefined} returnReview={params.review&&UUID.test(params.review)?params.review:undefined} initialSymbol={params.symbol??'TCS'} initialUpdate={params.update} initialAdmin={params.admin==='1'} authConfig={local?null:publicAuthConfig({SUPABASE_URL:process.env.SUPABASE_URL,SUPABASE_PUBLISHABLE_KEY:process.env.SUPABASE_PUBLISHABLE_KEY})}/>;
}
