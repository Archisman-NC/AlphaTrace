import { headers } from 'next/headers';
import { localAccessAllowed,UUID } from '@/lib/access';
import { publicAuthConfig } from '@/lib/invites';
import AccountWorkspace from '@/components/account-workspace';
export const dynamic='force-dynamic';
export default async function WorkspacePage({searchParams}:{searchParams:Promise<{view?:string}>}){
 const h=await headers(),id=process.env.ALPHATRACE_DEV_USER_ID,params=await searchParams;
 const local=localAccessAllowed(process.env,h.get('host'))&&!!id&&UUID.test(id);
 return <AccountWorkspace companiesView={params.view==='companies'} local={local} authConfig={local?null:publicAuthConfig({SUPABASE_URL:process.env.SUPABASE_URL,SUPABASE_PUBLISHABLE_KEY:process.env.SUPABASE_PUBLISHABLE_KEY})}/>;
}
