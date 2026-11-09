import { headers } from 'next/headers';
import { localAccessAllowed,UUID } from '@/lib/access';
import { publicAuthConfig } from '@/lib/invites';
import { inboxFilters,type InboxFilter } from '@/lib/inbox-contract';
import InboxWorkspace from '@/components/inbox-workspace';
export const dynamic='force-dynamic';
export default async function InboxPage({searchParams}:{searchParams:Promise<{update?:string;filter?:string}>}){
 const h=await headers(),id=process.env.ALPHATRACE_DEV_USER_ID,params=await searchParams;
 const local=localAccessAllowed(process.env,h.get('host'))&&!!id&&UUID.test(id);
 return <InboxWorkspace initialFilter={inboxFilters.includes(params.filter as InboxFilter)?params.filter as InboxFilter:params.update?'all':'pending'} local={local} initialUpdate={params.update&&UUID.test(params.update)?params.update:undefined} authConfig={local?null:publicAuthConfig({SUPABASE_URL:process.env.SUPABASE_URL,SUPABASE_PUBLISHABLE_KEY:process.env.SUPABASE_PUBLISHABLE_KEY})}/>;
}
