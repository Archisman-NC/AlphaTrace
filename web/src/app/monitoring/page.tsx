import { headers } from 'next/headers';
import { localAccessAllowed,UUID } from '@/lib/access';
import { publicAuthConfig } from '@/lib/invites';
import MonitoringWorkspace from '@/components/monitoring-workspace';
export const dynamic='force-dynamic';
export default async function MonitoringPage(){
 const h=await headers(),id=process.env.ALPHATRACE_DEV_USER_ID;
 const local=localAccessAllowed(process.env,h.get('host'))&&!!id&&UUID.test(id);
 return <MonitoringWorkspace local={local} authConfig={local?null:publicAuthConfig({SUPABASE_URL:process.env.SUPABASE_URL,SUPABASE_PUBLISHABLE_KEY:process.env.SUPABASE_PUBLISHABLE_KEY})}/>;
}
