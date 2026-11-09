import { headers } from 'next/headers';
import { localAccessAllowed,UUID } from '@/lib/access';
import { publicAuthConfig } from '@/lib/invites';
import IssuesWorkspace from '@/components/issues-workspace';
export const dynamic='force-dynamic';
export default async function IssuesPage({searchParams}:{searchParams:Promise<{report?:string;scope?:string}>}){
 const h=await headers(),id=process.env.ALPHATRACE_DEV_USER_ID,params=await searchParams;
 const local=localAccessAllowed(process.env,h.get('host'))&&!!id&&UUID.test(id);
 return <IssuesWorkspace local={local} authConfig={local?null:publicAuthConfig(process.env)} initialReport={params.report} initialScope={params.scope==='queue'?'queue':'mine'}/>;
}
