import { headers } from 'next/headers';
import { localAccessAllowed,UUID } from '@/lib/access';
import { publicAuthConfig } from '@/lib/invites';
import EmailSettings from '@/components/email-settings';
export const dynamic='force-dynamic';
export default async function SettingsPage(){
 const h=await headers(),id=process.env.ALPHATRACE_DEV_USER_ID;
 const local=localAccessAllowed(process.env,h.get('host'))&&!!id&&UUID.test(id);
 return <EmailSettings local={local} authConfig={local?null:publicAuthConfig(process.env)}/>;
}
