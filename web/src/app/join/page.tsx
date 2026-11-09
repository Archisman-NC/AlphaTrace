import {publicAuthConfig} from '@/lib/invites';
import JoinBeta from '@/components/join-beta';
export const dynamic='force-dynamic';
export const metadata={title:'Join the AlphaTrace beta',referrer:'no-referrer'};
export default function JoinPage(){return <JoinBeta authConfig={publicAuthConfig(process.env)}/>;}
