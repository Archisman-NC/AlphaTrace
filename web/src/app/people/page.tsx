import {publicAuthConfig} from '@/lib/invites';
import PeopleWorkspace from '@/components/people-workspace';
export const dynamic='force-dynamic';
export default function PeoplePage(){return <PeopleWorkspace authConfig={publicAuthConfig(process.env)}/>;}
