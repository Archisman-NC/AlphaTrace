import { headers } from 'next/headers';
import { localAccessAllowed, UUID } from '@/lib/access';
import { getWorkspace } from '@/lib/workspaces';
import { WorkspaceForm } from './workspace-form';
import ResearchApp from '@/components/research-app';
export const dynamic='force-dynamic';
export default async function Home() {
 const h=await headers(),id=process.env.ALPHATRACE_DEV_USER_ID;
 const local=localAccessAllowed(process.env,h.get('host'))&&!!id&&UUID.test(id);
 let name:string|undefined;let unavailable=false;
 if(local){try{name=(await getWorkspace(id!))?.name;}catch{unavailable=true;}}
 return <ResearchApp workspaceName={name} workspaceSettings={local?<><WorkspaceForm name={name}/>{unavailable&&<p className="warning">PostgreSQL is offline. Sample research still works; start the database to save your workspace name.</p>}</>:<p>Private account sign-in is planned for the next milestone. This sample contains no private account data.</p>}/>;
}
