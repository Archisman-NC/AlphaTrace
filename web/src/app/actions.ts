'use server';
import { headers } from 'next/headers';
import { revalidatePath } from 'next/cache';
import { localAccessAllowed, UUID, workspaceName } from '@/lib/access';
import { saveWorkspace } from '@/lib/workspaces';
export type FormState={error?:string;success?:string};
export async function updateWorkspace(_previous:FormState,form:FormData):Promise<FormState> {
  const h=await headers();
  const id=process.env.ALPHATRACE_DEV_USER_ID;
  if(!localAccessAllowed(process.env,h.get('host')) || !id || !UUID.test(id)) return {error:'Local development access is disabled.'};
  // Do not allow another website to submit against the local development workspace.
  const origin=h.get('origin');
  if(!origin || new URL(origin).host!==h.get('host')) return {error:'Invalid request origin.'};
  let name:string;
  try { name=workspaceName(form.get('name')); } catch(error) { return {error:(error as Error).message}; }
  try { await saveWorkspace(id,name); revalidatePath('/demo'); return {success:'Workspace saved to PostgreSQL.'}; }
  catch { return {error:'Could not save. Check the database connection and run migrations.'}; }
}
