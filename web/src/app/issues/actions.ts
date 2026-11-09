'use server';
import { headers } from 'next/headers';
import { localAccessAllowed,UUID } from '@/lib/access';
import { getIssues,saveIssue } from '@/lib/issues';
import { issueError } from '@/lib/issue-response';
import type { IssueQuery,IssueResult } from '@/lib/issue-contract';
export async function localIssues(query:IssueQuery={},command?:unknown):Promise<IssueResult>{
 const h=await headers(),id=process.env.ALPHATRACE_DEV_USER_ID;
 if(!localAccessAllowed(process.env,h.get('host'))||!id||!UUID.test(id))return {error:'Local access is disabled.',status:401};
 try{if(!h.get('origin')||new URL(h.get('origin')!).host!==h.get('host'))return {error:'Invalid request origin.',status:403};}catch{return {error:'Invalid request origin.',status:403};}
 try{return command===undefined?{issues:await getIssues(id,query)}:await saveIssue(id,command);}catch(e){return issueError(e);}
}
