import {apiFetch} from '@/lib/api-fetch';
import { localIssues } from '@/app/issues/actions';
import type { IssueQuery,IssueResult } from './issue-contract';
export async function requestIssues(local:boolean,token:string|undefined,query:IssueQuery={},command?:unknown):Promise<IssueResult>{
 if(local)return localIssues(query,command);
 const params=new URLSearchParams();for(const [key,value] of Object.entries(query))if(value)params.set(key,value);
 const response=await apiFetch('/api/issues?'+params,{method:command?'POST':'GET',headers:{Authorization:`Bearer ${token}`,...(command?{'Content-Type':'application/json'}:{})},cache:'no-store',signal:AbortSignal.timeout(20000),...(command?{body:JSON.stringify(command)}:{})});
 return {...await response.json(),status:response.status};
}
