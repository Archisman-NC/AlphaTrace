import { IssueInputError,IssueConflictError,IssueForbiddenError } from './issue-contract';
import { RequestBodyError } from './http-input';
export const issueJson=(body:unknown,status=200)=>Response.json(body,{status,headers:{'Cache-Control':'no-store'}});
export function issueError(e:unknown){
 if(e instanceof RequestBodyError)return {error:e.message,status:e.status};
 if(e instanceof IssueInputError)return {error:e.message,status:400};
 if(e instanceof IssueConflictError)return {error:e.message,status:409};
 if(e instanceof IssueForbiddenError)return {error:e.message,status:403};
 return {error:'Report storage is unavailable. The save could not be confirmed; retry the same action before editing it.',status:503};
}
