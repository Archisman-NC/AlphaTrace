import { ResearchInputError,ResearchConflictError,ResearchForbiddenError } from './research-contract';
import { RequestBodyError } from './http-input';
export const researchJson=(body:unknown,status=200)=>Response.json(body,{status,headers:{'Cache-Control':'no-store'}});
export function researchError(e:unknown){
 if(e instanceof RequestBodyError)return {error:e.message,status:e.status};
 if(e instanceof ResearchInputError)return {error:e.message,status:400};
 if(e instanceof ResearchConflictError)return {error:e.message,status:409};
 if(e instanceof ResearchForbiddenError)return {error:e.message,status:403};
 return {error:'Research storage is unavailable. Reload before retrying.',status:503};
}
