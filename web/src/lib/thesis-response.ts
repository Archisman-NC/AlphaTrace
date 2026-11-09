import { ThesisInputError,ThesisConflictError } from './thesis-input';
import { RequestBodyError } from './http-input';
export function thesisError(e:unknown){
 if(e instanceof RequestBodyError)return {error:e.message,status:e.status};
 if(e instanceof ThesisInputError)return {error:e.message,status:400};
 if(e instanceof ThesisConflictError)return {error:e.message,status:409};
 return {error:'Could not confirm your research save. Your edits are still here; retry the same save or load the latest revision.',status:503};
}
