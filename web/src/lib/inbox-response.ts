import { InboxInputError,InboxConflictError } from './inbox-contract';
import { RequestBodyError } from './http-input';
export function inboxError(e:unknown){
 if(e instanceof RequestBodyError)return {error:e.message,status:e.status};
 if(e instanceof InboxInputError)return {error:e.message,status:400};
 if(e instanceof InboxConflictError)return {error:e.message,status:409};
 return {error:'The inbox request could not be confirmed. Your saved reviews remain stored; retry when connected.',status:503};
}
