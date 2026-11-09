'use client';
import { useActionState } from 'react';
import { updateWorkspace,type FormState } from './actions';
export function WorkspaceForm({name}:{name?:string}) {
  const [state,action,pending]=useActionState<FormState,FormData>(updateWorkspace,{});
  return <form action={action}>
    <label htmlFor="name">Workspace name</label>
    <div className="form-row"><input id="name" name="name" defaultValue={name} placeholder="My investment research" maxLength={80} required autoComplete="off"/>
    <button disabled={pending}>{pending?'Saving…':name?'Save changes':'Create workspace'}</button></div>
    <p role="status" className={state.error?'error':'success'}>{state.error??state.success??'Stored locally in PostgreSQL. Refresh to verify persistence.'}</p>
  </form>;
}
