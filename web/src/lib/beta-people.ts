import {createHash} from 'node:crypto';
import {asUser} from './db';
import {betaCommand,betaToken,BetaConflictError,BetaForbiddenError,BetaInvitationError,type BetaPeople,type BetaInvitation} from './beta-contract';
import {recoveryPassword} from './password-recovery';
import {activateAuth,type ActivationIdentity} from './beta-auth';
const capability='00000000-0000-4000-8000-000000000001';
const hash=(token:string)=>createHash('sha256').update(betaToken(token)).digest('hex');
function databaseError(error:unknown):never{const e=error as {code?:string;message?:string};if(e.code==='42501')throw new BetaForbiddenError('Only research operators can manage beta access.');if(e.code==='23505')throw new BetaConflictError('Access or seat availability changed. Refresh People and check the existing email record.');throw error;}
export async function getPeople(owner:string):Promise<BetaPeople>{try{return await asUser(owner,async c=>(await c.query('SELECT alphatrace.beta_people() AS people')).rows[0].people,{readOnlySnapshot:true});}catch(e){return databaseError(e);}}
export async function managePeople(owner:string,input:unknown){
 const command=betaCommand(input);
 try{await asUser(owner,async c=>{
  if(command.action==='invite')await c.query('SELECT alphatrace.beta_invite($1,$2,$3)',[command.id,command.email,hash(command.token)]);
  else await c.query('SELECT alphatrace.beta_manage($1,$2,$3,$4)',[command.id,command.version,command.action,command.token?hash(command.token):null]);
 });return await getPeople(owner);}catch(e){return databaseError(e);}
}
export async function inspectInvitation(token:unknown):Promise<BetaInvitation>{
 const digest=hash(betaToken(token));
 const invite=await asUser(capability,async c=>(await c.query('SELECT alphatrace.beta_invitation($1) AS invite',[digest])).rows[0].invite,{accountControl:true});
 if(!invite)throw new BetaInvitationError('This invitation has expired, was used, or was revoked. Ask the operator for a new link.');
 return {email:invite.email,expiresAt:invite.expiresAt,existing:invite.existing};
}
export async function acceptInvitation(input:unknown,bearer?:string,activate=activateAuth){
 const b=input as Record<string,unknown>,digest=hash(betaToken(b?.token));
 if(!bearer){try{if(typeof b?.password!=='string'||typeof b?.confirmPassword!=='string')throw Error('Password required');recoveryPassword(b.password,b.confirmPassword);}catch{throw new BetaInvitationError('Choose matching passwords of 12–128 characters.');}}
 return asUser(capability,async c=>{
  // Serialize acceptance, reissue and revocation while the bounded Auth request runs.
  const invite=(await c.query('SELECT alphatrace.beta_invitation($1,true) AS invite',[digest])).rows[0].invite as ActivationIdentity|null;
  if(!invite)throw new BetaInvitationError('This invitation has expired, was used, or was revoked. If activation was interrupted, try signing in with the password you chose.');
  const user=await activate(invite,bearer?undefined:String(b.password),bearer);
  try{await c.query('SELECT alphatrace.beta_activate($1,$2)',[digest,user]);}catch(e){if((e as {code?:string}).code==='23505')throw new BetaConflictError('This account cannot accept the invitation. Ask the operator to check access.');throw e;}
  return {activated:true};
 },{accountControl:true});
}
