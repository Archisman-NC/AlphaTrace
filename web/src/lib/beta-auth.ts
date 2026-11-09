import {createClient,type User} from '@supabase/supabase-js';
import {BetaConflictError,BetaInvitationError} from './beta-contract';
const options={auth:{persistSession:false,autoRefreshToken:false,detectSessionInUrl:false},global:{fetch:(url:RequestInfo|URL,init?:RequestInit)=>fetch(url,{...init,signal:AbortSignal.timeout(12000)})}};
function clients(){const url=process.env.SUPABASE_URL,key=process.env.SUPABASE_SECRET_KEY,publishable=process.env.SUPABASE_PUBLISHABLE_KEY;if(!url||!key||!publishable)throw Error('Account activation unavailable');return {admin:createClient(url,key,options),auth:createClient(url,publishable,options)};}
export type ActivationIdentity={id:string;userId:string;email:string;existing:boolean};
function confirmed(user:User|null,email:string){return !!user?.email_confirmed_at&&user.email?.toLowerCase()===email;}
/** Only validated invitation holders reach this function. Existing passwords are never replaced. */
export async function activateAuth(invitation:ActivationIdentity,password?:string,bearer?:string):Promise<string>{
 const {admin,auth}=clients();
 if(bearer){const {data,error}=await auth.auth.getUser(bearer);if(error||!confirmed(data.user,invitation.email))throw new BetaInvitationError('Sign in with the email address on this invitation.');return data.user!.id;}
 if(invitation.existing||!password)throw new BetaInvitationError('Sign in with your existing account to accept this invitation.');
 const known=await admin.auth.admin.getUserById(invitation.userId);
 if(known.error&&known.error.status!==404&&known.error.code!=='user_not_found')throw Error('Account verification unavailable');
 if(known.data.user){
  if(!confirmed(known.data.user,invitation.email)||known.data.user.app_metadata?.alphatrace_invitation!==invitation.id)throw new BetaConflictError('Use your existing account to accept this invitation.');
  // Reconcile an interrupted earlier creation without silently changing its password.
  const signed=await auth.auth.signInWithPassword({email:invitation.email,password});
  if(signed.error||signed.data.user?.id!==invitation.userId)throw new BetaConflictError('An earlier activation may have created your login. Use the password you chose then, or recover it from sign-in.');
  await auth.auth.signOut({scope:'local'}).catch(()=>{});return invitation.userId;
 }
 const created=await admin.auth.admin.createUser({id:invitation.userId,email:invitation.email,password,email_confirm:true,app_metadata:{alphatrace_invitation:invitation.id}});
 if(created.error){if(['email_exists','user_already_exists'].includes(created.error.code??''))throw new BetaConflictError('This email already has a login. Choose “I already have an account” and sign in.');if(created.error.status&&created.error.status<500&&created.error.status!==429)throw new BetaConflictError('The login could not be created. Check the password requirements or sign in with an existing account.');throw Error('Account creation not confirmed');}
 if(created.data.user?.id!==invitation.userId||!confirmed(created.data.user,invitation.email))throw Error('Account creation not confirmed');
 return invitation.userId;
}
