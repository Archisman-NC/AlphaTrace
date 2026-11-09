import { createClient } from '@supabase/supabase-js';
import { invitedUser } from './invites';
export function confirmedRecipient(owner:string,user:{id:string;email?:string;email_confirmed_at?:string;deleted_at?:string;banned_until?:string}|null):string|null{
 if(!user||user.id!==owner||!user.email_confirmed_at||!user.email||user.deleted_at||user.banned_until&&Date.parse(user.banned_until)>Date.now())return null;
 return user.email;
}
/** Recheck verified Auth identity immediately before delivery; never accept a browser-supplied address. */
export async function verifiedEmailRecipient(owner:string):Promise<string|null>{
 if(!invitedUser(owner,process.env.ALPHATRACE_INVITED_USER_IDS))return null;
 const url=process.env.SUPABASE_URL,key=process.env.SUPABASE_SECRET_KEY;
 if(!url||!key)throw Error('Account verification is unavailable.');
 const auth=createClient(url,key,{auth:{persistSession:false,autoRefreshToken:false},global:{fetch:(url,init)=>fetch(url,{...init,signal:AbortSignal.timeout(10_000)})}});
 const {data,error}=await auth.auth.admin.getUserById(owner);
 if(error){if(error.status===404)return null;throw Error('Account verification is unavailable.');}
 return confirmedRecipient(owner,data.user);
}
