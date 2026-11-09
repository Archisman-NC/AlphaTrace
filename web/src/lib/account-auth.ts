import {createClient} from '@supabase/supabase-js';
import {publicAuthConfig} from './invites';
import {AccountInputError} from './account-contract';
const options={auth:{persistSession:false,autoRefreshToken:false,detectSessionInUrl:false},global:{fetch:(input:RequestInfo|URL,init?:RequestInit)=>fetch(input,{...init,signal:AbortSignal.timeout(15000)})}};
export async function reauthenticate(owner:string,token:string,password:string){
 const config=publicAuthConfig(process.env);if(!config)throw Error('Account authentication unavailable');
 const client=createClient(config.url,config.key,options),verified=await client.auth.getUser(token);
 if(verified.error||verified.data.user?.id!==owner||!verified.data.user?.email)throw new AccountInputError('Your session expired. Sign in again.');
 const signed=await client.auth.signInWithPassword({email:verified.data.user.email,password});
 if(signed.error||signed.data.user?.id!==owner||!signed.data.session)throw new AccountInputError('Current password could not be verified. Check it and try again.');
 return client;
}
export async function removeAccountLogin(owner:string){
 const url=process.env.SUPABASE_URL,key=process.env.SUPABASE_SECRET_KEY;
 if(!url||!key)throw Error('Account deletion is not configured');
 const client=createClient(url,key,options);
 const result=await client.auth.admin.deleteUser(owner,false);
 if(result.error&&result.error.code!=='user_not_found')throw Error('Login deletion is pending retry');
}
