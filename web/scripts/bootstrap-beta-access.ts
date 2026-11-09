import {createClient} from '@supabase/supabase-js';
import {asUser,closeDatabase} from '../src/lib/db';
import {getPeople} from '../src/lib/beta-people';
import {invitedUser} from '../src/lib/invites';
async function main(){
 const owner=process.env.ALPHATRACE_INGEST_USER_ID,url=process.env.SUPABASE_URL,key=process.env.SUPABASE_SECRET_KEY;
 if(!owner||!url||!key)throw Error('Required settings missing');
 const existing=await getPeople(owner);if(existing.initialized){console.log(`Beta access already initialized: ${existing.used}/${existing.limit} seats.`);return;}
 const ids=[...new Set((process.env.ALPHATRACE_INVITED_USER_IDS??'').split(',').map(id=>id.trim()))];
 if(!ids.includes(owner)||!ids.length||ids.some(id=>!invitedUser(id,process.env.ALPHATRACE_INVITED_USER_IDS)))throw Error('Invalid existing allowlist');
 const auth=createClient(url,key,{auth:{persistSession:false,autoRefreshToken:false},global:{fetch:(url,init)=>fetch(url,{...init,signal:AbortSignal.timeout(10000)})}});const profiles:{id:string;email:string}[]=[];
 for(const id of ids){const {data,error}=await auth.auth.admin.getUserById(id);if(error||!data.user.email||!data.user.email_confirmed_at)throw Error('Existing identity could not be verified');profiles.push({id,email:data.user.email});}
 await asUser(owner,c=>c.query('SELECT alphatrace.beta_initialize($1::jsonb)',[JSON.stringify(profiles)]));
 console.log(`Moved ${profiles.length} verified pilot accounts into beta access. No passwords or permissions changed.`);
}
try{await main();}catch{console.error('Beta initialization did not complete. Verify migration 020 and existing operator/allowlist settings. No credentials were logged.');process.exitCode=1;}finally{await closeDatabase();}
