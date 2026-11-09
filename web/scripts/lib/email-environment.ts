import { emailsAvailable } from '../../src/lib/email-contract';
import { invitedUser } from '../../src/lib/invites';
/** Only named runtime values; never export the whole local environment. */
export function emailEnvironment(env:Record<string,string|undefined>):Record<string,string>{
 if(env.ALPHATRACE_EMAILS_ENABLED!=='true')return {ALPHATRACE_EMAILS_ENABLED:'false'};
 if(!emailsAvailable(env))throw Error('Email delivery requires a sender, Resend API key, webhook secret, HTTPS app origin and Supabase account verification.');
 const ids=(env.ALPHATRACE_INVITED_USER_IDS??'').split(',').map(s=>s.trim());
 if(!ids.length||ids.some(id=>!invitedUser(id,env.ALPHATRACE_INVITED_USER_IDS)))throw Error('Configure the email pilot allowlist.');
 return Object.fromEntries(['ALPHATRACE_EMAILS_ENABLED','ALPHATRACE_APP_URL','RESEND_API_KEY','RESEND_FROM_EMAIL','RESEND_WEBHOOK_SECRET','ALPHATRACE_INVITED_USER_IDS'].map(k=>[k,env[k]!]));
}
