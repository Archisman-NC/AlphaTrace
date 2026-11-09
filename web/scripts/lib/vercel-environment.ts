import {runtimeEnvironment} from './trigger-environment';
import {UUID} from '../../src/lib/access';
export function vercelEnvironment(env:Record<string,string|undefined>){
 const values=Object.fromEntries(Object.entries(env).filter((entry):entry is [string,string]=>typeof entry[1]==='string'));
 const required=['DATABASE_URL','DATABASE_SSL','DATABASE_SSL_CA_PEM','SUPABASE_URL','SUPABASE_PUBLISHABLE_KEY','SUPABASE_SECRET_KEY','TRIGGER_SECRET_KEY','ALPHATRACE_INGEST_USER_ID','ALPHATRACE_INVITED_USER_IDS','ALPHATRACE_LOCAL_DEV','ALPHATRACE_CLOUD_MONITORING','ALPHATRACE_DOCUMENT_JOBS'];
 const missing=required.filter(name=>!values[name]?.trim());
 if(missing.length)throw Error(`Missing Vercel environment variables: ${missing.join(', ')}. Add them in Project Settings > Environment Variables for Production, save, then redeploy. The private .env.local file is intentionally not uploaded.`);
 const base=runtimeEnvironment(values);
 if(!/^https:\/\/[a-z]{20}\.supabase\.co\/?$/.test(values.SUPABASE_URL??'')||!values.SUPABASE_PUBLISHABLE_KEY?.startsWith('sb_publishable_'))throw Error('Configure the Supabase project URL and publishable key.');
 if(!values.SUPABASE_SECRET_KEY?.startsWith('sb_secret_'))throw Error('Configure the server-only Supabase secret key.');
 if(!values.TRIGGER_SECRET_KEY?.startsWith('tr_prod_'))throw Error('TRIGGER_SECRET_KEY must be the Trigger.dev Production key. Keep the development key in TRIGGER_DEV_SECRET_KEY.');
 const invited=(values.ALPHATRACE_INVITED_USER_IDS??'').split(',').map(v=>v.trim());
 if(!invited.length||invited.length>5||invited.some(v=>!UUID.test(v))||!invited.includes(base.ALPHATRACE_INGEST_USER_ID))throw Error('Configure one to five invited users, including the enrolled operator.');
 if(values.ALPHATRACE_LOCAL_DEV!=='false')throw Error('Disable local authentication before deployment.');
 if(values.ALPHATRACE_CLOUD_MONITORING!=='true'||values.ALPHATRACE_DOCUMENT_JOBS!=='true')throw Error('ALPHATRACE_CLOUD_MONITORING and ALPHATRACE_DOCUMENT_JOBS must both be true on Vercel.');
 return {...base,SUPABASE_URL:values.SUPABASE_URL,SUPABASE_PUBLISHABLE_KEY:values.SUPABASE_PUBLISHABLE_KEY,SUPABASE_SECRET_KEY:values.SUPABASE_SECRET_KEY,TRIGGER_SECRET_KEY:values.TRIGGER_SECRET_KEY,ALPHATRACE_INVITED_USER_IDS:invited.join(','),ALPHATRACE_LOCAL_DEV:'false',ALPHATRACE_CLOUD_MONITORING:'true',ALPHATRACE_DOCUMENT_JOBS:'true'};
}
