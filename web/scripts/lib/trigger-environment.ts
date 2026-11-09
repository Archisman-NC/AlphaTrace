import { readFileSync } from 'node:fs';
import { readLocalEnvironment } from './local-environment.mjs';
import { UUID } from '../../src/lib/access';

export function runtimeEnvironment(runtime:Record<string,string>) {
 const url=new URL(runtime.DATABASE_URL);
 if(!/^alphatrace_app(?:\.[a-z]{20})?$/.test(url.username)||url.search||runtime.DATABASE_SSL!=='true'||!UUID.test(runtime.ALPHATRACE_INGEST_USER_ID??''))throw Error('Invalid runtime configuration.');
 const ca=runtime.DATABASE_SSL_CA_PEM||(runtime.DATABASE_SSL_CA_FILE?readFileSync(runtime.DATABASE_SSL_CA_FILE,'utf8'):'');
 if(!ca.includes('-----BEGIN CERTIFICATE-----'))throw Error('A trusted database certificate is required.');
 return {DATABASE_URL:runtime.DATABASE_URL,DATABASE_SSL:'true',DATABASE_SSL_CA_PEM:ca,ALPHATRACE_INGEST_USER_ID:runtime.ALPHATRACE_INGEST_USER_ID};
}
/** Explicit cloud allowlist. Setup/admin credentials, test settings and Trigger keys stay local. */
export function productionEnvironment(directory=process.cwd()) {
 try {
  const runtime=readLocalEnvironment(directory),base=runtimeEnvironment(runtime);
  if(!runtime.INDIANAPI_API_KEY?.trim())throw Error();
  const storage:Record<string,string>={};
  if(runtime.SUPABASE_SECRET_KEY){
   if(!/^https:\/\/[a-z]{20}\.supabase\.co\/?$/.test(runtime.SUPABASE_URL)||!runtime.SUPABASE_SECRET_KEY.startsWith('sb_secret_'))throw Error();
   storage.SUPABASE_URL=runtime.SUPABASE_URL;storage.SUPABASE_SECRET_KEY=runtime.SUPABASE_SECRET_KEY;
  }
  return Object.entries({...base,INDIANAPI_API_KEY:runtime.INDIANAPI_API_KEY.trim(),...storage}).map(([name,value])=>({name,value,isSecret:true}));
 } catch { throw Error('Cloud environment preflight failed. Check .env.local app-role, certificate, operator, IndianAPI and private storage configuration.'); }
}
