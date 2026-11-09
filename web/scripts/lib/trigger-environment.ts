import { readFileSync } from 'node:fs';
import { readLocalEnvironment } from './local-environment.mjs';
import { UUID } from '../../src/lib/access';

export function runtimeEnvironment(runtime:Record<string,string>) {
 const connection=runtime.DATABASE_URL;
 if(!connection?.trim())throw Error('DATABASE_URL is missing. Add it to this deployment environment.');
 if(connection!==connection.trim()||/^["']|["']$/.test(connection))throw Error('DATABASE_URL contains surrounding quotes or whitespace. Paste only the connection URL value.');
 let url:URL;
 try{url=new URL(connection);}catch{throw Error('DATABASE_URL is not a valid PostgreSQL URL. Paste its value, without DATABASE_URL= or surrounding quotes.');}
 if(!['postgres:','postgresql:'].includes(url.protocol)||!url.hostname)throw Error('DATABASE_URL must be a PostgreSQL connection URL.');
 if(!/^alphatrace_app(?:\.[a-z]{20})?$/.test(url.username))throw Error('DATABASE_URL must use the restricted alphatrace_app role.');
 if(url.search)throw Error('DATABASE_URL must not contain query parameters; configure TLS through DATABASE_SSL and DATABASE_SSL_CA_PEM.');
 if(runtime.DATABASE_SSL!=='true')throw Error('DATABASE_SSL must be true for the hosted database.');
 if(!UUID.test(runtime.ALPHATRACE_INGEST_USER_ID??''))throw Error('ALPHATRACE_INGEST_USER_ID must be the enrolled operator UUID.');
 let ca=runtime.DATABASE_SSL_CA_PEM||'';
 if(!ca&&runtime.DATABASE_SSL_CA_FILE){try{ca=readFileSync(runtime.DATABASE_SSL_CA_FILE,'utf8');}catch{throw Error('DATABASE_SSL_CA_FILE cannot be read. On Vercel, configure DATABASE_SSL_CA_PEM instead of a local file path.');}}
 if(!ca.includes('-----BEGIN CERTIFICATE-----'))throw Error('DATABASE_SSL_CA_PEM must contain the trusted database certificate.');
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
