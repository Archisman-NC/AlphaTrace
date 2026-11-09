import {vercelEnvironment} from './lib/vercel-environment';
try{const env=vercelEnvironment(process.env);console.log(`Vercel production configuration ready: ${Object.keys(env).length} allowlisted settings. No credentials were printed or uploaded.`);}
catch(e){console.error(e instanceof Error?e.message:'Production configuration is incomplete.');process.exitCode=1;}
