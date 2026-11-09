import {createClient} from '@supabase/supabase-js';
import {documentBucket} from '../src/lib/document-job-contract';
import {manualFileLimit} from '../src/lib/manual-tcs-contract';
try{
 const url=process.env.SUPABASE_URL,key=process.env.SUPABASE_SECRET_KEY;
 if(!url||!/^https:\/\/[a-z]{20}\.supabase\.co\/?$/.test(url)||!key?.startsWith('sb_secret_'))throw Error();
 const storage=createClient(url,key,{auth:{persistSession:false,autoRefreshToken:false}}).storage;
 const {data:existing,error:lookup}=await storage.getBucket(documentBucket);
 if(lookup&&String(lookup.status)!=='404'&&String((lookup as {statusCode?:string}).statusCode)!=='404')throw Error();
 if(existing?.public)throw Error();
 const options={public:false,fileSizeLimit:manualFileLimit,allowedMimeTypes:['application/octet-stream']};
 const {error}=existing?await storage.updateBucket(documentBucket,options):await storage.createBucket(documentBucket,options);
 if(error)throw Error();
 console.log('Private document storage configured: 20 MB per object, signed uploads only. No public access policy was added.');
}catch{console.error('Private storage setup failed. Check the server-only Supabase secret and bucket configuration; no credentials were logged.');process.exitCode=1;}
