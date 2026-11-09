import { backfillProviderReviews } from '../src/lib/provider-reviews';
import { closeDatabase } from '../src/lib/db';
import { UUID } from '../src/lib/access';
try{
 const owner=process.env.ALPHATRACE_INGEST_USER_ID;if(!owner||!UUID.test(owner))throw Error();
 const result=await backfillProviderReviews(owner);
 console.log(`Import review backfill: ${result.observations} saved observations processed; ${result.created} queue items created. No provider calls or publication.`);
}catch{console.error('Import review backfill failed. Check migrations and operator configuration; no credentials were logged.');process.exitCode=1;}
finally{await closeDatabase();}
