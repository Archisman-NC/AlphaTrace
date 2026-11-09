import { readFile,writeFile,mkdir } from 'node:fs/promises';
import { resolve } from 'node:path';
import { createHash } from 'node:crypto';
import { getResearch } from '../src/lib/research';
import { attachResearchSource,checkResearchSources,inspectDocument } from '../src/lib/research-source';
import { reviseResearch } from '../src/lib/research-edit';
import { closeDatabase } from '../src/lib/db';
import { researchChanges } from '../src/lib/research-edit-contract';
import { comparisonResult,type Source } from '../src/lib/research-contract';
import { completeWiproDraft,wiproReviewOriginals } from './lib/wipro-review';
import type { ResearchView } from '../src/lib/research-view';
const prepare=process.argv.includes('--prepare'),directory=resolve(process.argv.find(a=>a.startsWith('--directory='))?.slice(12)??'.local/wipro-review');
try{
 const owner=process.env.ALPHATRACE_INGEST_USER_ID;if(!owner)throw Error('Configure the enrolled research operator.');
 const view:ResearchView=await getResearch(owner,'WIPRO',true),base=view.drafts.find(d=>d.payload.symbol==='WIPRO'&&d.state!=='published'&&!d.revised_by&&!d.superseded_by);
 if(!base)throw Error('No current Wipro draft is available.');
 const sources:Source[]=[];
 for(const ref of wiproReviewOriginals){
  const bytes=await readFile(resolve(directory,ref.key+'.html'));
  if(createHash('sha256').update(bytes).digest('hex')!==ref.sha256)throw Error(`The ${ref.key} original differs; reconcile it before continuing.`);
  await inspectDocument(bytes);
  const metadata=JSON.parse(await readFile(resolve(directory,ref.key+'.metadata.json'),'utf8'));
  if(metadata.sha256!==ref.sha256||metadata.url!==ref.url||metadata.bytes!==bytes.length)throw Error('Original metadata mismatch.');
  if(prepare){sources.push((await attachResearchSource(owner,{id:base.id,sourceId:ref.id,url:ref.url,title:ref.title,kind:ref.kind,publishedOn:ref.publishedOn},new File([bytes],ref.key+'.html',{type:'text/html'}))).source);}
  else sources.push({...ref,fetchedAt:metadata.fetchedAt,bytes:bytes.length,pages:1,mediaType:'text/html',fingerprintKind:'document_bytes',retrieval:'operator_file'});
 }
 const payload=completeWiproDraft(base.payload,sources);
 await mkdir(directory,{recursive:true});await writeFile(resolve(directory,'prepared-draft.json'),JSON.stringify(payload,null,2));
 let id=base.id;
 if(prepare){if(researchChanges(base.payload,payload).length)id=(await reviseResearch(owner,{id:base.id,payload,reason:'Completed the Wipro quarterly evidence with both official releases and an explicitly scoped IT-services margin comparison; reconciled profit definitions and retained exact original snapshots.'})).id;await checkResearchSources(owner,id);}
 console.log(JSON.stringify({mode:prepare?'saved-unapproved':'preview',draftId:id,metrics:payload.comparisons.map(c=>({metric:c.metric,previous:c.previous.value,current:c.current.value,...comparisonResult(c,payload.comparison)})),sourceCount:payload.sources.length,publication:'No approval or publication performed.'},null,2));
}catch(e){console.error(e instanceof Error?e.message:'Review preparation failed.');process.exitCode=1;}finally{await closeDatabase();}
