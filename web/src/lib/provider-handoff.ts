import { asUser } from './db';
import { UUID } from './access';
import { contentHash } from './research';
import { providerResearchDraft } from './provider-draft';
import { evaluateProvider,type ProviderSnapshot } from './provider-evaluation';
import { validateResearch,ResearchConflictError,ResearchForbiddenError,ResearchInputError,type ResearchPayload } from './research-contract';

export async function prepareProviderReview(owner:string,input:unknown){
 const b=input as {id?:string};
 if(!b||typeof b.id!=='string'||!UUID.test(b.id))throw new ResearchInputError('Choose an import review.');
 return asUser(owner,async c=>{
  if(!(await c.query('SELECT 1 FROM alphatrace.research_operators WHERE user_id=$1',[owner])).rowCount)throw new ResearchForbiddenError('Evidence preparation requires operator access.');
  // Match the provider -> research lock order used by import preparation.
  await c.query("SELECT pg_advisory_xact_lock(hashtext('indianapi-quota'))");
  const q=(await c.query('SELECT q.*,i.response_text,i.endpoint,i.sha256,h.sequence AS current_sequence FROM alphatrace.provider_reviews q JOIN alphatrace.provider_imports i ON i.id=q.current_import_id JOIN alphatrace.provider_review_heads h ON h.symbol=q.symbol WHERE q.id=$1',[b.id])).rows[0];
  if(!q)throw new ResearchInputError('Import review not found.');
  const linked=(await c.query('SELECT draft_id FROM alphatrace.provider_review_drafts WHERE review_id=$1',[q.id])).rows[0];
  if(linked)return {reviewId:q.id,symbol:q.symbol,message:'Existing evidence work retained.'};
  if(q.sequence!==q.current_sequence)throw new ResearchConflictError('A newer observation exists. Start evidence review from the latest change.');
  await c.query('SELECT pg_advisory_xact_lock(hashtext($1))',['research:'+q.symbol]);
  const s:ProviderSnapshot={requestedSymbol:q.symbol,endpoint:q.endpoint,responseText:q.response_text,sha256:q.sha256,fetchedAt:new Date(q.observed_at).toISOString()};
  const evaluation=evaluateProvider(s);
  let template:ResearchPayload;
  if(evaluation.reconciled)template=providerResearchDraft(s);
  else {
   const prior=(await c.query('SELECT payload FROM alphatrace.research_drafts WHERE symbol=$1 ORDER BY created_at DESC,id DESC LIMIT 1',[q.symbol])).rows[0];
   if(!prior)throw new ResearchConflictError('No official evidence template is available for this company. Attach and ingest its first official results package before preparing a draft.');
   template=validateResearch(prior.payload);
  }
  // The API is a private investigation cue, never a cited official source.
  // Blank every value so a prior period or prior approval cannot silently carry over.
  const payload=validateResearch({...template,title:`${q.symbol}: evidence work for import ${q.id.slice(0,8)}`,
   comparisons:template.comparisons.map(p=>({...p,previous:{...p.previous,value:null},current:{...p.current,value:null},note:'Enter values checked against official evidence; provider metric definitions may differ.'})),
   limitations:[`Investigation started from provider observation ${q.id}. Provider reporting periods: ${evaluation.periods.join(', ')}.`,
    'This is an incomplete evidence template. Confirm all values, reporting dates, units and definitions against the documents. Replace older sources before changing reporting periods.',
    'IndianAPI values remain in the private import review. They have not been mapped into reported financial metrics.'],extractorVersion:'provider-review-handoff-v1'});
  const draft=(await c.query('INSERT INTO alphatrace.research_drafts(symbol,content_hash,payload,created_by) VALUES($1,$2,$3,$4) RETURNING id',[q.symbol,contentHash(payload),JSON.stringify(payload),owner])).rows[0];
  await c.query('INSERT INTO alphatrace.provider_review_drafts(review_id,draft_id,created_by) VALUES($1,$2,$3)',[q.id,draft.id,owner]);
  return {reviewId:q.id,symbol:q.symbol,message:'Unapproved evidence template prepared. Complete values and sources in the evidence workbench.'};
 });
}
