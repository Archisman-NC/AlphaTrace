import { metrics,publicationGaps,comparisonResult,type ResearchPayload,type Source } from './research-contract';
// These are factual retrieval fields. Editable document labels cannot fabricate them.
export function sourceIdentity(s:Source){return JSON.stringify([s.url,s.sha256,s.fetchedAt,s.mediaType,s.fingerprintKind,s.retrieval,s.pages,s.bytes]);}
export function researchChanges(before:ResearchPayload,after:ResearchPayload):string[]{
 const changes:string[]=[];
 for(const key of ['title','periodEnd','comparison'] as const)if(before[key]!==after[key])changes.push(`${key}: ${before[key]} → ${after[key]}`);
 for(const s of after.sources){const old=before.sources.find(o=>o.id===s.id);if(!old)changes.push(`Added source: ${s.title}`);else if(JSON.stringify(old)!==JSON.stringify(s))changes.push(`Updated source: ${s.title}`);}
 for(const s of before.sources)if(!after.sources.some(n=>n.id===s.id))changes.push(`Removed source: ${s.title}`);
 for(const c of after.comparisons){const old=before.comparisons.find(o=>o.metric===c.metric);if(!old){changes.push(`Added metric: ${metrics[c.metric]}`);continue;}
  for(const side of ['previous','current'] as const)for(const key of [...new Set([...Object.keys(old[side]),...Object.keys(c[side])])] as (keyof typeof c.previous)[])if(old[side][key]!==c[side][key])changes.push(`${metrics[c.metric]} · ${side} ${key}: ${old[side][key]??'—'} → ${c[side][key]??'—'}`);
  if(old.note!==c.note)changes.push(`Updated note: ${metrics[c.metric]}`);
 }
 for(const c of before.comparisons)if(!after.comparisons.some(n=>n.metric===c.metric))changes.push(`Removed metric: ${metrics[c.metric]}`);
 if(JSON.stringify(before.limitations)!==JSON.stringify(after.limitations))changes.push('Updated limitations.');
 return changes;
}
export function draftReadiness(p:ResearchPayload,check:{status:string;checked_at:string;fingerprints:string[]}|null,now=Date.now()){
 const gaps=publicationGaps(p);
 for(const c of p.comparisons){const r=comparisonResult(c,p.comparison);if(!r.available)gaps.push(`${metrics[c.metric]}: ${r.reason}`);}
 if(!check||check.status!=='success'||now-Date.parse(check.checked_at)>7*86400000||!p.sources.every(s=>check.fingerprints.includes(s.sha256)))gaps.push('Recheck every exact source successfully before approval.');
 return gaps;
}
