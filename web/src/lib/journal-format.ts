import type { ThesisRevision } from './thesis-input';
import type { InboxSnapshot,ReviewEvent,ReviewState } from './inbox-contract';
import type { PublishedResearch } from './research-view';
import { comparisonResult } from './research-contract';

export type Journal={
 schemaVersion:1;exportedAt:string;scope:string;complete:true;
 workspace:{name:string;portfolio_version:number;created_at:string;updated_at:string}|null;
 portfolio:{symbol:string;quantity:string|null;followed:boolean;updated_at:string}[];
 theses:{symbol:string;version:number;updated_at:string}[];
 thesisRevisions:(ThesisRevision&{symbol:string})[];
 reviews:{update_id:string;snapshot:InboxSnapshot;version:number;state:ReviewState;revisit_at:string|null;created_at:string;updated_at:string}[];
 reviewEvents:(ReviewEvent&{update_id:string})[];
 publications:(PublishedResearch&{newer_update_id:string|null})[];
 sourceChecks:{symbol:string;status:string;checked_at:string;message:string;fingerprints:string[]}[];
};
export type JournalFormat='json'|'md';
export class JournalLimitError extends Error {}
export function journalFormat(value:unknown):JournalFormat{
 if(value===null||value===undefined||value==='json')return 'json';
 if(value==='md')return 'md';
 throw new Error('Choose JSON or Markdown.');
}
// Preserve private text as text when the Markdown is opened in another reader.
const text=(value:string|null)=>(value??'Not specified').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/([\\`*_{}\[\]()#+.!|~-])/g,'\\$1');
const link=(url:string)=>{try{const parsed=new URL(url);return parsed.protocol==='https:'?`<${parsed.href.replace(/</g,'%3C').replace(/>/g,'%3E')}>`:'Source URL unavailable';}catch{return 'Source URL unavailable';}};
function thesis(lines:string[],revision:ThesisRevision){
 lines.push(`Saved: ${revision.created_at}`,`Reason: ${text(revision.reason)}`,'',text(revision.payload.thesis)||'No investment case recorded.','',`Risks: ${text(revision.payload.risks)||'Not recorded'}`);
 for(const a of revision.payload.assumptions)lines.push('',`- Assumption: ${text(a.statement)}`,`  - Topic: ${text(a.metric)||'Not specified'}`,`  - Baseline: ${text(a.baseline)||'Not specified'}`,`  - Direction: ${a.direction}`,`  - Horizon: ${text(a.horizon)||'Not specified'}`,`  - Revisit condition: ${text(a.reviewCondition)||'Not specified'}`);
 lines.push('',`Evidence IDs: ${revision.payload.evidenceIds.join(', ')||'None'}`,'');
}
export function journalMarkdown(j:Journal):string{
 const lines=['# AlphaTrace research journal','',`Exported: ${j.exportedAt} (UTC)`,'',text(j.scope),'','Saved data only. Unsaved edits and unpublished operator drafts are excluded. This is a snapshot, not a live feed or a restore file.','',`Workspace: ${text(j.workspace?.name??'No workspace saved yet')}`,'','## Portfolio',''];
 for(const p of j.portfolio)lines.push(`- ${p.symbol}: quantity ${p.quantity??'not held'}; watchlist ${p.followed?'yes':'no'}; saved ${p.updated_at}`);
 if(!j.portfolio.length)lines.push('No saved holdings or watchlist entries.');
 lines.push('','## Thesis revision history','');
 for(const r of j.thesisRevisions){lines.push(`### ${r.symbol} · revision ${r.version}${j.theses.some(h=>h.symbol===r.symbol&&h.version===r.version)?' · current':''}`,'');thesis(lines,r);}
 if(!j.thesisRevisions.length)lines.push('No saved thesis revisions.');
 lines.push('','## Review decisions','');
 for(const item of j.reviews){
  const p=j.publications.find(p=>p.id===item.update_id);
  lines.push(`### ${p?.symbol??'Evidence'} · publication ${p?.version??'unknown'}`,'',`Evidence ID: ${item.update_id}`,`State: ${item.state}; response version: ${item.version}`,`Created: ${item.created_at}; last saved: ${item.updated_at}`,`Revisit: ${item.revisit_at??'Not scheduled'}`,'',`Included because of: ${item.snapshot.reason}. Matching rule: ${item.snapshot.ruleVersion}.`,'');
  lines.push('Topic matching does not establish support, contradiction or a breached threshold.','');
  for(const m of item.snapshot.matches)lines.push(`- Recorded assumption: ${text(m.assumption.statement)}; matched topics: ${m.metrics.join(', ')}`);
  if(item.snapshot.thesis){lines.push('','#### Original research when this review was created','');thesis(lines,item.snapshot.thesis);}
  const events=j.reviewEvents.filter(e=>e.update_id===item.update_id);
  for(const e of events){lines.push(`#### Response ${e.version} · ${e.action}`,'',`${e.created_at}; revisit: ${e.revisit_at??'Not scheduled'}`,'',text(e.note),'');if(e.thesis_revision){lines.push(`Linked thesis revision: ${e.thesis_revision.version}`,'');thesis(lines,e.thesis_revision);}}
  if(!events.length)lines.push('No response saved.','');
 }
 if(!j.reviews.length)lines.push('No saved review items.');
 lines.push('','## Referenced publications and sources','');
 for(const p of j.publications){
  lines.push(`### ${p.symbol} · publication ${p.version} · ${text(p.payload.title)}`,'',`Evidence ID: ${p.id}`,`Quarter ended: ${p.payload.periodEnd}; comparison: ${p.payload.comparison}; published: ${p.published_at}`,`Withdrawal: ${p.withdrawn_at?`${p.withdrawn_at} — ${text(p.withdrawal_reason??'')}`:'Not withdrawn at export time'}`,`Newer publication ID: ${p.newer_update_id??'None at export time'}`,`Supersedes: ${p.supersedes??'None'}`,'');
  const check=j.sourceChecks.find(c=>c.symbol===p.symbol);
  lines.push(check?`Latest recorded source check: ${check.status} at ${check.checked_at}. ${text(check.message)}`:'No source check recorded.','Source-check dates do not establish that no newer report exists.','');
  for(const c of p.payload.comparisons){
   const result=comparisonResult(c,p.payload.comparison);
   lines.push(`- ${text(c.metric)}: ${c.previous.value??'Unavailable'} ${text(c.previous.unit)} → ${c.current.value??'Unavailable'} ${text(c.current.unit)}. ${result.available?`${result.change}${result.changeUnit==='pp'?' pp':'%'} ${p.payload.comparison}`:`Not comparable: ${text(result.reason)}`}`);
   for(const [label,v] of [['Previous',c.previous],['Current',c.current]] as const)lines.push(`  - ${label}: ${v.periodEnd}; ${v.periodType}; ${v.basis}; ${v.scope?text(v.scope)+'; ':''}${v.standard}; ${text(v.treatment)}. Source ${text(v.sourceId)}, ${v.page?`physical page ${v.page}, `:''}${text(v.locator)}.`);
   if(c.note)lines.push(`  - Context: ${text(c.note)}`);
  }
  lines.push('');for(const limitation of p.payload.limitations)lines.push(`- Limitation: ${text(limitation)}`);
  for(const s of p.payload.sources)lines.push('',`Source ${text(s.id)}: ${text(s.title)}`,link(s.url),`Published: ${s.publishedOn}; retrieved: ${s.fetchedAt}`,`Retrieval: ${s.retrieval}; fingerprint of ${s.fingerprintKind}: ${s.sha256}`);
  lines.push('');
 }
 if(!j.publications.length)lines.push('No published evidence linked to saved research or reviews.');
 return lines.join('\n')+'\n';
}
export function journalFile(j:Journal,format:JournalFormat){
 const content=format==='json'?JSON.stringify(j,null,2)+'\n':journalMarkdown(j);
 if(new TextEncoder().encode(content).byteLength>20*1024*1024)throw new JournalLimitError('This journal exceeds the download limit. No partial file was created. Ask the operator for a full export.');
 return {content,filename:`alphatrace-journal-${j.exportedAt.slice(0,10)}.${format}`,mediaType:format==='json'?'application/json; charset=utf-8':'text/markdown; charset=utf-8'};
}
