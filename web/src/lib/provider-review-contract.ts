import type { ProviderChanges } from './provider-changes';
import type { ProviderSymbol } from './provider-contract';
import type { ProviderEvaluation } from './provider-contract';
export type ImportReviewStatus='pending'|'reviewed';
export type ImportReview={
 id:string;symbol:ProviderSymbol;sequence:number;current_import_id:string;previous_import_id:string|null;
 observed_at:string;previous_observed_at:string|null;detected_at:string;diff:ProviderChanges;
 status:ImportReviewStatus;version:number;reviewed_at:string|null;reviewed_by:string|null;note:string;
 endpoint:string;current_sha256:string;previous_sha256:string|null;sourceUrl:string|null;
 isCurrent:boolean;draft_id:string|null;
 evaluation:ProviderEvaluation;
 handoff:{id:string;state:string;periodEnd:string;updateId:string|null}|null;
};
export type ImportReviewView={items:ImportReview[];counts:Record<ImportReviewStatus,number>;nextCursor:string|null};
export function importReviewTitle(diff:ProviderChanges){
 if(diff.baseline)return 'First saved baseline';
 const parts:string[]=[];
 if(diff.newQuarters.length)parts.push(`${diff.newQuarters.length} new quarter${diff.newQuarters.length===1?'':'s'}`);
 const revised=diff.changes.filter(c=>c.kind==='changed').length;
 if(revised)parts.push(`${revised} revised value${revised===1?'':'s'}`);
 if(diff.removedQuarters.length||diff.changes.some(c=>c.kind==='missing'))parts.push('Missing data');
 if(diff.addedHistory.length)parts.push('History added');
 return parts.join(' · ')||'Values added';
}
