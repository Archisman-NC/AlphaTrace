import type { CompanyCoverage } from './research-view';
import type { Portfolio } from './portfolio-input';
import { freshness } from './research-contract';

export function coverageStatus(company:CompanyCoverage,now=Date.now()):{label:string;tone:string;available:boolean} {
 if(company.status==='withdrawn')return {label:'Publication withdrawn',tone:'amber',available:false};
 if(company.status!=='published')return {label:'No published research',tone:'neutral',available:false};
 if(company.sourceChanged)return {label:'Source changed',tone:'amber',available:true};
 const state=freshness(company.check,now);
 return {label:state==='checked'?'Sources checked':state==='failed'?'Source check failed':state==='stale'?'Source check overdue':'Sources unchecked',tone:state==='checked'?'neutral':'amber',available:true};
}

export function trackedCoverage(portfolio:Portfolio,catalog:CompanyCoverage[]|null) {
 const symbols=new Set(portfolio.entries.filter(e=>e.followed||e.quantity!==null).map(e=>e.symbol));
 return catalog===null?null:catalog.filter(c=>symbols.has(c.symbol));
}

/** Group decimal strings without passing exact published values through Number. */
export function evidenceNumber(value:string|null):string {
 if(value===null)return 'Unavailable';
 const [whole,fraction]=value.split('.');
 return whole.replace(/\B(?=(\d{3})+(?!\d))/g,',')+(fraction===undefined?'':'.'+fraction);
}

export const displayDate=(value:string)=>new Date(value).toLocaleDateString('en-IN',{day:'numeric',month:'short',year:'numeric',timeZone:'Asia/Kolkata'});
