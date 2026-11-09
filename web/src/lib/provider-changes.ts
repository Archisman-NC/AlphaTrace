import { createHash } from 'node:crypto';
import { quarterEnd } from '../../scripts/lib/upstox';
import { inspectIndianStats } from '../../scripts/lib/indianapi';
import type { ProviderSnapshot } from './provider-evaluation';

export const trackedMetrics=['Sales','Operating Profit','Net Profit'] as const;
export type TrackedMetric=typeof trackedMetrics[number];
export type QuarterValues={period:string;values:Record<TrackedMetric,string|null>};
export type ProviderChange={period:string;metric:TrackedMetric;before:string|null;after:string|null;kind:'baseline'|'added'|'changed'|'missing'};
export type ProviderChanges={baseline:boolean;newQuarters:string[];addedHistory:string[];removedQuarters:string[];missingMetrics:TrackedMetric[];changes:ProviderChange[]};

/** Canonical decimal text: no rounding, float subtraction or formatting-only alerts. */
function decimal(value:unknown):string|null {
 if(value===null||value===undefined||value===''||value==='-'||value==='—')return null;
 let text=String(value).trim().replaceAll(',','');
 if(typeof value==='number'&&/[eE]/.test(text)){
  const [coefficient,exponent]=text.toLowerCase().split('e'),negative=coefficient.startsWith('-');
  const [whole,fraction='']=coefficient.replace(/^-/,'').split('.'),digits=whole+fraction,point=whole.length+Number(exponent);
  text=(negative?'-':'')+(point<=0?'0.'+'0'.repeat(-point)+digits:point>=digits.length?digits+'0'.repeat(point-digits.length):digits.slice(0,point)+'.'+digits.slice(point));
 }
 if(!/^-?\d+(?:\.\d+)?$/.test(text))throw Error('Invalid provider decimal.');
 const [integer,fraction='']=text.replace(/^-/,'').split('.');
 const normalized=integer.replace(/^0+(?=\d)/,'')+(fraction.replace(/0+$/,'')?'.'+fraction.replace(/0+$/,''):'');
 return (text.startsWith('-')&&normalized!=='0'?'-':'')+normalized;
}
export function normalizeProvider(s:ProviderSnapshot):QuarterValues[] {
 const raw=JSON.parse(s.responseText);
 inspectIndianStats(raw,s.requestedSymbol,new Date(s.fetchedAt));
 const quarters=new Map<string,QuarterValues>();
 for(const metric of trackedMetrics)for(const [label,value] of Object.entries(raw[metric]??{})){
  const period=quarterEnd(label);
  const q=quarters.get(period)??{period,values:{Sales:null,'Operating Profit':null,'Net Profit':null}};
  q.values[metric]=decimal(value);quarters.set(period,q);
 }
 return [...quarters.values()].sort((a,b)=>a.period.localeCompare(b.period));
}
export const providerSemanticHash=(quarters:QuarterValues[])=>createHash('sha256').update(JSON.stringify(quarters)).digest('hex');
export function detectProviderChanges(previous:QuarterValues[]|null,current:QuarterValues[]):ProviderChanges {
 const old=new Map(previous?.map(q=>[q.period,q])??[]),fresh=new Map(current.map(q=>[q.period,q]));
 const latest=previous?.at(-1)?.period;
 const added=current.filter(q=>!old.has(q.period)).map(q=>q.period);
 const result:ProviderChanges={baseline:previous===null,newQuarters:previous?added.filter(p=>!latest||p>latest):[],addedHistory:previous?added.filter(p=>!!latest&&p<=latest):[],removedQuarters:previous?.filter(q=>!fresh.has(q.period)).map(q=>q.period)??[],missingMetrics:trackedMetrics.filter(m=>current.every(q=>q.values[m]===null)),changes:[]};
 // A first observation is a baseline, not a claim that every historic value changed.
 const periods=previous?[...new Set([...old.keys(),...fresh.keys()])].sort():current.slice(-2).map(q=>q.period);
 for(const period of periods)for(const metric of trackedMetrics){
  const before=old.get(period)?.values[metric]??null,after=fresh.get(period)?.values[metric]??null;
  if(previous===null||before!==after)result.changes.push({period,metric,before,after,kind:previous===null?'baseline':after===null?'missing':before===null?'added':'changed'});
 }
 return result;
}
export const meaningfulProviderChange=(d:ProviderChanges)=>d.baseline||!!(d.changes.length||d.newQuarters.length||d.addedHistory.length||d.removedQuarters.length);
