import { universe } from './companies';

export const metrics = {
  revenue:'Revenue', operating_profit:'Operating profit', net_profit:'Net profit attributable to owners',
  operating_margin:'Operating margin', deal_tcv:'Total contract value', employees:'Employees',
  operating_cash_flow:'Operating cash flow', debt:'Debt',
  net_interest_income:'Net interest income', net_interest_margin:'Net interest margin',
  gross_npa:'Gross non-performing assets ratio', net_npa:'Net non-performing assets ratio',
} as const;
export type Metric = keyof typeof metrics;
export const spreadsheetMediaType='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet';
export type Source = { id:string; title:string; url:string; sha256:string; publishedOn:string; fetchedAt:string; kind:'results'|'release'|'facts'; mediaType:'application/pdf'|'text/html'|typeof spreadsheetMediaType; fingerprintKind:'document_bytes'|'extracted_text'; retrieval:'direct'|'operator_snapshot'|'operator_file'; pages:number; bytes:number };
export type EvidencePoint = { value:string|null; unit:'USD million'|'USD billion'|'INR crore'|'%'|'people'; basis:'consolidated'|'standalone'; standard:'IFRS'|'Ind AS'|'operational'; treatment:'reported'|'excluding exceptional items'|'guidance'; periodEnd:string; periodType:'quarter'|'half_year'|'year'|'point_in_time'; sourceId:string; locator:string; page:number|null; worksheet?:string; cell?:string; scope?:string };
export type Comparison = { metric:Metric; previous:EvidencePoint; current:EvidencePoint; note:string };
export type ResearchPayload = { schemaVersion:1; symbol:string; title:string; periodEnd:string; comparison:'QoQ'|'YoY'; sources:Source[]; comparisons:Comparison[]; limitations:string[]; extractorVersion:string };
export type ResearchCheck = { symbol:string; checkedAt:string; status:'success'|'failed'; message:string; fingerprints:string[] };
export class ResearchInputError extends Error {}
export class ResearchConflictError extends Error {}
export class ResearchForbiddenError extends Error {}
const fail=(message:string):never=>{throw new ResearchInputError(message);};
function record(v:unknown):Record<string,unknown>{if(!v||typeof v!=='object'||Array.isArray(v))return fail('Expected an object.');return v as Record<string,unknown>;}
function text(v:unknown,max=500):string{if(typeof v!=='string'||!v.trim()||v.length>max||/[\u0000-\u0008\u000b\u000c\u000e-\u001f]/.test(v))return fail('Invalid or oversized text.');return v.trim();}
function choice<T extends string>(v:unknown,options:readonly T[]):T{if(typeof v!=='string'||!options.includes(v as T))return fail('Unsupported field value.');return v as T;}
function integer(v:unknown,min:number,max:number):number{if(typeof v!=='number'||!Number.isInteger(v)||v<min||v>max)return fail('Invalid numeric limit.');return v;}
function date(v:unknown):string{const s=text(v,10);if(!/^\d{4}-\d{2}-\d{2}$/.test(s)||!Number.isFinite(Date.parse(s))||new Date(s).toISOString().slice(0,10)!==s)return fail('Invalid calendar date.');return s;}
function timestamp(v:unknown):string{const s=text(v,40);if(!/^\d{4}-\d{2}-\d{2}T/.test(s)||!Number.isFinite(Date.parse(s))||Date.parse(s)>Date.now()+300000)return fail('Invalid or future check time.');return new Date(s).toISOString();}
export function researchSymbol(v:unknown):string{const s=text(v,20);if(!universe.some(c=>c.symbol===s))return fail('Company is outside the research universe.');return s;}
const sourceHosts:Record<string,string[]> = {
 TCS:['www.tcs.com'],INFY:['www.infosys.com'],HCLTECH:['www.hcltech.com'],WIPRO:['www.wipro.com'],
 HINDUNILVR:['www.hul.co.in'],ITC:['itcportal.com','www.itcportal.com'],ASIANPAINT:['www.asianpaints.com'],TITAN:['www.titancompany.in'],MARUTI:['www.marutisuzuki.com'],LT:['investors.larsentoubro.com'],
 RELIANCE:['www.ril.com'],BHARTIARTL:['www.airtel.in'],SUNPHARMA:['sunpharma.com'],CIPLA:['www.cipla.com'],NTPC:['ntpc.co.in'],POWERGRID:['www.powergrid.in'],ULTRACEMCO:['www.ultratechcement.com'],
 'BAJAJ-AUTO':['www.bajajauto.com'],HEROMOTOCO:['www.heromotocorp.com'],NESTLEIND:['www.nestle.in'],HDFCBANK:['www.hdfc.bank.in'],ICICIBANK:['www.icici.bank.in'],SBIN:['sbi.bank.in'],AXISBANK:['www.axis.bank.in'],KOTAKBANK:['www.kotak.bank.in'],
};
export function officialSourceUrl(value:unknown,symbol:string):string {
 const raw=text(value,1500);let url:URL;try{url=new URL(raw);}catch{return fail('Invalid source URL.');}
 const issuer=symbol==='INFY'?'1067491':symbol==='WIPRO'?'1123799':null;
 const sec=!!issuer&&url.hostname==='www.sec.gov'&&new RegExp(`^/Archives/edgar/data/${issuer}/\\d{18}/[a-zA-Z0-9_-]+\\.html?$`).test(url.pathname);
 if(url.protocol!=='https:'||url.username||url.password||url.port||url.hash||url.search||!(sourceHosts[symbol]?.includes(url.hostname)||sec))return fail('Use an approved official company source URL.');
 return url.href;
}
export function validateResearch(value:unknown):ResearchPayload {
 const p=record(value);if(p.schemaVersion!==1)return fail('Unsupported research schema.');
 const symbol=researchSymbol(p.symbol),periodEnd=date(p.periodEnd);
 if(periodEnd>new Date().toISOString().slice(0,10))return fail('Future reporting periods cannot be published.');
 if(!Array.isArray(p.sources)||p.sources.length<1||p.sources.length>8)return fail('Provide 1–8 official sources.');
 const sources=p.sources.map(item=>{
  const s=record(item),sha256=text(s.sha256,64),publishedOn=date(s.publishedOn),fetchedAt=timestamp(s.fetchedAt);
  if(!/^[a-f0-9]{64}$/.test(sha256))return fail('Invalid source fingerprint.');
  if(publishedOn>fetchedAt.slice(0,10))return fail('Source publication is later than its retrieval.');
  const id=text(s.id,50);if(!/^[a-z0-9_-]+$/.test(id))return fail('Invalid source identifier.');
  const mediaType=choice(s.mediaType,['application/pdf','text/html',spreadsheetMediaType]),fingerprintKind=choice(s.fingerprintKind,['document_bytes','extracted_text']),retrieval=choice(s.retrieval,['direct','operator_snapshot','operator_file']);
  if((retrieval==='operator_file'||mediaType===spreadsheetMediaType)&&fingerprintKind!=='document_bytes')return fail('Supplied files require a fingerprint of original document bytes.');
  if(mediaType===spreadsheetMediaType&&retrieval!=='operator_file')return fail('Spreadsheet evidence currently requires an operator-supplied file.');
  return {id,title:text(s.title,160),url:officialSourceUrl(s.url,symbol),sha256,publishedOn,fetchedAt,kind:choice(s.kind,['results','release','facts']),mediaType,fingerprintKind,retrieval,pages:mediaType===spreadsheetMediaType?integer(s.pages,0,0):integer(s.pages,1,100),bytes:integer(s.bytes,1,20*1024*1024)} satisfies Source;
 });
 if(new Set(sources.map(s=>s.id)).size!==sources.length||new Set(sources.map(s=>s.url)).size!==sources.length)return fail('Duplicate source identifiers or URLs.');
 function point(v:unknown):EvidencePoint {
  const a=record(v),sourceId=text(a.sourceId,50),source=sources.find(s=>s.id===sourceId);
  if(!source)return fail('Evidence must reference a source in this draft.');
  const val=a.value===null?null:text(a.value,30);if(val!==null&&!/^-?\d{1,15}(?:\.\d{1,6})?$/.test(val))return fail('Financial values must be exact decimal strings.');
  const page=a.page===null?null:integer(a.page,1,source.pages);
  if(source.mediaType==='application/pdf'&&page===null)return fail('PDF evidence needs a page number.');
  let spreadsheetLocation:{}|{worksheet:string;cell:string}={};
  if(source.mediaType===spreadsheetMediaType){
   if(page!==null)return fail('Spreadsheet evidence uses cells, not page numbers.');
   const worksheet=text(a.worksheet,31),cell=text(a.cell,10);
   if(/[\\/\[\]:*?]/.test(worksheet)||!/^\$?[A-Z]{1,3}\$?[1-9]\d{0,6}$/.test(cell))return fail('Provide a valid worksheet and cell reference.');
   const match=cell.replaceAll('$','').match(/^([A-Z]+)(\d+)$/)!;
   const column=[...match[1]].reduce((n,c)=>n*26+c.charCodeAt(0)-64,0);
   if(column>16384||Number(match[2])>1048576)return fail('Spreadsheet cell is outside Excel limits.');
   spreadsheetLocation={worksheet,cell};
  }else if(a.worksheet!==undefined||a.cell!==undefined)return fail('Cell references require spreadsheet evidence.');
  const end=date(a.periodEnd);if(end>source.publishedOn)return fail('Evidence period is later than its source.');
  return {value:val,unit:choice(a.unit,['USD million','USD billion','INR crore','%','people']),basis:choice(a.basis,['consolidated','standalone']),standard:choice(a.standard,['IFRS','Ind AS','operational']),treatment:choice(a.treatment,['reported','excluding exceptional items','guidance']),periodEnd:end,periodType:choice(a.periodType,['quarter','half_year','year','point_in_time']),sourceId,locator:text(a.locator,240),page,...spreadsheetLocation,...(a.scope===undefined?{}:{scope:text(a.scope,100)})};
 }
 if(!Array.isArray(p.comparisons)||p.comparisons.length<1||p.comparisons.length>12)return fail('Provide 1–12 comparisons.');
 const comparisons=p.comparisons.map(item=>{
  const c=record(item),metric=choice(c.metric,Object.keys(metrics) as Metric[]),previous=point(c.previous),current=point(c.current);
  if(current.periodEnd!==periodEnd)return fail('Current evidence must match the update period.');
  const isBank=universe.find(c=>c.symbol===symbol)?.sector==='Banking';
  if(isBank&&['revenue','operating_profit','operating_margin','debt'].includes(metric))return fail('Bank research requires bank-specific metrics.');
  if(!isBank&&['net_interest_income','net_interest_margin','gross_npa','net_npa'].includes(metric))return fail('Bank metrics cannot be used for this company.');
  for(const q of [previous,current]){
   if(['operating_margin','net_interest_margin','gross_npa','net_npa'].includes(metric)&&q.unit!=='%')return fail('Ratio metrics require percent units.');
   if(metric==='employees'&&(q.unit!=='people'||q.periodType!=='point_in_time'))return fail('Employee counts require point-in-time people units.');
   if(!['operating_margin','net_interest_margin','gross_npa','net_npa','employees'].includes(metric)&&['%','people'].includes(q.unit))return fail('Monetary metrics require currency units.');
  }
  return {metric,previous,current,note:typeof c.note==='string'&&c.note.length<=800?c.note.trim():fail('Comparison notes must be at most 800 characters.')};
 });
 if(new Set(comparisons.map(c=>c.metric)).size!==comparisons.length)return fail('Duplicate comparison metrics.');
 if(!Array.isArray(p.limitations)||p.limitations.length<1||p.limitations.length>8)return fail('Record 1–8 limitations.');
 return {schemaVersion:1,symbol,title:text(p.title,140),periodEnd,comparison:choice(p.comparison,['QoQ','YoY']),sources,comparisons,limitations:p.limitations.map(s=>text(s,600)),extractorVersion:text(p.extractorVersion,80)};
}
function scaled(s:string):bigint{const negative=s.startsWith('-');const [whole,fraction='']=s.replace(/^-/,'').split('.');const n=BigInt(whole)*1000000n+BigInt(fraction.padEnd(6,'0'));return negative?-n:n;}
function rounded(n:bigint,d:bigint):string{const negative=n<0n;n=negative?-n:n;const q=(n+d/2n)/d;return `${negative&&q!==0n?'-':''}${q/100n}.${(q%100n).toString().padStart(2,'0')}`;}
function quarter(end:string):boolean{const d=new Date(end+'T00:00:00Z');return [2,5,8,11].includes(d.getUTCMonth())&&new Date(Date.UTC(d.getUTCFullYear(),d.getUTCMonth()+1,0)).toISOString().slice(0,10)===end;}
export function comparisonResult(c:Comparison,mode:'QoQ'|'YoY'):{available:boolean;change:string|null;changeUnit:string;reason:string|null} {
 const a=c.previous,b=c.current;
 const unavailable=(reason:string)=>({available:false,change:null,changeUnit:'',reason});
 if(a.value===null||b.value===null)return unavailable('A source value is missing; no change is inferred.');
 if((a.scope??'Whole company')!==(b.scope??'Whole company'))return unavailable('Not comparable: business scope differs between periods.');
 for(const field of ['unit','basis','standard','treatment','periodType'] as const)if(a[field]!==b[field])return unavailable(`Not comparable: ${field} differs between periods.`);
 if(a.treatment==='guidance')return unavailable('Guidance is not a reported result.');
 if(!['quarter','point_in_time'].includes(a.periodType))return unavailable('This release compares quarterly periods, not year-to-date totals.');
 if(!quarter(a.periodEnd)||!quarter(b.periodEnd))return unavailable('Evidence must use calendar quarter-end dates.');
 const end=new Date(b.periodEnd+'T00:00:00Z');
 const expected=mode==='QoQ'?new Date(Date.UTC(end.getUTCFullYear(),end.getUTCMonth()-2,0)):new Date(Date.UTC(end.getUTCFullYear()-1,end.getUTCMonth()+1,0));
 if(expected.toISOString().slice(0,10)!==a.periodEnd)return unavailable(`Dates do not form a ${mode} comparison.`);
 const old=scaled(a.value),current=scaled(b.value);
 if(a.unit==='%')return {available:true,change:rounded((current-old)*100n,1000000n),changeUnit:'pp',reason:null};
 if(old<=0n)return unavailable('Percentage growth is undefined for a zero or negative baseline.');
 return {available:true,change:rounded((current-old)*10000n,old),changeUnit:'%',reason:null};
}
export function publicationGaps(p:ResearchPayload):string[]{
 const gaps:string[]=[];
 if(p.sources.length<2||!p.sources.some(s=>s.kind==='results')||!p.sources.some(s=>s.kind==='release'||s.kind==='facts'))gaps.push('Add official results and supporting release/fact-sheet sources.');
 if(p.comparisons.filter(c=>comparisonResult(c,p.comparison).available).length<3)gaps.push('At least three comparable, source-backed metrics are required.');
 return gaps;
}
export function validateCheck(value:unknown):ResearchCheck {
 const c=record(value);if(!Array.isArray(c.fingerprints)||c.fingerprints.length>8||!c.fingerprints.every(s=>typeof s==='string'&&/^[a-f0-9]{64}$/.test(s)))return fail('Invalid check fingerprints.');
 const status=choice(c.status,['success','failed']);if(status==='success'&&!c.fingerprints.length)return fail('A successful source check requires fingerprints.');
 return {symbol:researchSymbol(c.symbol),checkedAt:timestamp(c.checkedAt),status,message:text(c.message,500),fingerprints:c.fingerprints as string[]};
}
export function freshness(check:{status:string;checked_at:string}|null,now=Date.now()):'unchecked'|'failed'|'stale'|'checked'{
 if(!check)return 'unchecked';if(check.status!=='success')return 'failed';return now-Date.parse(check.checked_at)>7*86400000?'stale':'checked';
}
