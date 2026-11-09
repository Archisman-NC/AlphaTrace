import { ResearchInputError } from './research-contract';

export const manualFileLimit=20*1024*1024;
export const manualRequestLimit=3*manualFileLimit+65536;
export const tcsImportFiles=[
 {key:'workbook',label:'Q2 FY27 data workbook',extension:'.xlsx',url:'https://www.tcs.com/content/dam/tcs/pdf/discover-tcs/investor-relations/tcs-data-sheet/TCS-Data-Sheet-Q2FY27.xlsx'},
 {key:'q1',label:'Q1 FY27 consolidated IFRS results (USD)',extension:'.pdf',url:'https://www.tcs.com/content/dam/tcs/investor-relations/financial-statements/2026-27/q1/IFRS/Condensed%20Consolidated,%20Unaudited%20-%20USD.pdf'},
 {key:'q2',label:'Q2 FY27 consolidated IFRS results (USD)',extension:'.pdf',url:'https://www.tcs.com/content/dam/tcs/investor-relations/financial-statements/2026-27/q2/IFRS/Condensed%20Consolidated,%20Unaudited%20-%20USD.pdf'},
] as const;
export function manualDownloadTime(value:unknown,now=Date.now()):string {
 if(typeof value!=='string'||!/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2}(\.\d{1,3})?)?(Z|[+-]\d{2}:\d{2})$/.test(value))throw new ResearchInputError('Enter the actual download time for each file, including its timezone.');
 const time=Date.parse(value);
 if(!Number.isFinite(Date.parse(value.slice(0,10)+'T00:00:00Z'))||new Date(value.slice(0,10)+'T00:00:00Z').toISOString().slice(0,10)!==value.slice(0,10))throw new ResearchInputError('Enter a valid calendar date for the download.');
 if(!Number.isFinite(time)||time>now+300000||now-time>7*86400000)throw new ResearchInputError('Each actual download time must be within the last seven days. Download a fresh original if necessary.');
 return new Date(time).toISOString();
}
export function validateManualFiles(form:FormData){
 if(form.get('originalsConfirmed')!=='true')throw new ResearchInputError('Confirm these are unchanged originals from the linked official sources.');
 const allowed=new Set<string>(['originalsConfirmed',...tcsImportFiles.flatMap(f=>[f.key,f.key+'Time'])]);
 for(const key of form.keys())if(!allowed.has(key)||form.getAll(key).length!==1)throw new ResearchInputError('Unexpected or duplicate import field.');
 return tcsImportFiles.map(spec=>{
  const file=form.get(spec.key);
  if(!(file instanceof File)||!file.size||file.size>manualFileLimit||!file.name.toLowerCase().endsWith(spec.extension))throw new ResearchInputError(`Choose ${spec.label} as a ${spec.extension} file, up to 20 MB.`);
  return {...spec,file,retrievedAt:manualDownloadTime(form.get(spec.key+'Time'))};
 });
}
