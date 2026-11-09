import { universe } from './companies';
export type Holding = {symbol:string;quantity:string};
export type PortfolioEntry = {symbol:string;quantity:string|null;followed:boolean};
export type Portfolio = {version:number;entries:PortfolioEntry[]};
export type PortfolioCommand =
 | {type:'follow';symbol:string;followed:boolean}
 | {type:'holding';symbol:string;quantity:string}
 | {type:'remove_holding';symbol:string}
 | {type:'import';holdings:Holding[]};
export class PortfolioInputError extends Error {}
export class PortfolioConflictError extends Error {}
export function validSymbol(input:unknown):string {
 if(typeof input!=='string')throw new PortfolioInputError('Choose a supported company.');
 const symbol=input.trim().toUpperCase();
 if(!universe.some(c=>c.symbol===symbol))throw new PortfolioInputError('This symbol is outside the 25-company universe.');
 return symbol;
}
/** Decimal strings keep quantities exact through CSV, JSON and PostgreSQL. */
export function validQuantity(input:unknown):string {
 if(typeof input!=='string'||!/^\d{1,12}(\.\d{1,6})?$/.test(input.trim()))throw new PortfolioInputError('Use a positive quantity with up to 12 whole and 6 decimal digits.');
 const [whole,decimal='']=input.trim().split('.');
 const cleanedWhole=whole.replace(/^0+(?=\d)/,''),cleanedDecimal=decimal.replace(/0+$/,'');
 if(!/[1-9]/.test(cleanedWhole+cleanedDecimal))throw new PortfolioInputError('Quantity must be greater than zero.');
 return cleanedWhole+(cleanedDecimal?'.'+cleanedDecimal:'');
}
export function validateMutation(input:unknown):{version:number;command:PortfolioCommand} {
 if(!input||typeof input!=='object')throw new PortfolioInputError('Invalid portfolio update.');
 const {version,command}=input as {version:unknown;command:Record<string,unknown>};
 if(!Number.isSafeInteger(version)||(version as number)<0||!command||typeof command!=='object')throw new PortfolioInputError('A current portfolio version is required.');
 let parsed:PortfolioCommand;
 switch(command.type){
  case 'follow':
   if(typeof command.followed!=='boolean')throw new PortfolioInputError('Choose whether to follow the company.');
   parsed={type:'follow',symbol:validSymbol(command.symbol),followed:command.followed};break;
  case 'holding':parsed={type:'holding',symbol:validSymbol(command.symbol),quantity:validQuantity(command.quantity)};break;
  case 'remove_holding':parsed={type:'remove_holding',symbol:validSymbol(command.symbol)};break;
  case 'import':{
   if(!Array.isArray(command.holdings)||!command.holdings.length||command.holdings.length>100)throw new PortfolioInputError('Import 1–100 holdings.');
   const holdings=command.holdings.map(h=>{
    if(!h||typeof h!=='object')throw new PortfolioInputError('Invalid import row.');
    return {symbol:validSymbol(h.symbol),quantity:validQuantity(h.quantity)};
   });
   if(new Set(holdings.map(h=>h.symbol)).size!==holdings.length)throw new PortfolioInputError('Resolve duplicate symbols before importing.');
   parsed={type:'import',holdings};break;
  }
  default:throw new PortfolioInputError('Unknown portfolio action.');
 }
 return {version:version as number,command:parsed};
}
export type CsvTable={headers:string[];rows:string[][]};
export type ImportRow={row:number;symbol:string;quantity:string;error:string};
export function readHoldingsCsv(text:string):CsvTable {
 if(new TextEncoder().encode(text).length>1024*1024)throw new PortfolioInputError('CSV exceeds the 1 MB limit.');
 // Embedded line breaks are intentionally rejected; multiline broker notes are not imported.
 const lines=text.replace(/^\uFEFF/,'').trim().split(/\r?\n/);
 if(lines.length<2||lines.length>101)throw new PortfolioInputError('Use a header and 1–100 holdings rows.');
 const split=(line:string)=>{
  const cells:string[]=[];let value='',quoted=false,closed=false;
  for(let i=0;i<line.length;i++){
   const c=line[i];
   if(c==='"'){
    if(quoted&&line[i+1]==='"'){value+='"';i++;}
    else if(quoted){quoted=false;closed=true;}
    else if(!value.trim()&&!closed){quoted=true;value='';}
    else throw new PortfolioInputError('Invalid CSV quoting. Use one holding per line.');
   }else if(c===','&&!quoted){cells.push(value.trim());value='';closed=false;}
   else if(closed&&!quoted&&c.trim())throw new PortfolioInputError('Invalid text after a quoted CSV cell.');
   else value+=c;
  }
  if(quoted)throw new PortfolioInputError('Unclosed CSV quote. Use one holding per line.');
  cells.push(value.trim());return cells;
 };
 const headers=split(lines[0]);
 if(headers.length>30||headers.some(h=>!h)||new Set(headers.map(h=>h.toLowerCase())).size!==headers.length)throw new PortfolioInputError('Use up to 30 uniquely named column headers.');
 const rows=lines.slice(1).map(split);
 if(rows.some(row=>row.length!==headers.length))throw new PortfolioInputError('Every CSV row must have the same columns as the header.');
 return {headers,rows};
}
export function previewHoldings(table:CsvTable,symbolColumn:number,quantityColumn:number):ImportRow[] {
 if(symbolColumn===quantityColumn||![symbolColumn,quantityColumn].every(i=>Number.isInteger(i)&&i>=0&&i<table.headers.length))throw new PortfolioInputError('Map symbol and quantity to different columns.');
 const counts=new Map<string,number>();
 for(const row of table.rows){const key=row[symbolColumn].trim().toUpperCase();counts.set(key,(counts.get(key)??0)+1);}
 return table.rows.map((row,i)=>{
  const symbol=row[symbolColumn].trim().toUpperCase();let quantity=row[quantityColumn].trim(),error='';
  try{validSymbol(symbol);quantity=validQuantity(quantity);if(counts.get(symbol)!>1)throw new PortfolioInputError('Duplicate symbol: fix the CSV before importing.');}
  catch(e){error=(e as Error).message;}
  return {row:i+2,symbol,quantity,error};
 });
}
