import { PortfolioInputError,validSymbol,validateMutation,type Holding,type Portfolio } from './portfolio-input';
export const onboardingTopics={revenue:'Revenue growth',operating_margin:'Operating margins',operating_cash_flow:'Cash generation',debt:'Debt'} as const;
export type OnboardingTopic=keyof typeof onboardingTopics;
export type OnboardingState={version:number;step:'companies'|'topics'|'review';status:'active'|'completed'|'skipped';topics:OnboardingTopic[]};
export type FirstReview={id:string;symbol:string;title:string;periodEnd:string;publishedAt:string;metrics:string[]};
export type OnboardingView={state:OnboardingState|null;monitoring:{symbol:string;enabled:boolean}[];reviews:FirstReview[]};
export type OnboardingResult={onboarding?:OnboardingView;portfolio?:Portfolio;error?:string;status?:number};
export type OnboardingCommand=
 | {action:'companies';version:number;portfolioVersion:number;symbols:string[];holdings:Holding[];importConfirmed:boolean}
 | {action:'topics';version:number;topics:OnboardingTopic[]}
 | {action:'complete'|'skip'|'restart';version:number};
export function validateOnboarding(input:unknown):OnboardingCommand {
 if(!input||typeof input!=='object')throw new PortfolioInputError('Invalid setup request.');
 const v=input as Record<string,unknown>,version=v.version;
 if(!Number.isSafeInteger(version)||(version as number)<0)throw new PortfolioInputError('Reload setup before saving.');
 const base={version:version as number};
 if(v.action==='companies'){
  if(!Number.isSafeInteger(v.portfolioVersion)||(v.portfolioVersion as number)<0||!Array.isArray(v.symbols)||v.symbols.length>20||!Array.isArray(v.holdings))throw new PortfolioInputError('Choose up to 20 companies to follow.');
  const symbols=[...new Set(v.symbols.map(validSymbol))];
  let holdings:Holding[]=[];
  if(v.holdings.length){
   if(v.importConfirmed!==true)throw new PortfolioInputError('Confirm the CSV quantities before importing.');
   const parsed=validateMutation({version:v.portfolioVersion,command:{type:'import',holdings:v.holdings}});
   if(parsed.command.type==='import')holdings=parsed.command.holdings;
  }
  return {...base,action:'companies',portfolioVersion:v.portfolioVersion as number,symbols,holdings,importConfirmed:v.importConfirmed===true};
 }
 if(v.action==='topics'){
  if(!Array.isArray(v.topics)||v.topics.length>4||v.topics.some(t=>typeof t!=='string'||!Object.hasOwn(onboardingTopics,t)))throw new PortfolioInputError('Choose a supported research topic.');
  return {...base,action:'topics',topics:[...new Set(v.topics)] as OnboardingTopic[]};
 }
 if(v.action==='complete'||v.action==='skip'||v.action==='restart')return {...base,action:v.action};
 throw new PortfolioInputError('Unknown setup action.');
}
export function rankFirstReviews(reviews:FirstReview[],topics:OnboardingTopic[]):FirstReview[]{
 const score=(r:FirstReview)=>topics.filter(t=>r.metrics.includes(t)).length;
 return [...reviews].sort((a,b)=>score(b)-score(a)||Date.parse(b.publishedAt)-Date.parse(a.publishedAt)||a.symbol.localeCompare(b.symbol));
}
