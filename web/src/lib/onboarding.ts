import type { PoolClient } from 'pg';
import { asUser } from './db';
import { portfolioSnapshot } from './portfolios';
import { PortfolioConflictError,PortfolioInputError } from './portfolio-input';
import { validateOnboarding,type OnboardingView,type OnboardingState } from './onboarding-contract';
async function view(c:PoolClient,id:string):Promise<OnboardingView>{
 const state=(await c.query('SELECT version,step,status,topics FROM alphatrace.user_onboarding WHERE owner_id=$1',[id])).rows[0]??null;
 const monitoring=(await c.query('SELECT symbol,enabled FROM alphatrace.onboarding_monitoring_status()')).rows;
 const reviews=(await c.query(`SELECT u.id,u.symbol,u.payload->>'title' AS title,u.payload->>'periodEnd' AS "periodEnd",u.published_at AS "publishedAt",
  ARRAY(SELECT v->>'metric' FROM jsonb_array_elements(u.payload->'comparisons') v) AS metrics
  FROM (SELECT DISTINCT ON(symbol) * FROM alphatrace.research_updates ORDER BY symbol,version DESC) u
  JOIN alphatrace.portfolio_entries p ON p.symbol=u.symbol AND p.owner_id=$1
  WHERE u.withdrawn_at IS NULL ORDER BY u.published_at DESC`,[id])).rows;
 return {state,monitoring,reviews};
}
export async function getOnboarding(id:string):Promise<OnboardingView>{return asUser(id,c=>view(c,id),{readOnlySnapshot:true});}
export async function saveOnboarding(id:string,input:unknown){
 const command=validateOnboarding(input);
 return asUser(id,async c=>{
  // Consistent parent-first locking with portfolio writes prevents cross-tab races.
  await c.query("INSERT INTO alphatrace.workspaces(owner_id,name) VALUES($1,'My research') ON CONFLICT(owner_id) DO NOTHING",[id]);
  await c.query('SELECT id FROM alphatrace.workspaces WHERE owner_id=$1 FOR UPDATE',[id]);
  const current:OnboardingState=(await c.query('SELECT version,step,status,topics FROM alphatrace.user_onboarding WHERE owner_id=$1 FOR UPDATE',[id])).rows[0]??{version:0,step:'companies',status:'active',topics:[]};
  if(current.version!==command.version)throw new PortfolioConflictError('Setup changed in another tab. Reload before continuing.');
  let portfolio=await portfolioSnapshot(c,id),step=current.step,status=current.status,topics=current.topics;
  if(command.action==='companies'){
   if(portfolio.version!==command.portfolioVersion)throw new PortfolioConflictError('Your companies changed in another tab. Reload setup before saving.');
   if(new Set([...portfolio.entries.filter(e=>e.followed).map(e=>e.symbol),...command.symbols]).size>20)throw new PortfolioInputError('You can follow up to 20 companies. Existing follows are kept.');
   const additions=command.symbols.filter(s=>!portfolio.entries.some(e=>e.symbol===s&&e.followed));
   // Batch writes keep a 20-company setup bounded to a few database round trips.
   // Input uses the same symbol and exact-quantity validation as ordinary imports.
   if(additions.length)await c.query(`INSERT INTO alphatrace.portfolio_entries(owner_id,symbol,followed)
    SELECT $1,symbol,true FROM unnest($2::text[]) symbol
    ON CONFLICT(owner_id,symbol) DO UPDATE SET followed=true,updated_at=now()`,[id,additions]);
   if(command.holdings.length)await c.query(`INSERT INTO alphatrace.portfolio_entries(owner_id,symbol,quantity)
    SELECT $1,symbol,quantity::numeric FROM unnest($2::text[],$3::text[]) h(symbol,quantity)
    ON CONFLICT(owner_id,symbol) DO UPDATE SET quantity=EXCLUDED.quantity,updated_at=now()`,[id,command.holdings.map(h=>h.symbol),command.holdings.map(h=>h.quantity)]);
   if(additions.length||command.holdings.length){
    await c.query('UPDATE alphatrace.workspaces SET portfolio_version=portfolio_version+1,updated_at=now() WHERE owner_id=$1',[id]);
    portfolio=await portfolioSnapshot(c,id);
   }
   if(!portfolio.entries.length)throw new PortfolioInputError('Choose a company or import a supported holding to continue.');
   step='topics';status='active';
  }else if(command.action==='topics'){
   if(!portfolio.entries.length||current.step==='companies')throw new PortfolioInputError('Save your companies before choosing topics.');
   topics=command.topics;step='review';status='active';
  }else if(command.action==='complete'){
   if(current.step!=='review'||!portfolio.entries.length)throw new PortfolioInputError('Finish the company and topic steps first.');
   status='completed';
  }else if(command.action==='skip')status='skipped';
  else {step='companies';status='active';}
  await c.query(`INSERT INTO alphatrace.user_onboarding(owner_id,version,step,status,topics) VALUES($1,$2,$3,$4,$5)
   ON CONFLICT(owner_id) DO UPDATE SET version=EXCLUDED.version,step=EXCLUDED.step,status=EXCLUDED.status,topics=EXCLUDED.topics,updated_at=now()`,[id,current.version+1,step,status,topics]);
  return {onboarding:await view(c,id),portfolio};
 });
}
