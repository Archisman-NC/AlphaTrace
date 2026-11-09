import type { PoolClient } from 'pg';
import { asUser } from './db';
import { validateMutation,PortfolioConflictError,PortfolioInputError,type Portfolio } from './portfolio-input';
async function snapshot(client:PoolClient,ownerId:string):Promise<Portfolio> {
 const workspace=await client.query('SELECT portfolio_version FROM alphatrace.workspaces WHERE owner_id=$1',[ownerId]);
 const rows=await client.query('SELECT symbol, quantity::text, followed FROM alphatrace.portfolio_entries WHERE owner_id=$1 ORDER BY symbol',[ownerId]);
 return {version:workspace.rows[0]?.portfolio_version??0,entries:rows.rows};
}
export async function getPortfolio(ownerId:string):Promise<Portfolio> {
 return asUser(ownerId,async client=>{
  // Lock the parent when present so rows and version describe the same snapshot.
  const parent=await client.query('SELECT id FROM alphatrace.workspaces WHERE owner_id=$1 FOR SHARE',[ownerId]);
  if(!parent.rowCount)return {version:0,entries:[]};
  return snapshot(client,ownerId);
 });
}
export async function updatePortfolio(ownerId:string,input:unknown):Promise<Portfolio> {
 const {version,command}=validateMutation(input);
 return asUser(ownerId,async client=>{
  await client.query("INSERT INTO alphatrace.workspaces(owner_id,name) VALUES ($1,'My research') ON CONFLICT(owner_id) DO NOTHING",[ownerId]);
  const locked=await client.query('SELECT portfolio_version FROM alphatrace.workspaces WHERE owner_id=$1 FOR UPDATE',[ownerId]);
  if(locked.rows[0].portfolio_version!==version)throw new PortfolioConflictError('This portfolio changed in another tab. Reload it before saving again.');
  if(command.type==='follow'){
   if(command.followed){
    const count=await client.query('SELECT count(*)::int AS total FROM alphatrace.portfolio_entries WHERE owner_id=$1 AND followed AND symbol<>$2',[ownerId,command.symbol]);
    if(count.rows[0].total>=20)throw new PortfolioInputError('You can follow up to 20 companies.');
    await client.query('INSERT INTO alphatrace.portfolio_entries(owner_id,symbol,followed) VALUES ($1,$2,true) ON CONFLICT(owner_id,symbol) DO UPDATE SET followed=true,updated_at=now()',[ownerId,command.symbol]);
   }else{
    await client.query('DELETE FROM alphatrace.portfolio_entries WHERE owner_id=$1 AND symbol=$2 AND quantity IS NULL',[ownerId,command.symbol]);
    await client.query('UPDATE alphatrace.portfolio_entries SET followed=false,updated_at=now() WHERE owner_id=$1 AND symbol=$2',[ownerId,command.symbol]);
   }
  }else if(command.type==='remove_holding'){
   await client.query('DELETE FROM alphatrace.portfolio_entries WHERE owner_id=$1 AND symbol=$2 AND NOT followed',[ownerId,command.symbol]);
   await client.query('UPDATE alphatrace.portfolio_entries SET quantity=NULL,updated_at=now() WHERE owner_id=$1 AND symbol=$2',[ownerId,command.symbol]);
  }else{
   const holdings=command.type==='import'?command.holdings:[command];
   for(const holding of holdings){
    await client.query('INSERT INTO alphatrace.portfolio_entries(owner_id,symbol,quantity) VALUES ($1,$2,$3) ON CONFLICT(owner_id,symbol) DO UPDATE SET quantity=EXCLUDED.quantity,updated_at=now()',[ownerId,holding.symbol,holding.quantity]);
   }
  }
  await client.query('UPDATE alphatrace.workspaces SET portfolio_version=portfolio_version+1,updated_at=now() WHERE owner_id=$1',[ownerId]);
  return snapshot(client,ownerId);
 });
}
