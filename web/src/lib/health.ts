import {asUser} from './db';
import {ResearchForbiddenError} from './research-contract';
import {healthView,type HealthSnapshot} from './health-contract';
import {emailsAvailable} from './email-contract';
export async function getHealth(owner:string){return asUser(owner,async c=>{
 if(!(await c.query('SELECT 1 FROM alphatrace.research_operators WHERE user_id=$1',[owner])).rowCount)throw new ResearchForbiddenError('Service health requires operator access.');
 const snapshot=(await c.query('SELECT alphatrace.operator_health_snapshot() AS snapshot')).rows[0].snapshot as HealthSnapshot;
 return healthView(snapshot,emailsAvailable(process.env));
},{readOnlySnapshot:true,statementTimeoutMs:5000});}
