import { asUser } from './db';
import { workspaceName } from './access';
export type Workspace={ id:string; name:string; created_at:Date; updated_at:Date };
export async function getWorkspace(ownerId:string):Promise<Workspace|null> {
  return asUser(ownerId,async db=>(await db.query<Workspace>(
    'SELECT id,name,created_at,updated_at FROM alphatrace.workspaces WHERE owner_id=$1',[ownerId])).rows[0]??null);
}
export async function saveWorkspace(ownerId:string,value:unknown):Promise<Workspace> {
  const name=workspaceName(value);
  return asUser(ownerId,async db=>(await db.query<Workspace>(
    `INSERT INTO alphatrace.workspaces(owner_id,name) VALUES ($1,$2)
     ON CONFLICT (owner_id) DO UPDATE SET name=EXCLUDED.name, updated_at=now()
     RETURNING id,name,created_at,updated_at`,[ownerId,name])).rows[0]);
}
