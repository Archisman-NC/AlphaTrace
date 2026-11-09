import {serviceBoundary} from '@/lib/service-errors';
import { requestIdentity } from '@/lib/auth';
import { getWorkspace,saveWorkspace } from '@/lib/workspaces';
import { workspaceName } from '@/lib/access';
export const runtime='nodejs';
export const dynamic='force-dynamic';
const json=(data:unknown,status=200)=>Response.json(data,{status,headers:{'Cache-Control':'no-store'}});
async function GETHandler(request:Request) {
  const id=await requestIdentity(request);
  if (!id) return json({error:'Authentication required'},401);
  try { return json({workspace:await getWorkspace(id)}); }
  catch { return json({error:'Workspace storage unavailable'},503); }
}
async function POSTHandler(request:Request) {
  const id=await requestIdentity(request);
  if (!id) return json({error:'Authentication required'},401);
  if (!request.headers.get('content-type')?.startsWith('application/json')) return json({error:'JSON required'},415);
  let name:string;
  try {
    // Read a bounded stream, not an unbounded body followed by a length check.
    const reader=request.body?.getReader(); if (!reader) throw new Error();
    const chunks:Uint8Array[]=[]; let size=0;
    while(true) { const part=await reader.read(); if(part.done) break;
      size+=part.value.byteLength; if(size>1024) { await reader.cancel(); return json({error:'Body too large'},413); }
      chunks.push(part.value);
    }
    const body=JSON.parse(Buffer.concat(chunks).toString('utf8'));
    name=workspaceName(body.name);
  } catch { return json({error:'Provide a name containing 1–80 characters.'},400); }
  try { return json({workspace:await saveWorkspace(id,name)}); }
  catch { return json({error:'Workspace storage unavailable'},503); }
}

export const GET=serviceBoundary(GETHandler);
export const POST=serviceBoundary(POSTHandler);
