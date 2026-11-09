import { unsubscribeEmail } from '@/lib/emails';
import { PortfolioInputError } from '@/lib/portfolio-input';
export const runtime='nodejs';
export const dynamic='force-dynamic';
const headers={'Cache-Control':'no-store','Referrer-Policy':'no-referrer'};
export async function GET(request:Request){
 const token=new URL(request.url).searchParams.get('token')??'',url=new URL('/email/unsubscribe',request.url);url.searchParams.set('token',token);
 return new Response(null,{status:303,headers:{...headers,Location:url.href}});
}
export async function POST(request:Request){
 try{await unsubscribeEmail(new URL(request.url).searchParams.get('token'));return new Response(null,{status:200,headers});}
 catch(e){return Response.json({error:e instanceof PortfolioInputError?e.message:'Could not confirm unsubscribe. Please retry.'},{status:e instanceof PortfolioInputError?400:503,headers});}
}
