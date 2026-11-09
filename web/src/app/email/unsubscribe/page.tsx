import type { Metadata } from 'next';
import EmailUnsubscribe from '@/components/email-unsubscribe';
export const dynamic='force-dynamic';
export const metadata:Metadata={title:'Unsubscribe · AlphaTrace',referrer:'no-referrer',robots:{index:false,follow:false}};
export default async function UnsubscribePage({searchParams}:{searchParams:Promise<{token?:string}>}){
 const token=(await searchParams).token;
 return <EmailUnsubscribe token={typeof token==='string'&&/^[A-Za-z0-9_-]{43}$/.test(token)?token:null}/>;
}
