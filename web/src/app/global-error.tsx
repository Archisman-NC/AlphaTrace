'use client';
import RecoveryScreen from '@/components/recovery-screen';
import './reliability.css';
export default function GlobalError({retry}:{error:unknown;retry:()=>void}){return <html lang="en"><body style={{margin:0,background:'#0b0e13',color:'#ebedf3',fontFamily:'system-ui,sans-serif'}}><RecoveryScreen retry={retry}/></body></html>;}
