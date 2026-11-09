'use client';
import RecoveryScreen from '@/components/recovery-screen';
// Next 16.4 retries by re-fetching and rendering the failed segment.
export default function ErrorPage({retry}:{error:unknown;retry:()=>void}){return <RecoveryScreen retry={retry}/>;}
