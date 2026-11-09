import type { Metadata } from 'next';
import './globals.css';
import './workspace.css';
import './onboarding.css';
import './email.css';
import './issues.css';
import './account.css';
export const metadata:Metadata={title:'AlphaTrace · Research workspace',description:'An evidence-backed investment research workspace.',robots:{index:false,follow:false}};
export default function Layout({children}:{children:React.ReactNode}) {
  return <html lang="en"><body>{children}</body></html>;
}
