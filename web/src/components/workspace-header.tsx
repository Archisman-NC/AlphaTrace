import type { ReactNode } from 'react';

export default function WorkspaceHeader({active,local,children}:{active:'overview'|'companies'|'inbox'|'research'|'monitoring';local:boolean;children?:ReactNode}) {
 const links=[['overview','Overview','/workspace'],['companies','My companies','/workspace?view=companies'],['inbox','Review inbox','/inbox'],['research','Research','/research'],['monitoring','Monitoring','/monitoring']] as const;
 return <><a className="skip-link workspace-skip" href="#workspace-content">Skip to content</a><header className="workspace-header">
  <a href="/workspace" className="private-brand" aria-label="AlphaTrace overview"><span className="brand-mark">αT</span>AlphaTrace</a>
  <nav aria-label="Workspace navigation">{links.map(([id,label,href])=><a key={id} href={href} aria-current={active===id?'page':undefined}>{label}</a>)}</nav>
  <div className="workspace-account"><span className="badge">{local?'Local development':'Private beta'}</span>{children}</div>
 </header></>;
}
