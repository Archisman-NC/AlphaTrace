import { timingSafeEqual } from 'node:crypto';
export const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
export function localAccessAllowed(env: NodeJS.ProcessEnv, host: string | null) {
  return env.NODE_ENV === 'development' && env.ALPHATRACE_LOCAL_DEV === 'true'
    && /^(127\.0\.0\.1|localhost)(:\d+)?$/.test(host ?? '');
}
export function validDevToken(provided: string, expected: string | undefined) {
  if (!expected || expected.length < 32) return false;
  const a=Buffer.from(provided), b=Buffer.from(expected);
  return a.length === b.length && timingSafeEqual(a,b);
}
export function workspaceName(value: unknown): string {
  if (typeof value !== 'string') throw new Error('Enter a workspace name.');
  const name=value.trim();
  if (name.length < 1 || name.length > 80 || /[\u0000-\u001f\u007f]/.test(name))
    throw new Error('Use 1–80 characters without control characters.');
  return name;
}
