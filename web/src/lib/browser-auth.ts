import { createClient, type SupabaseClient } from '@supabase/supabase-js';

type BrowserAuthCache = { url: string; key: string; client: SupabaseClient };

/** One auth client per tab, including React Strict Mode and development refreshes. */
export function browserAuth(url: string, key: string): SupabaseClient {
  if (typeof window === 'undefined') throw new Error('Browser auth requires a browser.');
  const tab = window as Window & { alphaTraceAuth?: BrowserAuthCache };
  const cached = tab.alphaTraceAuth;
  if (cached?.url === url && cached.key === key) return cached.client;
  cached?.client.auth.stopAutoRefresh();
  const client = createClient(url, key, { auth: {
    persistSession: true, autoRefreshToken: true, detectSessionInUrl: true,
    storage: window.sessionStorage, storageKey: 'alphatrace.auth', flowType: 'implicit',
  } });
  tab.alphaTraceAuth = { url, key, client };
  return client;
}
