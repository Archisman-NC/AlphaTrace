CREATE TABLE alphatrace.user_onboarding (
 owner_id uuid PRIMARY KEY REFERENCES alphatrace.workspaces(owner_id) ON DELETE CASCADE,
 version integer NOT NULL DEFAULT 0 CHECK(version>=0),
 step text NOT NULL DEFAULT 'companies' CHECK(step IN ('companies','topics','review')),
 status text NOT NULL DEFAULT 'active' CHECK(status IN ('active','completed','skipped')),
 topics text[] NOT NULL DEFAULT '{}',
 updated_at timestamptz NOT NULL DEFAULT now(),
 CHECK(topics <@ ARRAY['revenue','operating_margin','operating_cash_flow','debt']::text[])
);
ALTER TABLE alphatrace.user_onboarding ENABLE ROW LEVEL SECURITY;
ALTER TABLE alphatrace.user_onboarding FORCE ROW LEVEL SECURITY;
CREATE POLICY own_onboarding ON alphatrace.user_onboarding FOR ALL TO alphatrace_app
 USING(owner_id=NULLIF(current_setting('app.user_id',true),'')::uuid)
 WITH CHECK(owner_id=NULLIF(current_setting('app.user_id',true),'')::uuid);
REVOKE ALL ON alphatrace.user_onboarding FROM PUBLIC;
GRANT SELECT,INSERT,UPDATE ON alphatrace.user_onboarding TO alphatrace_app;

-- Only configuration status is public to invited readers. Provider responses,
-- credentials, runs and operator controls retain their existing RLS policies.
CREATE FUNCTION alphatrace.onboarding_monitoring_status()
 RETURNS TABLE(symbol text,enabled boolean)
 LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog
 AS $$
 SELECT t.symbol,bool_or(t.enabled) FROM (
  SELECT p.symbol,p.enabled FROM alphatrace.provider_targets p
  UNION ALL SELECT m.symbol,m.enabled FROM alphatrace.monitor_targets m
 ) t WHERE NULLIF(current_setting('app.user_id',true),'') IS NOT NULL GROUP BY t.symbol
 $$;
REVOKE ALL ON FUNCTION alphatrace.onboarding_monitoring_status() FROM PUBLIC;
GRANT EXECUTE ON FUNCTION alphatrace.onboarding_monitoring_status() TO alphatrace_app;
