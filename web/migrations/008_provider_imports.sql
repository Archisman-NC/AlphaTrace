-- Provider evaluation is private to research operators, including raw responses.
CREATE TABLE alphatrace.provider_imports (
 id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
 symbol text NOT NULL CHECK(symbol IN ('TCS','INFY','HCLTECH','WIPRO','HINDUNILVR')),
 provider text NOT NULL DEFAULT 'indianapi' CHECK(provider='indianapi'),
 endpoint text NOT NULL, response_text text NOT NULL CHECK(octet_length(response_text)<=5242880),
 sha256 text NOT NULL CHECK(sha256 ~ '^[a-f0-9]{64}$'),
 fetched_at timestamptz NOT NULL, staged_at timestamptz NOT NULL DEFAULT now(), staged_by uuid NOT NULL,
 UNIQUE(symbol,sha256)
);
CREATE TABLE alphatrace.provider_targets (
 symbol text PRIMARY KEY CHECK(symbol IN ('TCS','INFY','HCLTECH','WIPRO','HINDUNILVR')),
 enabled boolean NOT NULL DEFAULT false, next_due_at timestamptz, worker_seen_at timestamptz
);
CREATE TABLE alphatrace.provider_runs (
 id uuid PRIMARY KEY DEFAULT gen_random_uuid(), symbol text NOT NULL REFERENCES alphatrace.provider_targets(symbol),
 started_at timestamptz NOT NULL DEFAULT now(), lease_until timestamptz NOT NULL DEFAULT now()+interval '2 minutes',
 finished_at timestamptz, state text NOT NULL DEFAULT 'running' CHECK(state IN ('running','success','failed')),
 message text NOT NULL DEFAULT 'Fetching quarterly data.', import_id uuid REFERENCES alphatrace.provider_imports(id),
 CHECK((state='running')=(finished_at IS NULL))
);
CREATE UNIQUE INDEX provider_one_running ON alphatrace.provider_runs(symbol) WHERE state='running';
CREATE INDEX provider_quota_window ON alphatrace.provider_runs(started_at);
CREATE INDEX provider_import_recent ON alphatrace.provider_imports(symbol,fetched_at DESC);
CREATE TABLE alphatrace.provider_draft_links (
 import_id uuid PRIMARY KEY REFERENCES alphatrace.provider_imports(id),
 draft_id uuid NOT NULL REFERENCES alphatrace.research_drafts(id), created_by uuid NOT NULL,
 created_at timestamptz NOT NULL DEFAULT now()
);
DO $$ DECLARE t text; BEGIN
 FOREACH t IN ARRAY ARRAY['provider_imports','provider_targets','provider_runs','provider_draft_links'] LOOP
  EXECUTE format('ALTER TABLE alphatrace.%I ENABLE ROW LEVEL SECURITY',t);
  EXECUTE format('ALTER TABLE alphatrace.%I FORCE ROW LEVEL SECURITY',t);
  EXECUTE format('CREATE POLICY operator_only ON alphatrace.%I FOR ALL TO alphatrace_app USING (EXISTS (SELECT 1 FROM alphatrace.research_operators WHERE user_id=NULLIF(current_setting(''app.user_id'',true),'''')::uuid)) WITH CHECK (EXISTS (SELECT 1 FROM alphatrace.research_operators WHERE user_id=NULLIF(current_setting(''app.user_id'',true),'''')::uuid))',t);
  EXECUTE format('REVOKE ALL ON alphatrace.%I FROM PUBLIC',t);
 END LOOP;
END $$;
GRANT SELECT,INSERT ON alphatrace.provider_imports,alphatrace.provider_draft_links TO alphatrace_app;
GRANT SELECT,INSERT,UPDATE ON alphatrace.provider_runs TO alphatrace_app;
GRANT SELECT,UPDATE ON alphatrace.provider_targets TO alphatrace_app;
INSERT INTO alphatrace.provider_targets(symbol) VALUES('TCS'),('INFY'),('HCLTECH'),('WIPRO'),('HINDUNILVR');
