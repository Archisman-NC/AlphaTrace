-- Immutable change observations; mutable triage state with append-only action history.
CREATE TABLE alphatrace.provider_review_heads (
 symbol text PRIMARY KEY REFERENCES alphatrace.provider_targets(symbol),
 import_id uuid NOT NULL REFERENCES alphatrace.provider_imports(id),
 observed_at timestamptz NOT NULL, semantic_hash text NOT NULL CHECK(semantic_hash ~ '^[a-f0-9]{64}$'),
 sequence integer NOT NULL CHECK(sequence>0)
);
CREATE TABLE alphatrace.provider_reviews (
 id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
 symbol text NOT NULL REFERENCES alphatrace.provider_targets(symbol),
 sequence integer NOT NULL CHECK(sequence>0),
 current_import_id uuid NOT NULL REFERENCES alphatrace.provider_imports(id),
 previous_import_id uuid REFERENCES alphatrace.provider_imports(id),
 observed_at timestamptz NOT NULL, previous_observed_at timestamptz,
 detected_at timestamptz NOT NULL DEFAULT now(),
 diff jsonb NOT NULL CHECK(jsonb_typeof(diff)='object' AND octet_length(diff::text)<=1048576),
 status text NOT NULL DEFAULT 'pending' CHECK(status IN ('pending','reviewed')),
 version integer NOT NULL DEFAULT 1 CHECK(version>0),
 reviewed_at timestamptz, reviewed_by uuid, note text NOT NULL DEFAULT '' CHECK(length(note)<=500),
 CHECK((status='reviewed')=(reviewed_at IS NOT NULL AND reviewed_by IS NOT NULL)),
 UNIQUE(symbol,sequence)
);
CREATE INDEX provider_reviews_queue ON alphatrace.provider_reviews(status,observed_at DESC,id DESC);
CREATE TABLE alphatrace.provider_review_actions (
 id uuid PRIMARY KEY DEFAULT gen_random_uuid(),review_id uuid NOT NULL REFERENCES alphatrace.provider_reviews(id),
 version integer NOT NULL,action text NOT NULL CHECK(action IN ('review','reopen')),note text NOT NULL CHECK(length(note)<=500),
 actor uuid NOT NULL,created_at timestamptz NOT NULL DEFAULT now(),UNIQUE(review_id,version)
);
DO $$ DECLARE t text; BEGIN
 FOREACH t IN ARRAY ARRAY['provider_review_heads','provider_reviews','provider_review_actions'] LOOP
  EXECUTE format('ALTER TABLE alphatrace.%I ENABLE ROW LEVEL SECURITY',t);
  EXECUTE format('ALTER TABLE alphatrace.%I FORCE ROW LEVEL SECURITY',t);
  EXECUTE format('CREATE POLICY operator_only ON alphatrace.%I FOR ALL TO alphatrace_app USING (EXISTS (SELECT 1 FROM alphatrace.research_operators WHERE user_id=NULLIF(current_setting(''app.user_id'',true),'''')::uuid)) WITH CHECK (EXISTS (SELECT 1 FROM alphatrace.research_operators WHERE user_id=NULLIF(current_setting(''app.user_id'',true),'''')::uuid))',t);
  EXECUTE format('REVOKE ALL ON alphatrace.%I FROM PUBLIC',t);
 END LOOP;
END $$;
GRANT SELECT,INSERT,UPDATE ON alphatrace.provider_review_heads TO alphatrace_app;
GRANT SELECT,INSERT ON alphatrace.provider_reviews,alphatrace.provider_review_actions TO alphatrace_app;
GRANT UPDATE(status,version,reviewed_at,reviewed_by,note) ON alphatrace.provider_reviews TO alphatrace_app;
