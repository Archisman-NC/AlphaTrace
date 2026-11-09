-- Research is shared; draft visibility and all publication writes require an operator.
CREATE TABLE alphatrace.research_operators (user_id uuid PRIMARY KEY, created_at timestamptz NOT NULL DEFAULT now());
ALTER TABLE alphatrace.research_operators ENABLE ROW LEVEL SECURITY;
ALTER TABLE alphatrace.research_operators FORCE ROW LEVEL SECURITY;
CREATE POLICY own_operator_record ON alphatrace.research_operators FOR SELECT TO alphatrace_app
  USING (user_id = NULLIF(current_setting('app.user_id', true), '')::uuid);
GRANT SELECT ON alphatrace.research_operators TO alphatrace_app;

CREATE TABLE alphatrace.research_drafts (
 id uuid PRIMARY KEY DEFAULT gen_random_uuid(), symbol text NOT NULL, content_hash text UNIQUE NOT NULL CHECK (length(content_hash)=64),
 payload jsonb NOT NULL CHECK (jsonb_typeof(payload)='object'), created_by uuid NOT NULL, created_at timestamptz NOT NULL DEFAULT now(),
 state text NOT NULL DEFAULT 'draft' CHECK (state IN ('draft','needs_correction','approved','published')),
 review_note text, reviewed_by uuid, reviewed_at timestamptz, published_update_id uuid
);
CREATE TABLE alphatrace.research_updates (
 id uuid PRIMARY KEY DEFAULT gen_random_uuid(), draft_id uuid UNIQUE NOT NULL REFERENCES alphatrace.research_drafts(id),
 symbol text NOT NULL, version integer NOT NULL CHECK(version>0), period_end date NOT NULL,
 payload jsonb NOT NULL, published_by uuid NOT NULL, published_at timestamptz NOT NULL DEFAULT now(),
 supersedes uuid REFERENCES alphatrace.research_updates(id), withdrawn_at timestamptz, withdrawal_reason text,
 UNIQUE(symbol,version), CHECK ((withdrawn_at IS NULL) = (withdrawal_reason IS NULL))
);
ALTER TABLE alphatrace.research_drafts ADD CONSTRAINT published_update_reference FOREIGN KEY (published_update_id) REFERENCES alphatrace.research_updates(id);
CREATE TABLE alphatrace.research_checks (
 id uuid PRIMARY KEY DEFAULT gen_random_uuid(), symbol text NOT NULL, checked_at timestamptz NOT NULL,
 status text NOT NULL CHECK(status IN ('success','failed')), message text NOT NULL, fingerprints jsonb NOT NULL,
 recorded_by uuid NOT NULL, recorded_at timestamptz NOT NULL DEFAULT now(), UNIQUE(symbol,checked_at)
);
CREATE INDEX research_update_company ON alphatrace.research_updates(symbol,version DESC);
CREATE INDEX research_check_company ON alphatrace.research_checks(symbol,checked_at DESC);
CREATE INDEX research_draft_created ON alphatrace.research_drafts(created_at DESC);

ALTER TABLE alphatrace.research_drafts ENABLE ROW LEVEL SECURITY;
ALTER TABLE alphatrace.research_drafts FORCE ROW LEVEL SECURITY;
ALTER TABLE alphatrace.research_updates ENABLE ROW LEVEL SECURITY;
ALTER TABLE alphatrace.research_updates FORCE ROW LEVEL SECURITY;
ALTER TABLE alphatrace.research_checks ENABLE ROW LEVEL SECURITY;
ALTER TABLE alphatrace.research_checks FORCE ROW LEVEL SECURITY;

CREATE POLICY operator_drafts ON alphatrace.research_drafts FOR ALL TO alphatrace_app
 USING (EXISTS (SELECT 1 FROM alphatrace.research_operators WHERE user_id=NULLIF(current_setting('app.user_id',true),'')::uuid))
 WITH CHECK (EXISTS (SELECT 1 FROM alphatrace.research_operators WHERE user_id=NULLIF(current_setting('app.user_id',true),'')::uuid));
CREATE POLICY read_shared_updates ON alphatrace.research_updates FOR SELECT TO alphatrace_app
 USING (NULLIF(current_setting('app.user_id',true),'') IS NOT NULL);
CREATE POLICY operator_publish ON alphatrace.research_updates FOR INSERT TO alphatrace_app
 WITH CHECK (published_by=NULLIF(current_setting('app.user_id',true),'')::uuid AND EXISTS (SELECT 1 FROM alphatrace.research_operators WHERE user_id=published_by));
CREATE POLICY operator_withdraw ON alphatrace.research_updates FOR UPDATE TO alphatrace_app
 USING (EXISTS (SELECT 1 FROM alphatrace.research_operators WHERE user_id=NULLIF(current_setting('app.user_id',true),'')::uuid))
 WITH CHECK (EXISTS (SELECT 1 FROM alphatrace.research_operators WHERE user_id=NULLIF(current_setting('app.user_id',true),'')::uuid));
CREATE POLICY read_shared_checks ON alphatrace.research_checks FOR SELECT TO alphatrace_app
 USING (NULLIF(current_setting('app.user_id',true),'') IS NOT NULL);
CREATE POLICY operator_checks ON alphatrace.research_checks FOR INSERT TO alphatrace_app
 WITH CHECK (recorded_by=NULLIF(current_setting('app.user_id',true),'')::uuid AND EXISTS (SELECT 1 FROM alphatrace.research_operators WHERE user_id=recorded_by));

GRANT SELECT,INSERT ON alphatrace.research_drafts TO alphatrace_app;
GRANT UPDATE(state,review_note,reviewed_by,reviewed_at,published_update_id) ON alphatrace.research_drafts TO alphatrace_app;
GRANT SELECT,INSERT ON alphatrace.research_updates TO alphatrace_app;
GRANT UPDATE(withdrawn_at,withdrawal_reason) ON alphatrace.research_updates TO alphatrace_app;
GRANT SELECT,INSERT ON alphatrace.research_checks TO alphatrace_app;
REVOKE ALL ON alphatrace.research_operators,alphatrace.research_drafts,alphatrace.research_updates,alphatrace.research_checks FROM PUBLIC;

-- Even a buggy server query cannot rewrite a published version or skip approval.
CREATE FUNCTION alphatrace.guard_research_publication() RETURNS trigger LANGUAGE plpgsql SET search_path=pg_catalog,alphatrace AS $$
DECLARE d alphatrace.research_drafts;
BEGIN
 SELECT * INTO d FROM alphatrace.research_drafts WHERE id=NEW.draft_id FOR UPDATE;
 IF d.id IS NULL OR d.state <> 'approved' OR d.reviewed_by IS NULL OR d.reviewed_at IS NULL OR d.symbol <> NEW.symbol OR d.payload <> NEW.payload OR NEW.period_end <> (d.payload->>'periodEnd')::date THEN
   RAISE EXCEPTION 'Publication requires the exact approved draft';
 END IF;
 RETURN NEW;
END $$;
CREATE TRIGGER check_research_publication BEFORE INSERT ON alphatrace.research_updates FOR EACH ROW EXECUTE FUNCTION alphatrace.guard_research_publication();
CREATE FUNCTION alphatrace.guard_research_review() RETURNS trigger LANGUAGE plpgsql SET search_path=pg_catalog,alphatrace AS $$
BEGIN
 IF TG_OP='INSERT' THEN
   IF NEW.state <> 'draft' OR NEW.reviewed_by IS NOT NULL OR NEW.reviewed_at IS NOT NULL OR NEW.published_update_id IS NOT NULL THEN RAISE EXCEPTION 'New drafts must be unreviewed'; END IF;
 ELSE
   IF OLD.state='published' THEN RAISE EXCEPTION 'Published drafts are immutable'; END IF;
   IF NEW.state='published' AND (OLD.state<>'approved' OR NOT EXISTS(SELECT 1 FROM alphatrace.research_updates WHERE id=NEW.published_update_id AND draft_id=NEW.id)) THEN RAISE EXCEPTION 'Missing publication'; END IF;
   IF NEW.state='approved' AND (NEW.reviewed_by IS DISTINCT FROM NULLIF(current_setting('app.user_id',true),'')::uuid OR NEW.reviewed_at IS NULL OR coalesce(length(btrim(NEW.review_note)),0)<10) THEN RAISE EXCEPTION 'Approval requires an identified reviewer and note'; END IF;
 END IF;
 RETURN NEW;
END $$;
CREATE TRIGGER check_research_review BEFORE INSERT OR UPDATE ON alphatrace.research_drafts FOR EACH ROW EXECUTE FUNCTION alphatrace.guard_research_review();
REVOKE ALL ON FUNCTION alphatrace.guard_research_publication(),alphatrace.guard_research_review() FROM PUBLIC;
