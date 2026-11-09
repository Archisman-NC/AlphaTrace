CREATE TABLE alphatrace.research_issues (
 id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
 owner_id uuid NOT NULL REFERENCES alphatrace.workspaces(owner_id) ON DELETE CASCADE,
 publication_id uuid NOT NULL REFERENCES alphatrace.research_updates(id), metric text,
 category text NOT NULL CHECK(category IN ('incorrect_value','period_or_units','source_link','unclear_explanation')),
 note text NOT NULL CHECK(char_length(note) BETWEEN 3 AND 2000), context jsonb NOT NULL DEFAULT '{}',
 status text NOT NULL DEFAULT 'open' CHECK(status IN ('open','in_review','resolved')),
 version integer NOT NULL DEFAULT 0 CHECK(version>=0),
 outcome text CHECK(outcome IN ('corrected','explained','withdrawn')), response text,
 correction_id uuid REFERENCES alphatrace.research_updates(id),
 last_command_id uuid NOT NULL, last_command jsonb NOT NULL, last_note text NOT NULL,
 created_at timestamptz NOT NULL DEFAULT now(), updated_at timestamptz NOT NULL DEFAULT now(),
 CHECK((status='resolved')=(outcome IS NOT NULL)),
 CHECK((status='resolved')=(response IS NOT NULL)),
 CHECK((outcome='corrected') IS NOT TRUE OR correction_id IS NOT NULL),
 CHECK(correction_id IS NULL OR outcome='corrected')
);
CREATE UNIQUE INDEX one_active_issue_per_subject ON alphatrace.research_issues(owner_id,publication_id,COALESCE(metric,''),category) WHERE status<>'resolved';
CREATE INDEX research_issue_queue ON alphatrace.research_issues(status,created_at DESC,id DESC);
CREATE INDEX research_issue_owner ON alphatrace.research_issues(owner_id,created_at DESC,id DESC);
CREATE TABLE alphatrace.research_issue_events (
 issue_id uuid NOT NULL REFERENCES alphatrace.research_issues(id) ON DELETE CASCADE,
 version integer NOT NULL, actor_id uuid NOT NULL, command_id uuid NOT NULL, command jsonb NOT NULL,
 action text NOT NULL CHECK(action IN ('submitted','in_review','resolved','reopened')),
 note text NOT NULL, outcome text, correction_id uuid REFERENCES alphatrace.research_updates(id),
 created_at timestamptz NOT NULL DEFAULT now(), PRIMARY KEY(issue_id,version), UNIQUE(actor_id,command_id)
);
ALTER TABLE alphatrace.research_issues ENABLE ROW LEVEL SECURITY;
ALTER TABLE alphatrace.research_issues FORCE ROW LEVEL SECURITY;
ALTER TABLE alphatrace.research_issue_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE alphatrace.research_issue_events FORCE ROW LEVEL SECURITY;
CREATE POLICY read_own_or_operator_issue ON alphatrace.research_issues FOR SELECT TO alphatrace_app
 USING(owner_id=NULLIF(current_setting('app.user_id',true),'')::uuid OR EXISTS(SELECT 1 FROM alphatrace.research_operators WHERE user_id=NULLIF(current_setting('app.user_id',true),'')::uuid));
CREATE POLICY create_own_issue ON alphatrace.research_issues FOR INSERT TO alphatrace_app
 WITH CHECK(owner_id=NULLIF(current_setting('app.user_id',true),'')::uuid);
CREATE POLICY operator_update_issue ON alphatrace.research_issues FOR UPDATE TO alphatrace_app
 USING(EXISTS(SELECT 1 FROM alphatrace.research_operators WHERE user_id=NULLIF(current_setting('app.user_id',true),'')::uuid))
 WITH CHECK(EXISTS(SELECT 1 FROM alphatrace.research_operators WHERE user_id=NULLIF(current_setting('app.user_id',true),'')::uuid));
CREATE POLICY read_issue_events ON alphatrace.research_issue_events FOR SELECT TO alphatrace_app
 USING(EXISTS(SELECT 1 FROM alphatrace.research_issues i WHERE i.id=issue_id));
REVOKE ALL ON alphatrace.research_issues,alphatrace.research_issue_events FROM PUBLIC;
GRANT SELECT,INSERT ON alphatrace.research_issues TO alphatrace_app;
GRANT UPDATE(status,version,outcome,response,correction_id,last_command_id,last_command,last_note) ON alphatrace.research_issues TO alphatrace_app;
GRANT SELECT ON alphatrace.research_issue_events TO alphatrace_app;

-- Derive immutable evidence context from the stored publication, not client JSON.
CREATE FUNCTION alphatrace.guard_research_issue() RETURNS trigger LANGUAGE plpgsql SET search_path=pg_catalog AS $$
 DECLARE original record; comparison jsonb; corrected record;
 BEGIN
  SELECT * INTO original FROM alphatrace.research_updates WHERE id=NEW.publication_id;
  IF NOT FOUND THEN RAISE EXCEPTION 'A published research version is required'; END IF;
  IF TG_OP='INSERT' THEN
   IF NEW.status<>'open' OR NEW.version<>0 OR NEW.outcome IS NOT NULL OR NEW.response IS NOT NULL OR NEW.correction_id IS NOT NULL THEN RAISE EXCEPTION 'Reports must start open'; END IF;
   IF NEW.metric IS NOT NULL THEN
    SELECT value INTO comparison FROM jsonb_array_elements(original.payload->'comparisons') WHERE value->>'metric'=NEW.metric;
    IF comparison IS NULL THEN RAISE EXCEPTION 'Metric is not in this publication'; END IF;
   END IF;
   NEW.context=jsonb_build_object('symbol',original.symbol,'publicationVersion',original.version,'title',original.payload->>'title','periodEnd',original.payload->>'periodEnd','comparisonType',original.payload->>'comparison','comparison',comparison,'sources',original.payload->'sources');
   NEW.created_at=now();NEW.last_note=NEW.note;
  ELSE
   IF (NEW.id,NEW.owner_id,NEW.publication_id,NEW.metric,NEW.category,NEW.note,NEW.context,NEW.created_at) IS DISTINCT FROM (OLD.id,OLD.owner_id,OLD.publication_id,OLD.metric,OLD.category,OLD.note,OLD.context,OLD.created_at) THEN RAISE EXCEPTION 'Original reports are immutable'; END IF;
   IF NEW.version<>OLD.version+1 OR NOT ((OLD.status='open' AND NEW.status='in_review') OR (OLD.status IN ('open','in_review') AND NEW.status='resolved') OR (OLD.status='resolved' AND NEW.status='open')) THEN RAISE EXCEPTION 'Invalid report transition'; END IF;
   IF NEW.status='resolved' THEN
    IF char_length(trim(NEW.response))<10 THEN RAISE EXCEPTION 'Explain the resolution'; END IF;
    IF NEW.outcome='corrected' THEN
     SELECT * INTO corrected FROM alphatrace.research_updates WHERE id=NEW.correction_id;
     IF NOT FOUND OR corrected.symbol<>original.symbol OR corrected.version<=original.version OR corrected.withdrawn_at IS NOT NULL THEN RAISE EXCEPTION 'Link a newer non-withdrawn publication for the same company'; END IF;
    ELSIF NEW.outcome='withdrawn' AND original.withdrawn_at IS NULL THEN RAISE EXCEPTION 'This publication has not been withdrawn'; END IF;
   END IF;
  END IF;
  NEW.updated_at=now();RETURN NEW;
 END $$;
CREATE TRIGGER guard_research_issue BEFORE INSERT OR UPDATE ON alphatrace.research_issues FOR EACH ROW EXECUTE FUNCTION alphatrace.guard_research_issue();

-- Only the trigger may append audit history; the application role cannot edit it.
CREATE FUNCTION alphatrace.append_research_issue_event() RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog AS $$
 BEGIN
  INSERT INTO alphatrace.research_issue_events(issue_id,version,actor_id,command_id,command,action,note,outcome,correction_id)
  VALUES(NEW.id,NEW.version,NULLIF(current_setting('app.user_id',true),'')::uuid,NEW.last_command_id,NEW.last_command,
   CASE WHEN TG_OP='INSERT' THEN 'submitted' WHEN NEW.status='open' THEN 'reopened' ELSE NEW.status END,NEW.last_note,NEW.outcome,NEW.correction_id);
  RETURN NEW;
 END $$;
REVOKE ALL ON FUNCTION alphatrace.append_research_issue_event() FROM PUBLIC;
CREATE TRIGGER append_research_issue_event AFTER INSERT OR UPDATE ON alphatrace.research_issues FOR EACH ROW EXECUTE FUNCTION alphatrace.append_research_issue_event();
