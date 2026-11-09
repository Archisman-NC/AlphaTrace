CREATE TABLE alphatrace.review_items (
 owner_id uuid NOT NULL REFERENCES alphatrace.workspaces(owner_id) ON DELETE CASCADE,
 update_id uuid NOT NULL REFERENCES alphatrace.research_updates(id),
 snapshot jsonb NOT NULL CHECK(jsonb_typeof(snapshot)='object'),
 created_at timestamptz NOT NULL DEFAULT now(),
 version integer NOT NULL DEFAULT 0 CHECK(version>=0),
 state text NOT NULL DEFAULT 'unreviewed' CHECK(state IN ('unreviewed','kept','dismissed','deferred','revised')),
 revisit_at timestamptz,
 updated_at timestamptz NOT NULL DEFAULT now(),
 PRIMARY KEY(owner_id,update_id),
 CHECK((state='deferred')=(revisit_at IS NOT NULL))
);
CREATE TABLE alphatrace.review_events (
 owner_id uuid NOT NULL, update_id uuid NOT NULL,
 version integer NOT NULL CHECK(version>0), command_id uuid NOT NULL,
 action text NOT NULL CHECK(action IN ('keep','dismiss','defer','revise','reopen')),
 note text NOT NULL CHECK(char_length(note) BETWEEN 3 AND 2000),
 revisit_at timestamptz, thesis_revision jsonb,
 command jsonb NOT NULL, created_at timestamptz NOT NULL DEFAULT now(),
 PRIMARY KEY(owner_id,update_id,version), UNIQUE(owner_id,command_id),
 FOREIGN KEY(owner_id,update_id) REFERENCES alphatrace.review_items(owner_id,update_id) ON DELETE CASCADE,
 CHECK((action='defer')=(revisit_at IS NOT NULL)),
 CHECK((action='revise')=(thesis_revision IS NOT NULL))
);
CREATE INDEX review_items_queue ON alphatrace.review_items(owner_id,created_at DESC,update_id DESC);
CREATE INDEX review_items_due ON alphatrace.review_items(owner_id,state,revisit_at);
ALTER TABLE alphatrace.review_items ENABLE ROW LEVEL SECURITY;
ALTER TABLE alphatrace.review_items FORCE ROW LEVEL SECURITY;
ALTER TABLE alphatrace.review_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE alphatrace.review_events FORCE ROW LEVEL SECURITY;
CREATE POLICY own_review_items ON alphatrace.review_items FOR ALL TO alphatrace_app
 USING(owner_id=NULLIF(current_setting('app.user_id',true),'')::uuid)
 WITH CHECK(owner_id=NULLIF(current_setting('app.user_id',true),'')::uuid);
CREATE POLICY own_review_events ON alphatrace.review_events FOR ALL TO alphatrace_app
 USING(owner_id=NULLIF(current_setting('app.user_id',true),'')::uuid)
 WITH CHECK(owner_id=NULLIF(current_setting('app.user_id',true),'')::uuid);
GRANT SELECT,INSERT ON alphatrace.review_items,alphatrace.review_events TO alphatrace_app;
GRANT UPDATE(version,state,revisit_at,updated_at) ON alphatrace.review_items TO alphatrace_app;
REVOKE ALL ON alphatrace.review_items,alphatrace.review_events FROM PUBLIC;
COMMENT ON TABLE alphatrace.review_items IS 'One private item per owner/publication. Original relevance and research snapshot cannot be edited by the runtime.';
COMMENT ON TABLE alphatrace.review_events IS 'Append-only private decisions referencing an exact publication and, for revisions, the exact saved thesis wording.';
