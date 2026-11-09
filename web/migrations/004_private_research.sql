CREATE TABLE alphatrace.private_research (
 owner_id uuid NOT NULL REFERENCES alphatrace.workspaces(owner_id) ON DELETE CASCADE,
 symbol text NOT NULL, version integer NOT NULL DEFAULT 0 CHECK(version>=0),
 updated_at timestamptz NOT NULL DEFAULT now(), PRIMARY KEY(owner_id,symbol)
);
CREATE TABLE alphatrace.private_research_revisions (
 owner_id uuid NOT NULL, symbol text NOT NULL, version integer NOT NULL CHECK(version>0),
 command_id uuid NOT NULL, payload jsonb NOT NULL CHECK(jsonb_typeof(payload)='object'),
 reason text NOT NULL CHECK(char_length(reason) BETWEEN 3 AND 1000), created_at timestamptz NOT NULL DEFAULT now(),
 PRIMARY KEY(owner_id,symbol,version), UNIQUE(owner_id,command_id),
 FOREIGN KEY(owner_id,symbol) REFERENCES alphatrace.private_research(owner_id,symbol) ON DELETE CASCADE
);
ALTER TABLE alphatrace.private_research ENABLE ROW LEVEL SECURITY;
ALTER TABLE alphatrace.private_research FORCE ROW LEVEL SECURITY;
ALTER TABLE alphatrace.private_research_revisions ENABLE ROW LEVEL SECURITY;
ALTER TABLE alphatrace.private_research_revisions FORCE ROW LEVEL SECURITY;
CREATE POLICY own_private_research ON alphatrace.private_research FOR ALL TO alphatrace_app
 USING(owner_id=NULLIF(current_setting('app.user_id',true),'')::uuid)
 WITH CHECK(owner_id=NULLIF(current_setting('app.user_id',true),'')::uuid);
CREATE POLICY own_research_revisions ON alphatrace.private_research_revisions FOR ALL TO alphatrace_app
 USING(owner_id=NULLIF(current_setting('app.user_id',true),'')::uuid)
 WITH CHECK(owner_id=NULLIF(current_setting('app.user_id',true),'')::uuid);
GRANT SELECT,INSERT ON alphatrace.private_research TO alphatrace_app;
GRANT UPDATE(version,updated_at) ON alphatrace.private_research TO alphatrace_app;
GRANT SELECT,INSERT ON alphatrace.private_research_revisions TO alphatrace_app;
REVOKE ALL ON alphatrace.private_research,alphatrace.private_research_revisions FROM PUBLIC;
COMMENT ON TABLE alphatrace.private_research_revisions IS 'Private user-confirmed research, append-only to the runtime role. Never shared with research ingestion or a model.';
