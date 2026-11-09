-- Editing creates immutable successors; it never overwrites an approved payload.
CREATE TABLE alphatrace.research_revisions (
 draft_id uuid PRIMARY KEY REFERENCES alphatrace.research_drafts(id),
 parent_id uuid UNIQUE NOT NULL REFERENCES alphatrace.research_drafts(id),
 reason text NOT NULL CHECK(length(btrim(reason)) BETWEEN 10 AND 1000),
 created_by uuid NOT NULL, created_at timestamptz NOT NULL DEFAULT now(),
 CHECK(draft_id<>parent_id)
);
CREATE TABLE alphatrace.research_attachments (
 id uuid PRIMARY KEY DEFAULT gen_random_uuid(), symbol text NOT NULL,
 source jsonb NOT NULL, created_by uuid NOT NULL, created_at timestamptz NOT NULL DEFAULT now()
);
ALTER TABLE alphatrace.research_revisions ENABLE ROW LEVEL SECURITY;
ALTER TABLE alphatrace.research_revisions FORCE ROW LEVEL SECURITY;
ALTER TABLE alphatrace.research_attachments ENABLE ROW LEVEL SECURITY;
ALTER TABLE alphatrace.research_attachments FORCE ROW LEVEL SECURITY;
CREATE POLICY operator_revisions ON alphatrace.research_revisions FOR ALL TO alphatrace_app
 USING(EXISTS(SELECT 1 FROM alphatrace.research_operators WHERE user_id=NULLIF(current_setting('app.user_id',true),'')::uuid))
 WITH CHECK(created_by=NULLIF(current_setting('app.user_id',true),'')::uuid AND EXISTS(SELECT 1 FROM alphatrace.research_operators WHERE user_id=created_by));
CREATE POLICY own_attachments ON alphatrace.research_attachments FOR ALL TO alphatrace_app
 USING(created_by=NULLIF(current_setting('app.user_id',true),'')::uuid AND EXISTS(SELECT 1 FROM alphatrace.research_operators WHERE user_id=created_by))
 WITH CHECK(created_by=NULLIF(current_setting('app.user_id',true),'')::uuid AND EXISTS(SELECT 1 FROM alphatrace.research_operators WHERE user_id=created_by));
GRANT SELECT,INSERT ON alphatrace.research_revisions,alphatrace.research_attachments TO alphatrace_app;
REVOKE ALL ON alphatrace.research_revisions,alphatrace.research_attachments FROM PUBLIC;
