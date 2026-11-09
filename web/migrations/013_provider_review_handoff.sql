-- Immutable link from a detected observation to its initial evidence draft.
-- Successors are resolved through the existing append-only revision history.
CREATE TABLE alphatrace.provider_review_drafts (
 review_id uuid PRIMARY KEY REFERENCES alphatrace.provider_reviews(id),
 draft_id uuid UNIQUE NOT NULL REFERENCES alphatrace.research_drafts(id),
 created_by uuid NOT NULL,created_at timestamptz NOT NULL DEFAULT now()
);
ALTER TABLE alphatrace.provider_review_drafts ENABLE ROW LEVEL SECURITY;
ALTER TABLE alphatrace.provider_review_drafts FORCE ROW LEVEL SECURITY;
CREATE POLICY operator_only ON alphatrace.provider_review_drafts FOR ALL TO alphatrace_app
 USING(EXISTS(SELECT 1 FROM alphatrace.research_operators WHERE user_id=NULLIF(current_setting('app.user_id',true),'')::uuid))
 WITH CHECK(created_by=NULLIF(current_setting('app.user_id',true),'')::uuid AND EXISTS(SELECT 1 FROM alphatrace.research_operators WHERE user_id=created_by));
REVOKE ALL ON alphatrace.provider_review_drafts FROM PUBLIC;
GRANT SELECT,INSERT ON alphatrace.provider_review_drafts TO alphatrace_app;
