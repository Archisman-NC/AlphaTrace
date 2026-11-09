-- Keep all occupied seats visible alongside recent invitation history.
CREATE OR REPLACE FUNCTION alphatrace.beta_people() RETURNS jsonb LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog AS $$
 DECLARE result jsonb;
 BEGIN
 PERFORM alphatrace.beta_require_operator();
 SELECT jsonb_build_object('initialized',(SELECT initialized FROM alphatrace.beta_configuration),'limit',5,'checkedAt',now(),
 'used',(SELECT count(*) FROM alphatrace.beta_members WHERE state='active' OR (state='invited' AND expires_at>now())),
 'people',coalesce((SELECT jsonb_agg(row_to_json(t) ORDER BY t."createdAt" DESC) FROM (
 SELECT m.id,m.email,m.version,CASE WHEN m.state='invited' AND m.expires_at<=now() THEN 'expired' ELSE m.state END AS state,
 m.created_at AS "createdAt",m.expires_at AS "expiresAt",m.activated_at AS "activatedAt",
 EXISTS(SELECT 1 FROM alphatrace.research_operators WHERE user_id=m.user_id) AS "isOperator",
 EXISTS(SELECT 1 FROM alphatrace.user_onboarding WHERE owner_id=m.user_id AND status='completed') AS "setupCompleted",
 EXISTS(SELECT 1 FROM alphatrace.review_events WHERE owner_id=m.user_id) AS "firstResponseSaved"
 FROM alphatrace.beta_members m ORDER BY (m.state='active' OR (m.state='invited' AND m.expires_at>now())) DESC,m.created_at DESC LIMIT 100) t),'[]'::jsonb)) INTO result;
 RETURN result;
 END $$;

-- An account export contains only the caller's membership, never invitation hashes.
CREATE FUNCTION alphatrace.beta_self() RETURNS jsonb LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog AS $$
 SELECT jsonb_build_object('email',email,'state',state,'createdAt',created_at,'activatedAt',activated_at,'updatedAt',updated_at)
 FROM alphatrace.beta_members WHERE user_id=NULLIF(current_setting('app.user_id',true),'')::uuid;
$$;
REVOKE ALL ON FUNCTION alphatrace.beta_self() FROM PUBLIC;
GRANT EXECUTE ON FUNCTION alphatrace.beta_self() TO alphatrace_app;
