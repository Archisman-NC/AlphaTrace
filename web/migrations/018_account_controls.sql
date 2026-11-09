-- A minimal closure receipt survives private-data removal so failed Auth deletion
-- can be retried and an old session cannot recreate a workspace.
CREATE TABLE alphatrace.account_closures (
 owner_id uuid PRIMARY KEY, requested_at timestamptz NOT NULL DEFAULT now(),
 completed_at timestamptz, attempts integer NOT NULL DEFAULT 0,
 next_attempt_at timestamptz NOT NULL DEFAULT now(), last_attempt_at timestamptz
);
ALTER TABLE alphatrace.account_closures ENABLE ROW LEVEL SECURITY;
ALTER TABLE alphatrace.account_closures FORCE ROW LEVEL SECURITY;
CREATE POLICY own_closure ON alphatrace.account_closures FOR SELECT TO alphatrace_app
 USING(owner_id=NULLIF(current_setting('app.user_id',true),'')::uuid);
REVOKE ALL ON alphatrace.account_closures FROM PUBLIC,alphatrace_app;
GRANT SELECT ON alphatrace.account_closures TO alphatrace_app;

CREATE FUNCTION alphatrace.account_deletion_blocked() RETURNS boolean
 LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog AS $$
 SELECT EXISTS(SELECT 1 FROM alphatrace.research_operators WHERE user_id=NULLIF(current_setting('app.user_id',true),'')::uuid)
 OR EXISTS(SELECT 1 FROM alphatrace.document_jobs WHERE owner_id=NULLIF(current_setting('app.user_id',true),'')::uuid AND cleaned_at IS NULL);
$$;
CREATE FUNCTION alphatrace.request_account_closure() RETURNS void
 LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog AS $$
 DECLARE target uuid:=NULLIF(current_setting('app.user_id',true),'')::uuid;
 BEGIN
  IF target IS NULL THEN RAISE EXCEPTION 'Account identity required'; END IF;
  PERFORM pg_advisory_xact_lock(hashtext('account:'||target::text));
  IF alphatrace.account_deletion_blocked() THEN RAISE EXCEPTION 'Operator handover or document cleanup required'; END IF;
  INSERT INTO alphatrace.account_closures(owner_id) VALUES(target) ON CONFLICT DO NOTHING;
  -- Delivery payloads contain addresses and unsubscribe capabilities: remove them
  -- with the workspace; webhook receipts are removed before that cascade.
  DELETE FROM alphatrace.email_webhook_events WHERE provider_id IN (SELECT provider_id FROM alphatrace.email_deliveries WHERE owner_id=target);
  DELETE FROM alphatrace.workspaces WHERE owner_id=target;
  DELETE FROM alphatrace.document_jobs WHERE owner_id=target AND cleaned_at IS NOT NULL;
 END $$;
CREATE FUNCTION alphatrace.complete_account_closure() RETURNS void
 LANGUAGE sql SECURITY DEFINER SET search_path=pg_catalog AS $$
 UPDATE alphatrace.account_closures SET completed_at=COALESCE(completed_at,now())
 WHERE owner_id=NULLIF(current_setting('app.user_id',true),'')::uuid;
$$;
-- The worker can enumerate only pending closure IDs; no private content or email.
CREATE FUNCTION alphatrace.pending_account_closures() RETURNS TABLE(owner_id uuid)
 LANGUAGE sql SECURITY DEFINER SET search_path=pg_catalog AS $$
 SELECT owner_id FROM alphatrace.account_closures WHERE completed_at IS NULL AND next_attempt_at<=now() ORDER BY next_attempt_at LIMIT 20;
$$;
CREATE FUNCTION alphatrace.claim_account_closure() RETURNS boolean
 LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog AS $$
 BEGIN
  UPDATE alphatrace.account_closures SET attempts=attempts+1,last_attempt_at=now(),next_attempt_at=now()+interval '1 hour'
  WHERE owner_id=NULLIF(current_setting('app.user_id',true),'')::uuid AND completed_at IS NULL AND next_attempt_at<=now();
  RETURN FOUND;
 END $$;
-- Also guard callers from an older worker deployment that has not loaded the
-- application-level closure gate yet.
CREATE FUNCTION alphatrace.prevent_closed_workspace() RETURNS trigger
 LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog AS $$
 BEGIN
  PERFORM pg_advisory_xact_lock_shared(hashtext('account:'||NEW.owner_id::text));
  IF EXISTS(SELECT 1 FROM alphatrace.account_closures WHERE owner_id=NEW.owner_id) THEN RAISE EXCEPTION 'Account closed'; END IF;
  RETURN NEW;
 END $$;
CREATE TRIGGER prevent_closed_workspace BEFORE INSERT OR UPDATE ON alphatrace.workspaces FOR EACH ROW EXECUTE FUNCTION alphatrace.prevent_closed_workspace();
REVOKE ALL ON FUNCTION alphatrace.account_deletion_blocked(),alphatrace.request_account_closure(),alphatrace.complete_account_closure(),alphatrace.pending_account_closures(),alphatrace.claim_account_closure(),alphatrace.prevent_closed_workspace() FROM PUBLIC;
GRANT EXECUTE ON FUNCTION alphatrace.account_deletion_blocked(),alphatrace.request_account_closure(),alphatrace.complete_account_closure(),alphatrace.pending_account_closures(),alphatrace.claim_account_closure() TO alphatrace_app;
