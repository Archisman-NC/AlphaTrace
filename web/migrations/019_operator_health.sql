-- Aggregate operational status only. Never expose notes, recipients, account IDs,
-- document inputs, raw provider responses or credentials to this endpoint.
CREATE FUNCTION alphatrace.operator_health_snapshot() RETURNS jsonb
 LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog AS $$
 DECLARE result jsonb;
 BEGIN
  IF NOT EXISTS(SELECT 1 FROM alphatrace.research_operators WHERE user_id=NULLIF(current_setting('app.user_id',true),'')::uuid) THEN RAISE EXCEPTION 'Operator access required' USING ERRCODE='42501'; END IF;
  SELECT jsonb_build_object(
   'checkedAt',now(),
   'official',COALESCE((SELECT jsonb_agg(row_to_json(t)) FROM (
    SELECT symbol,enabled,next_due_at,worker_seen_at,worker_mode,lease_until,
     (SELECT state FROM alphatrace.monitor_runs r WHERE r.symbol=t.symbol ORDER BY started_at DESC,id DESC LIMIT 1) AS latest_state
    FROM alphatrace.monitor_targets t ORDER BY symbol) t),'[]'::jsonb),
   'provider',COALESCE((SELECT jsonb_agg(row_to_json(t)) FROM (
    SELECT symbol,enabled,next_due_at,worker_seen_at,worker_mode,
     (SELECT lease_until FROM alphatrace.provider_runs r WHERE r.symbol=t.symbol AND state='running' LIMIT 1) AS lease_until,
     (SELECT state FROM alphatrace.provider_runs r WHERE r.symbol=t.symbol ORDER BY started_at DESC,id DESC LIMIT 1) AS latest_state
    FROM alphatrace.provider_targets t ORDER BY symbol) t),'[]'::jsonb),
   'quotaUsed',(SELECT count(*) FROM alphatrace.provider_runs WHERE started_at>now()-interval '24 hours'),
   'documents',jsonb_build_object(
    'stalled',(SELECT count(*) FROM alphatrace.document_jobs WHERE (state='running' AND lease_until<now()) OR (state='queued' AND created_at<now()-interval '30 minutes')),
    'failed',(SELECT count(*) FROM alphatrace.document_jobs WHERE state IN ('failed','expired') AND finished_at>now()-interval '24 hours'),
    'cleanupOverdue',(SELECT count(*) FROM alphatrace.document_jobs WHERE cleaned_at IS NULL AND cleanup_after<now()-interval '150 minutes')),
   'closures',jsonb_build_object(
    'pending',(SELECT count(*) FROM alphatrace.account_closures WHERE completed_at IS NULL),
    'overdue',(SELECT count(*) FROM alphatrace.account_closures WHERE completed_at IS NULL AND requested_at<now()-interval '150 minutes')),
   'emails',jsonb_build_object(
    'failed',(SELECT count(*) FROM alphatrace.email_deliveries WHERE state IN ('failed','uncertain') AND created_at>now()-interval '24 hours'),
    'overdue',(SELECT count(*) FROM alphatrace.email_deliveries WHERE (state IN ('pending','retry') AND next_attempt_at<now()-interval '150 minutes') OR (state='sending' AND lease_until<now()))),
   'reviewQueue',jsonb_build_object(
    'pending',(SELECT count(*) FROM alphatrace.provider_reviews WHERE status='pending'),
    'olderThanDay',(SELECT count(*) FROM alphatrace.provider_reviews WHERE status='pending' AND detected_at<now()-interval '24 hours')),
   'coverage',(SELECT jsonb_build_object('published',count(*) FILTER(WHERE withdrawn_at IS NULL),'withdrawn',count(*) FILTER(WHERE withdrawn_at IS NOT NULL),
    'needsSourceCheck',count(*) FILTER(WHERE withdrawn_at IS NULL AND (checked_at IS NULL OR checked_at<now()-interval '7 days' OR check_status<>'success')))
    FROM (SELECT DISTINCT ON(u.symbol) u.symbol,u.withdrawn_at,
     (SELECT checked_at FROM alphatrace.research_checks c WHERE c.symbol=u.symbol ORDER BY checked_at DESC LIMIT 1) AS checked_at,
     (SELECT status FROM alphatrace.research_checks c WHERE c.symbol=u.symbol ORDER BY checked_at DESC LIMIT 1) AS check_status
     FROM alphatrace.research_updates u ORDER BY u.symbol,u.version DESC) t)
  ) INTO result;
  RETURN result;
 END $$;
REVOKE ALL ON FUNCTION alphatrace.operator_health_snapshot() FROM PUBLIC;
GRANT EXECUTE ON FUNCTION alphatrace.operator_health_snapshot() TO alphatrace_app;
