ALTER TABLE alphatrace.workspaces ADD COLUMN portfolio_version integer NOT NULL DEFAULT 0;
CREATE TABLE alphatrace.portfolio_entries (
  owner_id uuid NOT NULL REFERENCES alphatrace.workspaces(owner_id) ON DELETE CASCADE,
  symbol text NOT NULL CHECK (symbol IN (
    'TCS','INFY','HCLTECH','WIPRO','HINDUNILVR','ITC','ASIANPAINT','TITAN','MARUTI','LT',
    'RELIANCE','BHARTIARTL','SUNPHARMA','CIPLA','NTPC','POWERGRID','ULTRACEMCO','BAJAJ-AUTO',
    'HEROMOTOCO','NESTLEIND','HDFCBANK','ICICIBANK','SBIN','AXISBANK','KOTAKBANK'
  )),
  quantity numeric(18,6) CHECK (quantity > 0 AND quantity <= 999999999999.999999),
  followed boolean NOT NULL DEFAULT false,
  updated_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (owner_id, symbol),
  CHECK (quantity IS NOT NULL OR followed)
);
ALTER TABLE alphatrace.portfolio_entries ENABLE ROW LEVEL SECURITY;
ALTER TABLE alphatrace.portfolio_entries FORCE ROW LEVEL SECURITY;
CREATE POLICY own_portfolio ON alphatrace.portfolio_entries
  FOR ALL TO alphatrace_app
  USING (owner_id = NULLIF(current_setting('app.user_id', true), '')::uuid)
  WITH CHECK (owner_id = NULLIF(current_setting('app.user_id', true), '')::uuid);
GRANT SELECT, INSERT, UPDATE, DELETE ON alphatrace.portfolio_entries TO alphatrace_app;
REVOKE ALL ON alphatrace.portfolio_entries FROM PUBLIC;
COMMENT ON TABLE alphatrace.portfolio_entries IS 'Private absolute holdings and watchlist membership. No trades, live prices or sample fixtures.';
