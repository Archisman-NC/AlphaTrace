# Baseline audit — 2026-10-31

Scope: Step 1 of the Next.js transformation. This is an observed local baseline, not a production-readiness certification. Runtime checks used the installed environment; no dependencies were installed or pinned during this audit.

## Repository and environment

- Branch: `main`; starting HEAD: `54bd1fc`.
- No tracked modifications at the start of the audit.
- 190 pre-existing untracked files named `xaa` through `xhh`, totaling 1,958 bytes. Sample contents are module names (`os`, `sys`, `json`, `logging`). They appear to be fragments of an import listing; their origin is unverified. Left untouched.
- `.python-version` specifies 3.12; the available `python3` is 3.13.0. The quickstart still says Python 3.9+. These are inconsistent; a clean supported setup is not yet established.
- Available Node: v20.20.0; npm: 10.8.2. No Next.js project or JavaScript dependency manifest exists yet.
- All 16 declared runtime module names are discoverable in the installed environment. Discoverability does not establish import compatibility.
- Parsed all 65 Python files across `app/`, `scratch/`, `research/`, and root launchers without syntax errors. Parsing does not execute imports or verify behavior.
- Requirements are unpinned. No lockfile, automated CI configuration, or license was found in the inspected repository files.

## Runtime results

| Check | Observed result | Interpretation |
| --- | --- | --- |
| `python3 run_verification.py` | Exit 1; reported 9/11 passed in 74.55 seconds | Reported passes are not reliable; see false-positive examples below |
| Default Streamlit AppTest startup | No tabs; exception while importing pandas-ta/Numba | Cache could not be created in this restricted environment |
| AppTest with writable Numba/Matplotlib caches | No app exceptions; all four expected tabs rendered | Initial UI startup verified, not interactive or provider-backed workflows |
| Direct market-data verification | Empty data/prices; Yahoo hostname DNS errors; script exit 0 | External-data check blocked by this environment and falsely treated as a pass |
| Direct signals verification | Numba cache RuntimeError; script exit 0 | Another confirmed false positive |
| Synthetic backtest assertions with writable caches | Passed: positions `[1, -1, 0]`, strategy returns `[0.1, 0.1, 0]`, ending equity 121,000 from 100,000 | Narrow evidence for one-bar lag and compounding; not comprehensive backtest correctness |
| Synthetic indicators with writable caches | Passed: 100 input rows produced 67 non-NaN rows with signals in `{-1, 0, 1}` | Indicator path works on a local fixture; indicator values were not independently benchmarked |

The verification runner marked `verify_regime_detector.py` and `verify_watchdog.py` as failures. Direct reruns confirmed both exited 1 after Yahoo DNS failures left their market inputs empty. Many other scripts catch exceptions or print `FAILURE` without returning a nonzero status; the runner only checks exit codes and hides successful subprocess output. Its summary cannot establish analytical correctness.

### Reproduce startup with writable caches

```bash
NUMBA_CACHE_DIR=/private/tmp/alphatrace-numba-cache \
MPLCONFIGDIR=/private/tmp/alphatrace-matplotlib \
python3 - <<'PY'
from streamlit.testing.v1 import AppTest
app = AppTest.from_file('app/ui/dashboard.py', default_timeout=30).run()
assert not app.exception, [e.message for e in app.exception]
assert len(app.tabs) == 4
print([t.label for t in app.tabs])
PY
```

This uses Streamlit's headless test harness; it is not a browser-based visual review. No live AI responses, successful Yahoo downloads, trained PPO performance, or fresh Python 3.12 installation were verified.

## Reuse map

| Existing area | Migration decision |
| --- | --- |
| `app/quant/backtester.py`, `diagnostics.py`, `watchdog.py` | Potential optional analytics library after isolated tests; not the thesis comparison engine |
| `app/data/market_data.py` | Preserve for legacy price context; remove Streamlit cache coupling before API reuse |
| `app/data/portfolio_builder.py` | Existing predefined portfolios are demo inputs, not a user/workspace data model |
| `app/quant/signals.py`, `signal_generator.py`, `portfolio_signals.py`, `regime_detector.py` | Keep in legacy Research Lab; prices and technical indicators cannot assess document-based investment assumptions |
| `app/reasoning/` | Reference for context assembly only; rebuild around structured evidence, persistence, and provider boundaries |
| `app/ui/dashboard.py` | Preserve legacy entrypoint; build the new primary interface in Next.js |
| `app/rl/`, `research/` | Preserve experiments; exclude from new web/API runtime dependencies |
| `scratch/verify_*.py` | Useful scenarios, but convert selected logic into fixture-based assertions before relying on it |

## Architectural gaps

The repository currently lacks persistent accounts, workspace authorization, a database, thesis revisions, document upload/extraction, source-linked evidence, comparison jobs, and a web API. These are new product capabilities, not cosmetic changes to Streamlit.

Some reasoning imports initialize provider/telemetry clients; several utilities use permissive fallback values, such as converting invalid numeric inputs to zero. The new evidence domain must represent missing or invalid facts explicitly. It must not inherit legacy fallback semantics that could make absent evidence appear factual.

## Priority follow-up work

1. Isolate and lock the new frontend/backend environment during Step 2. Avoid pulling the legacy RL/technical-analysis dependencies into it.
2. Replace any reused verification logic with tests that fail on assertions and use local data fixtures. Separate external-data checks from deterministic tests.
3. Establish migration-backed storage and ownership boundaries before introducing user documents.
4. Build citation and comparison fixtures before claiming thesis-analysis accuracy.
5. Preserve existing files and entrypoints; no unexplained untracked files should be deleted as part of migration.

Step 1 establishes the baseline and scope. It does not claim that the legacy platform is fixed, that a clean installation is reproducible, or that the Next.js product exists yet.
