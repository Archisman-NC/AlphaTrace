import type { inspectIndianStats } from './indianapi';

type Pair = readonly [number, number]; // June, September; never half-year columns.
type Reference = {
  sourceUrl: string; sha256: string; locator: string; unit: 'INR crore' | 'INR million';
  revenue: Pair; expenses: Pair; finance: Pair; depreciation: Pair; associateLossIncluded: Pair;
  totalProfit: Pair; ownersProfit: Pair; note: string;
};

/** Manually transcribed official statement observations, not provider metadata.
 * Restricted to these companies and quarters. No production ingestion permission.
 */
const references: Record<string, Reference> = {
  INFY: {
    sourceUrl: 'https://www.sec.gov/Archives/edgar/data/1067491/000106749126000042/exv99w03.htm',
    sha256: '32367a47093d415c03a00e8add41acdd6c92444910ad58d3c891ac54bc4492d0',
    locator: 'Opening consolidated audited Ind AS statement; June and September 2026 quarter columns', unit: 'INR crore',
    revenue: [48211, 49244], expenses: [38167, 39029], finance: [119, 115], depreciation: [1246, 1359], associateLossIncluded: [0, 0],
    totalProfit: [7775, 7838], ownersProfit: [7769, 7823],
    note: 'Operating Profit reconciles to a derived pre-depreciation figure, not reported operating income. Labour-code exceptional charges are absent in these two quarters; do not import the full-year charge.',
  },
  HCLTECH: {
    sourceUrl: 'https://www.hcltech.com/sites/default/files/documents/investor-reports/FinancialResults_4.pdf',
    sha256: 'f84f09361ca35e7d29135cf998e32a03e7a70b29986d9efc73ce6fcbfcdf4135',
    locator: 'Physical PDF pages 2–3, consolidated Ind AS; notes page 9', unit: 'INR crore',
    revenue: [34579, 35620], expenses: [28832, 29349], finance: [84, 194], depreciation: [1039, 1012], associateLossIncluded: [0, 38],
    totalProfit: [4626, 5132], ownersProfit: [4624, 5132],
    note: 'Operating Profit reconciles only after subtracting September associate loss of 38. Segment operating results are 5831 / 6465. Note 6 explains tax benefits affecting September profit; this is not adjusted recurring profit.',
  },
  WIPRO: {
    sourceUrl: 'https://www.sec.gov/Archives/edgar/data/1123799/000119312526426839/d118455dex991.htm',
    sha256: '23449d48c185d2f07de7f634a3db8f3d9afa280ad262e35fece394cbfe2166ad',
    locator: 'Audited consolidated financial results under Ind AS; three-month columns (not standalone or IFRS tables)', unit: 'INR million',
    revenue: [244786, 247527], expenses: [211226, 213121], finance: [4728, 5454], depreciation: [8044, 8767], associateLossIncluded: [0, 0],
    totalProfit: [33563, 31997], ownersProfit: [33520, 32006],
    note: 'Divide INR millions by 10; API values reconcile after rounding to whole crores. Derived Operating Profit adds back depreciation, amortisation AND impairment; it excludes the separate associate/JV result. It is not IT-services segment margin.',
  },
  HINDUNILVR: {
    sourceUrl: 'https://www.hul.co.in/files/hul-sq-26-results.pdf',
    sha256: '287b875db757be294ef3990448bd986c92dda34486576e41532ce41acfd1ee58',
    locator: 'Physical PDF pages 7–8 consolidated financial/segment results; notes page 11', unit: 'INR crore',
    revenue: [17341, 18226], expenses: [13822, 14574], finance: [75, 74], depreciation: [353, 363], associateLossIncluded: [0, 0],
    totalProfit: [2680, 2796], ownersProfit: [2673, 2790],
    note: 'Sales includes other operating revenue (157 / 155); turnover is 17184 / 18071. September Operating Profit also matches disclosed EBITDA. Net Profit includes exceptional items and NCI. Continuing and total profit coincide only because discontinued profit is nil in these two quarters; historical periods differ.',
  },
};

function comparison(actual: string | null | undefined, expected: number, allowWholeCroreRounding: boolean) {
  if (actual === null || actual === undefined) return 'missing';
  const value = Number(actual);
  if (!Number.isFinite(value)) return 'mismatch';
  if (value === expected) return 'exact';
  const rounded = Math.sign(expected) * Math.floor(Math.abs(expected) + 0.5);
  return allowWholeCroreRounding && Number.isInteger(value) && value === rounded ? 'rounded' : 'mismatch';
}

export function officialReferenceChecks(report: ReturnType<typeof inspectIndianStats>) {
  const ref = references[report.requestedSymbol];
  if (!ref || report.periods.join(',') !== '2026-06-30,2026-09-30') return null;
  const divisor = ref.unit === 'INR million' ? 10 : 1;
  const normalized = (values: Pair) => values.map(value => value / divisor);
  const operating = ref.revenue.map((revenue, i) =>
    revenue - ref.expenses[i] + ref.finance[i] + ref.depreciation[i] - ref.associateLossIncluded[i]) as [number, number];
  const series = [
    { providerLabel: 'Sales', officialDefinition: 'Consolidated revenue from operations', values: ref.revenue },
    { providerLabel: 'Operating Profit', officialDefinition: 'Derived: revenue - total expenses + finance costs + D&A/impairment - specified associate loss', values: operating },
    { providerLabel: 'Net Profit', officialDefinition: 'Consolidated profit for the period, including non-controlling interests', values: ref.totalProfit },
  ];
  const checks = series.map(series => {
    const row = report.rows.find(row => row.providerLabel === series.providerLabel);
    const expected = normalized(series.values);
    return { providerLabel: series.providerLabel, officialDefinition: series.officialDefinition,
      sourceValues: series.values, expectedInrCrore: expected,
      observed: [row?.previous ?? null, row?.current ?? null],
      results: [comparison(row?.previous, expected[0], divisor === 10), comparison(row?.current, expected[1], divisor === 10)],
      directMappingAllowed: false as const };
  });
  return { checkedAt: '2026-11-08', periods: report.periods, statementBasis: 'consolidated Ind AS',
    sourceUrl: ref.sourceUrl, sourceSha256: ref.sha256, locator: ref.locator, sourceUnit: ref.unit,
    comparisonUnit: 'INR crore', operandsInSourceUnit: { revenue: ref.revenue, expenses: ref.expenses,
      finance: ref.finance, depreciation: ref.depreciation, associateLossIncluded: ref.associateLossIncluded },
    ownersProfitInrCrore: normalized(ref.ownersProfit), note: ref.note, checks,
    limitation: 'Dated numerical reconciliation; provider semantics are inferred, not contractually confirmed. Identity/provenance, later quarters, correction behavior and usage rights remain unverified. No publication attestation.' };
}
