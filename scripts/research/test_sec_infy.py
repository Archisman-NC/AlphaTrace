"""Synthetic table/feed fixtures; no network or copied full filings in tests."""
import copy
import datetime as dt
import unittest
from unittest.mock import patch

import sec_infy as sec


def table(rows):
    return '<table>' + ''.join('<tr>' + ''.join('<td>' + c + '</td>' for c in r) + '</tr>' for r in rows) + '</table>'


FACTS = ('Consolidated Statement of Comprehensive Income for three months ended, '
         '(Extracted from IFRS Financial Statement) In US $ million, except per equity share data' + table([
             ['Particulars', 'Sep 30, 2026', 'Sep 30, 2025', 'Growth % YoY', 'Jun 30, 2026', 'Growth % QoQ'],
             ['Revenues', '1,200', '1,100', '9.1%', '1,000', '20%'],
             ['Operating profit', '240', '220', '9.1%', '210', '14.3%'],
             ['Net profit (after non-controlling interests)', '180', '160', '12.5%', '170', '5.9%'],
             ['Operating margin %', '20.0', '20.0', '0%', '21.0', '-1%'],
         ]))
RESULTS = ('International Financial Reporting Standards (IFRS) in US Dollars '
           'Dollars in millions except equity share data' + table([
               ['Condensed Consolidated Statement of Comprehensive Income for the', 'Note', 'Three months ended', 'Six months ended'],
               ['', '', 'September 30, 2026', 'September 30, 2025', 'September 30, 2026', 'September 30, 2025'],
               ['Revenues', '2.16', '1,200', '1,100', '2,200', '2,100'],
               ['Operating profit', '', '240', '220', '450', '430'],
               ['Profit attributable to:', '', '', '', '', ''],
               ['Owners of the Company', '', '180', '160', '350', '330'],
               ['Total comprehensive income attributable to:', '', '', '', '', ''],
               ['Owners of the Company', '', '181', '161', '351', '331'],
           ]))


def feed():
    today = dt.datetime.now(dt.timezone.utc).date()
    rows = [('0001067491-26-000044', '6-K', today.isoformat(), '2026-09-30', 'index.htm', 'DISCLOSURE OF RESULTS OF OPERATIONS'),
            ('0001067491-26-000042', '6-K', (today - dt.timedelta(days=5)).isoformat(), '2026-09-30', 'index.htm', 'DISCLOSURE OF RESULTS OF OPERATIONS'),
            ('0001067491-26-000045', '6-K', today.isoformat(), today.isoformat(), 'index.htm', 'INDEX')]
    keys = ['accessionNumber', 'form', 'filingDate', 'reportDate', 'primaryDocument', 'primaryDocDescription']
    return {'cik': '0001067491', 'name': 'Infosys Ltd', 'tickers': ['INFY'], 'filings': {'recent': {k: [r[i] for r in rows] for i, k in enumerate(keys)}}}


class SecTests(unittest.TestCase):
    def test_extracts_quarterly_values_by_headers_and_crosschecks_statement(self):
        previous, metrics = sec.extract(FACTS, RESULTS, '2026-09-30')
        self.assertEqual(previous, '2026-06-30')
        self.assertEqual(metrics['revenue'], ('1000', '1200'))
        self.assertEqual(metrics['net_profit'], ('170', '180'))
        self.assertEqual(metrics['operating_margin'], ('21.0', '20.0'))

    def test_ambiguous_currency_period_missing_values_and_disagreement_fail_closed(self):
        for facts, results in [(FACTS + FACTS, RESULTS), (FACTS.replace('US $ million', 'INR crore'), RESULTS),
                               (FACTS.replace('Jun 30, 2026', 'Mar 31, 2026'), RESULTS),
                               (FACTS.replace('three months', 'six months'), RESULTS),
                               (FACTS.replace('1,200', '–'), RESULTS),
                               (FACTS, RESULTS.replace('1,200', '1,201')),
                               (FACTS, RESULTS.replace('Three months ended', 'Six months ended')),
                               (FACTS, RESULTS.replace('US Dollars', 'Indian Rupees'))]:
            with self.subTest(facts=facts[:50], results=results[:50]), self.assertRaises(ValueError):
                sec.extract(facts, results, '2026-09-30')

    def test_feed_identity_newest_results_and_bounds(self):
        data = feed()
        d = sec.discover(data, sec.now(), 'a' * 64)
        self.assertEqual(d['accession'], '0001067491-26-000044')
        self.assertEqual(d['candidateCount'], 2)
        for change in ('issuer', 'path', 'arrays', 'empty'):
            bad = copy.deepcopy(data)
            if change == 'issuer': bad['cik'] = '123'
            if change == 'path': bad['filings']['recent']['primaryDocument'][0] = '../../evil.htm'
            if change == 'arrays': bad['filings']['recent']['form'].pop()
            if change == 'empty': bad['filings']['recent']['primaryDocDescription'] = ['INDEX'] * 3
            with self.assertRaises(ValueError): sec.discover(bad, sec.now(), 'a' * 64)

    def test_exhibits_must_be_declared_and_linked_in_same_filing(self):
        filing = {'periodEnd': '2026-09-30', 'filedOn': '2026-10-28', 'url': sec.ARCHIVE + '000106749126000044/index.htm'}
        cover = 'Infosys Limited September 30, 2026' + table([
            ['99.4', 'Fact Sheet'], ['99.7', 'Consolidated financial statements IFRS in US Dollars']])
        links = '<a href="exv99w04.htm">Facts</a><a href="exv99w07.htm">Statements</a>'
        self.assertTrue(sec.exhibit_links(cover + links, filing)['facts'].endswith('/exv99w04.htm'))
        with self.assertRaises(ValueError): sec.exhibit_links(cover, filing)
        with self.assertRaises(ValueError): sec.exhibit_links((cover + links).replace('US Dollars', 'Indian Rupees'), filing)

    def test_latest_unextractable_filing_preserves_discovery_without_fallback(self):
        import json
        calls = []
        def fetch(url, media):
            calls.append(url)
            return (json.dumps(feed()).encode() if url == sec.FEED else b'New release without full exhibits', sec.now(), 'a' * 64)
        with patch.object(sec, 'fetch', fetch), self.assertRaises(sec.DiscoveryError) as caught:
            sec.build()
        self.assertEqual(caught.exception.discovery['accession'], '0001067491-26-000044')
        self.assertEqual(len(calls), 2)

    def test_sec_allowlist_is_company_and_path_specific(self):
        good = sec.ARCHIVE + '000106749126000044/exv99w04.htm'
        self.assertEqual(sec.source_url(good, 'INFY'), good)
        for url, symbol in [(good, 'TCS'), (good.replace('/1067491/', '/1234/'), 'INFY'),
                            (good + '?redirect=x', 'INFY'), ('https://www.sec.gov/anything', 'INFY')]:
            with self.assertRaises(ValueError): sec.source_url(url, symbol)
