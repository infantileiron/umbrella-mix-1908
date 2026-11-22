"""Invariants for the matched-process (common-support) comparison.

Run from backend/:  python -m unittest discover -s tests -v
"""

import itertools
import json
import random
import unittest

import numpy as np

from scripts.seed_cases import demo_tickets
from src.services.decomposition import AnalysisError, analyze
from src.services.exports import csv_result
from src.services.grouping import plan
from src.utils.serialization import canonical

COMPARISON = {'focus_branch': 'south', 'control_branch': 'north'}


def plan_row(**overrides):
    base = dict(id=1, plan_key='RAIN-W46', version=1, label='测试', focus_branch='south',
                control_branch='north', coefficient_ref='control', date_from='2026-11-09',
                date_to='2026-11-15', min_samples=8, created_by='analyst')
    base.update(overrides)
    return base


def ticket(code, branch, ribs, fabric, mechanism, minutes=20):
    return dict(code=code, branch=branch, worked_on='2026-11-10', ribs=ribs, fabric=fabric,
                mechanism=mechanism, minutes=minutes, note='', revision=1)


def demo_rows():
    return [dict(code=code, branch=branch, worked_on=day.isoformat(), ribs=ribs, fabric=fabric,
                 mechanism=mechanism, minutes=minutes, note=note, revision=1)
            for code, branch, day, ribs, fabric, mechanism, minutes, note in demo_tickets()]


def run(rows, plan_overrides=None):
    snapshot = json.loads(canonical(dict(method='ob-twofold-ols-v1', plan=plan_row(**(plan_overrides or {})),
                                         tickets=rows, grouping=plan(rows, COMPARISON))))
    return analyze(snapshot), snapshot


class MatchedDecompositionTest(unittest.TestCase):
    def test_demo_reconciles_to_full_gap(self):
        result, _ = run(demo_rows())
        matched = result['matched']
        self.assertAlmostEqual(result['gap'], 16.5)
        self.assertAlmostEqual(result['composition'], 5.0)
        self.assertAlmostEqual(result['coefficient'], 11.5)
        self.assertAlmostEqual(matched['totals']['actual'], 5.90625)
        self.assertAlmostEqual(matched['exclusive']['contribution'], 10.59375)
        self.assertAlmostEqual(matched['reconciliation']['remainder'], 0.0, places=10)
        self.assertAlmostEqual(matched['totals']['actual'] + matched['exclusive']['contribution'], 16.5)

    def test_matched_cells_exclude_single_branch_processes(self):
        result, _ = run(demo_rows())
        matched = result['matched']
        matched_cells = {tuple(cell) for cell in matched['matched_cells']}
        exclusive = set()
        for side in ('focus', 'control'):
            exclusive |= {tuple(cell) for cell in matched['exclusive'][side]['cells']}
        self.assertFalse(matched_cells & exclusive)
        self.assertEqual(len(matched_cells), 18)
        for group in matched['groups']:
            for side in ('focus', 'control'):
                for item in group['tickets'][side]:
                    self.assertIn([item['ribs'], item['fabric'], item['mechanism']], group['cells'])

    def test_weights_fold_back_original_counts(self):
        result, snapshot = run(demo_rows())
        totals = {side: sum(1 for row in snapshot['tickets'] if row['branch'] == branch)
                  for side, branch in (('focus', 'south'), ('control', 'north'))}
        for group in result['matched']['groups']:
            for side in ('focus', 'control'):
                self.assertAlmostEqual(group['weights'][side],
                                       group['counts'][side] / totals[side], places=12)
        # Weighted group means sum to the matched share of the full gap, not the
        # single-branch exclusive remainder.
        actual = sum(group['weights']['focus'] * group['means']['focus']
                     - group['weights']['control'] * group['means']['control']
                     for group in result['matched']['groups'])
        self.assertAlmostEqual(actual, result['matched']['totals']['actual'], places=10)
        covered = sum(group['counts']['focus'] for group in result['matched']['groups'])
        self.assertLess(covered, totals['focus'])

    def test_group_contributions_conserve(self):
        result, _ = run(demo_rows())
        for group in result['matched']['groups']:
            if group['estimated']:
                self.assertAlmostEqual(group['actual'], group['composition'] + group['coefficient'], places=10)
            else:
                self.assertEqual(group['missing'], ['composition', 'coefficient'])

    def test_unfittable_group_keeps_actual_and_names_missing(self):
        rows = [ticket('S0', 'south', 2, 2, 0, 30), ticket('S1', 'south', 2, 2, 0, 32),
                ticket('N0', 'north', 2, 2, 0, 20), ticket('N1', 'north', 2, 2, 0, 22)]
        for index, (ribs, fabric, mechanism) in enumerate(itertools.product((1, 3, 5, 7), (4, 6), (0, 1))):
            rows.append(ticket(f'SX{index}', 'south', 10 + ribs, 10 + fabric, mechanism, 40 + 3 * ribs + mechanism))
            rows.append(ticket(f'NX{index}', 'north', 30 + ribs, 30 + fabric, mechanism, 30 + 3 * ribs + mechanism))
        result, _ = run(rows)
        matched = result['matched']
        self.assertEqual(matched['totals']['unestimated_groups'], 1)
        self.assertEqual(matched['totals']['missing'], ['composition', 'coefficient'])
        self.assertAlmostEqual(matched['totals']['unestimated_actual'], matched['totals']['actual'])
        self.assertAlmostEqual(matched['totals']['composition'], 0.0)
        self.assertTrue(matched['groups'][0]['reason'])
        self.assertAlmostEqual(matched['reconciliation']['remainder'], 0.0, places=9)

    def test_no_shared_process_reports_zero_comparable(self):
        rows = []
        for index, (ribs, fabric, mechanism) in enumerate(itertools.product((1, 2, 3), (1, 2), (0, 1))):
            rows.append(ticket(f'S{index}', 'south', ribs, fabric, mechanism, 10 + 4 * ribs + 3 * fabric + mechanism))
            rows.append(ticket(f'N{index}', 'north', ribs + 5, fabric + 5, mechanism,
                               8 + 3 * ribs + 2 * fabric + mechanism))
        result, _ = run(rows)
        matched = result['matched']
        self.assertEqual(matched['matched_cells'], [])
        self.assertEqual(matched['groups'], [])
        self.assertEqual(matched['totals']['actual'], 0.0)
        self.assertAlmostEqual(matched['exclusive']['contribution'], result['gap'], places=10)
        self.assertAlmostEqual(matched['reconciliation']['remainder'], 0.0, places=10)

    def test_grouping_is_frozen_in_snapshot(self):
        rows = demo_rows()
        snapshot = json.loads(canonical(dict(method='ob-twofold-ols-v1', plan=plan_row(),
                                             tickets=rows, grouping=plan(rows, COMPARISON))))
        first = analyze(json.loads(canonical(snapshot)))
        retry = analyze(json.loads(canonical(snapshot)))
        self.assertEqual(canonical(first), canonical(retry))
        # A sealed snapshot carries its grouping plan; recomputing against a
        # *different* ticket set (e.g. after later ledger edits) would yield a
        # different plan, which the stored plan protects the old batch from.
        edited_rows = [dict(row) for row in rows]
        for row in edited_rows:
            if row['code'].startswith('NQ'):
                row['ribs'] = 9
        self.assertNotEqual(plan(rows, COMPARISON)['matched_cells'],
                            plan(edited_rows, COMPARISON)['matched_cells'])
        # analyze() trusts the stored plan and refuses snapshots whose tickets
        # no longer sit in the sealed cells rather than silently re-binning.
        tampered = json.loads(canonical(dict(snapshot, tickets=edited_rows)))
        with self.assertRaises(AnalysisError):
            analyze(tampered)

    def test_grouping_is_order_independent(self):
        rows = demo_rows()
        shuffled = rows[:]
        random.Random(4).shuffle(shuffled)
        self.assertEqual(canonical(plan(rows, COMPARISON)), canonical(plan(shuffled, COMPARISON)))

    def test_alternate_basis_moves_composition(self):
        control_basis, _ = run(demo_rows(), {'coefficient_ref': 'control'})
        focus_basis, _ = run(demo_rows(), {'coefficient_ref': 'focus'})
        self.assertAlmostEqual(control_basis['gap'], focus_basis['gap'])
        self.assertAlmostEqual(control_basis['composition'], 5.0)
        self.assertAlmostEqual(focus_basis['composition'], 7.0)
        self.assertAlmostEqual(focus_basis['matched']['reconciliation']['remainder'], 0.0, places=10)

    def test_csv_contains_reconciliation_rows(self):
        result, snapshot = run(demo_rows())
        text = csv_result(dict(result=result, plan=snapshot['plan'], snapshot_sha='a' * 64, draft=None))
        for key in ('matched', 'matched-group', 'exclusive', 'reconcile'):
            self.assertIn(f'\n{key},', text)

    def test_randomized_invariants(self):
        generator = random.Random(9)
        checked = 0
        while checked < 120:
            rows = []
            for branch, prefix in (('south', 'S'), ('north', 'N')):
                for index in range(generator.randint(8, 28)):
                    rows.append(ticket(f'{prefix}{index}', branch, generator.randint(1, 6),
                                       generator.randint(1, 6), generator.randint(0, 1),
                                       generator.randint(8, 90)))
            fits = True
            for branch in ('south', 'north'):
                sample = [row for row in rows if row['branch'] == branch]
                design = np.array([[1, row['ribs'], row['fabric'], row['mechanism']] for row in sample], float)
                if len(sample) < 8 or np.linalg.matrix_rank(design) < 4:
                    fits = False
            if not fits:
                continue
            checked += 1
            result, _ = run(rows)
            matched = result['matched']
            self.assertAlmostEqual(matched['reconciliation']['remainder'], 0.0, places=9)
            self.assertAlmostEqual(
                matched['totals']['actual'],
                matched['totals']['composition'] + matched['totals']['coefficient']
                + matched['totals']['unestimated_actual'], places=9)
            for group in matched['groups']:
                if group['estimated']:
                    self.assertAlmostEqual(group['actual'], group['composition'] + group['coefficient'], places=9)

    def test_sample_below_plan_threshold_still_fails(self):
        rows = [ticket(f'S{index}', 'south', 1 + index % 3, 1 + index // 3, index % 2, 20 + index)
                for index in range(7)]
        with self.assertRaises(AnalysisError):
            run(rows)


if __name__ == '__main__':
    unittest.main()
