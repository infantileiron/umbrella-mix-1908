"""Matched-process grouping for the common-support comparison.

Only process cells (ribs, fabric, mechanism) observed on *both* branches may
enter a comparable group, so a process the other branch never handled can
never leak into the comparable contribution.

Sparse cells are merged into rectangular bands (ribs x fabric) while the
mechanism flag stays a within-group covariate.  The search prefers the
partition that leaves the least actual contribution unestimated, then the
finest grouping, then the narrowest bands.  A group that still cannot carry
a full four-term fit is reported with its actual contribution and the missing
estimates rather than being silently dropped.

The resulting plan is frozen inside the run snapshot at submission time so
that retries of an existing batch and later ticket edits cannot change its
grouping.
"""

import numpy as np

from src.constants.domain import TERMS

GROUPING_VERSION = 'common-cells-v1'
GROUP_MIN = 5


def cell_key(ticket):
    return (int(ticket['ribs']), int(ticket['fabric']), int(ticket['mechanism']))


def _rows_by_cell(tickets):
    cells = {}
    for ticket in tickets:
        cells.setdefault(cell_key(ticket), {}).setdefault(ticket['branch'], []).append(ticket)
    return cells


def _mean_minutes(rows):
    return sum(row['minutes'] for row in rows) / len(rows)


def _fittable(rows):
    if len(rows) < GROUP_MIN:
        return False
    design = np.array([[1.0, row['ribs'], row['fabric'], row['mechanism']] for row in rows], dtype=float)
    return int(np.linalg.matrix_rank(design)) == len(TERMS)


def _bins(matched, rmin, fmin, rib_width, fabric_width):
    bins = {}
    for cell in matched:
        key = ((cell[0] - rmin) // rib_width, (cell[1] - fmin) // fabric_width)
        bins.setdefault(key, []).append(cell)
    return bins


def _evaluate(bins, cells, focus, control, total_focus, total_control):
    """Return per-bin rows and the absolute contribution left unestimated."""
    evaluated = []
    unestimated = 0.0
    for key in sorted(bins):
        member_cells = sorted(bins[key])
        rows_focus = [row for cell in member_cells for row in cells[cell].get(focus, [])]
        rows_control = [row for cell in member_cells for row in cells[cell].get(control, [])]
        if not rows_focus or not rows_control:
            continue
        actual = (len(rows_focus) / total_focus * _mean_minutes(rows_focus)
                  - len(rows_control) / total_control * _mean_minutes(rows_control))
        fittable = _fittable(rows_focus) and _fittable(rows_control)
        if not fittable:
            unestimated += abs(actual)
        evaluated.append(dict(key=key, cells=member_cells, focus=rows_focus, control=rows_control,
                              actual=actual, fittable=fittable))
    return evaluated, unestimated


def _search(matched, cells, focus, control, total_focus, total_control):
    rmin, rmax = min(cell[0] for cell in matched), max(cell[0] for cell in matched)
    fmin, fmax = min(cell[1] for cell in matched), max(cell[1] for cell in matched)
    span_ribs, span_fabric = rmax - rmin + 1, fmax - fmin + 1
    best = None
    for rib_width in range(1, span_ribs + 1):
        for fabric_width in range(1, span_fabric + 1):
            bins = _bins(matched, rmin, fmin, rib_width, fabric_width)
            evaluated, unestimated = _evaluate(
                bins, cells, focus, control, total_focus, total_control)
            if any(len(row['focus']) < GROUP_MIN or len(row['control']) < GROUP_MIN
                   for row in evaluated):
                continue
            score = (round(unestimated, 9), -len(evaluated), max(rib_width, fabric_width),
                     rib_width + fabric_width, rib_width, fabric_width)
            if best is None or score < best['score']:
                best = dict(score=score, rib_width=rib_width, fabric_width=fabric_width,
                            rows=evaluated, fallback=False)
    if best is not None:
        return best, (rmin, fmin, span_ribs, span_fabric)
    bins = _bins(matched, rmin, fmin, span_ribs, span_fabric)
    evaluated, _ = _evaluate(bins, cells, focus, control, total_focus, total_control)
    return dict(rib_width=span_ribs, fabric_width=span_fabric, rows=evaluated,
                fallback=True), (rmin, fmin, span_ribs, span_fabric)


def _group_payload(index, row, rib_width, fabric_width):
    member_cells = row['cells']
    return dict(
        id=f'G{index}',
        cells=[[cell[0], cell[1], cell[2]] for cell in member_cells],
        basis=dict(rib_width=rib_width, fabric_width=fabric_width,
                   rib_band=row['key'][0], fabric_band=row['key'][1], mechanism_mode='merged'),
        range=dict(ribs=[min(cell[0] for cell in member_cells), max(cell[0] for cell in member_cells)],
                   fabric=[min(cell[1] for cell in member_cells), max(cell[1] for cell in member_cells)],
                   mechanism=sorted({cell[2] for cell in member_cells})),
        counts=dict(focus=len(row['focus']), control=len(row['control'])),
        tickets=dict(focus=sorted(ticket['code'] for ticket in row['focus']),
                     control=sorted(ticket['code'] for ticket in row['control'])),
    )


def plan(tickets, comparison):
    """Build the frozen matched-process grouping plan for a run snapshot."""
    focus, control = comparison['focus_branch'], comparison['control_branch']
    cells = _rows_by_cell(tickets)
    matched = sorted(cell for cell, sides in cells.items()
                     if sides.get(focus) and sides.get(control))
    matched_set = set(matched)
    exclusive = {focus: [], control: []}
    for cell in sorted(cells):
        if cell in matched_set:
            continue
        for branch in (focus, control):
            if cells[cell].get(branch):
                exclusive[branch].append([cell[0], cell[1], cell[2]])
    total_focus = sum(1 for ticket in tickets if ticket['branch'] == focus)
    total_control = sum(1 for ticket in tickets if ticket['branch'] == control)
    base = dict(
        version=GROUPING_VERSION,
        min_per_side=GROUP_MIN,
        focus_branch=focus,
        control_branch=control,
        matched_cells=[[cell[0], cell[1], cell[2]] for cell in matched],
        exclusive_cells=exclusive,
        partition=None,
        groups=[],
    )
    if not matched:
        return base
    best, origin = _search(matched, cells, focus, control, total_focus, total_control)
    groups = [_group_payload(index, row, best['rib_width'], best['fabric_width'])
              for index, row in enumerate(best['rows'], start=1)]
    base['partition'] = dict(origin=list(origin[:2]), rib_width=best['rib_width'],
                             fabric_width=best['fabric_width'], fallback=best['fallback'])
    base['groups'] = groups
    return base
