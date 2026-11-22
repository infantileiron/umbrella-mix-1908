import math

import numpy as np

from src.constants.domain import BRANCH_LABEL, FEATURES, METHOD, TERMS
from src.services.grouping import plan as build_grouping


class AnalysisError(ValueError):
    pass


def _fit_rows(samples, branch, minimum, scope='组'):
    count = len(samples)
    if count < minimum:
        advice = '请补录工票或调整分组。' if scope == '组' else '请补录工票或另存日期版本。'
        raise AnalysisError(f'{BRANCH_LABEL[branch]}只有 {count} 张工票，本{scope}要求至少 {minimum} 张。{advice}')
    values = np.array([[row[key] for key in FEATURES] for row in samples], dtype=float)
    design = np.column_stack((np.ones(count), values))
    response = np.array([row['minutes'] for row in samples], dtype=float)
    beta, _, rank, singular_values = np.linalg.lstsq(design, response, rcond=None)
    if rank != len(TERMS):
        raise AnalysisError(f'{BRANCH_LABEL[branch]}特征不满秩（{rank}/{len(TERMS)}）。请补录工序组合更丰富的工票。')
    residuals = response - design @ beta
    sse = float(residuals @ residuals)
    centered = response - response.mean()
    sst = float(centered @ centered)
    return design, beta, dict(
        branch=branch, n=count, mean_minutes=float(response.mean()),
        rank=int(rank), condition=float(singular_values[0] / singular_values[-1]),
        r_squared=1 - sse / sst if sst else None, rmse=float(np.sqrt(sse / count)),
        terms={key: {'mean': float(design[:, index].mean()), 'coefficient': float(beta[index])}
               for index, key in enumerate(TERMS)},
    )


def _fit(rows, branch, minimum):
    return _fit_rows([row for row in rows if row['branch'] == branch], branch, minimum, scope='网点')


def _summary(rows, branch, failure):
    values = np.array([row['minutes'] for row in rows], dtype=float)
    design = np.column_stack((np.ones(len(rows)), np.array([[row[key] for key in FEATURES] for row in rows], dtype=float)))
    return dict(branch=branch, n=len(rows), mean_minutes=float(values.mean()), rank=0,
                condition=None, r_squared=None, rmse=None,
                terms={key: {'mean': float(design[:, index].mean()), 'coefficient': None}
                       for index, key in enumerate(TERMS)}, error=str(failure))


def _design(rows):
    return np.column_stack((np.ones(len(rows)), np.array([[row[key] for key in FEATURES] for row in rows], dtype=float)))


def _support(evaluated, basis):
    low, high = basis.min(axis=0), basis.max(axis=0)
    outside = (evaluated < low) | (evaluated > high)
    terms = {key: {'outside': int(outside[:, index].sum()), 'min': int(low[index]), 'max': int(high[index])}
             for index, key in enumerate(FEATURES, start=1)}
    count = int(outside.any(axis=1).sum())
    return dict(outside=count, total=len(evaluated), share=count / len(evaluated), terms=terms)


def _grouping(snapshot, plan):
    """Return the grouping frozen at submission.

    The grouping travels inside the sealed snapshot, so retries of an existing
    batch keep their original partition and later ticket edits cannot leak in.
    Older snapshots without one fall back to a recomputed plan.
    """
    stored = snapshot.get('grouping')
    if stored is None:
        return build_grouping(snapshot['tickets'], dict(focus_branch=plan['focus_branch'],
                                                        control_branch=plan['control_branch']))
    if stored.get('focus_branch') != plan['focus_branch'] or stored.get('control_branch') != plan['control_branch']:
        raise AnalysisError('分组快照与方案网点不一致，请重新提交运行。')
    return stored


def _group_tickets(group, by_code, branch, side):
    rows = []
    for code in group['tickets'][side]:
        row = by_code.get(code)
        if row is None or row['branch'] != branch:
            raise AnalysisError('分组快照引用了不存在的工票，请重新提交运行。')
        cell = [int(row['ribs']), int(row['fabric']), int(row['mechanism'])]
        if cell not in group['cells']:
            raise AnalysisError('分组快照与工票工序不一致，请重新提交运行。')
        rows.append(row)
    return rows


def _mean(rows):
    return math.fsum(row['minutes'] for row in rows) / len(rows)


def _rule_description(grouping, plan):
    partition = grouping.get('partition')
    if not partition:
        return '本次窗口内没有两边都做过的工序组合，可比工序贡献为 0。'
    tail = '样本过散，已退回全窗口合并组。' if partition['fallback'] else ''
    return (f'仅纳入两边都做过的工序组合；换骨按 {partition["rib_width"]} 根、'
            f'布损按 {partition["fabric_width"]} 处分带，开合机构在组内区分；'
            f'每侧至少 {grouping["min_per_side"]} 张组成一组。' + tail)


def _matched(plan, rows, grouping):
    focus, control = plan['focus_branch'], plan['control_branch']
    total_focus = sum(1 for row in rows if row['branch'] == focus)
    total_control = sum(1 for row in rows if row['branch'] == control)
    if not total_focus or not total_control:
        raise AnalysisError('比较网点或对照网点在本窗口没有工票，无法核对可比工序。')
    by_code = {row['code']: row for row in rows}
    use_control = plan['coefficient_ref'] == 'control'
    minimum = grouping['min_per_side']
    groups = []
    term_composition = {key: [] for key in TERMS}
    term_coefficient = {key: [] for key in TERMS}
    actuals, compositions, coefficients, unestimated = [], [], [], []
    for group in grouping['groups']:
        rows_focus = _group_tickets(group, by_code, focus, 'focus')
        rows_control = _group_tickets(group, by_code, control, 'control')
        weight_focus = len(rows_focus) / total_focus
        weight_control = len(rows_control) / total_control
        mean_focus, mean_control = _mean(rows_focus), _mean(rows_control)
        actual = weight_focus * mean_focus - weight_control * mean_control
        actuals.append(actual)
        item = dict(
            **{key: group[key] for key in ('id', 'cells', 'basis', 'range')},
            counts=dict(focus=len(rows_focus), control=len(rows_control)),
            weights=dict(focus=weight_focus, control=weight_control),
            means=dict(focus=mean_focus, control=mean_control),
            actual=actual,
            tickets=dict(
                focus=sorted((dict(code=row['code'], worked_on=row['worked_on'], ribs=row['ribs'],
                                   fabric=row['fabric'], mechanism=row['mechanism'], minutes=row['minutes'])
                              for row in rows_focus), key=lambda row: row['code']),
                control=sorted((dict(code=row['code'], worked_on=row['worked_on'], ribs=row['ribs'],
                                     fabric=row['fabric'], mechanism=row['mechanism'], minutes=row['minutes'])
                                for row in rows_control), key=lambda row: row['code']),
            ),
        )
        try:
            design_focus, beta_focus, fit_focus = _fit_rows(rows_focus, focus, minimum)
            design_control, beta_control, fit_control = _fit_rows(rows_control, control, minimum)
        except AnalysisError as exc:
            unestimated.append(actual)
            item.update(composition=None, coefficient=None, terms=[], estimated=False,
                        missing=['composition', 'coefficient'], reason=str(exc),
                        fit=dict(focus=None, control=None))
        else:
            mean_x_focus, mean_x_control = design_focus.mean(axis=0), design_control.mean(axis=0)
            basis = beta_control if use_control else beta_focus
            composition_terms = (weight_focus * mean_x_focus - weight_control * mean_x_control) * basis
            coefficient_terms = ((weight_focus * mean_x_focus if use_control else weight_control * mean_x_control)
                                 * (beta_focus - beta_control))
            composition = float(composition_terms.sum())
            coefficient = float(coefficient_terms.sum())
            if abs(actual - composition - coefficient) > 1e-8:
                raise AnalysisError(f'分组 {group["id"]} 未通过守恒检查，请核查样本。')
            compositions.append(composition)
            coefficients.append(coefficient)
            for index, key in enumerate(TERMS):
                term_composition[key].append(float(composition_terms[index]))
                term_coefficient[key].append(float(coefficient_terms[index]))
            item.update(
                composition=composition, coefficient=coefficient, estimated=True, missing=[],
                reason=None, fit=dict(focus=fit_focus, control=fit_control),
                terms=[dict(term=key, composition=float(composition_terms[index]),
                            coefficient=float(coefficient_terms[index]))
                       for index, key in enumerate(TERMS)],
            )
        groups.append(item)

    focus_cells = {tuple(cell) for cell in grouping['exclusive_cells'].get(focus, [])}
    control_cells = {tuple(cell) for cell in grouping['exclusive_cells'].get(control, [])}
    rows_focus_exclusive = [row for row in rows if row['branch'] == focus
                            and (int(row['ribs']), int(row['fabric']), int(row['mechanism'])) in focus_cells]
    rows_control_exclusive = [row for row in rows if row['branch'] == control
                              and (int(row['ribs']), int(row['fabric']), int(row['mechanism'])) in control_cells]

    def _exclusive(branch, samples, cells):
        weight = len(samples) / (total_focus if branch == focus else total_control)
        mean = _mean(samples) if samples else None
        return dict(branch=branch, cells=sorted(cells), tickets=sorted(row['code'] for row in samples),
                    count=len(samples), weight=weight, mean=mean,
                    contribution=weight * mean if samples else 0.0)

    exclusive_focus = _exclusive(focus, rows_focus_exclusive, focus_cells)
    exclusive_control = _exclusive(control, rows_control_exclusive, control_cells)
    exclusive_contribution = exclusive_focus['contribution'] - exclusive_control['contribution']

    matched_actual = math.fsum(actuals)
    matched_composition = math.fsum(compositions)
    matched_coefficient = math.fsum(coefficients)
    unestimated_actual = math.fsum(unestimated)
    gap = _mean([row for row in rows if row['branch'] == focus]) - _mean(
        [row for row in rows if row['branch'] == control])
    remainder = gap - matched_actual - exclusive_contribution
    if abs(remainder) > 1e-8:
        raise AnalysisError('可比工序与单边工序之和未通过全量差额对账，请核查样本。')
    if abs(matched_actual - matched_composition - matched_coefficient - unestimated_actual) > 1e-8:
        raise AnalysisError('可比工序分组之和未通过守恒检查，请核查样本。')
    terms = [dict(term=key, composition=math.fsum(term_composition[key]),
                  coefficient=math.fsum(term_coefficient[key])) for key in TERMS]
    return dict(
        version=grouping['version'], min_per_side=minimum,
        rule=_rule_description(grouping, plan),
        partition=grouping.get('partition'), matched_cells=grouping['matched_cells'],
        groups=groups,
        totals=dict(
            tickets=dict(focus=sum(item['counts']['focus'] for item in groups),
                         control=sum(item['counts']['control'] for item in groups)),
            actual=matched_actual, composition=matched_composition,
            coefficient=matched_coefficient, terms=terms,
            estimated_groups=len(compositions), unestimated_groups=len(unestimated),
            unestimated_actual=unestimated_actual,
            missing=[] if not unestimated else ['composition', 'coefficient'],
        ),
        exclusive=dict(
            focus=exclusive_focus, control=exclusive_control,
            contribution=exclusive_contribution,
        ),
        reconciliation=dict(gap=gap, matched_actual=matched_actual,
                            exclusive_contribution=exclusive_contribution,
                            composition=matched_composition, coefficient=matched_coefficient,
                            unestimated_actual=unestimated_actual, remainder=remainder),
    )


def analyze(snapshot):
    plan, rows = snapshot['plan'], snapshot['tickets']
    focus_rows = [row for row in rows if row['branch'] == plan['focus_branch']]
    control_rows = [row for row in rows if row['branch'] == plan['control_branch']]
    if not focus_rows or not control_rows:
        raise AnalysisError('比较网点或对照网点在本窗口没有工票，无法核对可比工序。')
    xa, xb = _design(focus_rows), _design(control_rows)
    focus_error = control_error = None
    try:
        _, ba, a = _fit(rows, plan['focus_branch'], plan['min_samples'])
    except AnalysisError as exc:
        ba, a, focus_error = None, _summary(focus_rows, plan['focus_branch'], exc), exc
    try:
        _, bb, b = _fit(rows, plan['control_branch'], plan['min_samples'])
    except AnalysisError as exc:
        bb, b, control_error = None, _summary(control_rows, plan['control_branch'], exc), exc
    gap = a['mean_minutes'] - b['mean_minutes']
    grouping = _grouping(snapshot, plan)
    matched = _matched(plan, rows, grouping)
    if focus_error or control_error:
        reasons = [str(error) for error in (focus_error, control_error) if error]
        return dict(
            method=METHOD, gap=gap, composition=None, coefficient=None, remainder=None,
            decomposition_available=False, decomposition_reason='；'.join(reasons),
            focus=a, control=b,
            terms=[dict(term=key, composition=None, coefficient=None) for key in TERMS],
            counterfactual=None,
            extrapolation={**(_support(xa, xb) if plan['coefficient_ref'] == 'control'
                           else _support(xb, xa)),
                           'evaluated_branch': plan['focus_branch'] if plan['coefficient_ref'] == 'control' else plan['control_branch'],
                           'basis_branch': plan['control_branch'] if plan['coefficient_ref'] == 'control' else plan['focus_branch']},
            matched=matched,
        )
    ma, mb = xa.mean(axis=0), xb.mean(axis=0)
    use_control = plan['coefficient_ref'] == 'control'
    basis = bb if use_control else ba
    composition_terms = (ma - mb) * basis
    coefficient_terms = (ma if use_control else mb) * (ba - bb)
    composition, coefficient = float(composition_terms.sum()), float(coefficient_terms.sum())
    remainder = gap - composition - coefficient
    if abs(remainder) > 1e-8:
        raise AnalysisError('两项之和未通过守恒检查，请核查样本。')
    return dict(
        method=METHOD, gap=gap, composition=composition, coefficient=coefficient, remainder=remainder,
        decomposition_available=True, focus=a, control=b,
        terms=[dict(term=key, composition=float(composition_terms[index]), coefficient=float(coefficient_terms[index]))
               for index, key in enumerate(TERMS)],
        counterfactual=float(ma @ bb if use_control else mb @ ba),
        extrapolation={**(_support(xa, xb) if use_control else _support(xb, xa)),
                       'evaluated_branch': a['branch'] if use_control else b['branch'],
                       'basis_branch': b['branch'] if use_control else a['branch']},
        matched=matched,
    )
