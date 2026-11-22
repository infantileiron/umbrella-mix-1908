import { BRANCH } from '../constants/domain.js';
import { number, signed } from '../utils/format.js';
import { esc } from '../utils/html.js';

const quantity = value => `<strong class="mono">${signed(value)}</strong><small> 分钟/票</small>`;

const mechanismLabel = value => value === 1 ? '需修' : '—';

function bandLabel(group, partition, ticket) {
  const [ribOrigin, fabricOrigin] = partition?.origin ?? [0, 0];
  const ribBand = Math.floor((ticket.ribs - ribOrigin) / group.basis.rib_width);
  const fabricBand = Math.floor((ticket.fabric - fabricOrigin) / group.basis.fabric_width);
  return `带 ${ribBand}·${fabricBand}`;
}

function ticketTable(group, partition, side) {
  const rows = group.tickets[side].map(ticket => [
    `<span class="mono">${esc(ticket.worked_on)}</span>`, `<b class="mono">${esc(ticket.code)}</b>`,
    ticket.ribs, ticket.fabric, mechanismLabel(ticket.mechanism), `<b class="mono minutes">${ticket.minutes}</b>`,
    `<span class="mono">${ticket.ribs}·${ticket.fabric}·${ticket.mechanism}</span>`,
    `<span class="mono band">${esc(bandLabel(group, partition, ticket))}</span>`,
  ]).join('');
  return `<div class="table-wrap ticket-evidence"><table>
    <thead><tr><th scope="col">日期</th><th scope="col">工票</th><th scope="col">换骨</th><th scope="col">布损</th>
    <th scope="col">开合</th><th scope="col">工时</th><th scope="col">工序格</th><th scope="col">落入分带</th></tr></thead>
    <tbody>${rows}</tbody></table></div>`;
}

function ticketBlock(group, partition, result) {
  return `<div class="ticket-split">
    <div><h4>${BRANCH[result.focus.branch]} · ${group.counts.focus} 张</h4>${ticketTable(group, partition, 'focus')}</div>
    <div><h4>${BRANCH[result.control.branch]} · ${group.counts.control} 张</h4>${ticketTable(group, partition, 'control')}</div>
  </div>`;
}

function fitLine(fit) {
  if (!fit) return '<span class="muted">该组未能拟合</span>';
  return `<span class="mono">n=${fit.n} · 秩 ${fit.rank}/4 · 条件数 ${number(fit.condition)} ·
    R² ${number(fit.r_squared, 3)} · RMSE ${number(fit.rmse, 3)}</span>`;
}

function groupCard(group, partition, result) {
  const estimated = group.estimated;
  const status = estimated
    ? '<span class="status completed"><i></i>已估计</span>'
    : '<span class="status failed"><i></i>缺估计</span>';
  const missing = group.missing?.length
    ? `<p class="group-missing">保留实际工时贡献 ${signed(group.actual)} 分钟/票；缺 ${group.missing.map(key => key === 'composition' ? '构成差' : '系数差').join('、')}估计。${esc(group.reason ?? '')}</p>`
    : '';
  const contribution = (label, value, tone) =>
    `<span>${label} <b class="mono ${estimated ? tone : 'muted'}">${estimated ? signed(value) : '未估计'}</b></span>`;
  const ranges = group.range;
  const mechanismText = ranges.mechanism.length > 1 ? '含需修/无需修' : ranges.mechanism[0] === 1 ? '需修' : '无需修';
  const rangeText = `换骨 ${ranges.ribs[0]}–${ranges.ribs[1]} · 布损 ${ranges.fabric[0]}–${ranges.fabric[1]} ·
    开合 ${mechanismText}`;
  return `<details class="group-card"${estimated ? '' : ' open'}>
    <summary>
      <span class="group-id mono">${esc(group.id)}</span>
      <span class="group-range"><b>${esc(rangeText)}</b><small>${group.cells.length} 个工序格</small></span>
      <span class="group-counts">${BRANCH[result.focus.branch]} ${group.counts.focus} · ${BRANCH[result.control.branch]} ${group.counts.control}</span>
      <span class="group-weight mono">折回权重 ${number(group.weights.focus, 3)} / ${number(group.weights.control, 3)}</span>
      <span class="group-actual mono warm">${signed(group.actual)}</span>
      ${status}
    </summary>
    <div class="group-body">
      <p class="group-means">组均值　${BRANCH[result.focus.branch]} <b class="mono">${number(group.means.focus)}</b> ·
        ${BRANCH[result.control.branch]} <b class="mono">${number(group.means.control)}</b>　按两点原票数
        ${group.counts.focus}/${group.counts.control} 折回后贡献 ${signed(group.actual)} 分钟/票。</p>
      <div class="group-terms">${contribution('构成差', group.composition, 'warm')}
        ${contribution('系数差', group.coefficient, 'cool')}
        <span>实际 <b class="mono">${signed(group.actual)}</b></span></div>
      ${missing}
      <p class="group-fit">${estimated ? `${BRANCH[result.focus.branch]} ${fitLine(group.fit.focus)}　${BRANCH[result.control.branch]} ${fitLine(group.fit.control)}`
        : '该组样本或工序组合撑不起四参数拟合，未生成构成差与系数差。'}</p>
      ${ticketBlock(group, partition, result)}
    </div></details>`;
}

export function matchedDetail(result) {
  const matched = result.matched;
  if (!matched) return '';
  const totals = matched.totals;
  const exclusive = matched.exclusive;
  const reconciliation = matched.reconciliation;
  const focus = result.focus.branch, control = result.control.branch;
  const reconcileRows = [
    ['全量差额', reconciliation.gap, `${BRANCH[focus]} 减 ${BRANCH[control]}`],
    ['可比工序', totals.actual, `两边都做过的 ${totals.tickets.focus}/${totals.tickets.control} 张按原票数折回`],
    ['单边工序', exclusive.contribution, '另一边从未做过，不计入可比工序'],
    ['对账残差', reconciliation.remainder, '全量差额减前两项，应接近 0'],
  ];
  const groupList = matched.groups.length
    ? matched.groups.map(group => groupCard(group, matched.partition, result)).join('')
    : `<p class="matched-empty">${esc(matched.rule)}</p>`;
  const exclusiveDetails = ['focus', 'control'].map(side => {
    const item = exclusive[side];
    if (!item.count) return '';
    return `<details class="exclusive-item"><summary>${BRANCH[item.branch]} 单边 ${item.count} 张 ·
      贡献 ${signed(item.contribution)} 分钟/票</summary>
      <p>这些工序组合另一边在本窗口从未做过，只以实际工时计入单边贡献，不进入任何可比分组。
      工序格：${item.cells.map(cell => `<span class="mono">${cell.join('·')}</span>`).join('、')}</p>
      <p class="mono ticket-codes">${item.tickets.map(code => esc(code)).join('、')}</p></details>`;
  }).join('');
  const missingTotal = totals.missing?.length
    ? `<p class="group-missing">可比工序中有 ${totals.unestimated_groups} 组撑不起拟合，实际贡献
       ${signed(totals.unestimated_actual)} 分钟/票已保留；缺 ${totals.missing.map(key => key === 'composition' ? '构成差' : '系数差').join('、')}估计。</p>`
    : '';
  return `<section class="matched-section" aria-label="可比工序对账">
    <header class="matched-head"><div><h2>可比工序核对</h2><p>${esc(matched.rule)}</p></div>
      <span class="method-chip mono">${esc(matched.version)}</span></header>
    <div class="reconcile-strip" aria-label="与全量差额对账">${reconcileRows.map(([label, value, note], index) =>
      `<article class="reconcile-cell${index === 0 ? ' total' : ''}${index === 3 ? ' residual' : ''}">
        <h3>${label}</h3><div class="value">${quantity(value)}</div><p>${esc(note)}</p></article>`).join('')}</div>
    <p class="reconcile-equation">${signed(reconciliation.gap)} = 可比工序 ${signed(totals.actual)} + 单边工序
      ${signed(exclusive.contribution)}<small>　残差 ${reconciliation.remainder.toExponential(2)}</small></p>
    <p class="reconcile-equation">可比工序 ${signed(totals.actual)} = 构成差 ${signed(totals.composition)} +
      系数差 ${signed(totals.coefficient)}${totals.unestimated_actual ? ` + 未估计 ${signed(totals.unestimated_actual)}` : ''}
      <small>　${totals.estimated_groups} 组可拟合 / ${totals.unestimated_groups} 组缺估计</small></p>
    ${missingTotal}
    <div class="group-list">${groupList}</div>
    ${exclusiveDetails ? `<div class="exclusive-list">${exclusiveDetails}</div>` : ''}
  </section>`;
}
