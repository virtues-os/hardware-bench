<script lang="ts">
	import type { Run } from '$lib/server/results';
	let { data } = $props<{ data: { run: Run } }>();
	const run = data.run;

	function fmt(v: unknown, decimals = 2): string {
		if (v === null || v === undefined) return '—';
		if (typeof v === 'boolean') return v ? '✓' : '✗';
		if (typeof v === 'number') return v.toFixed(decimals);
		return String(v);
	}

	type KpiRow = { label: string; value: unknown };
	function collectKpis(run: Run): { label: string; rows: KpiRow[] }[] {
		const sections: { label: string; rows: KpiRow[] }[] = [];
		const passes = [
			['Pass 1a (stock governor)', run.pass_1a_stock],
			['Pass 1b (performance governor)', run.pass_1b_performance]
		] as const;

		for (const [passLabel, pass] of passes) {
			if (!pass) continue;
			const p = pass as any;
			const rows: KpiRow[] = [];
			rows.push({ label: 'K0 — NPU gate passed', value: p.k0_npu_gate?.passed });
			for (const ep of Object.keys(p.embed ?? {})) {
				const epShort = ep.replace('ExecutionProvider', '');
				for (const variant of Object.keys(p.embed[ep] ?? {})) {
					const e = p.embed[ep][variant];
					const r = p.rerank?.[ep]?.[variant];
					if (e?.error) {
						rows.push({ label: `${ep}/${variant} — error`, value: 'see JSON' });
						continue;
					}
					rows.push({ label: `K1 rerank p95 ms (${epShort}/${variant})`, value: r?.k1_p95_ms });
					rows.push({ label: `K2 rerank p99 ms (${epShort}/${variant})`, value: r?.k2_p99_ms });
					rows.push({
						label: `K3 embed throughput emb/sec (${epShort}/${variant})`,
						value: e?.k3_throughput_emb_per_sec
					});
					rows.push({
						label: `K4 embed p95 ms batch=1 (${epShort}/${variant})`,
						value: e?.k4_p95_ms
					});
				}
			}
			rows.push({ label: 'K5 max temp °C', value: p.thermal_power?.k5_max_temp_c });
			rows.push({
				label: 'K5 below-max time (s)',
				value: p.thermal_power?.k5_throttle?.approx_below_max_s
			});
			rows.push({ label: 'K6 idle power W', value: p.thermal_power?.k6_idle_power_w });
			rows.push({ label: 'K7 peak power W', value: p.thermal_power?.k7_peak_power_w });
			rows.push({ label: 'K8 pgvector ANN p95 ms', value: p.pgvector?.k8_ann_p95_ms });
			rows.push({ label: 'K9 RAM free MB', value: p.memory?.k9_ram_free_mb });
			sections.push({ label: passLabel, rows });
		}
		return sections;
	}

	const sections = collectKpis(run);
	const sys = run.system as any;
	const bcfg = run.board_config as any;
</script>

<svelte:head>
	<title>{run.board} — Virtues Hardware Bench</title>
</svelte:head>

<main>
	<nav><a href="/">← all runs</a></nav>
	<header>
		<h1>{run.board}</h1>
		<p class="meta">
			{run.started_at} · {Math.round(run.elapsed_s)}s · harness v{run.harness_version}
		</p>
	</header>

	<section>
		<h2>System</h2>
		<dl>
			<dt>SoC</dt>
			<dd>{bcfg?.soc ?? '—'}</dd>
			<dt>CPU</dt>
			<dd>{bcfg?.cpu_model ?? '—'}</dd>
			<dt>RAM (SKU / total)</dt>
			<dd>{bcfg?.ram_gb ?? '—'} GB / {sys?.ram_total_mb ?? '—'} MB</dd>
			<dt>Storage</dt>
			<dd>{bcfg?.storage ?? '—'}</dd>
			<dt>Cooling</dt>
			<dd>{bcfg?.cooling ?? '—'}</dd>
			<dt>OS</dt>
			<dd>{sys?.os_release?.PRETTY_NAME ?? bcfg?.os?.intended ?? '—'}</dd>
			<dt>Kernel</dt>
			<dd>{sys?.kernel ?? '—'}</dd>
			<dt>Governor</dt>
			<dd>{sys?.governor ?? '—'}</dd>
			<dt>ORT version</dt>
			<dd>{sys?.ort_version ?? '—'}</dd>
			<dt>Available EPs</dt>
			<dd>{(sys?.available_eps ?? []).join(', ') || '—'}</dd>
		</dl>
	</section>

	{#each sections as section}
		<section>
			<h2>{section.label}</h2>
			<table>
				<thead>
					<tr><th>KPI</th><th>Value</th></tr>
				</thead>
				<tbody>
					{#each section.rows as row}
						<tr>
							<td>{row.label}</td>
							<td class="num">{fmt(row.value)}</td>
						</tr>
					{/each}
				</tbody>
			</table>
		</section>
	{/each}

	<section>
		<h2>Raw</h2>
		<p>
			<a href="https://github.com/virtues/hardware-bench/blob/main/results/{run.file}.json">
				results/{run.file}.json on GitHub
			</a>
		</p>
	</section>
</main>

<style>
	:global(body) {
		font-family: ui-monospace, 'SF Mono', Menlo, monospace;
		background: #fafaf8;
		color: #222;
		margin: 0;
	}
	main {
		max-width: 900px;
		margin: 0 auto;
		padding: 2rem 1.5rem;
	}
	nav {
		margin-bottom: 1rem;
	}
	nav a {
		color: #888;
		text-decoration: none;
		font-size: 0.9rem;
	}
	nav a:hover {
		color: #0a5fff;
	}
	header {
		margin-bottom: 2rem;
		padding-bottom: 1rem;
		border-bottom: 1px solid #ddd;
	}
	h1 {
		margin: 0 0 0.25rem;
		font-size: 1.5rem;
		font-weight: 600;
	}
	h2 {
		margin: 0 0 0.75rem;
		font-size: 1rem;
		font-weight: 500;
		color: #444;
	}
	.meta {
		color: #888;
		font-size: 0.85rem;
		margin: 0;
	}
	section {
		margin-bottom: 2rem;
	}
	dl {
		display: grid;
		grid-template-columns: 12rem 1fr;
		gap: 0.4rem 1rem;
		font-size: 0.9rem;
		margin: 0;
	}
	dt {
		color: #888;
	}
	dd {
		margin: 0;
		color: #222;
	}
	table {
		width: 100%;
		border-collapse: collapse;
		font-size: 0.9rem;
	}
	th,
	td {
		text-align: left;
		padding: 0.4rem 0.75rem;
		border-bottom: 1px solid #eee;
	}
	th {
		font-weight: 500;
		color: #666;
		font-size: 0.8rem;
		text-transform: uppercase;
		letter-spacing: 0.03em;
	}
	td.num {
		font-variant-numeric: tabular-nums;
		text-align: right;
	}
	a {
		color: #0a5fff;
	}
</style>
