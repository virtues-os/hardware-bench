<script lang="ts">
	import type { Run } from '$lib/server/results';
	let { data } = $props<{ data: { runs: Run[] } }>();

	function k1(run: Run): number | null {
		const ep = (run.pass_1a_stock as any)?.rerank?.CPUExecutionProvider?.int8;
		return ep?.k1_p95_ms ?? null;
	}
	function k3(run: Run): number | null {
		const ep = (run.pass_1a_stock as any)?.embed?.CPUExecutionProvider?.int8;
		return ep?.k3_throughput_emb_per_sec ?? null;
	}
	function fmt(v: number | null, decimals = 1): string {
		return v == null ? '—' : v.toFixed(decimals);
	}
	function shortTs(s: string): string {
		return s.replace('T', ' ').replace(/\..*$/, '').replace('Z', ' UTC');
	}
</script>

<svelte:head>
	<title>Virtues Hardware Bench</title>
</svelte:head>

<main>
	<header>
		<h1>Virtues Hardware Bench</h1>
		<p>
			Public benchmark suite for evaluating single-board computers as the V1 platform for
			<a href="https://virtues.com">Virtues</a>. Source:
			<a href="https://github.com/virtues/hardware-bench">github.com/virtues/hardware-bench</a>.
		</p>
	</header>

	<section>
		<h2>Runs ({data.runs.length})</h2>
		{#if data.runs.length === 0}
			<p class="muted">No completed runs yet.</p>
		{:else}
			<table>
				<thead>
					<tr>
						<th>Board</th>
						<th>Started</th>
						<th>OS</th>
						<th>Governor</th>
						<th>K1 rerank p95 (ms)</th>
						<th>K3 embed/sec</th>
						<th>Elapsed</th>
					</tr>
				</thead>
				<tbody>
					{#each data.runs as run}
						{@const sys = run.system as any}
						{@const os = sys?.os_release?.PRETTY_NAME ?? '—'}
						<tr>
							<td><a href="/runs/{run.file}">{run.board}</a></td>
							<td>{shortTs(run.started_at)}</td>
							<td class="muted">{os}</td>
							<td class="muted">{sys?.governor ?? '—'}</td>
							<td>{fmt(k1(run))}</td>
							<td>{fmt(k3(run))}</td>
							<td class="muted">{Math.round(run.elapsed_s)}s</td>
						</tr>
					{/each}
				</tbody>
			</table>
		{/if}
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
		max-width: 1100px;
		margin: 0 auto;
		padding: 2rem 1.5rem;
	}
	header {
		margin-bottom: 2rem;
		padding-bottom: 1.5rem;
		border-bottom: 1px solid #ddd;
	}
	h1 {
		margin: 0 0 0.5rem;
		font-size: 1.6rem;
		font-weight: 600;
	}
	h2 {
		margin: 0 0 1rem;
		font-size: 1.1rem;
		font-weight: 500;
		color: #444;
	}
	p {
		margin: 0;
		color: #555;
		font-size: 0.95rem;
	}
	table {
		width: 100%;
		border-collapse: collapse;
		font-size: 0.9rem;
	}
	th,
	td {
		text-align: left;
		padding: 0.5rem 0.75rem;
		border-bottom: 1px solid #eee;
	}
	th {
		font-weight: 500;
		color: #666;
		font-size: 0.85rem;
		text-transform: uppercase;
		letter-spacing: 0.03em;
	}
	td a {
		color: #0a5fff;
		text-decoration: none;
	}
	td a:hover {
		text-decoration: underline;
	}
	.muted {
		color: #888;
	}
	a {
		color: #0a5fff;
	}
</style>
