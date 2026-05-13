import { error } from '@sveltejs/kit';
import { loadResults, loadRun } from '$lib/server/results';

export const prerender = true;

export async function entries() {
	const runs = await loadResults();
	return runs.map((r) => ({ file: r.file }));
}

export async function load({ params }) {
	const run = await loadRun(params.file);
	if (!run) throw error(404, 'run not found');
	return { run };
}
