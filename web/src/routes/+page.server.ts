import { loadResults } from '$lib/server/results';

export const prerender = true;

export async function load() {
	const runs = await loadResults();
	return { runs };
}
