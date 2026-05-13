// Build-time loader: reads ../../results/*.json via Node fs.
// Only imported from +page.server.ts files which run during prerender.

import fs from 'node:fs/promises';
import path from 'node:path';

// At SvelteKit build/dev time process.cwd() is the `web/` directory,
// so the bench's results/ dir is one level up. Override via env if needed.
const RESULTS_DIR = process.env.RESULTS_DIR ?? path.resolve(process.cwd(), '../results');

export interface Run {
	file: string; // basename without .json
	harness_version: string;
	board: string;
	board_config: Record<string, unknown>;
	started_at: string;
	elapsed_s: number;
	system: Record<string, unknown>;
	pass_1a_stock: Record<string, unknown> | null;
	pass_1b_performance: Record<string, unknown> | null;
}

export async function loadResults(): Promise<Run[]> {
	let entries: string[] = [];
	try {
		entries = await fs.readdir(RESULTS_DIR);
	} catch {
		return [];
	}
	const runs: Run[] = [];
	for (const f of entries) {
		if (!f.endsWith('.json') || f.startsWith('.')) continue;
		try {
			const content = await fs.readFile(path.join(RESULTS_DIR, f), 'utf-8');
			const data = JSON.parse(content);
			runs.push({ file: f.replace(/\.json$/, ''), ...data });
		} catch {
			// skip malformed
		}
	}
	return runs.sort((a, b) => (b.started_at || '').localeCompare(a.started_at || ''));
}

export async function loadRun(file: string): Promise<Run | null> {
	const all = await loadResults();
	return all.find((r) => r.file === file) ?? null;
}
