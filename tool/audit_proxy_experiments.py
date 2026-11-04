"""Read-only audit/replay of the November 2 proxy experiments; never calls an API.

Run from the repository root: .venv/bin/python tool/audit_proxy_experiments.py
Outputs a machine-readable audit under docs/. Existing logs and code are untouched.
The trusted local benchmark pickle is used as an oracle ONLY in this post-hoc audit.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
import pickle
import re
import statistics
import sys
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from llm_enas_pkb.config import Config
from llm_enas_pkb.encoder import ArchitectureInformationEncoder
from llm_enas_pkb.evolution import EvolutionSearch
from llm_enas_pkb.nasbench201 import NASBench201
from llm_enas_pkb.predictor import RegressionPredictor
from llm_enas_pkb.search_space import arch_to_str, str_to_arch

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'data/NAS-Bench-201-v1_1-096897.acc_cache.pkl'
ROW = re.compile(r'code=(\[[^\]]+\]).*?val=([\d.]+)\s+\[(GT|pred)\]')
GEN = re.compile(r'Generation (\d+) \| queries (\d+)/\d+ \| best-valid ([\d.]+)')


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def parse_search(path):
    text = path.read_text()
    config_match = re.search(r'Hyperparameters \(effective Config, JSON\):\n(.*?)\nEnd hyperparameters', text, re.S)
    config = json.loads(config_match[1]) if config_match else None
    generations, initial, current, section = [], [], None, None
    headings = {'Population P_': 'population', 'PKB exemplars (': 'pkb',
                'Offspring (': 'offspring', 'Ground-truth queried this': 'queries',
                'Ground-truth elites (': 'elites', 'Next population P_': 'next'}
    for line in text.splitlines():
        match = GEN.search(line)
        if match:
            current = {'generation': int(match[1]), 'count': int(match[2]),
                       'best': float(match[3]), **{v: [] for v in headings.values()}}
            generations.append(current)
        if line.startswith('Initial population ('):
            section = 'initial'
        elif line.startswith('---') or line.startswith('==='):
            section = None
        else:
            for prefix, name in headings.items():
                if line.startswith(prefix):
                    section = name
                    break
        match = ROW.search(line)
        if match and section:
            row = {'arch': tuple(json.loads(match[1])), 'value': float(match[2]),
                   'tag': match[3], 'line': line}
            if section == 'initial':
                initial.append(row)
            else:
                current[section].append(row)
    return {'path': str(path.relative_to(ROOT)), 'hash': digest(text), 'config': config,
            'seed': int(re.search(r'_seed(\d+)_', path.name)[1]),
            'initial': initial, 'generations': generations}


def parse_exchanges(path):
    text = path.read_text()
    exchanges = []
    for block in text.split('=' * 78):
        if '--- user ---\n' not in block:
            continue
        parts = re.split(r'--- (user|assistant|system|ERROR) ---\n', block)
        entries = [(parts[i], parts[i + 1].rstrip('\n')) for i in range(1, len(parts), 2)]
        assert entries[-1][0] == 'assistant', path
        exchanges.append(( [{'role': role, 'content': content} for role, content in entries[:-1]], entries[-1][1]))
    return exchanges


class CachedBenchmark:
    sample = NASBench201.sample

    def __init__(self, cache, dataset):
        self.cache = cache
        self.index = cache['datasets'].index(dataset)
        self.valid_calls = []
        self.test_calls = []

    def valid_acc(self, arch):
        self.valid_calls.append(arch)
        return self.cache['valid'][arch_to_str(arch)][self.index]

    def test_acc(self, arch):
        self.test_calls.append(arch)
        return self.cache['test'][arch_to_str(arch)][self.index]


def codes(rows):
    return [r['arch'] for r in rows]


def audit_replay(parsed, exchanges, cache):
    """Rebuild every prompt and replay every recorded response through real search code."""
    if parsed['config'] is not None:
        cfg = Config(**parsed['config'])
        config_source = 'recorded full Config'
    else:
        # Older logs omit API settings and several search settings. Replaying
        # with current defaults demonstrates compatibility, not historical proof
        # of the unrecorded settings (in particular temperature/API endpoint).
        header = (ROOT / parsed['path']).read_text().split('Initial population (')[0]
        cfg = Config(seed=parsed['seed'], n_init=len(parsed['initial']))
        aliases = {'iterations': 'max_iterations', 'pop': 'population_size',
                   'offspring': 'offspring_size'}
        for key, value in re.findall(r'(\w+)=([^\s]+)', header):
            key = aliases.get(key, key)
            if not hasattr(cfg, key):
                continue
            default = getattr(cfg, key)
            if isinstance(default, bool):
                value = value == 'True'
            elif isinstance(default, (int, float)):
                value = type(default)(value)
            setattr(cfg, key, value)
        config_source = 'recorded partial header plus current defaults; API settings unverified'
    cfg.search_log_path = cfg.prompt_log_path = None
    bench = CachedBenchmark(cache, cfg.dataset)

    class ReplayClient:
        cursor = 0

        def chat(self, messages):
            expected, response = exchanges[self.cursor]
            assert messages == expected, (parsed['path'], 'prompt mismatch', self.cursor + 1)
            self.cursor += 1
            return response

    client = ReplayClient()
    predictor = RegressionPredictor.__new__(RegressionPredictor)
    predictor.config = cfg
    predictor.encoder = ArchitectureInformationEncoder()
    predictor.client = client

    class CheckLogger:
        def __init__(self, config):
            pass

        def log_start(self):
            pass

        def log_header(self, population, fitness):
            assert population == codes(parsed['initial']), 'initial seed mismatch'
            for r in parsed['initial']:
                assert abs(fitness[r['arch']] - r['value']) <= 0.000050001

        def log_generation(self, **kw):
            expected = parsed['generations'][kw['generation'] - 1]
            for key, actual in [('population', kw['population']), ('pkb', kw['pkb_archs']),
                                ('offspring', kw['offspring']), ('next', kw['next_pop']),
                                ('elites', kw['elites']),
                                ('queries', [r['arch'] for r in kw['query_records']])]:
                assert list(actual) == codes(expected[key]), (parsed['path'], kw['generation'], key)
            assert kw['queries'] == expected['count']
            assert abs(kw['best_valid'] - expected['best']) <= 0.000050001
            for row, pred in zip(expected['offspring'], kw['preds']):
                assert abs(row['value'] - pred) <= 0.000050001
            for row, record in zip(expected['queries'], kw['query_records']):
                assert abs(row['value'] - record['true_valid']) <= 0.000050001
                assert f"reason={record['reason']}" in row['line']
                assert f"prediction_gen={record['prediction_generation']} " in row['line']
            for row in expected['next']:
                assert abs(row['value'] - kw['fitness'][row['arch']]) <= 0.000050001
                assert (row['tag'] == 'GT') == (row['arch'] in kw['archive'])

    with patch('llm_enas_pkb.evolution.SearchLogger', CheckLogger):
        result = EvolutionSearch(cfg, bench, predictor).run()
    assert client.cursor == len(exchanges)
    assert len(bench.valid_calls) == 100 == len(set(bench.valid_calls))
    assert bench.test_calls == [result['best_arch']]
    return {'full_prompt_and_search_replay': 'PASS', 'valid_calls': len(bench.valid_calls),
            'test_calls_after_search': len(bench.test_calls), 'exchanges': client.cursor,
            'config_source': config_source}


def summarize(parsed, cache, ds):
    index = cache['datasets'].index(ds)
    truth = lambda a: cache['valid'][arch_to_str(a)][index]
    archive, records, errors = {}, [], []
    for generation, rows in [(0, parsed['initial'])] + [(g['generation'], g['queries']) for g in parsed['generations']]:
        for row in rows:
            a = row['arch']
            assert a not in archive
            archive[a] = truth(a)
            errors.append(abs(truth(a) - row['value']))
            reason = re.search(r'reason=(\w+)', row['line'])
            records.append({'arch': a, 'generation': generation,
                            'reason': reason[1] if reason else 'init', 'query': len(records)+1})
    assert len(archive) == 100
    best = max(archive, key=archive.get)
    maximum = max(v[index] for v in cache['valid'].values())
    maximum_test = max(v[index] for v in cache['test'].values())
    first = next(r for r in records if r['arch'] == best)
    return {'search_log': parsed['path'], 'sha256': parsed['hash'], 'seed': parsed['seed'],
            'best_arch': best, 'best_valid': truth(best),
            'selected_test': cache['test'][arch_to_str(best)][index],
            'validation_global_hit': truth(best) == maximum,
            'test_global_hit': cache['test'][arch_to_str(best)][index] == maximum_test,
            'first_best': first, 'initial_codes': codes(parsed['initial']),
            'queries': records, 'max_logged_gt_rounding_error': max(errors)}


def main():
    logging.basicConfig(level=logging.ERROR)
    with CACHE.open('rb') as stream:
        cache = pickle.load(stream)
    output = {'method': 'Post-hoc cache comparison; exact prompt/response replay through current code, no API calls.',
              'datasets': {}}
    all_prompts, all_responses = [], []
    for ds in ['cifar10', 'cifar100']:
        groups = {}
        for suffix in ['pkb-overestimate-10', 'pkb-overestimate-proxy-10']:
            rows = []
            directory = ROOT / 'logs' / f'{ds}-{suffix}'
            for path in sorted(directory.glob('search*.log')):
                parsed = parse_search(path)
                summary = summarize(parsed, cache, ds)
                if suffix.endswith('proxy-10'):
                    exchanges = parse_exchanges(path.with_name(path.name.replace('search_', 'llm_prompts_', 1)))
                    summary['replay'] = audit_replay(parsed, exchanges, cache)
                    summary['config'] = parsed['config']
                    all_prompts += [digest(json.dumps(m, sort_keys=True)) for m, _ in exchanges]
                    all_responses += [digest(r) for _, r in exchanges]
                    print(ds, parsed['seed'], 'replay PASS', flush=True)
                rows.append(summary)
            rows.sort(key=lambda r: r['seed'])
            groups[suffix] = {'runs': rows,
                'validation_hits': sum(r['validation_global_hit'] for r in rows),
                'test_hits': sum(r['test_global_hit'] for r in rows),
                'mean_valid_percent': statistics.mean(r['best_valid']*100 for r in rows),
                'sd_valid_percentage_points': statistics.stdev(r['best_valid']*100 for r in rows),
                'mean_test_percent': statistics.mean(r['selected_test']*100 for r in rows),
                'sd_test_percentage_points': statistics.stdev(r['selected_test']*100 for r in rows)}
        before, after = [groups[s]['runs'] for s in ['pkb-overestimate-10', 'pkb-overestimate-proxy-10']]
        assert [r['seed'] for r in before] == [r['seed'] for r in after] == list(range(42, 52))
        assert all(a['initial_codes'] == b['initial_codes'] for a,b in zip(before,after))
        assert len({json.dumps(r['initial_codes']) for r in after}) == 10
        index = cache['datasets'].index(ds)
        vmax = max(v[index] for v in cache['valid'].values())
        tmax = max(v[index] for v in cache['test'].values())
        output['datasets'][ds] = {'validation_optimum': vmax, 'test_optimum': tmax,
            'validation_optimal_archs': [str_to_arch(a) for a,v in cache['valid'].items() if v[index] == vmax],
            'test_optimal_archs': [str_to_arch(a) for a,v in cache['test'].items() if v[index] == tmax],
            'paired_initial_populations_match': True, 'distinct_initial_populations': 10, 'groups': groups}
    output['llm_exchanges'] = len(all_prompts)
    output['distinct_complete_prompts'] = len(set(all_prompts))
    output['distinct_complete_responses'] = len(set(all_responses))
    out = ROOT / 'docs' / 'experiment_rigor_audit_20261102.json'
    out.write_text(json.dumps(output, ensure_ascii=False, indent=2)+'\n')
    print('Wrote', out)


if __name__ == '__main__':
    main()
