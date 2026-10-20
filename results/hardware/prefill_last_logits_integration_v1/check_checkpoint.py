"""Actual evaluator CLI: legacy disabled resume, enabled mismatch rejection."""
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
sys.path.insert(0, str(ROOT / 'experiments'))
import longbench_eval as bench
spec = importlib.util.spec_from_file_location('tiny_fixture', ROOT / 'tests/test_prefill_output_lifetime.py')
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)


def main():
    output = BASE / 'checkpoint_checks.json'
    if output.exists():
        raise FileExistsError(output)
    model = fixture.MistralForCausalLM(fixture.MistralConfig(hidden_size=16,
        intermediate_size=32, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, vocab_size=32, sliding_window=None)).eval()
    tokenizer = fixture.Tokenizer(); tokenizer.pad_token = 'pad'
    calls = []

    def evaluate(*args, **options):
        calls.append(options['prefill_last_logits'])
        args[3].sample_records = [dict(prediction='checkpoint proof')]
        return 0., [0.]

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'run.json'; checkpoint = Path(str(path) + '.ckpt')
        (Path(directory) / 'qasper_e.jsonl').write_text('{"context":"checkpoint-only fixture"}\n')
        arguments = ['eval', '--model', 'tiny', '--tasks', 'qasper', '--methods', 'full',
            '--max-samples', '1', '--device', 'cpu', '--json', str(path), '--data-dir', directory]
        with patch.object(bench.AutoModelForCausalLM, 'from_pretrained', return_value=model), \
             patch.object(bench.AutoTokenizer, 'from_pretrained', return_value=tokenizer), \
             patch.object(bench, 'evaluate_task', side_effect=evaluate), \
             patch.object(sys, 'argv', arguments), redirect_stdout(io.StringIO()):
            bench.main()
            stored = json.loads(checkpoint.read_text())
            assert stored['_config_fingerprint']['prefill_last_logits'] is False
            del stored['_config_fingerprint']['prefill_last_logits']
            checkpoint.write_text(json.dumps(stored))
            bench.main()
            assert calls == [False], 'Legacy disabled checkpoint failed to skip completed work'
            with patch.object(sys, 'argv', [*arguments, '--prefill-last-logits']):
                try:
                    bench.main()
                except RuntimeError as error:
                    assert 'different config' in str(error)
                else:
                    raise AssertionError('Enabled flag reused legacy disabled checkpoint')
            assert calls == [False]
    report = dict(passed=True, legacy_disabled_checkpoint_resumes=True,
        enabled_checkpoint_mismatch_rejected=True, core_source_sha256=hashlib.sha256(
            (ROOT / 'experiments/longbench_eval.py').read_bytes()).hexdigest(),
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Actual CLI parsing/fingerprint/resume code, mocked model load/task evaluator, CPU tiny model; no GPU/data quality claim.')
    output.write_text(json.dumps(report, indent=2) + '\n')
    print('Legacy disabled resume passes; enabled mismatch rejects before task evaluation.')


if __name__ == '__main__':
    main()
