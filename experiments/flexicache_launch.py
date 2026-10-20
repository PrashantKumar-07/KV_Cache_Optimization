"""Explicit L40S launch-only adapter; released Triton kernel source is untouched."""


class StageOnePrefill:
    def __init__(self, kernel):
        self.kernel = kernel
        self.calls = []

    def __getitem__(self, grid):
        original = self.kernel[grid]

        def launch(*args, **kwargs):
            expected = {'BLOCK_M': 128, 'BLOCK_N': 128, 'num_warps': 8, 'num_stages': 3}
            if any(kwargs.get(key) != value for key, value in expected.items()):
                raise ValueError('Unexpected released prefill launch configuration')
            kwargs['num_stages'] = 1
            compiled = original(*args, **kwargs)
            self.calls.append({'original': expected, 'num_stages': 1,
                               'compiled_shared_bytes': compiled.metadata.shared})
            return compiled

        return launch
