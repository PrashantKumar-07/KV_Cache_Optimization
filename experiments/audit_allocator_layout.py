"""Fresh-process layout comparison with the same charged-cleanup caller."""
import os
import torch
import compare_cpu_batched as trial
import screen_prefill_reclaim as screen


def main():
    setting = os.environ.get('PYTORCH_CUDA_ALLOC_CONF')
    assert setting in ('expandable_segments:False', 'expandable_segments:True')
    assert not torch.cuda.is_initialized(), 'Allocator setting must precede CUDA initialization'
    cleaned, identity = screen.cleanup_profile()
    original_profile, original_factory = trial.profile_one, screen.cleanup_profile

    def profile(*args, **options):
        result = cleaned(*args, **options)
        segments = [s for s in torch.cuda.memory_snapshot() if s['device'] == torch.cuda.current_device()]
        totals = dict(reserved_bytes=sum(s['total_size'] for s in segments),
            allocated_bytes=sum(s['allocated_size'] for s in segments),
            active_bytes=sum(s['active_size'] for s in segments),
            inactive_in_live_segments_bytes=sum(b['size'] for s in segments if s['active_size'] > 0
                                               for b in s['blocks'] if b['state'] == 'inactive'),
            fully_inactive_segment_bytes=sum(s['total_size'] for s in segments if s['active_size'] == 0),
            expandable_segment_count=sum(s['is_expandable'] for s in segments),
            segment_count=len(segments))
        assert totals['reserved_bytes'] == torch.cuda.memory_reserved()
        assert totals['allocated_bytes'] == torch.cuda.memory_allocated()
        assert totals['active_bytes'] + totals['inactive_in_live_segments_bytes'] + totals['fully_inactive_segment_bytes'] == totals['reserved_bytes']
        assert bool(totals['expandable_segment_count']) == setting.endswith('True'), 'Requested allocator layout not observed'
        result['allocator_layout'] = dict(setting=setting, caller='charged_cleanup_in_both_rows',
            same_caller_in_both_rows=True, snapshot_totals=totals, segments=segments,
            memory_stats={k:v for k,v in torch.cuda.memory_stats().items()
                          if k.startswith(('inactive_split_bytes.', 'num_alloc_retries', 'num_ooms'))})
        return result

    # Both legacy arm labels run the same caller. Layout changes only between processes.
    trial.profile_one = profile
    screen.cleanup_profile = lambda: (profile, identity)
    try:
        screen.main()
    finally:
        trial.profile_one, screen.cleanup_profile = original_profile, original_factory


if __name__ == '__main__':
    main()
