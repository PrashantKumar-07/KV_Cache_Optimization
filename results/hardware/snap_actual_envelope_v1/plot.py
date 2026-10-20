"""Standalone diagnostic figures from the verified resource audit only."""
import csv
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    report=json.loads((BASE/'verified_report.json').read_text())
    assert report['verified'] and not report['exact_actual_byte_and_width_match']
    assert all(sha(BASE/p)==v for p,v in report['evidence_sha256'].items())
    names=['snap','cpu_bulk'];labels=['Faithful SnapKV','Ordinary CPU TieredKV']
    rows=[report['resources'][name] for name in names]
    resources=BASE/'resources.csv';widths=BASE/'forward_widths.csv'
    for p in (resources,widths,BASE/'resource_gap.png',BASE/'resource_gap.pdf',BASE/'figure_manifest.json'):
        assert not p.exists(),p
    with resources.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['arm',*rows[0]])
        writer.writeheader();writer.writerows(dict(arm=name,**row) for name,row in zip(names,rows))
    raw=json.loads((BASE/'snap.json').read_text());cpu=json.loads((BASE/'cpu_gpu.json').read_text())
    observed=cpu['ordinary_width_observations'][1:]
    sw=[x[0] for x in raw['forward_attended_rows_by_layer']]
    tw=[x['before_cache_rows'][0]+x['input_rows'] for x in observed]
    with widths.open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['decode_step','snap_forward_width','tiered_forward_width'])
        writer.writerows((i+1,a,b) for i,(a,b) in enumerate(zip(sw,tw)))
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(10,4.3),layout='constrained')
    gpu=[row['gpu_decode_active_mib'] for row in rows]
    host=[row['owned_pinned_allocator_capacity_mib'] for row in rows]
    axes[0].bar(labels,gpu,label='CUDA active, model storage subtracted',color='#3572A5')
    axes[0].bar(labels,host,bottom=gpu,label='Owned pinned allocator capacity',color='#E69F00')
    axes[0].axhline(512,color='#666666',linestyle='--',linewidth=1,label='512 MiB target')
    for i,row in enumerate(rows):
        n=row['combined_active_decode_envelope_mib'];axes[0].text(i,n+7,f'{n:.2f}',ha='center')
    axes[0].set(ylabel='Decode allocation envelope (MiB)',ylim=(0,720),title='Actual measured allocation scope')
    axes[0].legend(loc='upper left',fontsize=8)
    steps=range(1,51)
    axes[1].plot(steps,sw,label='Faithful SnapKV',color='#3572A5',linewidth=2)
    axes[1].plot(steps,tw,label='Ordinary CPU TieredKV',color='#E69F00',linewidth=2)
    axes[1].set(xlabel='Fixed teacher decode step',ylabel='KV rows consumed by each layer',
                ylim=(965,1040),title='Widths differ on 49 of 50 steps')
    axes[1].legend(loc='lower right',fontsize=9)
    fig.suptitle('Mistral-7B · 16K prompt · 50 fixed tokens · seed 0',fontsize=12)
    fig.savefig(BASE/'resource_gap.png',dpi=180)
    fig.savefig(BASE/'resource_gap.pdf')
    plt.close(fig)
    manifest=dict(verified_report_sha256=sha(BASE/'verified_report.json'),
        artifacts_sha256={p:sha(BASE/p) for p in ('resources.csv','forward_widths.csv','resource_gap.png','resource_gap.pdf')},
        caption='Diagnostic allocation/geometry comparison; excludes constant model storage, unowned host and driver memory. Owned pinned ranges counted. No matched-resource, timing, quality or physicalSTT claim.',
        script_sha256=sha(Path(__file__)))
    (BASE/'figure_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Standalone PNG/PDF and CSVs exported from verified observations')


if __name__=='__main__':main()
