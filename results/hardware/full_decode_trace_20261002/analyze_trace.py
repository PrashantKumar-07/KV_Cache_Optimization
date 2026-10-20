"""Attribute CUDA API and correlated device work to innermost host NVTX range."""
import bisect,collections,hashlib,json,sqlite3,tarfile
from pathlib import Path
P=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
r=json.loads((P/'result.json').read_text());m=json.loads((P/'source_manifest.json').read_text())
assert r['passed'] and r['decode_steps']==50 and len(r['cold_geometry'])==50
assert sha(P/'source.tar.gz')==m['bundle_sha256']
with tarfile.open(P/'source.tar.gz') as archive:
 for name,expected in r['source_sha256'].items():assert expected==hashlib.sha256(archive.extractfile(name).read()).hexdigest()==m['source_sha256'][name]
assert sha(P/'result.prof')==r['profile_sha256'] and sha(P/'result.prof.txt')==r['profile_text_sha256']
c=sqlite3.connect(P/'cuda_trace.sqlite');strings=dict(c.execute('select id,value from StringIds'))
ranges=c.execute('select start,end,globalTid,coalesce(text,(select value from StringIds where id=textId)) from NVTX_EVENTS where end is not null and eventType=59 order by start').fetchall()
assert len(ranges)==4900
threads=collections.defaultdict(list)
for i,(start,end,tid,label) in enumerate(ranges):
 threads[tid].extend(((start,1,i,label),(end,0,i,label)))
segments={}
for tid,events in threads.items():
 stack=[];starts=[];labels=[];ends=[];previous=None
 for time,kind,index,label in sorted(events):
  if previous is not None and time>previous:
   starts.append(previous);ends.append(time);labels.append(stack[-1][1] if stack else 'unlabelled_model_or_score_work')
  if kind:stack.append((index,label))
  else:assert stack[-1][0]==index;stack.pop()
  previous=time
 assert not stack
 segments[tid]=(starts,ends,labels)
api=collections.defaultdict(lambda:collections.Counter());correlation={};crossing=0
for start,end,tid,corr,nameid in c.execute('select start,end,globalTid,correlationId,nameId from CUPTI_ACTIVITY_KIND_RUNTIME order by start'):
 starts,ends,labels=segments.get(tid,([],[],[]));index=bisect.bisect_right(starts,start)-1
 label=labels[index] if index>=0 and start<ends[index] else 'unlabelled_model_or_score_work'
 if index>=0 and end>ends[index]:crossing+=1
 group=label.split('/')[-1];name=strings[nameid];api[group]['calls']+=1;api[group]['host_api_ns']+=end-start
 api[group][name+'_calls']+=1;api[group][name+'_ns']+=end-start
 correlation[(tid>>24,corr)]=group
# Nsight global thread ID packs process ID in upper bits; join verified IDs.
pids={pid for pid,_ in c.execute('select distinct globalPid,correlationId from CUPTI_ACTIVITY_KIND_KERNEL')}
# Use correlationId directly only when process is unique in this capture.
assert len(pids)==1
correlation_only={corr:group for (_,corr),group in correlation.items()}
device=collections.defaultdict(lambda:collections.Counter())
for start,end,corr in c.execute('select start,end,correlationId from CUPTI_ACTIVITY_KIND_KERNEL'):
 group=correlation_only.get(corr,'unmatched');device[group]['kernel_calls']+=1;device[group]['kernel_ns']+=end-start
for start,end,corr,size in c.execute('select start,end,correlationId,bytes from CUPTI_ACTIVITY_KIND_MEMCPY'):
 group=correlation_only.get(corr,'unmatched');device[group]['memcpy_calls']+=1;device[group]['memcpy_ns']+=end-start;device[group]['memcpy_bytes']+=size
out=dict(source_and_trace_geometry_verified=True,cold_first=r['cold_geometry'][0],cold_final=r['cold_geometry'][-1],
 any_cold_saturation=any(x['saturated_layers'] for x in r['cold_geometry']),nvtx_ranges=len(ranges),
 host_api_crossing_range_boundaries=crossing,
 stage_attribution={group:dict(api[group])|dict(device[group]) for group in set(api)|set(device)},
 hypothesis_cold_overflow_explains_gap=False,
 attribution_method='Innermost push/pop host NVTX range for runtime API start, correlated device launch by uniqueprocess correlation ID. Unlabelled includes model/score/diagnostic work. GPU completion may overlap ranges; timings are inclusive diagnostics and not additive savings.',
 limitations='CProfile/NVTX/CUPTI perturbations, one fixedteacherstream, no speed gate. No cold overflow occurred; this cannot measure fullfill reclaim performance or isolate safe optimizations.',
 evidence_sha256={name:sha(P/name) for name in ('result.json','result.prof','cuda_trace.nsys-rep','cuda_trace.sqlite','source.tar.gz')})
q=P/'verified_trace_analysis.json';assert not q.exists();q.write_text(json.dumps(out,indent=2)+'\n')
for group,row in sorted(out['stage_attribution'].items()):
 print(group, 'APIcalls',row.get('calls'), 'kernelcalls',row.get('kernel_calls'), 'APIhost_ms',round(row.get('host_api_ns',0)/1e6,2),'devicekernel_ms',round(row.get('kernel_ns',0)/1e6,2),'sync_calls',row.get('cudaStreamSynchronize_calls',0))
print('cold overflow',out['any_cold_saturation'])
