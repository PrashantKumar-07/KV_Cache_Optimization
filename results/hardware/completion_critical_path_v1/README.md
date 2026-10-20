# Completion critical path, 3 October 2026

The repeated10–20 workingdays forecast is withdrawn. Failed research gates and
external hardware do not support a fixed completion date. Scope is unchanged:
full P0/P1/P2 remains open. `audit.json` maps requirements to frozen authoritative
reports, distinguishing prior DirectCPU comparisons from current ordinarybulk.
`verified_audit.json` checks report/source/archive hashes and diagnostic replay.

The current ordinary backend policy-only GPU0 profile uses16K/50steps,
hot991/cold1090/quota32, actual-queryprefill/lastlogits. Predictions, finalcache,
counters, nativeH2D315,621,376/D2H117,571,584bytes and76/57calls replay the frozen
ordinary reference.967049 calls take1.920s under instrumentation. Eviction/
demotion11.48ms/decode, readprep11.10 andvisibility9.94ms overlap; nativevisible
gather0.76ms. These identify a structural bookkeeping target, not speed savings.

The initial run passed state/pred/counter/payload checks but looked up nativecall
fields at the wrong reference level and stopped before dumping the profile.
Original evidence is retained. The preregistered v2 repair changes only that
fieldpath and outputnames, recording the failed lineage. No production source
changed; profiled timing cannot establish a favorable latency retry.

Next: consolidate cross-layer eviction/demotion bookkeeping with exactselection/
state/transfer and lifecycle proof, then whole-loop/memory/faithfulbaseline gates.
Failed fullquery derivatives stay stopped. No physicalSTT, energy/endurance,
independentquality or complete512MiB claim. All owned jobs are terminal.
