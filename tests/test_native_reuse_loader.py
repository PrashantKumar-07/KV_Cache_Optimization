"""Cached native reuse must reject unverified binaries, sources and ABIs."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
import native_reuse_visible as native


@pytest.fixture
def cached_build(tmp_path,monkeypatch):
    source=tmp_path/'experiments';source.mkdir()
    build=tmp_path/'build';build.mkdir()
    binary=build/'tieredkv_reuse_hot_v1.so';binary.write_bytes(b'fixture binary')
    hashes={}
    for name in ('native_hot_compact.cpp','native_reuse_hot.cu'):
        p=source/name;p.write_text(name)
        hashes[f'experiments/{name}']=hashlib.sha256(p.read_bytes()).hexdigest()
    proof_dir=tmp_path/'results/hardware/reuse_visible_integrity_20261002';proof_dir.mkdir(parents=True)
    proof=dict(verified=True,candidate_ready_for_opt_in_cli=True,
        reuse_binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),native_source_sha256=hashes,
        torch_version=str(torch.__version__),python_major_minor=list(sys.version_info[:2]))
    proof_path=proof_dir/'verified_report.json';proof_path.write_text(json.dumps(proof))
    real_path=Path;filename=native.__file__
    def path(value):
        if value==filename:return source/'native_reuse_visible.py'
        if str(value)=='/data/nishant/tieredkv_external/reuse_visible_build_20261002':return build
        return real_path(value)
    monkeypatch.setattr(native,'Path',path)
    calls=[];module=SimpleNamespace()
    loader=SimpleNamespace(exec_module=lambda obj:calls.append(obj))
    monkeypatch.setattr(native.importlib.util,'spec_from_file_location',lambda *args:SimpleNamespace(loader=loader))
    monkeypatch.setattr(native.importlib.util,'module_from_spec',lambda spec:module)
    native.kernel.cache_clear()
    yield source,binary,proof_path,proof,calls,module
    native.kernel.cache_clear()


def test_verified_cached_load_does_not_need_compilation(cached_build):
    _,_,_,_,calls,module=cached_build
    assert native.kernel() is module
    assert native.kernel() is module and calls==[module]


@pytest.mark.parametrize('fault',('binary','source','unverified','abi'))
def test_cached_build_rejects_changed_identity_before_load(cached_build,fault):
    source,binary,proof_path,proof,calls,_=cached_build
    if fault=='binary':binary.write_bytes(b'changed')
    elif fault=='source':(source/'native_reuse_hot.cu').write_text('changed')
    elif fault=='unverified':proof['candidate_ready_for_opt_in_cli']=False
    else:proof['torch_version']='unsupported ABI'
    proof_path.write_text(json.dumps(proof))
    with pytest.raises(ValueError):native.kernel()
    assert calls==[]
