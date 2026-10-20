"""Paired localization only; failed candidate timing gate remains final."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'experiments'))
from profile_cpu_batched import ProfileCalls
from shared_decode_transition import SharedTransitionPolicy,SharedAppendLease
from bulk_promotion_policy import BulkPromotionPolicy
BASE=Path(__file__).resolve().parent
class ProfileControl(ProfileCalls,BulkPromotionPolicy):pass
class ProfileShared(ProfileCalls,SharedTransitionPolicy):pass

def main():
    plan=json.loads((BASE/'plan.json').read_text())
    for arm,cls in [('control',ProfileControl),('candidate',ProfileShared)]:
        output=BASE/(arm+'.json');assert not output.exists()
        source=(ROOT/'results/hardware/completion_critical_path_v1/profile_ordinary_v2.py').read_text()
        source=source.replace("BASE/'profile_v2.json'",f"BASE/'{arm}.json'").replace("BASE/'profile_plan_v2.json'","BASE/'plan.json'")
        source=source.replace("plan['reference']",f"plan['{arm}']").replace("plan['reference_sha256']",f"plan['{arm}_sha256']")
        source=source.replace("BASE/'policy_v2.prof'",f"BASE/'{arm}.prof'").replace("BASE/'policy_v2.prof.txt'",f"BASE/'{arm}.prof.txt'")
        if arm=='candidate':
            source=source.replace('capture=PromptQueryCapture(model,policy)','capture=PromptQueryCapture(model,policy);lease=SharedAppendLease(model,policy)')
            source=source.replace('capture.close();policy.reset()',"lease.close();report['lease_closed']=not lease.handles and not lease.pending;capture.close();policy.reset()")
            source=source.replace('if capture:capture.close()','if \'lease\' in locals():lease.close()\n        if capture:capture.close()')
        scope=dict(__name__='profile_'+arm,__file__=str(Path(__file__)))
        exec(compile(source,'<frozen-policy-profile-'+arm+'>','exec'),scope)
        scope.update(ProfileBulk=cls,SharedAppendLease=SharedAppendLease)
        scope['main']()
        print(arm,'finished',flush=True)
if __name__=='__main__':main()
