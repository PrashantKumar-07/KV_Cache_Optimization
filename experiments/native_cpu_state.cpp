// Research fixed CPU state owners; commands describe GPU payload/metadata work.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
namespace py=pybind11;
template<class T> using Array=py::array_t<T,py::array::c_style>;
using I32=Array<int32_t>;using I64=Array<int64_t>;using Bool=Array<bool>;using F32=Array<float>;

template<class T> Array<T> field(py::dict values,const char* name,int dimensions) {
  py::object raw=values[name];
  if (!py::isinstance<py::array>(raw)) throw std::invalid_argument("CPU state requires NumPy arrays");
  auto a=py::cast<py::array>(raw);
  if (!a.dtype().is(py::dtype::of<T>()) || a.ndim()!=dimensions ||
      !(a.flags()&py::array::c_style) || !a.writeable() || reinterpret_cast<uintptr_t>(a.data())%alignof(T))
    throw std::invalid_argument("CPU state dtype, dimensions, contiguity or writability invalid");
  return py::reinterpret_borrow<Array<T>>(a);
}
template<class T> void descriptor(const Array<T>& a,int ndim) {
  if (a.ndim()!=ndim) throw std::invalid_argument("Invalid command dimensions");
}
struct State {
  I64 hot,cold,version;Bool grace,shadow,resident;F32 streak;
  I32 slots,free,hot_n,cold_n,free_n;
  int64_t layers,hcap,ccap;
  std::vector<std::pair<uintptr_t,uintptr_t>> spans;
  State(py::dict values):hot(field<int64_t>(values,"hot",2)),cold(field<int64_t>(values,"cold",2)),version(field<int64_t>(values,"version",1)),
    grace(field<bool>(values,"grace",2)),shadow(field<bool>(values,"shadow",2)),resident(field<bool>(values,"resident",2)),streak(field<float>(values,"streak",2)),
    slots(field<int32_t>(values,"slots",2)),free(field<int32_t>(values,"free",2)),hot_n(field<int32_t>(values,"hot_n",1)),cold_n(field<int32_t>(values,"cold_n",1)),free_n(field<int32_t>(values,"free_n",1)) {
    layers=hot.shape(0);hcap=hot.shape(1);ccap=cold.shape(1);
    if (layers<1 || hcap<1 || ccap<1 || hcap>INT32_MAX || ccap>INT32_MAX) throw std::invalid_argument("Invalid owner geometry");
    for (auto a:std::vector<py::array>{grace}) if (a.shape(0)!=layers || a.shape(1)!=hcap) throw std::invalid_argument("Hot owner geometry mismatch");
    for (auto a:std::vector<py::array>{cold,shadow,resident,streak,slots,free}) if (a.shape(0)!=layers || a.shape(1)!=ccap) throw std::invalid_argument("Cold owner geometry mismatch");
    for (auto a:std::vector<py::array>{version,hot_n,cold_n,free_n}) if (a.shape(0)!=layers) throw std::invalid_argument("Count geometry mismatch");
    for (auto a:std::vector<py::array>{hot,cold,grace,shadow,resident,streak,slots,free,version,hot_n,cold_n,free_n}) {
      auto start=reinterpret_cast<uintptr_t>(a.data());auto end=start+a.nbytes();
      for (auto [other,last]:spans) if (start<last && other<end) throw std::invalid_argument("CPU state owners overlap");
      spans.emplace_back(start,end);
    }
    for (int64_t l=0;l<layers;++l) {
      int h=hot_n.data()[l],c=cold_n.data()[l],f=free_n.data()[l];
      if (h<0 || h>hcap || c<0 || c>ccap || f<0 || f>ccap || int64_t(c)+f!=ccap || version.data()[l]<0 || version.data()[l]==INT64_MAX)
        throw std::invalid_argument("Invalid logical CPU state counts/version");
      std::vector<bool> used(ccap,false);
      for (int i=0;i<c+f;++i) {
        int slot=i<c?slots.data()[l*ccap+i]:free.data()[l*ccap+i-c];
        if (slot<0 || slot>=ccap || used[slot]) throw std::invalid_argument("Invalid logical/free slot partition");
        used[slot]=true;
      }
    }
  }
  void command(py::array values,py::array counts,int64_t capacity) {
    if (values.ndim()!=2 || counts.ndim()!=1 || values.shape(0)!=layers || counts.shape(0)!=layers || values.shape(1)>capacity)
      throw std::invalid_argument("Command geometry mismatch");
    if (reinterpret_cast<uintptr_t>(values.data())%values.itemsize() || reinterpret_cast<uintptr_t>(counts.data())%counts.itemsize()) throw std::invalid_argument("Misaligned command arrays");
    for (auto a:std::vector<py::array>{values,counts}) {
      auto start=reinterpret_cast<uintptr_t>(a.data()),end=start+a.nbytes();
      for (auto [other,last]:spans) if (start<last && other<end) throw std::invalid_argument("Command aliases mutable CPU state");
    }
  }
};
struct Demotion {std::vector<int32_t> paid,restored,keep,writes;};
py::dict demote(py::dict owners,I32 victims,I32 counts,bool inclusive) {
  State s(owners);s.command(victims,counts,s.hcap);std::vector<Demotion> plans(s.layers);
  for (int64_t l=0;l<s.layers;++l) {
    auto& p=plans[l];int h=s.hot_n.data()[l],c=s.cold_n.data()[l],n=counts.data()[l];
    if (n<0 || n>victims.shape(1)) throw std::invalid_argument("Invalid victim count");
    std::unordered_map<int64_t,int32_t> backup;
    if (inclusive) for (int i=0;i<c;++i) if (s.shadow.data()[l*s.ccap+i]) backup[s.cold.data()[l*s.ccap+i]]=i;
    std::vector<bool> removed(h,false);
    for (int i=0;i<n;++i) {
      int v=victims.data()[l*victims.shape(1)+i];
      if (v<0 || v>=h || removed[v]) throw std::invalid_argument("Invalid or duplicate hot victim");
      removed[v]=true;auto found=backup.find(s.hot.data()[l*s.hcap+v]);
      if (found==backup.end()) p.paid.push_back(v);else p.restored.push_back(found->second);
    }
    if (int64_t(c)+p.paid.size()>s.ccap || p.paid.size()>static_cast<size_t>(s.free_n.data()[l])) throw std::invalid_argument("Cold append exceeds CPU owner");
    for (int i=0;i<h;++i) if (!removed[i]) p.keep.push_back(i);
    for (size_t i=0;i<p.paid.size();++i) p.writes.push_back(s.free.data()[l*s.ccap+i]);
  }
  // Allocate all descriptor outputs before the first owner mutation.
  I32 paid({s.layers,victims.shape(1)}),restored({s.layers,victims.shape(1)}),writes({s.layers,victims.shape(1)}),keep({s.layers,s.hcap}),sizes({s.layers,int64_t(3)});
  I64 versions(s.layers);
  for (int64_t l=0;l<s.layers;++l) versions.mutable_data()[l]=s.version.data()[l]+1;
  py::dict output;output["paid"]=paid;output["restored"]=restored;output["writes"]=writes;output["keep"]=keep;output["sizes"]=sizes;output["versions"]=versions;
  std::fill_n(paid.mutable_data(),paid.size(),-1);std::fill_n(restored.mutable_data(),restored.size(),-1);std::fill_n(writes.mutable_data(),writes.size(),-1);std::fill_n(keep.mutable_data(),keep.size(),-1);
  for (int64_t l=0;l<s.layers;++l) {
    auto& p=plans[l];int c=s.cold_n.data()[l],f=s.free_n.data()[l];
    sizes.mutable_data()[l*3]=p.paid.size();sizes.mutable_data()[l*3+1]=p.restored.size();sizes.mutable_data()[l*3+2]=p.keep.size();
    std::copy(p.paid.begin(),p.paid.end(),paid.mutable_data()+l*victims.shape(1));std::copy(p.restored.begin(),p.restored.end(),restored.mutable_data()+l*victims.shape(1));
    std::copy(p.writes.begin(),p.writes.end(),writes.mutable_data()+l*victims.shape(1));std::copy(p.keep.begin(),p.keep.end(),keep.mutable_data()+l*s.hcap);
    for (auto i:p.restored) {s.shadow.mutable_data()[l*s.ccap+i]=false;s.streak.mutable_data()[l*s.ccap+i]=0.0f;}
    for (size_t i=0;i<p.paid.size();++i) {
      auto destination=l*s.ccap+c+i;
      s.cold.mutable_data()[destination]=s.hot.data()[l*s.hcap+p.paid[i]];
      s.shadow.mutable_data()[destination]=false;s.resident.mutable_data()[destination]=true;s.streak.mutable_data()[destination]=0.0f;s.slots.mutable_data()[destination]=p.writes[i];
    }
    std::memmove(s.free.mutable_data()+l*s.ccap,s.free.data()+l*s.ccap+p.paid.size(),(f-p.paid.size())*sizeof(int32_t));
    for (size_t i=0;i<p.keep.size();++i) {s.hot.mutable_data()[l*s.hcap+i]=s.hot.data()[l*s.hcap+p.keep[i]];s.grace.mutable_data()[l*s.hcap+i]=false;}
    s.hot_n.mutable_data()[l]=p.keep.size();s.cold_n.mutable_data()[l]=c+p.paid.size();s.free_n.mutable_data()[l]=f-p.paid.size();++s.version.mutable_data()[l];
  }
  return output;
}
struct Reclaim {std::vector<int64_t> positions;std::vector<bool> shadow,resident;std::vector<uint32_t> streak;std::vector<int32_t> slots,freed;};
void reclaim(py::dict owners,I32 keeps,I32 counts) {
  State s(owners);s.command(keeps,counts,s.ccap);std::vector<Reclaim> plans(s.layers);
  for (int64_t l=0;l<s.layers;++l) {
    auto& p=plans[l];int c=s.cold_n.data()[l],n=counts.data()[l];
    if (n<0 || n>keeps.shape(1) || n>c) throw std::invalid_argument("Invalid cold keep count");
    std::vector<bool> selected(c,false);
    for (int i=0;i<n;++i) {
      int k=keeps.data()[l*keeps.shape(1)+i];
      if (k<0 || k>=c || selected[k]) throw std::invalid_argument("Invalid or duplicate cold keep");
      selected[k]=true;auto index=l*s.ccap+k;
      p.positions.push_back(s.cold.data()[index]);p.shadow.push_back(s.shadow.data()[index]);p.resident.push_back(s.resident.data()[index]);p.slots.push_back(s.slots.data()[index]);
      uint32_t bits;std::memcpy(&bits,s.streak.data()+index,sizeof(bits));p.streak.push_back(bits);
    }
    for (int i=0;i<c;++i) if (!selected[i]) p.freed.push_back(s.slots.data()[l*s.ccap+i]);
  }
  for (int64_t l=0;l<s.layers;++l) {
    auto& p=plans[l];int n=p.positions.size(),f=s.free_n.data()[l];
    for (int i=0;i<n;++i) {auto index=l*s.ccap+i;s.cold.mutable_data()[index]=p.positions[i];s.shadow.mutable_data()[index]=p.shadow[i];s.resident.mutable_data()[index]=p.resident[i];s.slots.mutable_data()[index]=p.slots[i];std::memcpy(s.streak.mutable_data()+index,&p.streak[i],sizeof(uint32_t));}
    std::copy(p.freed.begin(),p.freed.end(),s.free.mutable_data()+l*s.ccap+f);
    s.cold_n.mutable_data()[l]=n;s.free_n.mutable_data()[l]=f+p.freed.size();++s.version.mutable_data()[l];
  }
}
void append_hot(py::dict owners,I64 positions,I32 counts) {
  State s(owners);s.command(positions,counts,s.hcap);
  for (int64_t l=0;l<s.layers;++l) if (counts.data()[l]<0 || counts.data()[l]>positions.shape(1) || int64_t(s.hot_n.data()[l])+counts.data()[l]>s.hcap) throw std::invalid_argument("Hot append exceeds CPU owner");
  for (int64_t l=0;l<s.layers;++l) {int old=s.hot_n.data()[l],n=counts.data()[l];for (int i=0;i<n;++i) {s.hot.mutable_data()[l*s.hcap+old+i]=positions.data()[l*positions.shape(1)+i];s.grace.mutable_data()[l*s.hcap+old+i]=false;}s.hot_n.mutable_data()[l]=old+n;++s.version.mutable_data()[l];}
}
// GPU reads must consume these slots before a later phase reuses freed slots.
// Return exact pre-removal indices and physical slots as immutable commands.
py::dict promote(py::dict owners,I32 rows,I32 counts,bool inclusive,int32_t floor) {
  State s(owners);s.command(rows,counts,s.ccap);
  if (floor<0) throw std::invalid_argument("Negative live floor");
  struct Promotion {std::vector<int64_t> positions;std::vector<int32_t> keep,freed,read_slots;};
  std::vector<Promotion> plans(s.layers);
  for (int64_t l=0;l<s.layers;++l) {
    auto& p=plans[l];int c=s.cold_n.data()[l],h=s.hot_n.data()[l],n=counts.data()[l];
    if (n<0 || n>rows.shape(1) || int64_t(h)+n>s.hcap) throw std::invalid_argument("Invalid promotion count/capacity");
    int live=0;for (int i=0;i<c;++i) live+=!s.shadow.data()[l*s.ccap+i];
    if (inclusive && n>std::max(0,live-floor)) throw std::invalid_argument("Promotion breaches live floor");
    std::vector<bool> selected(c,false);
    for (int i=0;i<n;++i) {
      int k=rows.data()[l*rows.shape(1)+i];
      if (k<0 || k>=c || selected[k] || s.shadow.data()[l*s.ccap+k]) throw std::invalid_argument("Invalid, duplicate or shadow promotion");
      selected[k]=true;p.positions.push_back(s.cold.data()[l*s.ccap+k]);p.read_slots.push_back(s.slots.data()[l*s.ccap+k]);
    }
    for (int i=0;i<c;++i) {if (!selected[i]) p.keep.push_back(i);else p.freed.push_back(s.slots.data()[l*s.ccap+i]);}
  }
  I32 read_slots({s.layers,rows.shape(1)}),read_rows({s.layers,rows.shape(1)}),read_counts(s.layers),hot_start(s.layers);I64 versions(s.layers);
  std::fill_n(read_slots.mutable_data(),read_slots.size(),-1);
  std::copy_n(rows.data(),rows.size(),read_rows.mutable_data());
  for (int64_t l=0;l<s.layers;++l) {versions.mutable_data()[l]=s.version.data()[l]+1;read_counts.mutable_data()[l]=counts.data()[l];hot_start.mutable_data()[l]=s.hot_n.data()[l];std::copy(plans[l].read_slots.begin(),plans[l].read_slots.end(),read_slots.mutable_data()+l*rows.shape(1));}
  py::dict output;output["read_slots"]=read_slots;output["rows"]=read_rows;output["counts"]=read_counts;output["hot_start"]=hot_start;output["versions"]=versions;
  for (int64_t l=0;l<s.layers;++l) {
    auto& p=plans[l];int h=s.hot_n.data()[l],n=p.positions.size(),f=s.free_n.data()[l];
    for (int i=0;i<n;++i) {s.hot.mutable_data()[l*s.hcap+h+i]=p.positions[i];s.grace.mutable_data()[l*s.hcap+h+i]=true;s.resident.mutable_data()[l*s.ccap+rows.data()[l*rows.shape(1)+i]]=true;}
    s.hot_n.mutable_data()[l]=h+n;
    if (inclusive) {
      for (int i=0;i<n;++i) s.shadow.mutable_data()[l*s.ccap+rows.data()[l*rows.shape(1)+i]]=true;
    } else {
      // Keeps are ascending, so in-place left compaction never overwrites a future source.
      for (size_t i=0;i<p.keep.size();++i) {
        int64_t dst=l*s.ccap+static_cast<int64_t>(i),src=l*s.ccap+p.keep[i];
        s.cold.mutable_data()[dst]=s.cold.data()[src];s.shadow.mutable_data()[dst]=s.shadow.data()[src];s.resident.mutable_data()[dst]=s.resident.data()[src];s.slots.mutable_data()[dst]=s.slots.data()[src];
        std::memmove(s.streak.mutable_data()+dst,s.streak.data()+src,sizeof(float));
      }
      std::copy(p.freed.begin(),p.freed.end(),s.free.mutable_data()+l*s.ccap+f);
      s.cold_n.mutable_data()[l]=p.keep.size();s.free_n.mutable_data()[l]=f+p.freed.size();
    }
    ++s.version.mutable_data()[l];
  }
  return output;
}
PYBIND11_MODULE(tieredkv_cpu_state_v1,module) {
  module.def("demote",&demote,py::arg("owners"),py::arg("victims").noconvert(),py::arg("counts").noconvert(),py::arg("inclusive"));
  module.def("reclaim",&reclaim,py::arg("owners"),py::arg("keeps").noconvert(),py::arg("counts").noconvert());
  module.def("append_hot",&append_hot,py::arg("owners"),py::arg("positions").noconvert(),py::arg("counts").noconvert());
  module.def("promote",&promote,py::arg("owners"),py::arg("rows").noconvert(),py::arg("counts").noconvert(),py::arg("inclusive"),py::arg("floor"));
}
