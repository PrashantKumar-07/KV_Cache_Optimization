// Research CPU metadata controller. No Torch tensor API or GPU ownership changes.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>
namespace py = pybind11;
using Bool = py::array_t<bool, py::array::c_style>;
using Float = py::array_t<float, py::array::c_style>;
using Long = py::array_t<int64_t, py::array::c_style>;

py::list promotion(py::list frames, int budget, double hysteresis) {
  if (budget < 0) throw std::invalid_argument("Negative promotion budget");
  py::list output;
  for (auto frame_handle : frames) {
    auto frame=py::cast<py::dict>(frame_handle);
    Bool shadow=py::cast<Bool>(frame["shadow"]), resident=py::cast<Bool>(frame["resident"]);
    Float streak=py::cast<Float>(frame["streak"]);
    if (shadow.ndim()!=1 || resident.ndim()!=1 || streak.ndim()!=1 ||
        shadow.size()!=resident.size() || shadow.size()!=streak.size())
      throw std::invalid_argument("Invalid cold metadata geometry");
    const auto size=shadow.size();
    auto scored=py::cast<std::vector<std::pair<double,int64_t>>>(frame["attention"]);
    // Validate the complete frame before computing any transition.
    for (auto [score,index] : scored)
      if (std::isnan(score) || index < -size || index >= size)
        throw std::invalid_argument("Unsupported attention value or index");
    scored.erase(std::remove_if(scored.begin(),scored.end(),[&](auto item) {
      auto index=item.second<0 ? size+item.second : item.second;
      return shadow.data()[index];
    }),scored.end());
    py::dict result;
    result["hits"]=py::none(); result["deferred"]=0;
    result["rows"]=py::list();result["floor_deferred"]=0;
    Float cleared(size);
    std::fill(cleared.mutable_data(), cleared.mutable_data()+size, 0.0f);
    result["streak"]=cleared;
    if (!scored.empty()) {
      std::sort(scored.begin(),scored.end(),[](auto a,auto b) {
        return a.first>b.first || (a.first==b.first && a.second>b.second);
      });
      scored.resize(std::min<size_t>(scored.size(),budget));
      std::vector<bool> selected(size,false);
      std::vector<int64_t> candidates;
      for (auto item:scored) {
        auto index=item.second;
        if (index>=0 && !selected[index]) {selected[index]=true;candidates.push_back(index);}
      }
      Float next(size);
      for (int64_t i=0;i<size;++i) next.mutable_data()[i]=selected[i] ? streak.data()[i]+1.0f : 0.0f;
      std::vector<int64_t> promote; int deferred=0;
      for (auto index:candidates) {
        if (!resident.data()[index] || next.data()[index]>=hysteresis) promote.push_back(index);
        else ++deferred;
      }
      std::vector<std::pair<int64_t,int64_t>> hits;
      for (auto index:promote) hits.emplace_back(index,index+1);
      auto rows=promote;
      int floor_deferred=0;
      auto floor=py::cast<int64_t>(frame["floor"]);
      if (py::cast<bool>(frame["inclusive"]) && floor>0) {
        int64_t live=0;
        for (int64_t i=0;i<size;++i) live+=!shadow.data()[i];
        auto allowed=std::max<int64_t>(0,live-floor);
        if (rows.size()>static_cast<size_t>(allowed)) {
          floor_deferred=rows.size()-allowed;rows.resize(allowed);
        }
      }
      result["hits"]=py::cast(hits);result["deferred"]=deferred;
      result["rows"]=py::cast(rows);result["floor_deferred"]=floor_deferred;
      result["streak"]=next;
    }
    output.append(result);
  }
  return output;
}

py::list backups(py::list frames) {
  py::list output;
  for (auto h:frames) {
    auto frame=py::cast<py::dict>(h);
    auto hot=py::cast<std::vector<int64_t>>(frame["hot"]);
    auto cold=py::cast<std::vector<int64_t>>(frame["cold"]);
    auto victims=py::cast<std::vector<int64_t>>(frame["victims"]);
    Bool shadow=py::cast<Bool>(frame["shadow"]);
    if (shadow.ndim()!=1 || shadow.size()!=static_cast<py::ssize_t>(cold.size()))
      throw std::invalid_argument("Invalid backup geometry");
    std::unordered_map<int64_t,int64_t> lookup;
    if (py::cast<bool>(frame["inclusive"]))
      for (size_t i=0;i<cold.size();++i) if (shadow.data()[i]) lookup[cold[i]]=i;
    std::vector<int64_t> restored,paid;
    for (auto victim:victims) {
      if (victim<0 || victim>=static_cast<int64_t>(hot.size()))
        throw std::invalid_argument("Victim outside hot rows");
      auto found=lookup.find(hot[victim]);
      if (found==lookup.end()) paid.push_back(victim); else restored.push_back(found->second);
    }
    py::dict result;result["restored"]=py::cast(restored);result["paid"]=py::cast(paid);
    output.append(result);
  }
  return output;
}

Long addresses(py::list frames) {
  std::vector<int64_t> result;
  for (auto h:frames) {
    auto frame=py::cast<py::dict>(h);
    const auto heads=py::cast<int64_t>(frame["heads"]),capacity=py::cast<int64_t>(frame["capacity"]),base=py::cast<int64_t>(frame["base"]);
    auto slots=py::cast<std::vector<int64_t>>(frame["slots"]);
    if (heads<0 || capacity<0 || base<0) throw std::invalid_argument("Negative address geometry");
    if (base>std::numeric_limits<int32_t>::max()) throw std::invalid_argument("Address base exceeds int32");
    if (heads && capacity>(int64_t(std::numeric_limits<int32_t>::max())-base+1)/heads)
      throw std::invalid_argument("Address span exceeds int32");
    for (auto slot:slots) if (slot<0 || slot>=capacity) throw std::invalid_argument("Slot outside pool");
    for (int64_t head=0;head<heads;++head)
      for (auto slot:slots) result.push_back(base+head*capacity+slot);
  }
  Long output(result.size());std::copy(result.begin(),result.end(),output.mutable_data());return output;
}

PYBIND11_MODULE(tieredkv_controller_v1,module) {
  module.def("promotion",&promotion);
  module.def("backups",&backups);
  module.def("addresses",&addresses);
}
