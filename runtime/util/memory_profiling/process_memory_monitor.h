// Copyright 2026 The ODML Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_MEMORY_PROFILING_PROCESS_MEMORY_MONITOR_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_MEMORY_PROFILING_PROCESS_MEMORY_MONITOR_H_

#include <memory>
#include <optional>

#include "runtime/util/memory_profiling/accelerator_memory_monitor.h"

namespace tflite::profiling::memory {
class MemoryUsageMonitor;
}  // namespace tflite::profiling::memory

namespace litert::lm {

// The sampling interval of every `report_peak_memory_footprint` implementation.
inline constexpr int kDefaultMemorySamplingIntervalMs = 50;

// Tracks the peak host and accelerator memory of the current process by
// sampling both from background threads.
//
// Every `report_peak_memory_footprint` implementation (the benchmark tools and
// the C API) uses this class, so that they all report the same numbers.
class ProcessMemoryMonitor {
 public:
  // The peaks seen between Start() and Stop(). A field is std::nullopt when the
  // platform cannot measure it or nothing has been sampled yet.
  struct Peaks {
    // Peak resident memory (RSS / working set) in MB.
    std::optional<float> resident_mb;
    // Peak private memory footprint in MB. Its definition differs by OS, see
    // host_memory.h.
    std::optional<float> private_mb;
    // Peak accelerator memory, with its breakdown.
    std::optional<AcceleratorMemoryUsage> accelerator;
  };

  explicit ProcessMemoryMonitor(
      int sampling_interval_ms = kDefaultMemorySamplingIntervalMs);
  ~ProcessMemoryMonitor();

  ProcessMemoryMonitor(const ProcessMemoryMonitor&) = delete;
  ProcessMemoryMonitor& operator=(const ProcessMemoryMonitor&) = delete;

  // Starts sampling. A no-op if already running.
  void Start();
  // Stops sampling. A no-op if not running. The peaks remain available.
  void Stop();

  Peaks GetPeaks() const;

 private:
  std::unique_ptr<tflite::profiling::memory::MemoryUsageMonitor> host_;
  AcceleratorMemoryMonitor accelerator_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_MEMORY_PROFILING_PROCESS_MEMORY_MONITOR_H_
