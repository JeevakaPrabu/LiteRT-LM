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

#include "runtime/util/memory_profiling/process_memory_monitor.h"

#include <memory>

#include "absl/time/time.h"  // from @com_google_absl
#include "tflite/profiling/memory_usage_monitor.h"  // from @litert

namespace litert::lm {

ProcessMemoryMonitor::ProcessMemoryMonitor(int sampling_interval_ms)
    : host_(std::make_unique<tflite::profiling::memory::MemoryUsageMonitor>(
          sampling_interval_ms)),
      accelerator_(absl::Milliseconds(sampling_interval_ms)) {}

ProcessMemoryMonitor::~ProcessMemoryMonitor() = default;

void ProcessMemoryMonitor::Start() {
  host_->Start();
  accelerator_.Start();
}

void ProcessMemoryMonitor::Stop() {
  host_->Stop();
  accelerator_.Stop();
}

ProcessMemoryMonitor::Peaks ProcessMemoryMonitor::GetPeaks() const {
  Peaks peaks;
  // The host monitor reports a negative value if unsupported or not sampled.
  if (const float resident_mb = host_->GetPeakMemUsageInMB();
      resident_mb >= 0) {
    peaks.resident_mb = resident_mb;
  }
  if (const float private_mb = host_->GetPeakPrivateFootprintInMB();
      private_mb >= 0) {
    peaks.private_mb = private_mb;
  }
  peaks.accelerator = accelerator_.GetPeakUsage();
  return peaks;
}

}  // namespace litert::lm
