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

#include <cstddef>
#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include "runtime/util/memory_profiling/accelerator_memory_monitor.h"
#include "tflite/profiling/memory_info.h"  // from @litert

namespace litert::lm {
namespace {

constexpr int kSamplingIntervalMs = 1;
constexpr size_t kAllocationBytes = 32 * 1024 * 1024;
constexpr float kAllocationMb = 32.0f;

bool IsHostMemorySupported() {
  return tflite::profiling::memory::MemoryUsage::IsSupported();
}

TEST(ProcessMemoryMonitorTest, NoPeaksBeforeStart) {
  ProcessMemoryMonitor monitor(kSamplingIntervalMs);

  const ProcessMemoryMonitor::Peaks peaks = monitor.GetPeaks();

  EXPECT_FALSE(peaks.resident_mb.has_value());
  EXPECT_FALSE(peaks.private_mb.has_value());
  EXPECT_FALSE(peaks.accelerator.has_value());
}

TEST(ProcessMemoryMonitorTest, ReportsPeaksOfTheMonitoredWindow) {
  ProcessMemoryMonitor monitor(kSamplingIntervalMs);

  monitor.Start();
  // Stays alive past Stop() so that the last sample still sees it.
  std::vector<char> buffer(kAllocationBytes, 1);
  monitor.Stop();
  const ProcessMemoryMonitor::Peaks peaks = monitor.GetPeaks();

  EXPECT_EQ(buffer.back(), 1);
  EXPECT_EQ(peaks.resident_mb.has_value(), IsHostMemorySupported());
  EXPECT_EQ(peaks.private_mb.has_value(), IsHostMemorySupported());
  if (IsHostMemorySupported()) {
    EXPECT_GE(*peaks.resident_mb, kAllocationMb);
    EXPECT_GE(*peaks.private_mb, kAllocationMb);
  }
  EXPECT_EQ(peaks.accelerator.has_value(),
            GetProcessAcceleratorMemoryUsage().has_value());
}

TEST(ProcessMemoryMonitorTest, StopIsIdempotentAndKeepsPeaks) {
  ProcessMemoryMonitor monitor(kSamplingIntervalMs);
  monitor.Start();
  monitor.Stop();
  const ProcessMemoryMonitor::Peaks first = monitor.GetPeaks();

  monitor.Stop();
  const ProcessMemoryMonitor::Peaks second = monitor.GetPeaks();

  EXPECT_EQ(first.resident_mb, second.resident_mb);
  EXPECT_EQ(first.private_mb, second.private_mb);
}

TEST(ProcessMemoryMonitorTest, StartWhileRunningIsANoOp) {
  ProcessMemoryMonitor monitor(kSamplingIntervalMs);

  monitor.Start();
  monitor.Start();
  monitor.Stop();

  EXPECT_EQ(monitor.GetPeaks().resident_mb.has_value(),
            IsHostMemorySupported());
}

TEST(ProcessMemoryMonitorTest, DestructorStopsRunningMonitor) {
  auto monitor = std::make_unique<ProcessMemoryMonitor>(kSamplingIntervalMs);
  monitor->Start();

  monitor.reset();
}

}  // namespace
}  // namespace litert::lm
