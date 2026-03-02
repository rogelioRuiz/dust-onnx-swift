<p align="center">
  <img alt="dust" src="assets/dust_banner.png" width="400">
</p>

<p align="center">
  <strong>Device Unified Serving Toolkit</strong><br>
  <a href="https://github.com/rogelioRuiz/dust">dust ecosystem</a> · v0.1.0 · Apache 2.0
</p>

<p align="center">
  <a href="https://github.com/rogelioRuiz/dust/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache_2.0-blue.svg"></a>
  <img alt="Version" src="https://img.shields.io/badge/version-0.1.0-informational">
  <img alt="SPM" src="https://img.shields.io/badge/SPM-DustOnnx-F05138">
  <img alt="CocoaPods" src="https://img.shields.io/badge/CocoaPods-DustOnnx-EE3322">
  <a href="https://swift.org"><img alt="Swift" src="https://img.shields.io/badge/Swift-5.9-orange.svg"></a>
  <img alt="Platforms" src="https://img.shields.io/badge/Platforms-iOS_16+_|_macOS_13+-lightgrey">
  <img alt="ONNX" src="https://img.shields.io/badge/ONNX_Runtime-1.20-005CED">
</p>

---

<p align="center">
<strong>dust ecosystem</strong> —
<a href="../capacitor-core/README.md">capacitor-core</a> ·
<a href="../capacitor-llm/README.md">capacitor-llm</a> ·
<a href="../capacitor-onnx/README.md">capacitor-onnx</a> ·
<a href="../capacitor-serve/README.md">capacitor-serve</a> ·
<a href="../capacitor-embeddings/README.md">capacitor-embeddings</a>
<br>
<a href="../dust-core-kotlin/README.md">dust-core-kotlin</a> ·
<a href="../dust-llm-kotlin/README.md">dust-llm-kotlin</a> ·
<a href="../dust-onnx-kotlin/README.md">dust-onnx-kotlin</a> ·
<a href="../dust-embeddings-kotlin/README.md">dust-embeddings-kotlin</a> ·
<a href="../dust-serve-kotlin/README.md">dust-serve-kotlin</a>
<br>
<a href="../dust-core-swift/README.md">dust-core-swift</a> ·
<a href="../dust-llm-swift/README.md">dust-llm-swift</a> ·
<strong>dust-onnx-swift</strong> ·
<a href="../dust-embeddings-swift/README.md">dust-embeddings-swift</a> ·
<a href="../dust-serve-swift/README.md">dust-serve-swift</a>
</p>

---

# dust-onnx-swift

Standalone ONNX runtime session management and preprocessing for Dust — iOS/macOS.

**Version: 0.1.0**

## Overview

Wraps the ONNX Runtime for Apple platforms behind the [dust-core-swift](../dust-core-swift) protocols. Handles session lifecycle, image preprocessing, accelerator selection, and inference pipeline orchestration. Requires iOS 16+ / macOS 13+.

```
dust-onnx-swift/
├── Package.swift                           # SPM: product "DustOnnx", iOS 16+ / macOS 13+
├── DustOnnx.podspec                        # CocoaPods spec (module name: DustOnnx)
├── Sources/DustOnnx/
│   ├── ONNXSession.swift
│   ├── ONNXInferenceEngine.swift
│   ├── ImagePreprocessor.swift
│   ├── AcceleratorSelector.swift
│   ├── ONNXRegistry.swift
│   └── ONNXPipeline.swift
└── Tests/DustOnnxTests/
    └── Fixtures/
        └── tiny-test.onnx                  # Minimal model for integration tests
```

## Install

### Swift Package Manager — local

```swift
// Package.swift
dependencies: [
    .package(name: "dust-onnx-swift", path: "../dust-onnx-swift"),
],
targets: [
    .target(
        name: "MyTarget",
        dependencies: [
            .product(name: "DustOnnx", package: "dust-onnx-swift"),
        ]
    )
]
```

### Swift Package Manager — remote (when published)

```swift
.package(url: "https://github.com/rogelioRuiz/dust-onnx-swift.git", from: "0.1.0")
```

### CocoaPods

```ruby
pod 'DustOnnx', '~> 0.1'
```

## Dependencies

- [dust-core-swift](../dust-core-swift) (DustCore)

## Usage

```swift
import DustOnnx

// 1. Select accelerator
let accelerator = AcceleratorSelector.best()

// 2. Open a session
let session = try ONNXSession(modelPath: modelURL, accelerator: accelerator)

// 3. Preprocess an image
let tensor = ImagePreprocessor.imageToTensor(cgImage, width: 224, height: 224)

// 4. Run inference
let engine = ONNXInferenceEngine(session: session)
let outputs = try await engine.run(inputs: [tensor])

// 5. Clean up
session.close()
```

## Test

```bash
cd dust-onnx-swift
swift test    # 51 XCTest tests
```

Tests use the bundled `tiny-test.onnx` fixture. Requires macOS with Swift toolchain — no Xcode project needed.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, coding conventions, and PR guidelines.

## License

Copyright 2026 Rogelio Ruiz Perez. Licensed under the [Apache License 2.0](LICENSE).
