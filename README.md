<p align="center">
  <img alt="dust" src="assets/dust_banner.png" width="400">
</p>

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

Copyright 2026 T6X. Licensed under the [Apache License 2.0](LICENSE).
