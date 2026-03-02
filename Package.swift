// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "dust-onnx-swift",
    platforms: [.iOS(.v16), .macOS(.v13)],
    products: [
        .library(
            name: "DustOnnx",
            targets: ["DustOnnx"]
        )
    ],
    dependencies: [
        .package(url: "https://github.com/rogelioRuiz/dust-core-swift.git", from: "0.1.0"),
        .package(url: "https://github.com/microsoft/onnxruntime-swift-package-manager.git", from: "1.20.0"),
    ],
    targets: [
        .target(
            name: "DustOnnx",
            dependencies: [
                .product(name: "DustCore", package: "dust-core-swift"),
                .product(name: "onnxruntime", package: "onnxruntime-swift-package-manager"),
            ],
            path: "Sources/DustOnnx"
        ),
        .testTarget(
            name: "DustOnnxTests",
            dependencies: ["DustOnnx"],
            path: "Tests/DustOnnxTests",
            resources: [
                .copy("Fixtures/tiny-test.onnx"),
            ]
        )
    ],
    swiftLanguageVersions: [.v5]
)
