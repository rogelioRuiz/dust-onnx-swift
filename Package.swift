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
        .package(name: "dust-core-swift", path: "../dust-core-swift"),
    ],
    targets: [
        .target(
            name: "DustOnnx",
            dependencies: [
                .product(name: "DustCore", package: "dust-core-swift"),
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
