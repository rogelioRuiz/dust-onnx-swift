import XCTest
@testable import DustOnnx
import DustCore

final class ONNXAcceleratorTests: XCTestCase {
    func testO4T1AutoAcceleratorConfigReachesInjectedFactory() throws {
        let fixtureURL = try fixtureURL()
        var capturedAccelerator: String?
        let manager = ONNXSessionManager(
            sessionFactory: { _, modelId, config, priority in
                capturedAccelerator = config.accelerator
                return ONNXSession(
                    sessionId: modelId,
                    metadata: Self.metadata(accelerator: config.accelerator),
                    priority: priority
                )
            }
        )

        let session = try manager.loadModel(
            path: fixtureURL.path,
            modelId: "auto-model",
            config: ONNXConfig(accelerator: "auto"),
            priority: .interactive
        )

        XCTAssertEqual(capturedAccelerator, "auto")
        XCTAssertEqual(session.metadata.accelerator, "auto")
    }

    func testO4T2CpuAcceleratorMetadataMatchesFactory() throws {
        let fixtureURL = try fixtureURL()
        let manager = ONNXSessionManager(
            sessionFactory: { _, modelId, _, priority in
                ONNXSession(
                    sessionId: modelId,
                    metadata: Self.metadata(accelerator: "cpu"),
                    priority: priority
                )
            }
        )

        let session = try manager.loadModel(
            path: fixtureURL.path,
            modelId: "cpu-model",
            config: ONNXConfig(accelerator: "cpu"),
            priority: .interactive
        )

        XCTAssertEqual(session.metadata.accelerator, "cpu")
    }

    func testO4T3CoreMLAcceleratorPropagatesThroughInjectedFactory() throws {
        let fixtureURL = try fixtureURL()
        var capturedAccelerator: String?
        let manager = ONNXSessionManager(
            sessionFactory: { _, modelId, config, priority in
                capturedAccelerator = config.accelerator
                return ONNXSession(
                    sessionId: modelId,
                    metadata: Self.metadata(accelerator: "coreml"),
                    priority: priority
                )
            }
        )

        let session = try manager.loadModel(
            path: fixtureURL.path,
            modelId: "coreml-model",
            config: ONNXConfig(accelerator: "coreml"),
            priority: .interactive
        )

        XCTAssertEqual(capturedAccelerator, "coreml")
        XCTAssertEqual(session.metadata.accelerator, "coreml")
    }

    func testO4T4LoadingSameAutoModelTwiceReusesCachedSession() throws {
        let fixtureURL = try fixtureURL()
        let manager = ONNXSessionManager(
            sessionFactory: { _, modelId, config, priority in
                ONNXSession(
                    sessionId: modelId,
                    metadata: Self.metadata(accelerator: config.accelerator),
                    priority: priority
                )
            }
        )

        let first = try manager.loadModel(
            path: fixtureURL.path,
            modelId: "cached-auto",
            config: ONNXConfig(accelerator: "auto"),
            priority: .interactive
        )
        let second = try manager.loadModel(
            path: fixtureURL.path,
            modelId: "cached-auto",
            config: ONNXConfig(accelerator: "auto"),
            priority: .background
        )

        XCTAssertEqual(ObjectIdentifier(first), ObjectIdentifier(second))
        XCTAssertEqual(manager.refCount(for: "cached-auto"), 2)
    }

    func testO4T5LoadedSessionExposesFactoryResolvedAccelerator() throws {
        let fixtureURL = try fixtureURL()
        let manager = ONNXSessionManager(
            sessionFactory: { _, modelId, _, priority in
                ONNXSession(
                    sessionId: modelId,
                    metadata: Self.metadata(accelerator: "coreml"),
                    priority: priority
                )
            }
        )

        _ = try manager.loadModel(
            path: fixtureURL.path,
            modelId: "metadata-coreml",
            config: ONNXConfig(accelerator: "auto"),
            priority: .interactive
        )

        let metadata = try XCTUnwrap(manager.session(for: "metadata-coreml")?.metadata)
        XCTAssertEqual(metadata.accelerator, "coreml")
    }

    func testO4T6FactoryFailurePropagatesWithoutRetry() throws {
        let fixtureURL = try fixtureURL()
        var callCount = 0
        let manager = ONNXSessionManager(
            sessionFactory: { path, _, _, _ in
                callCount += 1
                throw ONNXError.loadFailed(path: path, detail: "simulated failure")
            }
        )

        XCTAssertThrowsError(
            try manager.loadModel(
                path: fixtureURL.path,
                modelId: "failing-model",
                config: ONNXConfig(accelerator: "auto"),
                priority: .interactive
            )
        ) { error in
            guard case .loadFailed(let path, let detail) = error as? ONNXError else {
                return XCTFail("Expected loadFailed, got \(error)")
            }
            XCTAssertEqual(path, fixtureURL.path)
            XCTAssertEqual(detail, "simulated failure")
        }

        XCTAssertEqual(callCount, 1)
        XCTAssertNil(manager.session(for: "failing-model"))
    }

    func testO4T7MetalAcceleratorPassesThroughToInjectedFactory() throws {
        let fixtureURL = try fixtureURL()
        var capturedAccelerator: String?
        let manager = ONNXSessionManager(
            sessionFactory: { _, modelId, config, priority in
                capturedAccelerator = config.accelerator
                return ONNXSession(
                    sessionId: modelId,
                    metadata: Self.metadata(accelerator: config.accelerator),
                    priority: priority
                )
            }
        )

        let session = try manager.loadModel(
            path: fixtureURL.path,
            modelId: "metal-model",
            config: ONNXConfig(accelerator: "metal"),
            priority: .interactive
        )

        XCTAssertEqual(capturedAccelerator, "metal")
        XCTAssertEqual(session.metadata.accelerator, "metal")
    }

    func testO4T8LegacyLoadPathCachesInjectedAcceleratorMetadata() throws {
        let fixtureURL = try fixtureURL()
        let manager = ONNXSessionManager(
            sessionFactory: { _, modelId, _, priority in
                ONNXSession(
                    sessionId: modelId,
                    metadata: Self.metadata(accelerator: "coreml"),
                    priority: priority
                )
            }
        )

        let session = try manager.loadModel(
            path: fixtureURL.path,
            modelId: "legacy-coreml",
            config: ONNXConfig(accelerator: "coreml"),
            priority: .interactive
        )

        XCTAssertEqual(session.metadata.accelerator, "coreml")
        XCTAssertEqual(manager.session(for: "legacy-coreml")?.metadata.accelerator, "coreml")
    }

    func testO4T9CpuAcceleratorRemainsVisibleOnCachedSessionLookup() throws {
        let fixtureURL = try fixtureURL()
        let manager = ONNXSessionManager(
            sessionFactory: { _, modelId, _, priority in
                ONNXSession(
                    sessionId: modelId,
                    metadata: Self.metadata(accelerator: "cpu"),
                    priority: priority
                )
            }
        )

        _ = try manager.loadModel(
            path: fixtureURL.path,
            modelId: "cached-cpu",
            config: ONNXConfig(accelerator: "cpu"),
            priority: .interactive
        )

        let session = try XCTUnwrap(manager.session(for: "cached-cpu"))
        XCTAssertEqual(session.metadata.accelerator, "cpu")
    }

    private func fixtureURL() throws -> URL {
        if let bundled = Bundle.module.url(forResource: "tiny-test", withExtension: "onnx") {
            return bundled
        }

        let fallback = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appendingPathComponent("Fixtures/tiny-test.onnx")

        if FileManager.default.fileExists(atPath: fallback.path) {
            return fallback
        }

        throw XCTSkip("tiny-test.onnx fixture was not found")
    }

    private static func metadata(accelerator: String) -> ONNXModelMetadataValue {
        ONNXModelMetadataValue(
            inputs: [
                ONNXTensorMetadata(name: "input_a", shape: [1, 3], dtype: "float32"),
            ],
            outputs: [
                ONNXTensorMetadata(name: "output", shape: [1, 3], dtype: "float32"),
            ],
            accelerator: accelerator,
            opset: 13
        )
    }
}
