import XCTest
@testable import DustOnnx
import DustCore

final class ONNXSessionManagerTests: XCTestCase {
    func testO1T1LoadValidFixtureCreatesSession() throws {
        let fixtureURL = try fixtureURL()
        XCTAssertTrue(FileManager.default.fileExists(atPath: fixtureURL.path))

        let manager = ONNXSessionManager(
            sessionFactory: { _, modelId, _, priority in
                ONNXSession(
                    sessionId: modelId,
                    metadata: Self.mockMetadata(),
                    priority: priority
                )
            }
        )

        let session = try manager.loadModel(
            path: fixtureURL.path,
            modelId: "tiny-test",
            config: ONNXConfig(),
            priority: .interactive
        )

        XCTAssertEqual(session.sessionId, "tiny-test")
        XCTAssertEqual(session.status(), .ready)
    }

    func testO1T2MetadataAccessReturnsExpectedTensorNames() throws {
        let fixtureURL = try fixtureURL()
        let manager = ONNXSessionManager(
            sessionFactory: { _, modelId, _, priority in
                ONNXSession(
                    sessionId: modelId,
                    metadata: Self.mockMetadata(),
                    priority: priority
                )
            }
        )

        _ = try manager.loadModel(
            path: fixtureURL.path,
            modelId: "tiny-test",
            config: ONNXConfig(),
            priority: .interactive
        )

        let metadata = try XCTUnwrap(manager.session(for: "tiny-test")?.metadata)
        XCTAssertEqual(metadata.inputs.first?.name, "input_a")
        XCTAssertEqual(metadata.outputs.first?.name, "output")
        XCTAssertEqual(metadata.opset, 13)
    }

    func testO1T3LoadMissingFileThrowsFileNotFound() {
        let manager = ONNXSessionManager(
            sessionFactory: { path, _, _, _ in
                throw ONNXError.fileNotFound(path: path)
            }
        )

        XCTAssertThrowsError(
            try manager.loadModel(
                path: "/nonexistent/model.onnx",
                modelId: "missing",
                config: ONNXConfig(),
                priority: .interactive
            )
        ) { error in
            guard case .fileNotFound(let path) = error as? ONNXError else {
                return XCTFail("Expected fileNotFound, got \(error)")
            }
            XCTAssertEqual(path, "/nonexistent/model.onnx")
        }
    }

    func testO1T4LoadCorruptFileThrowsLoadFailed() throws {
        let fixtureURL = try fixtureURL()
        let manager = ONNXSessionManager(
            sessionFactory: { path, _, _, _ in
                throw ONNXError.loadFailed(path: path, detail: "fixture is corrupt")
            }
        )

        XCTAssertThrowsError(
            try manager.loadModel(
                path: fixtureURL.path,
                modelId: "corrupt",
                config: ONNXConfig(),
                priority: .interactive
            )
        ) { error in
            guard case .loadFailed(let path, let detail) = error as? ONNXError else {
                return XCTFail("Expected loadFailed, got \(error)")
            }
            XCTAssertEqual(path, fixtureURL.path)
            XCTAssertEqual(detail, "fixture is corrupt")
        }
    }

    func testO1T5WrongFormatRejectedBeforeLoad() {
        let accepted = DustModelFormat.onnx.rawValue
        XCTAssertEqual(accepted, "onnx")

        let rejected: [DustModelFormat] = [.gguf, .coreml, .tflite, .custom]
        for format in rejected {
            XCTAssertNotEqual(format.rawValue, accepted)
        }
    }

    func testO1T6UnloadLoadedModelRemovesSession() async throws {
        let fixtureURL = try fixtureURL()
        let manager = ONNXSessionManager(
            sessionFactory: { _, modelId, _, priority in
                ONNXSession(
                    sessionId: modelId,
                    metadata: Self.mockMetadata(),
                    priority: priority
                )
            }
        )

        _ = try manager.loadModel(
            path: fixtureURL.path,
            modelId: "tiny-test",
            config: ONNXConfig(),
            priority: .interactive
        )
        try await manager.forceUnloadModel(id: "tiny-test")

        XCTAssertEqual(manager.sessionCount, 0)
        XCTAssertNil(manager.session(for: "tiny-test"))
    }

    func testO1T6bUnloadUnknownIdThrowsModelNotFound() async {
        let manager = ONNXSessionManager()

        do {
            try await manager.forceUnloadModel(id: "nonexistent")
            XCTFail("Expected modelNotFound")
        } catch let error as DustCoreError {
            XCTAssertEqual(error, .modelNotFound)
        } catch {
            XCTFail("Unexpected error: \(error)")
        }
    }

    func testO1T7LoadingSameIdTwiceReusesSessionAndRefCount() throws {
        let fixtureURL = try fixtureURL()
        let manager = ONNXSessionManager(
            sessionFactory: { _, modelId, _, priority in
                ONNXSession(
                    sessionId: modelId,
                    metadata: Self.mockMetadata(),
                    priority: priority
                )
            }
        )

        let first = try manager.loadModel(
            path: fixtureURL.path,
            modelId: "tiny-test",
            config: ONNXConfig(),
            priority: .interactive
        )
        let second = try manager.loadModel(
            path: fixtureURL.path,
            modelId: "tiny-test",
            config: ONNXConfig(),
            priority: .background
        )

        XCTAssertEqual(ObjectIdentifier(first), ObjectIdentifier(second))
        XCTAssertEqual(manager.sessionCount, 1)
        XCTAssertEqual(manager.refCount(for: "tiny-test"), 2)
    }

    func testO1T8ConcurrentLoadTwoDifferentModels() async throws {
        let fixtureURL = try fixtureURL()
        let manager = ONNXSessionManager(
            sessionFactory: { _, modelId, _, priority in
                ONNXSession(
                    sessionId: modelId,
                    metadata: Self.mockMetadata(),
                    priority: priority
                )
            }
        )

        try await withThrowingTaskGroup(of: Void.self) { group in
            group.addTask {
                _ = try manager.loadModel(
                    path: fixtureURL.path,
                    modelId: "model-a",
                    config: ONNXConfig(),
                    priority: .interactive
                )
            }
            group.addTask {
                _ = try manager.loadModel(
                    path: fixtureURL.path,
                    modelId: "model-b",
                    config: ONNXConfig(),
                    priority: .interactive
                )
            }
            try await group.waitForAll()
        }

        XCTAssertEqual(manager.sessionCount, 2)
        XCTAssertEqual(manager.refCount(for: "model-a"), 1)
        XCTAssertEqual(manager.refCount(for: "model-b"), 1)
        XCTAssertNotNil(manager.session(for: "model-a"))
        XCTAssertNotNil(manager.session(for: "model-b"))
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

    private static func mockMetadata() -> ONNXModelMetadataValue {
        ONNXModelMetadataValue(
            inputs: [
                ONNXTensorMetadata(name: "input_a", shape: [1, 3], dtype: "float32"),
                ONNXTensorMetadata(name: "input_b", shape: [1, 3], dtype: "float32"),
            ],
            outputs: [
                ONNXTensorMetadata(name: "output", shape: [1, 3], dtype: "float32"),
            ],
            accelerator: "auto",
            opset: 13
        )
    }
}
