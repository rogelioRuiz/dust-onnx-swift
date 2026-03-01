import XCTest
@testable import DustOnnx
import DustCore

final class ONNXRegistryTests: XCTestCase {
    override func setUp() {
        super.setUp()
        DustCoreRegistry.shared.resetForTesting()
    }

    override func tearDown() {
        DustCoreRegistry.shared.resetForTesting()
        super.tearDown()
    }

    func testO5T1RegistryRegistrationMakesManagerResolvable() throws {
        let manager = makeManager()

        DustCoreRegistry.shared.register(modelServer: manager)

        let resolved = try DustCoreRegistry.shared.resolveModelServer()
        XCTAssertTrue((resolved as AnyObject) === manager)
    }

    func testO5T2LoadModelForReadyDescriptorCreatesSessionAndRefCount() async throws {
        let fileURL = try makeTempModelFile()
        defer { try? FileManager.default.removeItem(at: fileURL) }

        let manager = makeManager()
        let descriptor = makeDescriptor(id: "model-a", path: fileURL.path)
        manager.register(descriptor: descriptor)

        let session = try await manager.loadModel(descriptor: descriptor, priority: .interactive)

        XCTAssertEqual(session.status(), .ready)
        XCTAssertEqual(manager.refCount(for: "model-a"), 1)
    }

    func testO5T3LoadModelForNotLoadedDescriptorThrowsModelNotReady() async throws {
        let manager = makeManager()
        let descriptor = makeDescriptor(id: "model-a", path: "/tmp/missing-model.onnx")
        manager.register(descriptor: descriptor)

        do {
            _ = try await manager.loadModel(descriptor: descriptor, priority: .interactive)
            XCTFail("Expected modelNotReady")
        } catch let error as DustCoreError {
            XCTAssertEqual(error, .modelNotReady)
        }
    }

    func testO5T4LoadModelForUnregisteredIdThrowsModelNotFound() async {
        let manager = makeManager()
        let descriptor = makeDescriptor(id: "ghost", path: "/tmp/ghost.onnx")

        do {
            _ = try await manager.loadModel(descriptor: descriptor, priority: .interactive)
            XCTFail("Expected modelNotFound")
        } catch let error as DustCoreError {
            XCTAssertEqual(error, .modelNotFound)
        } catch {
            XCTFail("Unexpected error: \(error)")
        }
    }

    func testO5T5UnloadModelDecrementsRefCountAndKeepsSessionCached() async throws {
        let fileURL = try makeTempModelFile()
        defer { try? FileManager.default.removeItem(at: fileURL) }

        let manager = makeManager()
        let descriptor = makeDescriptor(id: "model-a", path: fileURL.path)
        manager.register(descriptor: descriptor)

        _ = try await manager.loadModel(descriptor: descriptor, priority: .interactive)
        try await manager.unloadModel(id: "model-a")

        XCTAssertEqual(manager.refCount(for: "model-a"), 0)
        XCTAssertTrue(manager.hasCachedSession(for: "model-a"))
    }

    func testO5T6LoadModelTwiceReusesSameSessionAndIncrementsRefCount() async throws {
        let fileURL = try makeTempModelFile()
        defer { try? FileManager.default.removeItem(at: fileURL) }

        let manager = makeManager()
        let descriptor = makeDescriptor(id: "model-a", path: fileURL.path)
        manager.register(descriptor: descriptor)

        let first = try await manager.loadModel(descriptor: descriptor, priority: .interactive)
        let second = try await manager.loadModel(descriptor: descriptor, priority: .background)

        XCTAssertEqual(identity(of: first), identity(of: second))
        XCTAssertEqual(manager.refCount(for: "model-a"), 2)
    }

    func testO5T7EvictUnderPressureStandardRemovesBackgroundZeroRefSessions() async throws {
        let fileA = try makeTempModelFile()
        let fileB = try makeTempModelFile()
        defer {
            try? FileManager.default.removeItem(at: fileA)
            try? FileManager.default.removeItem(at: fileB)
        }

        let manager = makeManager()
        let descriptorA = makeDescriptor(id: "model-a", path: fileA.path)
        let descriptorB = makeDescriptor(id: "model-b", path: fileB.path)
        manager.register(descriptor: descriptorA)
        manager.register(descriptor: descriptorB)

        let loadedA = try await manager.loadModel(descriptor: descriptorA, priority: .background)
        let loadedB = try await manager.loadModel(descriptor: descriptorB, priority: .interactive)
        let sessionA = try XCTUnwrap(loadedA as? ONNXSession)
        let sessionB = try XCTUnwrap(loadedB as? ONNXSession)

        try await manager.unloadModel(id: "model-a")
        try await manager.unloadModel(id: "model-b")
        await manager.evictUnderPressure(level: .standard)

        XCTAssertTrue(sessionA.isModelEvicted)
        XCTAssertFalse(manager.hasCachedSession(for: "model-a"))
        XCTAssertFalse(sessionB.isModelEvicted)
        XCTAssertTrue(manager.hasCachedSession(for: "model-b"))
    }

    func testO5T8EvictUnderPressureCriticalRemovesAllZeroRefSessions() async throws {
        let fileURL = try makeTempModelFile()
        defer { try? FileManager.default.removeItem(at: fileURL) }

        let manager = makeManager()
        let descriptor = makeDescriptor(id: "model-a", path: fileURL.path)
        manager.register(descriptor: descriptor)

        let loaded = try await manager.loadModel(descriptor: descriptor, priority: .interactive)
        let session = try XCTUnwrap(loaded as? ONNXSession)
        try await manager.unloadModel(id: "model-a")
        await manager.evictUnderPressure(level: .critical)

        XCTAssertTrue(session.isModelEvicted)
        XCTAssertFalse(manager.hasCachedSession(for: "model-a"))
    }

    func testO5T9AllModelIdsReturnsOnlyLiveSessionsAfterEviction() async throws {
        let fileA = try makeTempModelFile()
        let fileB = try makeTempModelFile()
        defer {
            try? FileManager.default.removeItem(at: fileA)
            try? FileManager.default.removeItem(at: fileB)
        }

        let manager = makeManager()
        let descriptorA = makeDescriptor(id: "model-a", path: fileA.path)
        let descriptorB = makeDescriptor(id: "model-b", path: fileB.path)
        manager.register(descriptor: descriptorA)
        manager.register(descriptor: descriptorB)

        _ = try await manager.loadModel(descriptor: descriptorA, priority: .interactive)
        _ = try await manager.loadModel(descriptor: descriptorB, priority: .interactive)

        try await manager.unloadModel(id: "model-a")
        _ = await manager.evict(modelId: "model-a")

        XCTAssertEqual(manager.allModelIds(), ["model-b"])
    }

    private func makeManager() -> ONNXSessionManager {
        ONNXSessionManager(
            sessionFactory: { _, modelId, _, priority in
                ONNXSession(
                    sessionId: modelId,
                    metadata: Self.mockMetadata(),
                    priority: priority
                )
            }
        )
    }

    private func makeDescriptor(
        id: String,
        path: String
    ) -> DustModelDescriptor {
        DustModelDescriptor(
            id: id,
            name: id,
            format: .onnx,
            sizeBytes: 1,
            version: "1.0.0",
            metadata: ["localPath": path]
        )
    }

    private func makeTempModelFile() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".onnx")
        try Data([0x08, 0x01, 0x12, 0x00]).write(to: url)
        return url
    }

    private static func mockMetadata() -> ONNXModelMetadataValue {
        ONNXModelMetadataValue(
            inputs: [
                ONNXTensorMetadata(name: "input", shape: [1, 3], dtype: "float32"),
            ],
            outputs: [
                ONNXTensorMetadata(name: "output", shape: [1, 3], dtype: "float32"),
            ],
            accelerator: "cpu",
            opset: 13
        )
    }
}

private func identity(of session: any DustModelSession) -> ObjectIdentifier {
    ObjectIdentifier(session as AnyObject)
}
