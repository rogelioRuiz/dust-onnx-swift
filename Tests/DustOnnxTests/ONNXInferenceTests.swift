import XCTest
@testable import DustOnnx
import DustCore

final class ONNXInferenceTests: XCTestCase {
    func testO2T1RunInferenceFloat32ReturnsOutput() throws {
        let engine = MockONNXEngine()
        engine.outputTensors = [
            "output": TensorData(name: "output", dtype: "float32", shape: [1, 3], data: [5, 7, 9]),
        ]

        let session = makeSession(engine: engine)
        let outputs = try session.runInference(
            inputs: [
                "input_a": TensorData(name: "input_a", dtype: "float32", shape: [1, 3], data: [1, 2, 3]),
                "input_b": TensorData(name: "input_b", dtype: "float32", shape: [1, 3], data: [4, 5, 6]),
            ],
            outputNames: nil
        )

        XCTAssertEqual(outputs["output"]?.shape, [1, 3])
        XCTAssertEqual(outputs["output"]?.dtype, "float32")
        XCTAssertEqual(engine.lastInputs?["input_a"]?.data, [1, 2, 3])
    }

    func testO2T2RunInferenceUInt8ReturnsOutput() throws {
        let engine = MockONNXEngine(
            inputMetadata: [
                ONNXTensorMetadata(name: "pixels", shape: [1, 4], dtype: "uint8"),
            ],
            outputMetadata: [
                ONNXTensorMetadata(name: "output", shape: [1, 4], dtype: "uint8"),
            ]
        )
        engine.outputTensors = [
            "output": TensorData(name: "output", dtype: "uint8", shape: [1, 4], data: [8, 16, 32, 64]),
        ]

        let session = makeSession(
            engine: engine,
            metadata: ONNXModelMetadataValue(
                inputs: engine.inputMetadata,
                outputs: engine.outputMetadata,
                accelerator: "auto",
                opset: 13
            )
        )

        let outputs = try session.runInference(
            inputs: [
                "pixels": TensorData(name: "pixels", dtype: "uint8", shape: [1, 4], data: [1, 2, 3, 4]),
            ],
            outputNames: nil
        )

        XCTAssertEqual(outputs["output"]?.dtype, "uint8")
        XCTAssertEqual(engine.lastInputs?["pixels"]?.dtype, "uint8")
    }

    func testO2T3ShapeMismatchWrongRankThrowsShapeError() {
        let session = makeSession()

        XCTAssertThrowsError(
            try session.runInference(
                inputs: [
                    "input_a": TensorData(name: "input_a", dtype: "float32", shape: [3], data: [1, 2, 3]),
                    "input_b": TensorData(name: "input_b", dtype: "float32", shape: [1, 3], data: [4, 5, 6]),
                ],
                outputNames: nil
            )
        ) { error in
            guard case .shapeError(let name, let expected, let got) = error as? ONNXError else {
                return XCTFail("Expected shapeError, got \(error)")
            }
            XCTAssertEqual(name, "input_a")
            XCTAssertEqual(expected, [1, 3])
            XCTAssertEqual(got, [3])
        }
    }

    func testO2T4ShapeMismatchWrongStaticDimensionThrowsShapeError() {
        let session = makeSession()

        XCTAssertThrowsError(
            try session.runInference(
                inputs: [
                    "input_a": TensorData(name: "input_a", dtype: "float32", shape: [1, 4], data: [1, 2, 3, 4]),
                    "input_b": TensorData(name: "input_b", dtype: "float32", shape: [1, 3], data: [4, 5, 6]),
                ],
                outputNames: nil
            )
        ) { error in
            guard case .shapeError(let name, let expected, let got) = error as? ONNXError else {
                return XCTFail("Expected shapeError, got \(error)")
            }
            XCTAssertEqual(name, "input_a")
            XCTAssertEqual(expected, [1, 3])
            XCTAssertEqual(got, [1, 4])
        }
    }

    func testO2T5DynamicDimensionAcceptsAnySize() throws {
        let engine = MockONNXEngine(
            inputMetadata: [
                ONNXTensorMetadata(name: "tokens", shape: [-1, 3], dtype: "float32"),
            ],
            outputMetadata: [
                ONNXTensorMetadata(name: "output", shape: [-1, 3], dtype: "float32"),
            ]
        )
        engine.outputTensors = [
            "output": TensorData(name: "output", dtype: "float32", shape: [2, 3], data: [1, 2, 3, 4, 5, 6]),
        ]

        let session = makeSession(
            engine: engine,
            metadata: ONNXModelMetadataValue(
                inputs: engine.inputMetadata,
                outputs: engine.outputMetadata,
                accelerator: "auto",
                opset: 13
            )
        )

        let outputs = try session.runInference(
            inputs: [
                "tokens": TensorData(name: "tokens", dtype: "float32", shape: [2, 3], data: [1, 2, 3, 4, 5, 6]),
            ],
            outputNames: nil
        )

        XCTAssertEqual(outputs["output"]?.shape, [2, 3])
    }

    func testO2T6DtypeMismatchThrowsDtypeError() {
        let session = makeSession()

        XCTAssertThrowsError(
            try session.runInference(
                inputs: [
                    "input_a": TensorData(name: "input_a", dtype: "int32", shape: [1, 3], data: [1, 2, 3]),
                    "input_b": TensorData(name: "input_b", dtype: "float32", shape: [1, 3], data: [4, 5, 6]),
                ],
                outputNames: nil
            )
        ) { error in
            guard case .dtypeError(let name, let expected, let got) = error as? ONNXError else {
                return XCTFail("Expected dtypeError, got \(error)")
            }
            XCTAssertEqual(name, "input_a")
            XCTAssertEqual(expected, "float32")
            XCTAssertEqual(got, "int32")
        }
    }

    func testO2T7OutputNamesFiltersSubset() throws {
        let engine = MockONNXEngine(
            outputMetadata: [
                ONNXTensorMetadata(name: "first", shape: [1], dtype: "float32"),
                ONNXTensorMetadata(name: "second", shape: [1], dtype: "float32"),
            ]
        )
        engine.outputTensors = [
            "first": TensorData(name: "first", dtype: "float32", shape: [1], data: [1]),
            "second": TensorData(name: "second", dtype: "float32", shape: [1], data: [2]),
        ]

        let session = makeSession(
            engine: engine,
            metadata: ONNXModelMetadataValue(
                inputs: engine.inputMetadata,
                outputs: engine.outputMetadata,
                accelerator: "auto",
                opset: 13
            )
        )

        let outputs = try session.runInference(
            inputs: [
                "input_a": TensorData(name: "input_a", dtype: "float32", shape: [1, 3], data: [1, 2, 3]),
                "input_b": TensorData(name: "input_b", dtype: "float32", shape: [1, 3], data: [4, 5, 6]),
            ],
            outputNames: ["second"]
        )

        XCTAssertEqual(Array(outputs.keys), ["second"])
        XCTAssertEqual(outputs["second"]?.data, [2])
    }

    func testO2T8RunInferenceOnUnloadedModelMapsToModelNotFound() async throws {
        let temporaryURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("onnx")
        XCTAssertTrue(FileManager.default.createFile(atPath: temporaryURL.path, contents: Data(), attributes: nil))
        defer { try? FileManager.default.removeItem(at: temporaryURL) }

        let engine = MockONNXEngine()
        let manager = ONNXSessionManager(
            sessionFactory: { _, modelId, _, priority in
                ONNXSession(
                    sessionId: modelId,
                    engine: engine,
                    metadata: Self.defaultMetadata(),
                    priority: priority
                )
            }
        )

        _ = try manager.loadModel(
            path: temporaryURL.path,
            modelId: "tiny-test",
            config: ONNXConfig(),
            priority: .interactive
        )
        try await manager.forceUnloadModel(id: "tiny-test")

        do {
            guard manager.session(for: "tiny-test") != nil else {
                throw DustCoreError.modelNotFound
            }
            XCTFail("Expected modelNotFound")
        } catch let error as DustCoreError {
            XCTAssertEqual(error, .modelNotFound)
        } catch {
            XCTFail("Unexpected error: \(error)")
        }
    }

    func testO2T9EngineThrowsDuringRunMapsToInferenceError() {
        let engine = MockONNXEngine()
        engine.runError = NSError(domain: "ONNXInferenceTests", code: 1, userInfo: [
            NSLocalizedDescriptionKey: "mock engine failure",
        ])

        let session = makeSession(engine: engine)

        XCTAssertThrowsError(
            try session.runInference(
                inputs: [
                    "input_a": TensorData(name: "input_a", dtype: "float32", shape: [1, 3], data: [1, 2, 3]),
                    "input_b": TensorData(name: "input_b", dtype: "float32", shape: [1, 3], data: [4, 5, 6]),
                ],
                outputNames: nil
            )
        ) { error in
            guard case .inferenceError(let detail) = error as? ONNXError else {
                return XCTFail("Expected inferenceError, got \(error)")
            }
            XCTAssertEqual(detail, "mock engine failure")
        }
    }

    private func makeSession(
        engine: MockONNXEngine = MockONNXEngine(),
        metadata: ONNXModelMetadataValue = ONNXInferenceTests.defaultMetadata()
    ) -> ONNXSession {
        ONNXSession(
            sessionId: "tiny-test",
            engine: engine,
            metadata: metadata,
            priority: .interactive
        )
    }

    private static func defaultMetadata() -> ONNXModelMetadataValue {
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

private final class MockONNXEngine: ONNXEngine {
    let inputMetadata: [ONNXTensorMetadata]
    let outputMetadata: [ONNXTensorMetadata]
    let accelerator: String

    var outputTensors: [String: TensorData] = [
        "output": TensorData(name: "output", dtype: "float32", shape: [1, 3], data: [5, 7, 9]),
    ]
    var runError: Error?
    private(set) var lastInputs: [String: TensorData]?
    private(set) var closeCallCount = 0

    init(
        inputMetadata: [ONNXTensorMetadata] = [
            ONNXTensorMetadata(name: "input_a", shape: [1, 3], dtype: "float32"),
            ONNXTensorMetadata(name: "input_b", shape: [1, 3], dtype: "float32"),
        ],
        outputMetadata: [ONNXTensorMetadata] = [
            ONNXTensorMetadata(name: "output", shape: [1, 3], dtype: "float32"),
        ],
        accelerator: String = "auto"
    ) {
        self.inputMetadata = inputMetadata
        self.outputMetadata = outputMetadata
        self.accelerator = accelerator
    }

    func run(inputs: [String: TensorData]) throws -> [String: TensorData] {
        lastInputs = inputs
        if let runError {
            throw runError
        }
        return outputTensors
    }

    func close() {
        closeCallCount += 1
    }
}
