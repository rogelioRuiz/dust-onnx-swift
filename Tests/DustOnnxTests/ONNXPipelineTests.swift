import XCTest
@testable import DustOnnx
import DustCore

final class ONNXPipelineTests: XCTestCase {
    func testO6T1TwoStepPipelineBothResultsReturned() throws {
        let engine = ScriptedMockONNXEngine()
        engine.scriptedOutputs = [
            [
                "output": TensorData(name: "output", dtype: "float32", shape: [1, 3], data: [1, 2, 3]),
            ],
            [
                "output": TensorData(name: "output", dtype: "float32", shape: [1, 2], data: [4, 5]),
            ],
        ]

        let session = makeSession(engine: engine)
        let results = try session.runPipeline(steps: [
            literalStep(data: [1, 2, 3]),
            literalStep(data: [9, 8, 7]),
        ])

        XCTAssertEqual(results.count, 2)
        XCTAssertEqual(results[0]["output"]?.shape, [1, 3])
        XCTAssertEqual(results[1]["output"]?.shape, [1, 2])
        XCTAssertEqual(engine.callCount, 2)
    }

    func testO6T2PreviousOutputChainingSubstitutesCorrectly() throws {
        let engine = ScriptedMockONNXEngine(
            inputMetadata: [
                ONNXTensorMetadata(name: "output", shape: [1, 3], dtype: "float32"),
            ]
        )
        engine.scriptedOutputs = [
            [
                "output": TensorData(name: "output", dtype: "float32", shape: [1, 3], data: [9, 8, 7]),
            ],
            [
                "output": TensorData(name: "output", dtype: "float32", shape: [1, 3], data: [1, 1, 1]),
            ],
        ]

        let session = makeSession(engine: engine, metadata: Self.metadata(for: engine))
        _ = try session.runPipeline(steps: [
            literalStep(inputName: "output", data: [1, 2, 3]),
            PipelineStep(inputs: [.previousOutput(name: "output")], outputNames: nil),
        ])

        XCTAssertEqual(engine.allInputs[1]["output"]?.data, [9, 8, 7])
    }

    func testO6T3ExplicitFromStepChainingRoutesCorrectTensor() throws {
        let engine = ScriptedMockONNXEngine()
        engine.scriptedOutputs = [
            [
                "output": TensorData(name: "output", dtype: "float32", shape: [1, 3], data: [3, 2, 1]),
            ],
            [
                "output": TensorData(name: "output", dtype: "float32", shape: [1, 3], data: [0, 0, 0]),
            ],
        ]

        let session = makeSession(engine: engine)
        _ = try session.runPipeline(steps: [
            literalStep(data: [1, 2, 3]),
            PipelineStep(
                inputs: [
                    .stepReference(name: "input", fromStep: 0, outputName: "output"),
                ],
                outputNames: nil
            ),
        ])

        XCTAssertEqual(engine.allInputs[1]["input"]?.data, [3, 2, 1])
    }

    func testO6T4Step1FailsPipelineHaltsWithStepIndex0() {
        let engine = ScriptedMockONNXEngine()
        engine.scriptedErrors = [
            NSError(domain: "ONNXPipelineTests", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "step 0 boom",
            ]),
        ]

        let session = makeSession(engine: engine)

        XCTAssertThrowsError(
            try session.runPipeline(steps: [
                literalStep(data: [1, 2, 3]),
            ])
        ) { error in
            guard case .inferenceError(let detail) = error as? ONNXError else {
                return XCTFail("Expected inferenceError, got \(error)")
            }
            XCTAssertTrue(detail.contains("step 0"))
            XCTAssertEqual(engine.callCount, 1)
        }
    }

    func testO6T5Step2FailsErrorReportsStepIndex1() {
        let engine = ScriptedMockONNXEngine()
        engine.scriptedErrors = [
            nil,
            NSError(domain: "ONNXPipelineTests", code: 2, userInfo: [
                NSLocalizedDescriptionKey: "step 1 boom",
            ]),
        ]

        let session = makeSession(engine: engine)

        XCTAssertThrowsError(
            try session.runPipeline(steps: [
                literalStep(data: [1, 2, 3]),
                literalStep(data: [4, 5, 6]),
            ])
        ) { error in
            guard case .inferenceError(let detail) = error as? ONNXError else {
                return XCTFail("Expected inferenceError, got \(error)")
            }
            XCTAssertTrue(detail.contains("step 1"))
            XCTAssertEqual(engine.callCount, 2)
        }
    }

    func testO6T6SingleStepPipelineMatchesRunInference() throws {
        let expectedOutput: [String: TensorData] = [
            "output": TensorData(name: "output", dtype: "float32", shape: [1, 3], data: [5, 7, 9]),
        ]

        let pipelineEngine = ScriptedMockONNXEngine()
        pipelineEngine.scriptedOutputs = [expectedOutput]

        let directEngine = ScriptedMockONNXEngine()
        directEngine.scriptedOutputs = [expectedOutput]

        let pipelineSession = makeSession(engine: pipelineEngine)
        let directSession = makeSession(engine: directEngine)
        let inputs = [
            "input": TensorData(name: "input", dtype: "float32", shape: [1, 3], data: [1, 2, 3]),
        ]

        let pipelineOutputs = try pipelineSession.runPipeline(steps: [
            PipelineStep(
                inputs: [
                    .literal(inputs["input"]!),
                ],
                outputNames: nil
            ),
        ])[0]
        let directOutputs = try directSession.runInference(inputs: inputs, outputNames: nil)

        XCTAssertEqual(pipelineOutputs, directOutputs)
    }

    func testO6T7PipelineOnEvictedSessionThrowsAtFirstStep() {
        let engine = ScriptedMockONNXEngine()
        let session = makeSession(engine: engine)
        session.evict()

        XCTAssertThrowsError(
            try session.runPipeline(steps: [
                literalStep(data: [1, 2, 3]),
            ])
        ) { error in
            XCTAssertEqual(error as? ONNXError, .modelEvicted)
            XCTAssertEqual(engine.callCount, 0)
        }
    }

    private func makeSession(
        engine: ScriptedMockONNXEngine = ScriptedMockONNXEngine(),
        metadata: ONNXModelMetadataValue? = nil
    ) -> ONNXSession {
        ONNXSession(
            sessionId: "tiny-test",
            engine: engine,
            metadata: metadata ?? Self.metadata(for: engine),
            priority: .interactive
        )
    }

    private static func metadata(for engine: ScriptedMockONNXEngine) -> ONNXModelMetadataValue {
        ONNXModelMetadataValue(
            inputs: engine.inputMetadata,
            outputs: engine.outputMetadata,
            accelerator: "auto",
            opset: 13
        )
    }

    private func literalStep(
        inputName: String = "input",
        data: [Double],
        shape: [Int]? = nil,
        outputNames: [String]? = nil
    ) -> PipelineStep {
        PipelineStep(
            inputs: [
                .literal(
                    TensorData(
                        name: inputName,
                        dtype: "float32",
                        shape: shape ?? [1, data.count],
                        data: data
                    )
                ),
            ],
            outputNames: outputNames
        )
    }
}

private final class ScriptedMockONNXEngine: ONNXEngine {
    let inputMetadata: [ONNXTensorMetadata]
    let outputMetadata: [ONNXTensorMetadata]
    let accelerator: String

    private let defaultOutput: [String: TensorData] = [
        "output": TensorData(name: "output", dtype: "float32", shape: [1, 3], data: [5, 7, 9]),
    ]

    var scriptedOutputs: [[String: TensorData]] = []
    var scriptedErrors: [Error?] = []
    private(set) var allInputs: [[String: TensorData]] = []
    private(set) var callCount = 0

    init(
        inputMetadata: [ONNXTensorMetadata] = [
            ONNXTensorMetadata(name: "input", shape: [1, 3], dtype: "float32"),
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
        let index = callCount
        callCount += 1
        allInputs.append(inputs)

        if scriptedErrors.indices.contains(index), let error = scriptedErrors[index] {
            throw error
        }

        return scriptedOutputs.indices.contains(index) ? scriptedOutputs[index] : defaultOutput
    }

    func close() {}
}
