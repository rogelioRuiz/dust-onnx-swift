import Foundation
import DustCore

#if canImport(OnnxRuntimeBindings)
import OnnxRuntimeBindings
#elseif canImport(onnxruntime_objc)
import onnxruntime_objc
#endif

public struct ONNXTensorMetadata: Equatable, Sendable {
    public let name: String
    public let shape: [Int]
    public let dtype: String

    public init(name: String, shape: [Int], dtype: String) {
        self.name = name
        self.shape = shape
        self.dtype = dtype
    }

    public func toJSObject() -> [String: Any] {
        [
            "name": name,
            "shape": shape,
            "dtype": dtype,
        ]
    }
}

public struct ONNXModelMetadataValue: Equatable, Sendable {
    public let inputs: [ONNXTensorMetadata]
    public let outputs: [ONNXTensorMetadata]
    public let accelerator: String
    public let opset: Int?

    public init(inputs: [ONNXTensorMetadata], outputs: [ONNXTensorMetadata], accelerator: String, opset: Int?) {
        self.inputs = inputs
        self.outputs = outputs
        self.accelerator = accelerator
        self.opset = opset
    }

    public func toJSObject() -> [String: Any] {
        var object: [String: Any] = [
            "inputs": inputs.map { $0.toJSObject() },
            "outputs": outputs.map { $0.toJSObject() },
            "accelerator": accelerator,
        ]
        if let opset {
            object["opset"] = opset
        }
        return object
    }
}

public struct TensorData: Equatable, Sendable {
    public let name: String
    public let dtype: String
    public let shape: [Int]
    public let data: [Double]

    public init(name: String, dtype: String, shape: [Int], data: [Double]) {
        self.name = name
        self.dtype = dtype
        self.shape = shape
        self.data = data
    }

    public func toJSObject() -> [String: Any] {
        [
            "name": name,
            "dtype": dtype,
            "shape": shape,
            "data": data,
        ]
    }
}

public enum PipelineInputValue: Equatable, Sendable {
    case literal(TensorData)
    case previousOutput(name: String)
    case stepReference(name: String, fromStep: Int, outputName: String)

    public var inputName: String {
        switch self {
        case .literal(let tensor):
            return tensor.name
        case .previousOutput(let name):
            return name
        case .stepReference(let name, _, _):
            return name
        }
    }
}

public struct PipelineStep: Equatable, Sendable {
    public let inputs: [PipelineInputValue]
    public let outputNames: [String]?

    public init(inputs: [PipelineInputValue], outputNames: [String]?) {
        self.inputs = inputs
        self.outputNames = outputNames
    }
}

public final class ONNXSession: NSObject, DustModelSession, @unchecked Sendable {
    public let sessionId: String
    public let metadata: ONNXModelMetadataValue

    private let sessionPriority: DustSessionPriority
    private let lock = NSLock()
    private var engine: ONNXEngine?
    private var currentStatus: DustModelStatus = .ready
    private var evicted = false

    public init(
        sessionId: String,
        engine: ONNXEngine,
        metadata: ONNXModelMetadataValue,
        priority: DustSessionPriority
    ) {
        self.sessionId = sessionId
        self.engine = engine
        self.metadata = metadata
        self.sessionPriority = priority
        super.init()
    }

    public init(
        sessionId: String,
        metadata: ONNXModelMetadataValue,
        priority: DustSessionPriority
    ) {
        self.sessionId = sessionId
        self.engine = nil
        self.metadata = metadata
        self.sessionPriority = priority
        super.init()
    }

    public func predict(inputs: [DustInputTensor]) async throws -> [DustOutputTensor] {
        let tensors = inputs.map { input in
            TensorData(
                name: input.name,
                dtype: "float32",
                shape: input.shape,
                data: input.data.map(Double.init)
            )
        }
        let outputs = try runInference(
            inputs: Dictionary(uniqueKeysWithValues: tensors.map { ($0.name, $0) }),
            outputNames: nil
        )

        return metadata.outputs.compactMap { outputMetadata in
            guard let tensor = outputs[outputMetadata.name] else {
                return nil
            }
            return DustOutputTensor(
                name: tensor.name,
                data: tensor.data.map(Float.init),
                shape: tensor.shape
            )
        }
    }

    public func runInference(
        inputs: [String: TensorData],
        outputNames: [String]?
    ) throws -> [String: TensorData] {
        let activeEngine = try requireEngine()
        try validate(inputs: inputs, metadata: activeEngine.inputMetadata)

        let rawOutputs: [String: TensorData]
        do {
            rawOutputs = try activeEngine.run(inputs: inputs)
        } catch let error as ONNXError {
            if case .inferenceError = error {
                throw error
            }
            throw ONNXError.inferenceError(detail: Self.errorDetail(from: error))
        } catch {
            throw ONNXError.inferenceError(detail: Self.errorDetail(from: error))
        }

        return filterOutputs(rawOutputs, outputNames: outputNames)
    }

    public func runPipeline(steps: [PipelineStep]) throws -> [[String: TensorData]] {
        let activeEngine = try requireEngine()
        var resolvedResults: [[String: TensorData]?] = Array(repeating: nil, count: steps.count)
        var cachedResults: [[String: TensorData]?] = Array(repeating: nil, count: steps.count)

        for (stepIndex, step) in steps.enumerated() {
            var resolvedInputs: [String: TensorData] = [:]

            for inputValue in step.inputs {
                let tensor: TensorData

                switch inputValue {
                case .literal(let inputTensor):
                    tensor = inputTensor
                case .previousOutput(let name):
                    guard stepIndex > 0 else {
                        throw Self.pipelineResolutionError(
                            stepIndex: stepIndex,
                            detail: "previous_output requires a previous step"
                        )
                    }
                    guard let previousOutputs = cachedResults[stepIndex - 1] else {
                        throw Self.pipelineResolutionError(
                            stepIndex: stepIndex,
                            detail: "previous step \(stepIndex - 1) outputs are unavailable"
                        )
                    }
                    guard let previousTensor = previousOutputs[name] else {
                        throw Self.pipelineResolutionError(
                            stepIndex: stepIndex,
                            detail: "previous step output '\(name)' was not found"
                        )
                    }
                    tensor = TensorData(
                        name: name,
                        dtype: previousTensor.dtype,
                        shape: previousTensor.shape,
                        data: previousTensor.data
                    )
                case .stepReference(let name, let fromStep, let outputName):
                    guard (0..<stepIndex).contains(fromStep) else {
                        throw Self.pipelineResolutionError(
                            stepIndex: stepIndex,
                            detail: "fromStep \(fromStep) must reference an earlier step"
                        )
                    }
                    guard let referencedOutputs = cachedResults[fromStep] else {
                        throw Self.pipelineResolutionError(
                            stepIndex: stepIndex,
                            detail: "step \(fromStep) outputs are unavailable"
                        )
                    }
                    guard let referencedTensor = referencedOutputs[outputName] else {
                        throw Self.pipelineResolutionError(
                            stepIndex: stepIndex,
                            detail: "step \(fromStep) output '\(outputName)' was not found"
                        )
                    }
                    tensor = TensorData(
                        name: name,
                        dtype: referencedTensor.dtype,
                        shape: referencedTensor.shape,
                        data: referencedTensor.data
                    )
                }

                resolvedInputs[inputValue.inputName] = tensor
            }

            let filteredOutputs: [String: TensorData]
            do {
                try validate(inputs: resolvedInputs, metadata: activeEngine.inputMetadata)
                filteredOutputs = filterOutputs(
                    try activeEngine.run(inputs: resolvedInputs),
                    outputNames: step.outputNames
                )
            } catch {
                throw ONNXError.inferenceError(
                    detail: "Pipeline step \(stepIndex) failed: \(Self.errorDetail(from: error))"
                )
            }

            resolvedResults[stepIndex] = filteredOutputs
            cachedResults[stepIndex] = filteredOutputs

            if stepIndex > 0 {
                let previousIndex = stepIndex - 1
                let referencedLater: Bool
                if stepIndex + 1 < steps.count {
                    referencedLater = steps[(stepIndex + 1)...].contains { futureStep in
                        futureStep.inputs.contains {
                            if case .stepReference(_, let fromStep, _) = $0 {
                                return fromStep == previousIndex
                            }
                            return false
                        }
                    }
                } else {
                    referencedLater = false
                }
                if !referencedLater {
                    cachedResults[previousIndex] = nil
                }
            }
        }

        return resolvedResults.map { $0 ?? [:] }
    }

    public func status() -> DustModelStatus {
        lock.lock()
        defer { lock.unlock() }
        return currentStatus
    }

    public var isModelEvicted: Bool {
        lock.lock()
        defer { lock.unlock() }
        return evicted
    }

    public func priority() -> DustSessionPriority {
        sessionPriority
    }

    public func close() async throws {
        let activeEngine: ONNXEngine?

        lock.lock()
        activeEngine = engine
        engine = nil
        currentStatus = .notLoaded
        evicted = false
        lock.unlock()

        activeEngine?.close()
    }

    public func evict() {
        let activeEngine: ONNXEngine?

        lock.lock()
        activeEngine = engine
        engine = nil
        currentStatus = .notLoaded
        evicted = true
        lock.unlock()

        activeEngine?.close()
    }

    public func requireEngine() throws -> ONNXEngine {
        lock.lock()
        defer { lock.unlock() }

        guard let engine else {
            throw evicted ? ONNXError.modelEvicted : ONNXError.sessionClosed
        }

        return engine
    }

    public static func readMetadata(
        fromModelAt path: String,
        fallbackInputNames: [String],
        fallbackOutputNames: [String],
        accelerator: String,
        config _: ONNXConfig
    ) -> ONNXModelMetadataValue {
        if let parsed = try? ONNXModelParser.parseMetadata(at: path) {
            return ONNXModelMetadataValue(
                inputs: parsed.inputs,
                outputs: parsed.outputs,
                accelerator: accelerator,
                opset: parsed.opset
            )
        }

        return metadataFromNames(
            inputNames: fallbackInputNames,
            outputNames: fallbackOutputNames,
            accelerator: accelerator,
            opset: nil
        )
    }

    public static func metadataFromNames(
        inputNames: [String],
        outputNames: [String],
        accelerator: String,
        opset: Int?
    ) -> ONNXModelMetadataValue {
        ONNXModelMetadataValue(
            inputs: inputNames.map {
                ONNXTensorMetadata(name: $0, shape: [], dtype: "float32")
            },
            outputs: outputNames.map {
                ONNXTensorMetadata(name: $0, shape: [], dtype: "float32")
            },
            accelerator: accelerator,
            opset: opset
        )
    }

    private func validate(
        inputs: [String: TensorData],
        metadata: [ONNXTensorMetadata]
    ) throws {
        let metadataByName = Dictionary(uniqueKeysWithValues: metadata.map { ($0.name, $0) })

        for (name, tensor) in inputs {
            guard let expected = metadataByName[name] else {
                continue
            }

            if expected.dtype != "unknown", expected.dtype != tensor.dtype {
                throw ONNXError.dtypeError(name: name, expected: expected.dtype, got: tensor.dtype)
            }

            guard expected.shape.count == tensor.shape.count else {
                throw ONNXError.shapeError(name: name, expected: expected.shape, got: tensor.shape)
            }

            for (expectedDimension, actualDimension) in zip(expected.shape, tensor.shape) {
                if expectedDimension != -1, expectedDimension != actualDimension {
                    throw ONNXError.shapeError(name: name, expected: expected.shape, got: tensor.shape)
                }
            }
        }
    }

    private func filterOutputs(
        _ outputs: [String: TensorData],
        outputNames: [String]?
    ) -> [String: TensorData] {
        guard let outputNames else {
            return outputs
        }

        var filtered: [String: TensorData] = [:]
        for outputName in outputNames {
            guard let tensor = outputs[outputName] else {
                continue
            }
            filtered[outputName] = tensor
        }
        return filtered
    }

    private static func pipelineResolutionError(stepIndex: Int, detail: String) -> ONNXError {
        ONNXError.inferenceError(detail: "Pipeline step \(stepIndex): \(detail)")
    }

    private static func errorDetail(from error: Error) -> String {
        if let onnxError = error as? ONNXError {
            switch onnxError {
            case .fileNotFound(let path):
                return "Model file not found: \(path)"
            case .loadFailed(let path, let detail):
                if let detail, !detail.isEmpty {
                    return "Failed to load ONNX model at \(path): \(detail)"
                }
                return "Failed to load ONNX model at \(path)"
            case .formatUnsupported(let format):
                return "Unsupported model format: \(format)"
            case .sessionClosed:
                return "Model session is closed"
            case .modelEvicted:
                return "Model was evicted from memory"
            case .shapeError(let name, let expected, let got):
                return "Shape mismatch for \(name): expected \(expected), got \(got)"
            case .dtypeError(let name, let expected, let got):
                return "Dtype mismatch for \(name): expected \(expected), got \(got)"
            case .preprocessError(let detail), .inferenceError(let detail):
                return detail
            }
        }

        let description = (error as NSError).localizedDescription
        return description.isEmpty ? String(describing: error) : description
    }
}

private struct ParsedONNXModelMetadata {
    let inputs: [ONNXTensorMetadata]
    let outputs: [ONNXTensorMetadata]
    let opset: Int?
}

private enum ONNXModelParser {
    static func parseMetadata(at path: String) throws -> ParsedONNXModelMetadata {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        return try parseMetadata(from: data)
    }

    static func parseMetadata(from data: Data) throws -> ParsedONNXModelMetadata {
        var reader = ProtobufReader(data: data)
        var graphPayload: Data?
        var opset: Int?

        while let field = try reader.nextField() {
            switch field.number {
            case 7 where field.wireType == .lengthDelimited:
                graphPayload = try reader.readLengthDelimited()
            case 8 where field.wireType == .lengthDelimited:
                let payload = try reader.readLengthDelimited()
                if let version = try parseOpsetVersion(from: payload) {
                    opset = version
                }
            default:
                try reader.skipField(wireType: field.wireType)
            }
        }

        guard let graphPayload else {
            return ParsedONNXModelMetadata(inputs: [], outputs: [], opset: opset)
        }

        let graph = try parseGraph(from: graphPayload)
        let inputs = graph.inputs.filter { !graph.initializerNames.contains($0.name) }
        return ParsedONNXModelMetadata(inputs: inputs, outputs: graph.outputs, opset: opset)
    }

    private static func parseOpsetVersion(from data: Data) throws -> Int? {
        var reader = ProtobufReader(data: data)
        var version: Int?

        while let field = try reader.nextField() {
            switch field.number {
            case 2 where field.wireType == .varint:
                version = Int(try reader.readVarint())
            default:
                try reader.skipField(wireType: field.wireType)
            }
        }

        return version
    }

    private static func parseGraph(from data: Data) throws -> ParsedGraph {
        var reader = ProtobufReader(data: data)
        var inputs: [ONNXTensorMetadata] = []
        var outputs: [ONNXTensorMetadata] = []
        var initializerNames = Set<String>()

        while let field = try reader.nextField() {
            switch field.number {
            case 5 where field.wireType == .lengthDelimited:
                let payload = try reader.readLengthDelimited()
                if let name = try parseInitializerName(from: payload) {
                    initializerNames.insert(name)
                }
            case 11 where field.wireType == .lengthDelimited:
                inputs.append(try parseValueInfo(from: reader.readLengthDelimited()))
            case 12 where field.wireType == .lengthDelimited:
                outputs.append(try parseValueInfo(from: reader.readLengthDelimited()))
            default:
                try reader.skipField(wireType: field.wireType)
            }
        }

        return ParsedGraph(inputs: inputs, outputs: outputs, initializerNames: initializerNames)
    }

    private static func parseInitializerName(from data: Data) throws -> String? {
        var reader = ProtobufReader(data: data)

        while let field = try reader.nextField() {
            switch field.number {
            case 1 where field.wireType == .lengthDelimited:
                return try reader.readString()
            default:
                try reader.skipField(wireType: field.wireType)
            }
        }

        return nil
    }

    private static func parseValueInfo(from data: Data) throws -> ONNXTensorMetadata {
        var reader = ProtobufReader(data: data)
        var name = ""
        var shape: [Int] = []
        var dtype = "unknown"

        while let field = try reader.nextField() {
            switch field.number {
            case 1 where field.wireType == .lengthDelimited:
                name = try reader.readString()
            case 2 where field.wireType == .lengthDelimited:
                let typePayload = try reader.readLengthDelimited()
                let parsedType = try parseType(from: typePayload)
                shape = parsedType.shape
                dtype = parsedType.dtype
            default:
                try reader.skipField(wireType: field.wireType)
            }
        }

        return ONNXTensorMetadata(name: name, shape: shape, dtype: dtype)
    }

    private static func parseType(from data: Data) throws -> (shape: [Int], dtype: String) {
        var reader = ProtobufReader(data: data)

        while let field = try reader.nextField() {
            switch field.number {
            case 1 where field.wireType == .lengthDelimited:
                return try parseTensorType(from: reader.readLengthDelimited())
            default:
                try reader.skipField(wireType: field.wireType)
            }
        }

        return ([], "unknown")
    }

    private static func parseTensorType(from data: Data) throws -> (shape: [Int], dtype: String) {
        var reader = ProtobufReader(data: data)
        var elementType = "unknown"
        var shape: [Int] = []

        while let field = try reader.nextField() {
            switch field.number {
            case 1 where field.wireType == .varint:
                elementType = mapElementType(Int(try reader.readVarint()))
            case 2 where field.wireType == .lengthDelimited:
                shape = try parseShape(from: reader.readLengthDelimited())
            default:
                try reader.skipField(wireType: field.wireType)
            }
        }

        return (shape, elementType)
    }

    private static func parseShape(from data: Data) throws -> [Int] {
        var reader = ProtobufReader(data: data)
        var shape: [Int] = []

        while let field = try reader.nextField() {
            switch field.number {
            case 1 where field.wireType == .lengthDelimited:
                shape.append(try parseDimension(from: reader.readLengthDelimited()))
            default:
                try reader.skipField(wireType: field.wireType)
            }
        }

        return shape
    }

    private static func parseDimension(from data: Data) throws -> Int {
        var reader = ProtobufReader(data: data)

        while let field = try reader.nextField() {
            switch field.number {
            case 1 where field.wireType == .varint:
                return Int(try reader.readVarint())
            case 2 where field.wireType == .lengthDelimited:
                _ = try reader.readString()
                return -1
            default:
                try reader.skipField(wireType: field.wireType)
            }
        }

        return -1
    }

    private static func mapElementType(_ value: Int) -> String {
        switch value {
        case 1:
            return "float32"
        case 2:
            return "uint8"
        case 3:
            return "int8"
        case 5:
            return "int16"
        case 6:
            return "int32"
        case 7:
            return "int64"
        case 8:
            return "string"
        case 9:
            return "bool"
        case 10:
            return "float16"
        case 11:
            return "float64"
        default:
            return "unknown"
        }
    }
}

private struct ParsedGraph {
    let inputs: [ONNXTensorMetadata]
    let outputs: [ONNXTensorMetadata]
    let initializerNames: Set<String>
}

private struct ProtobufField {
    let number: Int
    let wireType: ProtobufWireType
}

private enum ProtobufWireType: Int {
    case varint = 0
    case fixed64 = 1
    case lengthDelimited = 2
    case fixed32 = 5
}

private struct ProtobufReader {
    private let data: Data
    private var index = 0

    init(data: Data) {
        self.data = data
    }

    mutating func nextField() throws -> ProtobufField? {
        guard index < data.count else {
            return nil
        }

        let key = try readVarint()
        guard let wireType = ProtobufWireType(rawValue: Int(key & 0x7)) else {
            throw ONNXError.inferenceError(detail: "Unsupported protobuf wire type")
        }

        return ProtobufField(number: Int(key >> 3), wireType: wireType)
    }

    mutating func readVarint() throws -> UInt64 {
        var result: UInt64 = 0
        var shift: UInt64 = 0

        while index < data.count {
            let byte = data[index]
            index += 1

            result |= UInt64(byte & 0x7f) << shift
            if byte & 0x80 == 0 {
                return result
            }

            shift += 7
            if shift > 63 {
                break
            }
        }

        throw ONNXError.inferenceError(detail: "Invalid protobuf varint")
    }

    mutating func readLengthDelimited() throws -> Data {
        let length = Int(try readVarint())
        guard length >= 0, index + length <= data.count else {
            throw ONNXError.inferenceError(detail: "Invalid protobuf length-delimited field")
        }

        let slice = data[index..<(index + length)]
        index += length
        return Data(slice)
    }

    mutating func readString() throws -> String {
        let payload = try readLengthDelimited()
        guard let value = String(data: payload, encoding: .utf8) else {
            throw ONNXError.inferenceError(detail: "Invalid UTF-8 string in protobuf payload")
        }
        return value
    }

    mutating func skipField(wireType: ProtobufWireType) throws {
        switch wireType {
        case .varint:
            _ = try readVarint()
        case .fixed64:
            try skip(bytes: 8)
        case .lengthDelimited:
            _ = try readLengthDelimited()
        case .fixed32:
            try skip(bytes: 4)
        }
    }

    private mutating func skip(bytes: Int) throws {
        guard index + bytes <= data.count else {
            throw ONNXError.inferenceError(detail: "Unexpected end of protobuf payload")
        }
        index += bytes
    }
}
