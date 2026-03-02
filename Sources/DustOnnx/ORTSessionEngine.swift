import Foundation

#if canImport(OnnxRuntimeBindings)
import OnnxRuntimeBindings
#elseif canImport(onnxruntime_objc)
import onnxruntime_objc
#endif

#if canImport(OnnxRuntimeBindings) || canImport(onnxruntime_objc)

final class ORTSessionEngine: ONNXEngine {
    let inputMetadata: [ONNXTensorMetadata]
    let outputMetadata: [ONNXTensorMetadata]
    let accelerator: String

    private let session: ORTSession

    init(
        session: ORTSession,
        metadata: ONNXModelMetadataValue
    ) {
        self.session = session
        inputMetadata = metadata.inputs
        outputMetadata = metadata.outputs
        accelerator = metadata.accelerator
    }

    func run(inputs: [String: TensorData]) throws -> [String: TensorData] {
        var ortInputs: [String: ORTValue] = [:]
        var retainedBuffers: [NSMutableData] = []

        for (name, tensor) in inputs {
            let prepared = try Self.makeORTValue(from: tensor)
            ortInputs[name] = prepared.value
            retainedBuffers.append(prepared.backingData)
        }

        let requestedOutputNames = Set(outputMetadata.map(\.name))
        let rawOutputs = try session.run(
            withInputs: ortInputs,
            outputNames: requestedOutputNames,
            runOptions: nil
        )

        var outputs: [String: TensorData] = [:]
        for (name, value) in rawOutputs {
            outputs[name] = try Self.makeTensorData(name: name, value: value)
        }

        _ = retainedBuffers
        return outputs
    }

    func close() {}

    private static func makeORTValue(from tensor: TensorData) throws -> (value: ORTValue, backingData: NSMutableData) {
        let shape = tensor.shape.map { NSNumber(value: $0) }

        switch tensor.dtype {
        case "float32":
            let values = tensor.data.map(Float.init)
            let bytes = mutableData(from: values)
            return (try ORTValue(tensorData: bytes, elementType: .float, shape: shape), bytes)
        case "int8":
            let values = tensor.data.map { Int8(clamping: Int($0.rounded())) }
            let bytes = mutableData(from: values)
            return (try ORTValue(tensorData: bytes, elementType: .int8, shape: shape), bytes)
        case "uint8":
            let values = tensor.data.map { UInt8(clamping: Int($0.rounded())) }
            let bytes = mutableData(from: values)
            return (try ORTValue(tensorData: bytes, elementType: .uInt8, shape: shape), bytes)
        case "int32":
            let values = tensor.data.map { Int32(clamping: Int64($0.rounded())) }
            let bytes = mutableData(from: values)
            return (try ORTValue(tensorData: bytes, elementType: .int32, shape: shape), bytes)
        case "int64":
            let values = tensor.data.map { Int64($0.rounded()) }
            let bytes = mutableData(from: values)
            return (try ORTValue(tensorData: bytes, elementType: .int64, shape: shape), bytes)
        default:
            throw ONNXError.inferenceError(detail: "Unsupported tensor dtype on iOS: \(tensor.dtype)")
        }
    }

    private static func makeTensorData(
        name: String,
        value: ORTValue
    ) throws -> TensorData {
        let tensorInfo = try value.tensorTypeAndShapeInfo()
        let rawData = try value.tensorData()
        let shape = tensorInfo.shape.map(\.intValue)
        let dtype = mapElementType(tensorInfo.elementType)

        let numbers: [Double]
        switch tensorInfo.elementType {
        case .float:
            numbers = decode(rawData, as: Float.self).map(Double.init)
        case .int8:
            numbers = decode(rawData, as: Int8.self).map(Double.init)
        case .uInt8:
            numbers = decode(rawData, as: UInt8.self).map(Double.init)
        case .int32:
            numbers = decode(rawData, as: Int32.self).map(Double.init)
        case .uInt32:
            numbers = decode(rawData, as: UInt32.self).map(Double.init)
        case .int64:
            numbers = decode(rawData, as: Int64.self).map(Double.init)
        case .uInt64:
            numbers = decode(rawData, as: UInt64.self).map(Double.init)
        default:
            throw ONNXError.inferenceError(detail: "Unsupported output tensor dtype on iOS: \(tensorInfo.elementType.rawValue)")
        }

        return TensorData(name: name, dtype: dtype, shape: shape, data: numbers)
    }

    private static func mutableData<T>(from values: [T]) -> NSMutableData {
        values.withUnsafeBufferPointer { buffer in
            guard let baseAddress = buffer.baseAddress else {
                return NSMutableData()
            }
            return NSMutableData(bytes: baseAddress, length: buffer.count * MemoryLayout<T>.stride)
        }
    }

    private static func decode<T>(_ data: NSMutableData, as _: T.Type) -> [T] {
        let count = data.length / MemoryLayout<T>.stride
        guard count > 0 else {
            return []
        }
        let pointer = data.bytes.bindMemory(to: T.self, capacity: count)
        return Array(UnsafeBufferPointer(start: pointer, count: count))
    }

    private static func mapElementType(_ type: ORTTensorElementDataType) -> String {
        switch type {
        case .float:
            return "float32"
        case .int8:
            return "int8"
        case .uInt8:
            return "uint8"
        case .int32:
            return "int32"
        case .uInt32:
            return "unknown"
        case .int64:
            return "int64"
        case .uInt64:
            return "unknown"
        case .string:
            return "string"
        default:
            return "unknown"
        }
    }
}
#endif
