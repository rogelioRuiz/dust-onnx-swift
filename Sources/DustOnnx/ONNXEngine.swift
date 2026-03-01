import Foundation

public protocol ONNXEngine: AnyObject {
    var inputMetadata: [ONNXTensorMetadata] { get }
    var outputMetadata: [ONNXTensorMetadata] { get }
    var accelerator: String { get }

    func run(inputs: [String: TensorData]) throws -> [String: TensorData]
    func close()
}
