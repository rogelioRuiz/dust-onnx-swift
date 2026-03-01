import Foundation

public enum ONNXError: Error, Equatable {
    case fileNotFound(path: String)
    case loadFailed(path: String, detail: String?)
    case formatUnsupported(format: String)
    case sessionClosed
    case modelEvicted
    case shapeError(name: String, expected: [Int], got: [Int])
    case dtypeError(name: String, expected: String, got: String)
    case inferenceError(detail: String)
    case preprocessError(detail: String)
}
