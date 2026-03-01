import Foundation

public struct ONNXConfig: Equatable, Sendable {
    public let accelerator: String
    public let interOpNumThreads: Int
    public let intraOpNumThreads: Int
    public let graphOptimizationLevel: String
    public let enableMemoryPattern: Bool

    public init(
        accelerator: String = "auto",
        interOpNumThreads: Int = 1,
        intraOpNumThreads: Int = max(1, ProcessInfo.processInfo.activeProcessorCount - 1),
        graphOptimizationLevel: String = "all",
        enableMemoryPattern: Bool = true
    ) {
        self.accelerator = accelerator
        self.interOpNumThreads = interOpNumThreads
        self.intraOpNumThreads = intraOpNumThreads
        self.graphOptimizationLevel = graphOptimizationLevel
        self.enableMemoryPattern = enableMemoryPattern
    }

    public init(jsObject: [String: Any]?) {
        let defaultIntra = max(1, ProcessInfo.processInfo.activeProcessorCount - 1)
        let threadsValue = jsObject?["threads"]
        let threadObject = threadsValue as? [String: Any]
        let threadCount = threadsValue as? Int

        self.init(
            accelerator: jsObject?["accelerator"] as? String ?? "auto",
            interOpNumThreads: threadObject?["interOp"] as? Int ?? threadCount ?? 1,
            intraOpNumThreads: threadObject?["intraOp"] as? Int ?? threadCount ?? defaultIntra,
            graphOptimizationLevel: jsObject?["graphOptLevel"] as? String
                ?? jsObject?["graphOptimizationLevel"] as? String
                ?? "all",
            enableMemoryPattern: jsObject?["memoryPattern"] as? Bool
                ?? jsObject?["enableMemoryPattern"] as? Bool
                ?? true
        )
    }
}
