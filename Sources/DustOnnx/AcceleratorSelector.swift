import Foundation

#if canImport(onnxruntime_objc)
import ObjectiveC.runtime
import onnxruntime_objc

struct AcceleratorResult {
    let options: ORTSessionOptions
    let resolvedAccelerator: String
}

struct AcceleratorSelector {
    static func configureSessionOptions(
        accelerator: String,
        modelId: String,
        cacheBaseDir: URL
    ) throws -> AcceleratorResult {
        let normalized = accelerator.lowercased()

        guard normalized == "auto" || normalized == "coreml" else {
            return AcceleratorResult(
                options: try ORTSessionOptions(),
                resolvedAccelerator: "cpu"
            )
        }

        do {
            let options = try ORTSessionOptions()
            let coreMLOptions = ORTCoreMLExecutionProviderOptions()
            let modelCacheDir = cacheBaseDir.appendingPathComponent(modelId, isDirectory: true)

            try FileManager.default.createDirectory(
                at: modelCacheDir,
                withIntermediateDirectories: true,
                attributes: nil
            )
            setCacheDirectoryIfSupported(modelCacheDir.path, on: coreMLOptions)
            try options.appendCoreMLExecutionProvider(with: coreMLOptions)

            return AcceleratorResult(
                options: options,
                resolvedAccelerator: "coreml"
            )
        } catch {
            return AcceleratorResult(
                options: try ORTSessionOptions(),
                resolvedAccelerator: "cpu"
            )
        }
    }

    private static func setCacheDirectoryIfSupported(
        _ path: String,
        on options: ORTCoreMLExecutionProviderOptions
    ) {
        let candidateKeys = [
            "modelCacheDirectory",
            "cacheDirectory",
            "modelCachePath",
        ]

        for key in candidateKeys {
            let hasProperty = key.withCString {
                class_getProperty(ORTCoreMLExecutionProviderOptions.self, $0) != nil
            }

            guard hasProperty else {
                continue
            }

            options.setValue(path, forKey: key)
            return
        }
    }
}
#endif
