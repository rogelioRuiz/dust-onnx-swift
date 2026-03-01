import CoreGraphics
import Foundation
import ImageIO

public struct ImagePreprocessor {
    private static let defaultMean = [0.485, 0.456, 0.406]
    private static let defaultStd = [0.229, 0.224, 0.225]

    public static func preprocess(
        imageData: Data,
        targetWidth: Int,
        targetHeight: Int,
        resize: String,
        normalization: String,
        customMean: [Double]?,
        customStd: [Double]?
    ) throws -> TensorData {
        guard targetWidth > 0, targetHeight > 0 else {
            throw ONNXError.preprocessError(detail: "Target dimensions must be greater than zero")
        }

        let sourceImage = try decodeImage(from: imageData)
        let context = try makeContext(width: targetWidth, height: targetHeight)
        let drawRect = try drawRect(
            for: sourceImage,
            targetWidth: targetWidth,
            targetHeight: targetHeight,
            resize: resize
        )

        if resize == "letterbox" {
            let padding = CGFloat(114.0 / 255.0)
            context.setFillColor(red: padding, green: padding, blue: padding, alpha: 1.0)
            context.fill(CGRect(x: 0, y: 0, width: CGFloat(targetWidth), height: CGFloat(targetHeight)))
        }

        context.interpolationQuality = .high
        context.draw(sourceImage, in: drawRect)

        let normalized = try extractNormalizedPixels(
            from: context,
            width: targetWidth,
            height: targetHeight,
            normalization: normalization,
            customMean: customMean,
            customStd: customStd
        )

        return TensorData(
            name: "image",
            dtype: "float32",
            shape: [1, 3, targetHeight, targetWidth],
            data: normalized
        )
    }

    private static func decodeImage(from imageData: Data) throws -> CGImage {
        guard let source = CGImageSourceCreateWithData(imageData as CFData, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw ONNXError.preprocessError(detail: "Unable to decode image data")
        }
        return image
    }

    private static func makeContext(width: Int, height: Int) throws -> CGContext {
        let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
        guard let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: bitmapInfo
        ) else {
            throw ONNXError.preprocessError(detail: "Unable to allocate image buffer")
        }

        // Match top-left image coordinates so the tensor rows are emitted in display order.
        context.translateBy(x: 0, y: CGFloat(height))
        context.scaleBy(x: 1, y: -1)
        return context
    }

    private static func drawRect(
        for image: CGImage,
        targetWidth: Int,
        targetHeight: Int,
        resize: String
    ) throws -> CGRect {
        let sourceWidth = CGFloat(image.width)
        let sourceHeight = CGFloat(image.height)
        let destinationWidth = CGFloat(targetWidth)
        let destinationHeight = CGFloat(targetHeight)

        switch resize {
        case "stretch":
            return CGRect(x: 0, y: 0, width: destinationWidth, height: destinationHeight)
        case "letterbox":
            let scale = min(destinationWidth / sourceWidth, destinationHeight / sourceHeight)
            let scaledWidth = sourceWidth * scale
            let scaledHeight = sourceHeight * scale
            return CGRect(
                x: (destinationWidth - scaledWidth) / 2.0,
                y: (destinationHeight - scaledHeight) / 2.0,
                width: scaledWidth,
                height: scaledHeight
            )
        case "crop_center":
            let scale = max(destinationWidth / sourceWidth, destinationHeight / sourceHeight)
            let scaledWidth = sourceWidth * scale
            let scaledHeight = sourceHeight * scale
            return CGRect(
                x: (destinationWidth - scaledWidth) / 2.0,
                y: (destinationHeight - scaledHeight) / 2.0,
                width: scaledWidth,
                height: scaledHeight
            )
        default:
            throw ONNXError.preprocessError(detail: "Unsupported resize mode: \(resize)")
        }
    }

    private static func extractNormalizedPixels(
        from context: CGContext,
        width: Int,
        height: Int,
        normalization: String,
        customMean: [Double]?,
        customStd: [Double]?
    ) throws -> [Double] {
        guard let rawData = context.data else {
            throw ONNXError.preprocessError(detail: "Image buffer is unavailable")
        }

        if let customMean, customMean.count != 3 {
            throw ONNXError.preprocessError(detail: "Custom mean must contain three values")
        }
        if let customStd, customStd.count != 3 {
            throw ONNXError.preprocessError(detail: "Custom std must contain three values")
        }

        let mean = customMean ?? defaultMean
        let std = customStd ?? defaultStd
        let useCustomStatistics = customMean != nil || customStd != nil

        if useCustomStatistics && std.contains(0.0) {
            throw ONNXError.preprocessError(detail: "Custom std values must be non-zero")
        }

        let planeSize = width * height
        var output = Array(repeating: 0.0, count: planeSize * 3)
        let bytesPerRow = context.bytesPerRow
        let pixelBuffer = rawData.assumingMemoryBound(to: UInt8.self)

        for y in 0..<height {
            for x in 0..<width {
                let pixelOffset = (y * bytesPerRow) + (x * 4)
                let pixelValues = [
                    Double(pixelBuffer[pixelOffset]),
                    Double(pixelBuffer[pixelOffset + 1]),
                    Double(pixelBuffer[pixelOffset + 2]),
                ]
                let tensorIndex = (y * width) + x

                for channel in 0..<3 {
                    output[channel * planeSize + tensorIndex] = try normalize(
                        pixel: pixelValues[channel],
                        channel: channel,
                        normalization: normalization,
                        mean: mean,
                        std: std,
                        useCustomStatistics: useCustomStatistics
                    )
                }
            }
        }

        return output
    }

    private static func normalize(
        pixel: Double,
        channel: Int,
        normalization: String,
        mean: [Double],
        std: [Double],
        useCustomStatistics: Bool
    ) throws -> Double {
        if useCustomStatistics {
            return ((pixel / 255.0) - mean[channel]) / std[channel]
        }

        switch normalization {
        case "imagenet":
            return ((pixel / 255.0) - mean[channel]) / std[channel]
        case "minus1_plus1":
            return (pixel / 127.5) - 1.0
        case "zero_to_1":
            return pixel / 255.0
        case "none":
            return pixel
        default:
            throw ONNXError.preprocessError(detail: "Unsupported normalization mode: \(normalization)")
        }
    }
}
