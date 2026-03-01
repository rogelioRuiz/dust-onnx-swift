import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers
import XCTest
@testable import DustOnnx

final class ONNXPreprocessTests: XCTestCase {
    func testO3T1SolidRedStretchImagenetNormalization() throws {
        let imageData = try createSolidColorPNG(width: 8, height: 8, r: 255, g: 0, b: 0)

        let tensor = try ImagePreprocessor.preprocess(
            imageData: imageData,
            targetWidth: 8,
            targetHeight: 8,
            resize: "stretch",
            normalization: "imagenet",
            customMean: nil,
            customStd: nil
        )

        XCTAssertEqual(tensor.shape, [1, 3, 8, 8])
        XCTAssertEqual(tensor.dtype, "float32")
        XCTAssertEqual(value(in: tensor, channel: 0, x: 0, y: 0), 2.2489082969, accuracy: 0.001)
        XCTAssertEqual(value(in: tensor, channel: 1, x: 0, y: 0), -2.0357142857, accuracy: 0.001)
        XCTAssertEqual(value(in: tensor, channel: 2, x: 0, y: 0), -1.8044444444, accuracy: 0.001)
    }

    func testO3T2LetterboxAppliesPaddingAndCentersImage() throws {
        let imageData = try createSolidColorPNG(width: 8, height: 16, r: 0, g: 0, b: 255)

        let tensor = try ImagePreprocessor.preprocess(
            imageData: imageData,
            targetWidth: 8,
            targetHeight: 8,
            resize: "letterbox",
            normalization: "imagenet",
            customMean: nil,
            customStd: nil
        )

        XCTAssertEqual(tensor.shape, [1, 3, 8, 8])
        XCTAssertEqual(value(in: tensor, channel: 0, x: 0, y: 4), -0.1656819937, accuracy: 0.001)
        XCTAssertEqual(value(in: tensor, channel: 2, x: 4, y: 4), 2.6399999999, accuracy: 0.001)
    }

    func testO3T3StretchUpscalesSmallImage() throws {
        let imageData = try createSolidColorPNG(width: 4, height: 4, r: 0, g: 255, b: 0)

        let tensor = try ImagePreprocessor.preprocess(
            imageData: imageData,
            targetWidth: 8,
            targetHeight: 8,
            resize: "stretch",
            normalization: "imagenet",
            customMean: nil,
            customStd: nil
        )

        XCTAssertEqual(value(in: tensor, channel: 1, x: 3, y: 3), 2.4285714286, accuracy: 0.001)
    }

    func testO3T4MinusOneToOneNormalization() throws {
        let imageData = try createSolidColorPNG(width: 8, height: 8, r: 255, g: 255, b: 255)

        let tensor = try ImagePreprocessor.preprocess(
            imageData: imageData,
            targetWidth: 8,
            targetHeight: 8,
            resize: "stretch",
            normalization: "minus1_plus1",
            customMean: nil,
            customStd: nil
        )

        XCTAssertEqual(value(in: tensor, channel: 0, x: 0, y: 0), 1.0, accuracy: 0.0001)
        XCTAssertEqual(value(in: tensor, channel: 1, x: 0, y: 0), 1.0, accuracy: 0.0001)
        XCTAssertEqual(value(in: tensor, channel: 2, x: 0, y: 0), 1.0, accuracy: 0.0001)
    }

    func testO3T5ZeroToOneNormalization() throws {
        let imageData = try createSolidColorPNG(width: 8, height: 8, r: 0, g: 0, b: 0)

        let tensor = try ImagePreprocessor.preprocess(
            imageData: imageData,
            targetWidth: 8,
            targetHeight: 8,
            resize: "stretch",
            normalization: "zero_to_1",
            customMean: nil,
            customStd: nil
        )

        XCTAssertEqual(value(in: tensor, channel: 0, x: 2, y: 2), 0.0, accuracy: 0.0001)
        XCTAssertEqual(value(in: tensor, channel: 1, x: 2, y: 2), 0.0, accuracy: 0.0001)
        XCTAssertEqual(value(in: tensor, channel: 2, x: 2, y: 2), 0.0, accuracy: 0.0001)
    }

    func testO3T6NoneNormalizationPreservesByteValues() throws {
        let imageData = try createSolidColorPNG(width: 8, height: 8, r: 255, g: 0, b: 0)

        let tensor = try ImagePreprocessor.preprocess(
            imageData: imageData,
            targetWidth: 8,
            targetHeight: 8,
            resize: "stretch",
            normalization: "none",
            customMean: nil,
            customStd: nil
        )

        XCTAssertEqual(value(in: tensor, channel: 0, x: 6, y: 6), 255.0, accuracy: 0.0001)
        XCTAssertEqual(value(in: tensor, channel: 1, x: 6, y: 6), 0.0, accuracy: 0.0001)
        XCTAssertEqual(value(in: tensor, channel: 2, x: 6, y: 6), 0.0, accuracy: 0.0001)
    }

    func testO3T7InvalidImageDataThrowsPreprocessError() {
        XCTAssertThrowsError(
            try ImagePreprocessor.preprocess(
                imageData: Data("not-an-image".utf8),
                targetWidth: 8,
                targetHeight: 8,
                resize: "stretch",
                normalization: "imagenet",
                customMean: nil,
                customStd: nil
            )
        ) { error in
            guard case .preprocessError(let detail) = error as? ONNXError else {
                return XCTFail("Expected preprocessError, got \(error)")
            }
            XCTAssertEqual(detail, "Unable to decode image data")
        }
    }

    func testO3T8CustomMeanAndStdOverrideNormalization() throws {
        let imageData = try createSolidColorPNG(width: 8, height: 8, r: 128, g: 128, b: 128)

        let tensor = try ImagePreprocessor.preprocess(
            imageData: imageData,
            targetWidth: 8,
            targetHeight: 8,
            resize: "stretch",
            normalization: "imagenet",
            customMean: [0.5, 0.5, 0.5],
            customStd: [0.5, 0.5, 0.5]
        )

        XCTAssertEqual(value(in: tensor, channel: 0, x: 1, y: 1), 0.0039215686, accuracy: 0.001)
        XCTAssertEqual(value(in: tensor, channel: 1, x: 1, y: 1), 0.0039215686, accuracy: 0.001)
        XCTAssertEqual(value(in: tensor, channel: 2, x: 1, y: 1), 0.0039215686, accuracy: 0.001)
    }

    private func value(in tensor: TensorData, channel: Int, x: Int, y: Int) -> Double {
        let width = tensor.shape[3]
        let height = tensor.shape[2]
        let planeSize = width * height
        let index = (channel * planeSize) + (y * width) + x
        return tensor.data[index]
    }

    private func createSolidColorPNG(width: Int, height: Int, r: UInt8, g: UInt8, b: UInt8) throws -> Data {
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
            throw TestImageError.encodingFailed
        }

        context.setFillColor(
            red: CGFloat(Double(r) / 255.0),
            green: CGFloat(Double(g) / 255.0),
            blue: CGFloat(Double(b) / 255.0),
            alpha: 1.0
        )
        context.fill(CGRect(x: 0, y: 0, width: CGFloat(width), height: CGFloat(height)))

        guard let image = context.makeImage() else {
            throw TestImageError.encodingFailed
        }

        let data = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(
            data as CFMutableData,
            UTType.png.identifier as CFString,
            1,
            nil
        ) else {
            throw TestImageError.encodingFailed
        }

        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else {
            throw TestImageError.encodingFailed
        }

        return data as Data
    }
}

private enum TestImageError: Error {
    case encodingFailed
}
