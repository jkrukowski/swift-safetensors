import CoreML
import Foundation

extension MLShapedArray: SafetensorsEncodable {
    public var tensorScalarCount: Int {
        scalarCount
    }

    public var tensorShape: [Int] {
        shape
    }

    public var dtype: String {
        get throws {
            switch Self.Scalar.multiArrayDataType {
            case .float64:
                return Constants.DataType.float64
            case .float32:
                return Constants.DataType.float32
            case .float16:
                return Constants.DataType.float16
            case .int32:
                return Constants.DataType.int32
            default:
                throw Safetensors.Error.unsupportedDataType(
                    String(describing: Self.Scalar.multiArrayDataType.rawValue))
            }
        }
    }

    public var scalarSize: Int {
        get throws {
            switch Self.Scalar.multiArrayDataType {
            case .float64:
                return MemoryLayout<Float64>.size
            case .float32:
                return MemoryLayout<Float>.size
            #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
                case .float16:
                    return MemoryLayout<Float16>.size
            #endif
            case .int32:
                return MemoryLayout<Int32>.size
            default:
                throw Safetensors.Error.unsupportedDataType(
                    String(describing: Self.Scalar.multiArrayDataType.rawValue))
            }
        }
    }

    public func toData() throws -> Data {
        switch Self.Scalar.multiArrayDataType {
        case .float64:
            return withUnsafeShapedBufferPointer { ptr, _, _ in
                Data(buffer: ptr)
            }
        case .float32:
            return withUnsafeShapedBufferPointer { ptr, _, _ in
                Data(buffer: ptr)
            }
        #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64)) && swift(>=6.0)
            case .float16:
                if #available(macOS 15.0, iOS 16.0, tvOS 16.0, watchOS 9.0, visionOS 1.0, *) {
                    return withUnsafeShapedBufferPointer { ptr, _, _ in
                        Data(buffer: ptr)
                    }
                } else {
                    throw Safetensors.Error.unsupportedDataType(
                        String(describing: Self.Scalar.multiArrayDataType.rawValue))
                }
        #endif
        case .int32:
            return withUnsafeShapedBufferPointer { ptr, _, _ in
                Data(buffer: ptr)
            }
        default:
            throw Safetensors.Error.unsupportedDataType(
                String(describing: Self.Scalar.multiArrayDataType.rawValue))
        }
    }
}

extension ParsedSafetensors {
    /// Get the MLShapedArray for the given key.
    /// - Parameters:
    ///   - key: key for the array
    ///   - noCopy: if true, the returned MLShapedArray will not copy the data from the original buffer
    /// - Returns: the MLShapedArray for the given key
    public func mlShapedArray<Scalar>(
        forKey key: String,
        noCopy: Bool = false
    ) throws -> MLShapedArray<Scalar> {
        let tensorData = try tensorData(forKey: key)
        let dataType = try toMLMultiArrayDataType(from: tensorData.dtype)
        if dataType != Scalar.multiArrayDataType {
            throw Safetensors.Error.dataTypeMismatch
        }
        let startIndex = tensorData.dataOffsets.start + headerOffset
        let endIndex = tensorData.dataOffsets.end + headerOffset
        let count = endIndex - startIndex
        if noCopy {
            return rawData.withUnsafeBytes { (rawBuffer: UnsafeRawBufferPointer) in
                let startPtr = rawBuffer.baseAddress!.advanced(by: startIndex)
                return MLShapedArray<Scalar>(
                    bytesNoCopy: startPtr,
                    shape: tensorData.shape,
                    deallocator: .none
                )
            }
        } else {
            return rawData.withUnsafeBytes { (rawBuffer: UnsafeRawBufferPointer) in
                MLShapedArray<Scalar>(
                    unsafeUninitializedShape: tensorData.shape,
                    initializingWith: { ptr, strides in
                        let startPtr = rawBuffer.baseAddress!.advanced(by: startIndex)
                        let rawBuffer = UnsafeRawBufferPointer(start: startPtr, count: count)
                        rawBuffer.copyBytes(to: ptr)
                    }
                )
            }
        }
    }
}
