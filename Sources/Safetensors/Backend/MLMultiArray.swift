import CoreML
import Foundation

extension MLMultiArray: SafetensorsEncodable {
    public var tensorScalarCount: Int {
        shape.reduce(1, { $0 * $1.intValue })
    }

    public var tensorShape: [Int] {
        shape.map { $0.intValue }
    }

    public var dtype: String {
        get throws {
            switch dataType {
            case .float64:
                return Constants.DataType.float64
            case .float32:
                return Constants.DataType.float32
            case .float16:
                return Constants.DataType.float16
            case .int32:
                return Constants.DataType.int32
            default:
                throw Safetensors.Error.unsupportedDataType(String(describing: dataType.rawValue))
            }
        }
    }

    public var scalarSize: Int {
        get throws {
            switch dataType {
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
                throw Safetensors.Error.unsupportedDataType(String(describing: dataType.rawValue))
            }
        }
    }

    public func toData() throws -> Data {
        switch dataType {
        case .float64:
            return withUnsafeBufferPointer(ofType: Float64.self) { ptr in
                Data(buffer: ptr)
            }
        case .float32:
            return withUnsafeBufferPointer(ofType: Float32.self) { ptr in
                Data(buffer: ptr)
            }
        #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64)) && swift(>=6.0)
            case .float16:
                if #available(macOS 15.0, iOS 16.0, tvOS 16.0, watchOS 9.0, visionOS 1.0, *) {
                    return withUnsafeBufferPointer(ofType: Float16.self) { ptr in
                        Data(buffer: ptr)
                    }
                } else {
                    throw Safetensors.Error.unsupportedDataType(
                        String(describing: dataType.rawValue))
                }
        #endif
        case .int32:
            return withUnsafeBufferPointer(ofType: Int32.self) { ptr in
                Data(buffer: ptr)
            }
        default:
            throw Safetensors.Error.unsupportedDataType(String(describing: dataType.rawValue))
        }
    }
}

extension ParsedSafetensors {
    /// Get the MLMultiArray for the given key.
    /// - Parameters:
    ///   - key: key for the array
    ///   - noCopy: if true, the returned MLMultiArray will not copy the data from the original buffer
    /// - Returns: the MLMultiArray for the given key
    public func mlMultiArray(forKey key: String, noCopy: Bool = false) throws -> MLMultiArray {
        let tensorData = try tensorData(forKey: key)
        let dataType = try toMLMultiArrayDataType(from: tensorData.dtype)
        let startIndex = tensorData.dataOffsets.start + headerOffset
        let endIndex = tensorData.dataOffsets.end + headerOffset
        let count = endIndex - startIndex
        var strides = [NSNumber]()
        var stride = 1
        for dimension in tensorData.shape.reversed() {
            strides.append(NSNumber(value: stride))
            stride *= dimension
        }
        strides.reverse()
        if noCopy {
            return try rawData.withUnsafeBytes { (rawBuffer: UnsafeRawBufferPointer) in
                let dataPtr = rawBuffer.baseAddress!.advanced(by: startIndex)
                return try MLMultiArray(
                    dataPointer: UnsafeMutableRawPointer(mutating: dataPtr),
                    shape: tensorData.shape.map { NSNumber(value: $0) },
                    dataType: dataType,
                    strides: strides
                )
            }
        } else {
            let dataPointer = UnsafeMutableRawPointer.allocate(
                byteCount: rawData.count,
                alignment: MemoryLayout<UInt8>.alignment
            )
            rawData.withUnsafeBytes { (rawBuffer: UnsafeRawBufferPointer) in
                dataPointer.copyMemory(
                    from: rawBuffer.baseAddress!.advanced(by: startIndex), byteCount: count)
            }
            return try MLMultiArray(
                dataPointer: dataPointer,
                shape: tensorData.shape.map { NSNumber(value: $0) },
                dataType: dataType,
                strides: strides,
                deallocator: { $0.deallocate() }
            )
        }
    }
}
