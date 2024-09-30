import CoreML
import Foundation

extension MLMultiArray: SafetensorsEncodable {
    public var scalarCount: Int {
        shape.reduce(1, { $0 * $1.intValue })
    }

    public var tensorShape: [Int] {
        shape.map { $0.intValue }
    }

    public func dtype() throws -> String {
        switch dataType {
        case .float64:
            return "F64"
        case .float32:
            return "F32"
        case .float16:
            return "F16"
        case .int32:
            return "I32"
        default:
            throw SafetensorsError.unsupportedDataType(String(describing: dataType.rawValue))
        }
    }

    public func scalarSize() throws -> Int {
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
            throw SafetensorsError.unsupportedDataType(String(describing: dataType.rawValue))
        }
    }

    public func toData() throws -> Data {
        switch dataType {
        case .double:
            return withUnsafeBufferPointer(ofType: Float64.self) { ptr in
                Data(buffer: ptr)
            }
        case .float32:
            return withUnsafeBufferPointer(ofType: Float32.self) { ptr in
                Data(buffer: ptr)
            }
        #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
            case .float16:
                if #available(macOS 15.0, iOS 16.0, tvOS 16.0, watchOS 9.0, visionOS 1.0, *) {
                    return withUnsafeBufferPointer(ofType: Float16.self) { ptr in
                        Data(buffer: ptr)
                    }
                } else {
                    throw SafetensorsError.unsupportedDataType(
                        String(describing: dataType.rawValue))
                }
        #endif
        case .int32:
            return withUnsafeBufferPointer(ofType: Int32.self) { ptr in
                Data(buffer: ptr)
            }
        @unknown default:
            throw SafetensorsError.unsupportedDataType(String(describing: dataType.rawValue))
        }
    }
}

extension MLMultiArray {
    static func toMLMultiArrayDataType(from dtype: String) throws -> MLMultiArrayDataType {
        switch dtype {
        case "F64":
            return .float64
        case "F32":
            return .float32
        case "F16":
            #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
                return .float16
            #else
                throw SafetensorsError.unsupportedDataType(dtype)
            #endif
        case "I32":
            return .int32
        default:
            throw SafetensorsError.unsupportedDataType(dtype)
        }
    }
}
