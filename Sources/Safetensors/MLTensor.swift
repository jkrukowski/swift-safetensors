import CoreML
import Foundation

@available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
extension MLTensor {
    static func toMLTensorScalarType(from dtype: String) throws -> MLTensorScalar.Type {
        switch dtype {
        case "F32":
            return Float32.self
        case "F16":
            #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
                return Float16.self
            #else
                throw SafetensorsError.unsupportedDataType(dtype)
            #endif
        case "I32":
            return Int32.self
        case "U32":
            return UInt32.self
        case "I16":
            return Int16.self
        case "U16":
            return UInt16.self
        case "I8":
            return Int8.self
        case "U8":
            return UInt8.self
        case "BOOL":
            return Bool.self
        default:
            throw SafetensorsError.unsupportedDataType(dtype)
        }
    }
}
