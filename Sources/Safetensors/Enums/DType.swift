import Foundation
import CoreML


public enum DType: String, Codable {
    case float64 = "F64"
    case float32 = "F32"
    case float16 = "F16"
    case int32 = "I32"
    case uint32 = "U32"
    case int16 = "I16"
    case uint16 = "U16"
    case int8 = "I8"
    case uint8 = "U8"
    case bool = "BOOL"
}

extension DType {
    var scalarSize: Int {
        switch self {
            case .float64:
                return MemoryLayout<Double>.size
            case .float32:
                return MemoryLayout<Float>.size
            case .float16:
                return MemoryLayout<UInt16>.size
            case .int32:
                return MemoryLayout<Int32>.size
            case .uint32:
                return MemoryLayout<UInt32>.size
            case .int16:
                return MemoryLayout<Int16>.size
            case .uint16:
                return MemoryLayout<UInt16>.size
            case .int8:
                return MemoryLayout<Int8>.size
            case .uint8:
                return MemoryLayout<UInt8>.size
            case .bool:
                return MemoryLayout<Bool>.size
        }
    }
}
