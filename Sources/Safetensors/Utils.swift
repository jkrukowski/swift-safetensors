import CoreML
import Foundation

enum Constants {
    static let metadataKey = "__metadata__"

    enum DataType {
        static let float64 = "F64"
        static let float32 = "F32"
        static let float16 = "F16"
        static let int32 = "I32"
        static let uint32 = "U32"
        static let int16 = "I16"
        static let uint16 = "U16"
        static let int8 = "I8"
        static let uint8 = "U8"
        static let bool = "BOOL"
    }
}

func toMLMultiArrayDataType(from dtype: String) throws -> MLMultiArrayDataType {
    switch dtype {
    case Constants.DataType.float64:
        return .float64
    case Constants.DataType.float32:
        return .float32
    case Constants.DataType.float16:
        #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
            return .float16
        #else
            throw Safetensors.Error.unsupportedDataType(dtype)
        #endif
    case Constants.DataType.int32:
        return .int32
    default:
        throw Safetensors.Error.unsupportedDataType(dtype)
    }
}

extension String {
    func zfill(_ width: Int) -> String {
        if self.count >= width {
            return self
        }
        let zerosNeeded = width - self.count
        if self.hasPrefix("-") {
            return "-" + String(repeating: "0", count: zerosNeeded) + self.dropFirst()
        } else {
            return String(repeating: "0", count: zerosNeeded) + self
        }
    }
}
