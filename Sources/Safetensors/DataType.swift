#if canImport(CoreML)
    import CoreML
#endif

public enum DataType {
    case float64
    case float32
    case float16
    case int64
    case uint64
    case int32
    case uint32
    case int16
    case uint16
    case int8
    case uint8
    case bool
    case notSupported(String)

    public init(rawValue: String) {
        switch rawValue {
        case Constants.F64:
            self = .float64
        case Constants.F32:
            self = .float32
        case Constants.F16:
            self = .float16
        case Constants.I64:
            self = .int64
        case Constants.U64:
            self = .uint64
        case Constants.I32:
            self = .int32
        case Constants.U32:
            self = .uint32
        case Constants.I16:
            self = .int16
        case Constants.U16:
            self = .uint16
        case Constants.I8:
            self = .int8
        case Constants.U8:
            self = .uint8
        case Constants.BOOL:
            self = .bool
        default:
            self = .notSupported(rawValue)
        }
    }

    public var rawValue: String {
        switch self {
        case .float64:
            return Constants.F64
        case .float32:
            return Constants.F32
        case .float16:
            return Constants.F16
        case .int64:
            return Constants.I64
        case .uint64:
            return Constants.U64
        case .int32:
            return Constants.I32
        case .uint32:
            return Constants.U32
        case .int16:
            return Constants.I16
        case .uint16:
            return Constants.U16
        case .int8:
            return Constants.I8
        case .uint8:
            return Constants.U8
        case .bool:
            return Constants.BOOL
        case .notSupported(let string):
            return string
        }
    }
}

extension DataType: Codable {
    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        let rawValue = try container.decode(String.self)
        self = DataType(rawValue: rawValue)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(rawValue)
    }
}

extension DataType: Equatable {
    public static func == (lhs: DataType, rhs: DataType) -> Bool {
        switch (lhs, rhs) {
        case (.float64, .float64), (.float32, .float32), (.float16, .float16),
            (.int64, .int64), (.uint64, .uint64), (.int32, .int32), (.uint32, .uint32),
            (.int16, .int16), (.uint16, .uint16), (.int8, .int8), (.uint8, .uint8),
            (.bool, .bool):
            return true
        case (.notSupported(let lhsString), .notSupported(let rhsString)):
            return lhsString == rhsString
        default:
            return false
        }
    }
}

extension DataType {
    public init(_ type: Any.Type) {
        switch type {
        case is Float64.Type:
            self = .float64
        case is Float32.Type:
            self = .float32
        #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
            case is Float16.Type:
                self = .float16
        #endif
        case is Int64.Type:
            self = .int64
        case is UInt64.Type:
            self = .uint64
        case is Int32.Type:
            self = .int32
        case is UInt32.Type:
            self = .uint32
        case is Int16.Type:
            self = .int16
        case is UInt16.Type:
            self = .uint16
        case is Int8.Type:
            self = .int8
        case is UInt8.Type:
            self = .uint8
        case is Bool.Type:
            self = .bool
        default:
            self = .notSupported("\(type)")
        }
    }

    #if canImport(CoreML)
        func toMLMultiArrayDataType() throws -> MLMultiArrayDataType {
            switch self {
            case .float64:
                return .float64
            case .float32:
                return .float32
            case .float16:
                #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
                    return .float16
                #else
                    throw Safetensors.Error.unsupportedDataType(self.rawValue)
                #endif
            case .int32:
                return .int32
            default:
                throw Safetensors.Error.unsupportedDataType(self.rawValue)
            }
        }
    #endif

    func toArrayDataType() throws -> Any.Type {
        switch self {
        case .float64:
            return Double.self
        case .float32:
            return Float.self
        case .float16:
            #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
                return Float16.self
            #else
                throw Safetensors.Error.unsupportedDataType(self.rawValue)
            #endif
        case .int64:
            return Int64.self
        case .uint64:
            return UInt64.self
        case .int32:
            return Int32.self
        case .uint32:
            return UInt32.self
        case .int16:
            return Int16.self
        case .uint16:
            return UInt16.self
        case .int8:
            return Int8.self
        case .uint8:
            return UInt8.self
        case .bool:
            return Bool.self
        case .notSupported(let value):
            throw Safetensors.Error.unsupportedDataType(value)
        }
    }

    #if canImport(CoreML) && swift(>=6)
        @available(macOS 15.0, macCatalyst 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
        func toMLTensorScalarType() throws -> MLTensorScalar.Type {
            switch self {
            case .float32:
                return Float32.self
            case .float16:
                #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
                    return Float16.self
                #else
                    throw Safetensors.Error.unsupportedDataType(self.rawValue)
                #endif
            case .int32:
                return Int32.self
            case .uint32:
                return UInt32.self
            case .int16:
                return Int16.self
            case .uint16:
                return UInt16.self
            case .int8:
                return Int8.self
            case .uint8:
                return UInt8.self
            case .bool:
                return Bool.self
            default:
                throw Safetensors.Error.unsupportedDataType(self.rawValue)
            }
        }
    #endif
}
