//
//  DType+MLMultiArrayDataType.swift
//  swift-safetensors
//
//  Created by Tomasz Stachowiak on 1.10.2024.
//

import CoreML


@available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
extension DType {
    var mlTensorScalarType: MLTensorScalar.Type {
        get throws {
            switch self {
                case .float32:
                    Float32.self
                case .int32:
                    Int32.self
                case .uint32:
                    UInt32.self
                case .int16:
                    Int16.self
                case .uint16:
                    UInt16.self
                case .int8:
                    Int8.self
                case .uint8:
                    UInt8.self
                case .bool:
                    Bool.self
#if !(os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64)
                case .float16:
                    Float16.self
#endif
                default:
                    throw SafetensorsError.unsupportedDataType(rawValue)
            }
        }
    }
}
