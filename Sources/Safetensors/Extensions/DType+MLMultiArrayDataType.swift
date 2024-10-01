//
//  DType+MLMultiArrayDataType.swift
//  swift-safetensors
//
//  Created by Tomasz Stachowiak on 1.10.2024.
//

import CoreML

extension DType {
    init(mlMultiArrayDataType dataType: MLMultiArrayDataType) throws {
        switch dataType {
            case .float64, .double:
                self = .float64
            case .float32, .float:
                self = .float32
            case .int32:
                self = .int32
#if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
            case .float16:
                self = .float16
#endif
            @unknown default:
                throw SafetensorsError.unsupportedDataType(String(describing: dataType.rawValue))
        }
    }
    
    var mlMultiArrayDataType: MLMultiArrayDataType {
        get throws {
            switch self {
                case .float64:
                    return .float64
                case .float32:
                    return .float32
                case .int32:
                    return .int32
#if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
                case .float16:
                    return .float16
#endif
                default:
                    throw SafetensorsError.unsupportedDataType(self.rawValue)
            }
        }
    }
}
