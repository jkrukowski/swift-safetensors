//
//  MLMultiArrayDataType+ScalarSize.swift
//  swift-safetensors
//
//  Created by Tomasz Stachowiak on 1.10.2024.
//

import CoreML

extension MLMultiArrayDataType {
    var scalarSize: Int {
        get throws {
            switch self {
                case .double:
                    return MemoryLayout<Double>.size
                case .float32:
                    return MemoryLayout<Float32>.size
                case .int32:
                    return MemoryLayout<Int32>.size
#if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
                case .float16:
                    return MemoryLayout<Float16>.size
#endif
                @unknown default:
                    throw SafetensorsError.unsupportedDataType(rawValue.description)
            }
        }
    }
}
