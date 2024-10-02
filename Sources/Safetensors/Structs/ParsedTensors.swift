//
// Created by Tomasz Stachowiak on 1.10.2024.
//

import Foundation
import CoreML

public struct ParsedSafetensors {
    typealias HeaderData = [String: HeaderElement]
    
    private let headerSize: Int
    private let headerData: HeaderData
    private let rawData: Data

    public init(
        headerSize: Int,
        headerData: [String: HeaderElement],
        rawData: Data
    ) {
        self.headerSize = headerSize
        self.headerData = headerData
        self.rawData = rawData
    }

    public var metadata: [String: String]? {
        headerData["__metadata__"]?.metadata
    }

    public func tensorData(forKey key: String) throws -> TensorData {
        guard let tensorData = headerData[key]?.tensorData else {
            throw SafetensorsError.missingTensorData
        }
        return tensorData
    }

    @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
    public func mlTensor(forKey key: String, noCopy: Bool = false) throws -> MLTensor {
        let tensorData = try tensorData(forKey: key)
        let scalarType = try tensorData.dtype.mlTensorScalarType
        let startIndex = tensorData.dataOffsets.start + headerSize
        let endIndex = tensorData.dataOffsets.end + headerSize
        let count = endIndex - startIndex
        if noCopy {
            return rawData.withUnsafeBytes { (ptr: UnsafeRawBufferPointer) in
                let startPtr = ptr.baseAddress!.advanced(by: startIndex)
                return MLTensor(
                    bytesNoCopy: UnsafeRawBufferPointer(start: startPtr, count: count),
                    shape: tensorData.shape,
                    scalarType: scalarType,
                    deallocator: .none
                )
            }
        } else {
            return rawData.withUnsafeBytes { (sourcePtr: UnsafeRawBufferPointer) in
                MLTensor(
                    unsafeUninitializedShape: tensorData.shape,
                    scalarType: scalarType,
                    initializingWith: { ptr in
                        ptr.copyMemory(
                            from: UnsafeRawBufferPointer(
                                start: sourcePtr.baseAddress!.advanced(by: startIndex), count: count
                            )
                        )
                    }
                )
            }
        }
    }

    public func mlMultiArray(forKey key: String, noCopy: Bool = false) throws -> MLMultiArray {
        let tensorData = try tensorData(forKey: key)
        let dataType = try tensorData.dtype.mlMultiArrayDataType
        let startIndex = tensorData.dataOffsets.start + headerSize
        let endIndex = tensorData.dataOffsets.end + headerSize
        let count = endIndex - startIndex
        var strides = [NSNumber]()
        var stride = 1
        for dimension in tensorData.shape.reversed() {
            strides.append(NSNumber(value: stride))
            stride *= dimension
        }
        strides.reverse()
        if noCopy {
            return try rawData.withUnsafeBytes { (ptr: UnsafeRawBufferPointer) in
                let dataPtr = ptr.baseAddress!.advanced(by: startIndex)
                return try MLMultiArray(
                    dataPointer: UnsafeMutableRawPointer(mutating: dataPtr),
                    shape: tensorData.shape.map {
                        NSNumber(value: $0)
                    },
                    dataType: dataType,
                    strides: strides
                )
            }
        } else {
            let rawDataCopy = rawData.withUnsafeBytes { (ptr: UnsafeRawBufferPointer) in
                let dataPtr = ptr.baseAddress!.advanced(by: startIndex)
                return Data(bytes: dataPtr, count: count)
            }
            return try rawDataCopy.withUnsafeBytes { (ptr: UnsafeRawBufferPointer) in
                try MLMultiArray(
                    dataPointer: UnsafeMutableRawPointer(mutating: ptr.baseAddress!),
                    shape: tensorData.shape.map {
                        NSNumber(value: $0)
                    },
                    dataType: dataType,
                    strides: strides
                )
            }
        }
    }
}
