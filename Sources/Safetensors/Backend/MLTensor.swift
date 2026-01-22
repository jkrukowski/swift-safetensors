#if canImport(CoreML) && swift(>=6)
    import CoreML
    import Foundation

    // NOTE: Right now `MLTensor` does not conform to `SafetensorsEncodable`.

    extension ParsedSafetensors {
        /// Get the MLTensor for the given key.
        /// - Parameters:
        ///   - key: key for the tensor
        ///   - noCopy: if true, the returned MLTensor will not copy the data from the original buffer
        /// - Returns: the MLTensor for the given key
        @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
        public func mlTensor(forKey key: String, noCopy: Bool = false) throws -> MLTensor {
            let tensorData = try tensorData(forKey: key)
            let scalarType = try tensorData.dtype.toMLTensorScalarType()
            let startIndex = tensorData.dataOffsets.start + headerOffset
            let endIndex = tensorData.dataOffsets.end + headerOffset
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
                                    start: sourcePtr.baseAddress!.advanced(by: startIndex),
                                    count: count
                                )
                            )
                        }
                    )
                }
            }
        }
    }
#endif
