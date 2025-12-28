#if swift(>=6.2)
    import Foundation

    extension ParsedSafetensors {
        /// Get the InlineArray for the given key.
        /// - Parameters:
        ///   - key: key for the tensor
        /// - Returns: the InlineArray for the given key
        @available(macOS 26.0, iOS 26.0, tvOS 26.0, watchOS 26.0, visionOS 26.0, *)
        public func inlineArray<let count: Int, Element>(
            forKey key: String
        ) throws -> InlineArray<count, Element> {
            let tensorData = try tensorData(forKey: key)
            let expectedType = try toArrayDataType(from: tensorData.dtype)
            guard expectedType == Element.self else {
                throw Safetensors.Error.dataTypeMismatch
            }
            let startIndex = tensorData.dataOffsets.start + headerOffset
            let endIndex = tensorData.dataOffsets.end + headerOffset
            let byteCount = endIndex - startIndex
            let elementCount = byteCount / MemoryLayout<Element>.size
            guard elementCount == count else {
                throw Safetensors.Error.dataTypeMismatch
            }
            return InlineArray<count, Element>(initializingWith: { span in
                span.withUnsafeMutableBufferPointer { buffer, initializedCount in
                    rawData.copyBytes(to: buffer, from: startIndex..<endIndex)
                    initializedCount = count
                }
            })
        }
    }
#endif
