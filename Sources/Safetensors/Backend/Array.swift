extension ParsedSafetensors {
    /// Get the array for the given key.
    /// - Parameters:
    ///   - key: key for the tensor
    /// - Returns: the array for the given key
    public func array<Element>(forKey key: String) throws -> [Element] {
        let tensorData = try tensorData(forKey: key)
        let expectedType = try toArrayDataType(from: tensorData.dtype)
        guard expectedType == Element.self else {
            throw Safetensors.Error.dataTypeMismatch
        }

        let startIndex = tensorData.dataOffsets.start + headerOffset
        let endIndex = tensorData.dataOffsets.end + headerOffset
        let byteCount = endIndex - startIndex
        let elementCount = byteCount / MemoryLayout<Element>.size
        return [Element](unsafeUninitializedCapacity: elementCount) { buffer, initializedCount in
            rawData.copyBytes(to: buffer, from: startIndex..<endIndex)
            initializedCount = elementCount
        }
    }
}
