import CoreML
import Foundation

public enum Safetensors {}

extension Safetensors {
    public enum Error: Swift.Error {
        case invalidHeaderSize
        case invalidHeaderData
        case missingTensorDataForKey(String)
        case unsupportedDataType(String)
        case dataTypeMismatch
        case metadataIncompleteBuffer
    }
}

extension Safetensors {
    ///  Validate the header data and ensure that the tensor data is contiguous.
    /// - Parameters:
    ///   - header: header data dictionary
    ///   - dataCount: total size of the data buffer, excluding header
    static func validate(header: [String: HeaderElement], dataCount: Int) throws {
        let allDataOffsets = header
            .values
            .compactMap { $0.tensorData?.dataOffsets }
            .sorted { $0.start < $1.start }
        guard let first = allDataOffsets.first, let last = allDataOffsets.last else {
            throw Error.metadataIncompleteBuffer
        }
        if first.start != 0 || last.end != dataCount {
            throw Error.metadataIncompleteBuffer
        }
        for (first, second) in zip(allDataOffsets, allDataOffsets.dropFirst()) {
            if first.end != second.start {
                throw Error.metadataIncompleteBuffer
            }
        }
    }
}

extension Safetensors {
    ///  Read file at given URL and return `ParsedSafetensors` object.
    /// - Parameter url: file URL to read the data from
    /// - Returns: `ParsedSafetensors` object containing the decoded data
    public static func read(at url: URL) throws -> ParsedSafetensors {
        precondition(url.isFileURL, "URL must be a file URL")
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        return try decode(data)
    }

    ///  Decode Data object to ParsedSafetensors object.
    /// - Parameter data: `Data` object containing the encoded data
    /// - Returns: `ParsedSafetensors` object containing the decoded data
    /// - Note: Spec is available [here](https://github.com/huggingface/safetensors/tree/main?tab=readme-ov-file#format)
    public static func decode(_ data: Data) throws -> ParsedSafetensors {
        guard data.count >= 8 else {
            throw Error.invalidHeaderSize
        }
        let headerOffset = data[0..<8].withUnsafeBytes { ptr in
            /// 8 bytes: N, an unsigned little-endian 64-bit integer, containing the size of the header
            let headerSize = ptr.load(as: UInt64.self)
            return Int(headerSize) + 8
        }
        guard data.count >= headerOffset else {
            throw Error.invalidHeaderData
        }
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let headerData = try decoder.decode(
            [String: HeaderElement].self, from: data[8..<headerOffset])
        try validate(header: headerData, dataCount: data.count - headerOffset)
        return ParsedSafetensors(
            headerOffset: headerOffset,
            headerData: headerData,
            rawData: data
        )
    }

    ///  Save dictionary of `SafetensorsEncodable` values to file.
    /// - Parameters:
    ///   - data: dictionary of `SafetensorsEncodable` values
    ///   - metadata: optional metadata dictionary to include in the encoded data
    ///   - url: file URL to save the data to
    public static func write(
        _ data: [String: any SafetensorsEncodable],
        metadata: [String: String]? = nil,
        to url: URL
    ) throws {
        precondition(url.isFileURL, "URL must be a file URL")
        let encodedData = try encode(data, metadata: metadata)
        try encodedData.write(to: url)
    }

    ///  Encode dictionary of `SafetensorsEncodable` values to `Data` object.
    /// - Parameters:
    ///   - data: dictionary of `SafetensorsEncodable` values
    ///   - metadata: optional metadata dictionary to include in the encoded data
    /// - Returns: `Data` object containing the encoded data
    public static func encode(
        _ data: [String: any SafetensorsEncodable],
        metadata: [String: String]? = nil
    ) throws -> Data {
        var headerData = [String: HeaderElement]()
        headerData.reserveCapacity(data.count + (metadata == nil ? 0 : 1))
        let dataBytesCount = try data.values.reduce(0) {
            try $0 + $1.scalarSize * $1.tensorScalarCount
        }
        var tensorData = Data(capacity: dataBytesCount)
        var previousOffset = 0
        for (key, tensor) in data {
            let tensorByteCount = try tensor.scalarSize * tensor.tensorScalarCount
            let tensorHeaderData = try TensorData(
                dtype: tensor.dtype,
                shape: tensor.tensorShape,
                dataOffsets: OffsetRange(
                    start: previousOffset,
                    end: tensorByteCount + previousOffset
                )
            )
            previousOffset += tensorByteCount
            headerData[key] = .tensorData(tensorHeaderData)
            try tensorData.append(tensor.toData())
        }
        if let metadata {
            headerData[Constants.metadataKey] = .metadata(metadata)
        }
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        let header = try encoder.encode(headerData)
        let headerSize = withUnsafeBytes(of: UInt64(header.count)) { Data($0) }
        return headerSize + header + tensorData
    }
}
