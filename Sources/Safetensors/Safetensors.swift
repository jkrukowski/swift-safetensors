import CoreML
import Foundation

public enum SafetensorsError: Error {
    case invalidHeaderSize
    case invalidHeaderData
    case missingTensorData
    case unsupportedDataType(String)
    case metadataIncompleteBuffer
}

public enum Safetensors {
    ///  Validate the header data and ensure that the tensor data is contiguous.
    /// - Parameters:
    ///   - header: header data dictionary
    ///   - dataCount: total size of the data buffer
    static func validate(header: [String: HeaderElement], dataCount: Int) throws {
        let allDataOffsets = header
            .values
            .compactMap {
                $0.tensorData?.dataOffsets
            }
            .sorted {
                $0.start < $1.start
            }
        
        guard let first = allDataOffsets.first, let last = allDataOffsets.last else {
            throw SafetensorsError.metadataIncompleteBuffer
        }
        
        if first.start != 0 || last.end != dataCount {
            throw SafetensorsError.metadataIncompleteBuffer
        }
        
        if zip(allDataOffsets, allDataOffsets.dropFirst()).contains(where: { $0.end != $1.start }) {
            throw SafetensorsError.metadataIncompleteBuffer
        }
        
        for (first, second) in zip(allDataOffsets, allDataOffsets.dropFirst()) {
            if first.end != second.start {
                throw SafetensorsError.metadataIncompleteBuffer
            }
        }
    }
    
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
    public static func decode(_ data: Data) throws -> ParsedSafetensors {
        guard data.count >= MemoryLayout<Int>.size else {
            throw SafetensorsError.invalidHeaderSize
        }

        let result = try HeaderDecoder().decode(data)
        
        try validate(header: result.header, dataCount: data.count - result.size)
        
        return ParsedSafetensors(
            headerSize: result.size,
            headerData: result.header,
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
        var headerData = ParsedSafetensors.HeaderData(
            minimumCapacity: data.count + (metadata == nil ? 0 : 1)
        )
    
        let totalDataSize = try data.values.reduce(0) { try $1.byteCount + $0 }
        var tensorData: Data = .init(capacity: totalDataSize)
        
        for (key, tensor) in data {
            headerData[key] = .tensorData(try tensor.tensorData(at: tensorData.count))
            tensorData.append(try tensor.toData())
        }
        
        if let metadata {
            headerData["__metadata__"] = .metadata(metadata)
        }

        return try HeaderEncoder().encode(headerData) + tensorData
    }
}
