import CoreML
import Foundation

public enum Safetensors {}

/// Protocol for types that can be encoded to a `Data` in `Safetensors` format.
public protocol SafetensorsEncodable {
    var scalarCount: Int { get }
    var tensorShape: [Int] { get }

    func dtype() throws -> String
    func scalarSize() throws -> Int
    func toData() throws -> Data
}

public enum SafetensorsError: Error {
    case invalidHeaderSize
    case invalidHeaderData
    case missingTensorData
    case unsupportedDataType(String)
    case metadataIncompleteBuffer
}

public enum HeaderElement {
    case metadata([String: String])
    case tensorData(TensorData)
}

extension HeaderElement {
    public var metadata: [String: String]? {
        if case .metadata(let metadata) = self {
            return metadata
        }
        return nil
    }

    public var tensorData: TensorData? {
        if case .tensorData(let tensorData) = self {
            return tensorData
        }
        return nil
    }
}

extension HeaderElement: Codable {
    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let metadata = try? container.decode([String: String].self) {
            self = .metadata(metadata)
        } else if let tensorData = try? container.decode(TensorData.self) {
            self = .tensorData(tensorData)
        } else {
            throw DecodingError.dataCorrupted(
                DecodingError.Context(
                    codingPath: decoder.codingPath,
                    debugDescription: "Invalid header element"))
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .metadata(let metadata):
            try container.encode(metadata)
        case .tensorData(let tensorData):
            try container.encode(tensorData)
        }
    }
}

public struct TensorData: Codable {
    public let dtype: String
    public let shape: [Int]
    public let dataOffsets: OffsetRange

    public init(dtype: String, shape: [Int], dataOffsets: OffsetRange) {
        self.dtype = dtype
        self.shape = shape
        self.dataOffsets = dataOffsets
    }
}

public struct ParsedSafetensors {
    private let headerOffset: Int
    private let headerData: [String: HeaderElement]
    private let rawData: Data

    public init(
        headerOffset: Int,
        headerData: [String: HeaderElement],
        rawData: Data
    ) {
        self.headerOffset = headerOffset
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
        let scalarType = try MLTensor.toMLTensorScalarType(from: tensorData.dtype)
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
        let dataType = try MLMultiArray.toMLMultiArrayDataType(from: tensorData.dtype)
        let startIndex = tensorData.dataOffsets.start + headerOffset
        let endIndex = tensorData.dataOffsets.end + headerOffset
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
                    shape: tensorData.shape.map { NSNumber(value: $0) },
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
                    shape: tensorData.shape.map { NSNumber(value: $0) },
                    dataType: dataType,
                    strides: strides
                )
            }
        }
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
            throw SafetensorsError.metadataIncompleteBuffer
        }
        if first.start != 0 || last.end != dataCount {
            throw SafetensorsError.metadataIncompleteBuffer
        }
        for (first, second) in zip(allDataOffsets, allDataOffsets.dropFirst()) {
            if first.end != second.start {
                throw SafetensorsError.metadataIncompleteBuffer
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
    public static func decode(_ data: Data) throws -> ParsedSafetensors {
        guard data.count >= 8 else {
            throw SafetensorsError.invalidHeaderSize
        }
        let headerOffset = data[0..<8].withUnsafeBytes { ptr in
            let headerSize = ptr.load(as: UInt64.self)
            return Int(headerSize) + 8
        }
        guard data.count >= headerOffset else {
            throw SafetensorsError.invalidHeaderData
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
        var previousOffset = 0
        var tensorData = [UInt8]()
        for (key, tensor) in data {
            let tensorByteCount = try tensor.scalarSize() * tensor.scalarCount
            let tensorHeaderData = try TensorData(
                dtype: tensor.dtype(),
                shape: tensor.tensorShape,
                dataOffsets: OffsetRange(
                    start: previousOffset,
                    end: tensorByteCount + previousOffset
                )
            )
            previousOffset += tensorByteCount
            headerData[key] = .tensorData(tensorHeaderData)
            try tensorData.append(contentsOf: tensor.toData())
        }
        if let metadata {
            headerData["__metadata__"] = .metadata(metadata)
        }
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        let header = try encoder.encode(headerData)
        let headerSize = withUnsafeBytes(of: UInt64(header.count)) { Data($0) }
        return headerSize + header + Data(tensorData)
    }
}
