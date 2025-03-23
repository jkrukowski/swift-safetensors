import Foundation

/// Represents the decoded data from a safetensors index file.
public struct ParsedSafetensorsIndexData: Codable {
    public var metadata: Metadata?
    public var weightMap: [String: String]

    public init(
        metadata: ParsedSafetensorsIndexData.Metadata?,
        weightMap: [String: String]
    ) {
        self.metadata = metadata
        self.weightMap = weightMap
    }
}

/// Represents a safetensors index with the ability to load referenced tensors.
public struct ParsedSafetensorsIndex {
    public var metadata: ParsedSafetensorsIndexData.Metadata?
    public var weightMap: [String: String]
    public var baseURL: URL

    public init(
        metadata: ParsedSafetensorsIndexData.Metadata?,
        weightMap: [String: String],
        baseURL: URL
    ) {
        self.metadata = metadata
        self.weightMap = weightMap
        self.baseURL = baseURL
    }

    public func parsedSafetensors(
        forWeightKey key: String,
        baseURL: URL? = nil
    ) throws -> ParsedSafetensors {
        guard let fileName = weightMap[key] else {
            throw Safetensors.Error.missingTensorDataForKey(key)
        }
        let url = baseURL ?? self.baseURL
        return try Safetensors.read(at: url.appendingPathComponent(fileName))
    }

    public func tensorData(
        forKey key: String,
        baseURL: URL? = nil
    ) throws -> TensorData {
        try parsedSafetensors(forWeightKey: key, baseURL: baseURL).tensorData(forKey: key)
    }
}

extension ParsedSafetensorsIndexData {
    public struct Metadata: Codable {
        public var totalSize: Int

        public init(totalSize: Int) {
            self.totalSize = totalSize
        }
    }
}
