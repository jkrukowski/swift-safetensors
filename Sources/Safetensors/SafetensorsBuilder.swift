import Foundation

public final class SafetensorsBuilder {
    public var tensors: [String: any SafetensorsEncodable]
    public var metadata: [String: String]
    public var maxShardSizeInBytes: Int?

    public init() {
        self.tensors = [:]
        self.metadata = [:]
        self.maxShardSizeInBytes = nil
    }

    public func addTensor(_ tensor: any SafetensorsEncodable, forKey key: String) -> Self {
        self.tensors[key] = tensor
        return self
    }

    public func withMetadata(_ metadata: [String: String]) -> Self {
        self.metadata = metadata
        return self
    }

    public func withMaxShardingSize(_ maxShardSizeInBytes: Int) -> Self {
        self.maxShardSizeInBytes = maxShardSizeInBytes
        return self
    }

    public func encode() throws -> EncodedSafetensors {
        try Safetensors.encode(
            tensors,
            metadata: metadata,
            maxShardSizeInBytes: maxShardSizeInBytes
        )
    }

    public func write(to url: URL) throws {
        try Safetensors.write(
            tensors,
            metadata: metadata,
            maxShardSizeInBytes: maxShardSizeInBytes,
            to: url
        )
    }
}
