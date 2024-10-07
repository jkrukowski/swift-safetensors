import Foundation

public enum HeaderElement {
    case metadata([String: String]?)
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
            try HeaderElement.validateMetadataKey(in: container)
            self = .metadata(metadata)
        } else if container.decodeNil() {
            try HeaderElement.validateMetadataKey(in: container)
            self = .metadata(nil)
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

    static func validateMetadataKey(in container: any SingleValueDecodingContainer) throws {
        guard let lastCodingPath = container.codingPath.last else {
            throw Safetensors.Error.metadataIncompleteBuffer
        }
        guard lastCodingPath.stringValue == Constants.metadataKey else {
            throw Safetensors.Error.metadataIncompleteBuffer
        }
    }
}
