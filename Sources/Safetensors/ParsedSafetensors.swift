import CoreML
import Foundation

public struct ParsedSafetensors {
    let headerOffset: Int
    let headerData: [String: HeaderElement]
    let rawData: Data

    public init(
        headerOffset: Int,
        headerData: [String: HeaderElement],
        rawData: Data
    ) {
        self.headerOffset = headerOffset
        self.headerData = headerData
        self.rawData = rawData
    }

    public var keys: Dictionary<String, HeaderElement>.Keys {
        headerData.keys
    }

    public var metadata: [String: String]? {
        headerData[Constants.metadataKey]?.metadata
    }

    public func tensorData(forKey key: String) throws -> TensorData {
        guard let tensorData = headerData[key]?.tensorData else {
            throw Safetensors.Error.missingTensorDataForKey(key)
        }
        return tensorData
    }
}
