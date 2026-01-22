import Foundation

public struct ShardedData {
    public let shards: [Data]
    public let tensorNames: [[String]]  // Array of tensor names for each shard
    public let totalSize: Int

    public init(shards: [Data], tensorNames: [[String]], totalSize: Int) {
        precondition(shards.count == tensorNames.count, "Shard count must match tensor name count")
        self.shards = shards
        self.tensorNames = tensorNames
        self.totalSize = totalSize
    }

    public func write(to url: URL) throws {
        let directoryURL = url.deletingLastPathComponent()
        let baseFileName = url.deletingPathExtension().lastPathComponent
        let fileExtension = url.pathExtension
        var shardSuffix = "\(shards.count)"
        if shardSuffix.count < 5 {
            shardSuffix = shardSuffix.zfill(5)
        }

        // Create a mapping from shard index to shard filename
        var shardFileNames = [String]()
        shardFileNames.reserveCapacity(shards.count)
        for index in 0..<shards.count {
            let shardPrefix = "\(index + 1)".zfill(shardSuffix.count)
            let shardFileName = "\(baseFileName)-\(shardPrefix)-of-\(shardSuffix).\(fileExtension)"
            shardFileNames.append(shardFileName)
        }

        // Write shard files
        for (index, shardData) in shards.enumerated() {
            let shardURL = directoryURL.appendingPathComponent(shardFileNames[index])
            try shardData.write(to: shardURL)
        }

        // Create weight map from tensor names and shard indices
        var weightMap = [String: String]()
        for (shardIndex, names) in tensorNames.enumerated() {
            for tensorName in names {
                weightMap[tensorName] = shardFileNames[shardIndex]
            }
        }

        // Create and write index file
        let modelIndex = ParsedSafetensorsIndexData(
            metadata: ParsedSafetensorsIndexData.Metadata(totalSize: totalSize),
            weightMap: weightMap
        )
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        let encodedIndex = try encoder.encode(modelIndex)
        try encodedIndex.write(
            to: directoryURL.appendingPathComponent("\(baseFileName).index.json")
        )
    }
}
