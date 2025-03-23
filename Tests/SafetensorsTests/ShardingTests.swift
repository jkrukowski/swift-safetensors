#if canImport(Testing)
    import CoreML
    import Foundation
    import Testing

    @testable import Safetensors

    @Suite struct ShardingTests {
        @Test func groupsForShardingBasic() throws {
            // Create test tensors with different sizes
            let tensors: [String: SafetensorsEncodable] = [
                "tensor1": TestTensor(size: 100, shape: [5, 5, 4]),
                "tensor2": TestTensor(size: 200, shape: [10, 5, 4]),
                "tensor3": TestTensor(size: 300, shape: [15, 5, 4]),
                "tensor4": TestTensor(size: 400, shape: [20, 5, 4]),
            ]

            // Group with a max shard size of 500 bytes
            let groups = try groupsForSharding(tensors, maxShardSizeInBytes: 500)

            // Should create 3 groups:
            // Group 1: tensor4 = 400 bytes
            // Group 2: tensor3 + tensor2 = 500 bytes
            // Group 3: tensor1 = 100 bytes
            #expect(groups.count == 3)

            // Verify first group
            let group1 = groups[0]
            #expect(group1.count == 1)
            #expect(group1.keys.contains("tensor4"))

            // Verify second group
            let group2 = groups[1]
            #expect(group2.count == 2)
            #expect(group2.keys.contains("tensor3"))
            #expect(group2.keys.contains("tensor2"))

            // Verify third group
            let group3 = groups[2]
            #expect(group3.count == 1)
            #expect(group3.keys.contains("tensor1"))
        }

        @Test func groupsForShardingLargeTensor() throws {
            // Create test tensors with one larger than max shard size
            let tensors: [String: SafetensorsEncodable] = [
                "small1": TestTensor(size: 100, shape: [5, 5, 4]),
                "small2": TestTensor(size: 200, shape: [10, 5, 4]),
                "large": TestTensor(size: 1000, shape: [50, 5, 4]),
                "small3": TestTensor(size: 300, shape: [15, 5, 4]),
            ]

            // Group with a max shard size of 500 bytes
            let groups = try groupsForSharding(tensors, maxShardSizeInBytes: 500)

            // Should create three groups:
            // Group 1: large = 1000 bytes (exceeds max size but must be in its own group)
            // Group 2: small3 + small2 = 500 bytes
            // Group 3: small1 = 100 bytes
            #expect(groups.count == 3)

            // Verify first group (large tensor only)
            let group1 = groups[0]
            #expect(group1.count == 1)
            #expect(group1.keys.contains("large"))

            // Verify second group
            let group2 = groups[1]
            #expect(group2.count == 2)
            #expect(group2.keys.contains("small3"))
            #expect(group2.keys.contains("small2"))

            // Verify third group
            let group3 = groups[2]
            #expect(group3.count == 1)
            #expect(group3.keys.contains("small1"))
        }

        @Test func groupsForShardingEmptyInput() throws {
            let tensors: [String: SafetensorsEncodable] = [:]

            let groups = try groupsForSharding(tensors, maxShardSizeInBytes: 500)

            #expect(groups.isEmpty)
        }

        @Test func groupsForShardingSingleTensor() throws {
            let tensors: [String: SafetensorsEncodable] = [
                "tensor1": TestTensor(size: 100, shape: [5, 5, 4])
            ]

            let groups = try groupsForSharding(tensors, maxShardSizeInBytes: 500)

            #expect(groups.count == 1)
            #expect(groups[0].count == 1)
            #expect(groups[0].keys.contains("tensor1"))
        }

        @Test func groupsForShardingMemoryUsage() throws {
            // This test verifies that we don't duplicate large tensors in memory
            // We'll create a large tensor and track memory addresses

            let large1 = TestTensor(size: 1_000_000, shape: [500, 500, 4])
            let large2 = TestTensor(size: 2_000_000, shape: [1000, 500, 4])

            let tensors: [String: SafetensorsEncodable] = [
                "large1": large1,
                "large2": large2,
            ]

            // Get the identifiers of our tensor objects
            let large1Id = ObjectIdentifier(large1)
            let large2Id = ObjectIdentifier(large2)

            // Group with a max shard size that forces separation
            let groups = try groupsForSharding(tensors, maxShardSizeInBytes: 1_500_000)

            // Should create two groups with one tensor each
            #expect(groups.count == 2)

            // Get the identifiers of the tensors in the groups
            let group1TensorId = ObjectIdentifier(groups[0]["large2"] as! TestTensor)
            let group2TensorId = ObjectIdentifier(groups[1]["large1"] as! TestTensor)

            // Verify that they're the same objects (not copies)
            #expect(group1TensorId == large2Id)
            #expect(group2TensorId == large1Id)
        }

        @Test func decodeTensorIndexData() throws {
            let stringIndex =
                #"{"metadata": {"total_size": 123}, "weight_map": {"simple_weight1": "model-00002-of-00002.safetensors","simple_weight2": "model-00001-of-00002.safetensors"}}"#
            let stringData = try #require(stringIndex.data(using: .utf8))
            let index = try Safetensors.decodeIndex(stringData)

            #expect(index.metadata?.totalSize == 123)
            #expect(index.weightMap["simple_weight1"] == "model-00002-of-00002.safetensors")
            #expect(index.weightMap["simple_weight2"] == "model-00001-of-00002.safetensors")
        }

        @Test func decodeTensorIndexDataFromFile() throws {
            let data1: [String: any SafetensorsEncodable] = [
                "test1": MLMultiArray(MLShapedArray<Int32>(repeating: 1, shape: [2, 2]))
            ]
            let data2: [String: any SafetensorsEncodable] = [
                "test2": MLMultiArray(MLShapedArray<Int32>(repeating: 2, shape: [9]))
            ]
            let fileURL1 = try writeToTemporaryFile("model-00001-of-00002.safetensors", data: data1)
            let fileURL2 = try writeToTemporaryFile("model-00002-of-00002.safetensors", data: data2)
            defer {
                try? FileManager.default.removeItem(at: fileURL1)
                try? FileManager.default.removeItem(at: fileURL2)
            }
            let stringIndex =
                #"{"metadata": {"total_size": 123}, "weight_map": {"test1": "model-00001-of-00002.safetensors","test2": "model-00002-of-00002.safetensors"}}"#
            let stringData = try #require(stringIndex.data(using: .utf8))
            let indexData = try Safetensors.decodeIndex(stringData)
            let index = ParsedSafetensorsIndex(
                metadata: indexData.metadata,
                weightMap: indexData.weightMap,
                baseURL: FileManager.default.temporaryDirectory
            )

            let parsedSafetensors1 = try index.parsedSafetensors(forWeightKey: "test1")
            let parsedSafetensors2 = try index.parsedSafetensors(forWeightKey: "test2")

            #expect(parsedSafetensors1.keys.elementsEqual(["test1"]))
            #expect(parsedSafetensors2.keys.elementsEqual(["test2"]))
        }

        @Test func readWriteWithSharding() throws {
            let data: [String: any SafetensorsEncodable] = [
                "small1": MLMultiArray(MLShapedArray<Int32>(repeating: 1, shape: [2, 2])),
                "large1": MLMultiArray(MLShapedArray<Int32>(repeating: 4, shape: [20, 20])),
                "medium": MLMultiArray(MLShapedArray<Float>(repeating: 3, shape: [10, 10])),
                "small2": MLMultiArray(MLShapedArray<Int32>(repeating: 2, shape: [2, 2])),
                "large2": MLMultiArray(MLShapedArray<Float32>(repeating: 5, shape: [20, 20])),
            ]
            let metadata = ["test_key": "test_value"]

            let tempDirectory = FileManager.default.temporaryDirectory
            let fileURL = tempDirectory.appendingPathComponent("model.safetensors")

            // Set small max shard size to force multiple shards
            let maxShardSizeBytes = 2_000
            try Safetensors.write(
                data, metadata: metadata, to: fileURL, maxShardSizeInBytes: maxShardSizeBytes)

            let indexURL = tempDirectory.appendingPathComponent("model.index.json")
            let shardURLs = [
                tempDirectory.appendingPathComponent("model-00001-of-00003.safetensors"),
                tempDirectory.appendingPathComponent("model-00002-of-00003.safetensors"),
                tempDirectory.appendingPathComponent("model-00003-of-00003.safetensors"),
            ]
            #expect(FileManager.default.fileExists(atPath: indexURL.path()))
            for shardURL in shardURLs {
                #expect(
                    FileManager.default.fileExists(atPath: shardURL.path()),
                    "File \(shardURL.path()) should exist")
            }
            #expect(
                !FileManager.default.fileExists(
                    atPath:
                        tempDirectory
                        .appendingPathComponent("model-00000-of-00003.safetensors")
                        .path()
                ),
                "Shard should be indexed from 1, not 0"
            )
            defer {
                for shardURL in shardURLs {
                    try? FileManager.default.removeItem(at: shardURL)
                }
                try? FileManager.default.removeItem(at: indexURL)
            }

            // Verify the index file contents
            let parsedIndex = try Safetensors.readFromIndex(at: indexURL)

            // Should have entries for all our tensors
            #expect(Set(parsedIndex.weightMap.keys) == Set(data.keys))
            #expect(parsedIndex.metadata?.totalSize == 3_632)

            let tensorData1 = try parsedIndex.tensorData(forKey: "small1", baseURL: tempDirectory)
            #expect(tensorData1.dtype == "I32")
            #expect(tensorData1.shape == [2, 2])
            #expect(
                try parsedIndex
                    .parsedSafetensors(forWeightKey: "small1", baseURL: tempDirectory)
                    .metadata == metadata
            )

            let tensorData2 = try parsedIndex.tensorData(forKey: "small2", baseURL: tempDirectory)
            #expect(tensorData2.dtype == "I32")
            #expect(tensorData2.shape == [2, 2])
            #expect(
                try parsedIndex
                    .parsedSafetensors(forWeightKey: "small2", baseURL: tempDirectory)
                    .metadata == metadata
            )

            let tensorData3 = try parsedIndex.tensorData(forKey: "medium", baseURL: tempDirectory)
            #expect(tensorData3.dtype == "F32")
            #expect(tensorData3.shape == [10, 10])
            #expect(
                try parsedIndex
                    .parsedSafetensors(forWeightKey: "medium", baseURL: tempDirectory)
                    .metadata == metadata
            )

            let tensorData4 = try parsedIndex.tensorData(forKey: "large1", baseURL: tempDirectory)
            #expect(tensorData4.dtype == "I32")
            #expect(tensorData4.shape == [20, 20])
            #expect(
                try parsedIndex
                    .parsedSafetensors(forWeightKey: "large1", baseURL: tempDirectory)
                    .metadata == metadata
            )

            let tensorData5 = try parsedIndex.tensorData(forKey: "large2", baseURL: tempDirectory)
            #expect(tensorData5.dtype == "F32")
            #expect(tensorData5.shape == [20, 20])
            #expect(
                try parsedIndex
                    .parsedSafetensors(forWeightKey: "large2", baseURL: tempDirectory)
                    .metadata == metadata
            )
        }
    }
#endif
