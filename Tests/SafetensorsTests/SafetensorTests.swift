#if canImport(Testing) && canImport(CoreML)
    import CoreML
    import Foundation
    import Testing

    @testable import Safetensors

    @Suite struct SafetensorsTests {
        @Test func decodeTensorData() throws {
            let data = createRawSafetensors(
                headerString: #"{"test":{"dtype":"I32","shape":[2,2],"data_offsets":[0,16]}}"#,
                tensorData: Data([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            )
            let safeTensors = try Safetensors.decode(data)
            let testTensor = try safeTensors.tensorData(forKey: "test")

            #expect(testTensor.dtype == .int32)
            #expect(testTensor.shape == [2, 2])
            #expect(testTensor.dataOffsets == OffsetRange(start: 0, end: 16))
            #expect(safeTensors.metadata == nil)
        }

        @Test func decodeTensorWithMetadata() throws {
            let data = createRawSafetensors(
                headerString:
                    #"{"test":{"dtype":"I32","shape":[2,2],"data_offsets":[0,16]},"__metadata__":{"key1":"value1","key2":"value2"}}"#,
                tensorData: Data([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            )
            let safeTensors = try Safetensors.decode(data)
            let testTensor = try safeTensors.tensorData(forKey: "test")

            #expect(testTensor.dtype == .int32)
            #expect(testTensor.shape == [2, 2])
            #expect(testTensor.dataOffsets == OffsetRange(start: 0, end: 16))
            #expect(safeTensors.metadata == ["key1": "value1", "key2": "value2"])
        }

        @Test func decodeTensorWithNullMetadata() throws {
            let data = createRawSafetensors(
                headerString:
                    #"{"test":{"dtype":"I32","shape":[2,2],"data_offsets":[0,16]},"__metadata__":null}"#,
                tensorData: Data([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            )
            let safeTensors = try Safetensors.decode(data)
            let testTensor = try safeTensors.tensorData(forKey: "test")

            #expect(testTensor.dtype == .int32)
            #expect(testTensor.shape == [2, 2])
            #expect(testTensor.dataOffsets == OffsetRange(start: 0, end: 16))
            #expect(safeTensors.metadata == nil)
        }

        @Test func readFromFile() async throws {
            let fileUrl = try #require(
                Bundle.module.url(forResource: "data", withExtension: "safetensors"))
            let safeTensors = try Safetensors.read(at: fileUrl)
            let testTensor: TensorData = try safeTensors.tensorData(forKey: "test")

            #expect(testTensor.dtype == .int32)
            #expect(testTensor.shape == [2, 2])
            #expect(testTensor.dataOffsets == OffsetRange(start: 0, end: 16))

            let tensor = try safeTensors.mlMultiArray(forKey: "test")
            #expect(tensor.shape == [2, 2])
            #expect(tensor.dataType == .int32)
            #expect(tensor[[0, 0] as [NSNumber]] == 0)
            #expect(tensor[[0, 1] as [NSNumber]] == 0)
            #expect(tensor[[1, 0] as [NSNumber]] == 0)
            #expect(tensor[[1, 1] as [NSNumber]] == 0)
        }

        @Test func writeToFile() throws {
            let data: [String: any SafetensorsEncodable] = [
                "test1": MLMultiArray(MLShapedArray<Int32>(repeating: 1, shape: [2, 2])),
                "test2": MLMultiArray(MLShapedArray<Int32>(repeating: 2, shape: [9])),
            ]
            let fileURL = try writeToTemporaryFile("data.safetensors", data: data)
            defer {
                try? FileManager.default.removeItem(at: fileURL)
            }
            #expect(FileManager.default.fileExists(atPath: fileURL.path))
        }

        @Test func decodeEmpty() throws {
            #expect(throws: Safetensors.Error.self) {
                _ = try Safetensors.decode(Data())
            }
        }

        @Test func emptyShapesAllowed() throws {
            let data = createRawSafetensors(
                headerString: #"{"test":{"dtype":"I32","shape":[],"data_offsets":[0,4]}}"#,
                tensorData: Data([0, 0, 0, 0])
            )
            let safeTensors = try Safetensors.decode(data)
            let testTensor = try safeTensors.tensorData(forKey: "test")

            #expect(testTensor.dtype == .int32)
            #expect(testTensor.shape == [])
            #expect(testTensor.dataOffsets == OffsetRange(start: 0, end: 4))
        }

        @Test func zeroSizeTensorAllowed() throws {
            let data = createRawSafetensors(
                headerString: #"{"test":{"dtype":"I32","shape":[],"data_offsets":[0,0]}}"#,
                tensorData: Data()
            )
            let safeTensors = try Safetensors.decode(data)
            let testTensor = try safeTensors.tensorData(forKey: "test")

            #expect(testTensor.dtype == .int32)
            #expect(testTensor.shape == [])
            #expect(testTensor.dataOffsets == OffsetRange(start: 0, end: 0))
        }

        @Test func notSupportedDataTypeAreAllowed() throws {
            let data = createRawSafetensors(
                headerString:
                    #"{"test":{"dtype":"X32","shape":[2,2],"data_offsets":[0,16]},"__metadata__":null}"#,
                tensorData: Data([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            )
            let safeTensors = try Safetensors.decode(data)
            let testTensor = try safeTensors.tensorData(forKey: "test")

            #expect(testTensor.dtype == .notSupported("X32"))
            #expect(testTensor.shape == [2, 2])
            #expect(testTensor.dataOffsets == OffsetRange(start: 0, end: 16))
        }

        @Test func metadataWrongKey() throws {
            let data = createRawSafetensors(
                headerString:
                    #"{"test":{"dtype":"I32","shape":[2,2],"data_offsets":[0,16]},"__wrong__":{"key1":"value1","key2":"value2"}}"#,
                tensorData: Data([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            )
            #expect(throws: Safetensors.Error.self) {
                _ = try Safetensors.decode(data)
            }
        }

        @Test func dataTooShort() throws {
            let data = createRawSafetensors(
                headerSize: 60,
                headerString: #"{"test":{"dtype":"I32","shape":[2,2],"data_offsets":[0,16]}}"#,
                tensorData: Data([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  // missing 2 bytes
            )
            #expect(throws: Safetensors.Error.self) {
                _ = try Safetensors.decode(data)
            }
        }

        @Test func headerTooShort() throws {
            let data = createRawSafetensors(
                headerSize: 10,
                headerString: #"{"test":{"dtype":"I32","shape":[2,2],"data_offsets":[0,16]}}"#,
                tensorData: Data([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            )
            #expect(throws: Swift.DecodingError.self) {
                _ = try Safetensors.decode(data)
            }
        }

        @Test func dataTooLong() throws {
            let data = createRawSafetensors(
                headerSize: 60,
                headerString: #"{"test":{"dtype":"I32","shape":[2,2],"data_offsets":[0,16]}}"#,
                tensorData: Data([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  // extra 2 bytes
            )
            #expect(throws: Safetensors.Error.self) {
                _ = try Safetensors.decode(data)
            }
        }

        @Test func headerTooLong() throws {
            let data = createRawSafetensors(
                headerSize: 1_000_000,
                headerString: #"{"test":{"dtype":"I32","shape":[2,2],"data_offsets":[0,16]}}"#,
                tensorData: Data([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            )
            #expect(throws: Safetensors.Error.self) {
                _ = try Safetensors.decode(data)
            }
        }

        @Test func invalidJSON() throws {
            let data1 = createRawSafetensors(
                headerString: #"[]"#,
                tensorData: Data([0, 0, 0, 0, 0, 0, 0, 0])
            )
            #expect(throws: Swift.DecodingError.self) {
                _ = try Safetensors.decode(data1)
            }

            let data2 = createRawSafetensors(
                headerString: #"{"#,
                tensorData: Data([0, 0, 0, 0, 0, 0, 0, 0])
            )
            #expect(throws: Swift.DecodingError.self) {
                _ = try Safetensors.decode(data2)
            }

            let data3 = createRawSafetensors(
                headerString: #" "#,
                tensorData: Data([0, 0, 0, 0, 0, 0, 0, 0])
            )
            #expect(throws: Swift.DecodingError.self) {
                _ = try Safetensors.decode(data3)
            }
        }

        @Test func convertToDataTypeTests() {
            #expect(DataType(Float.self) == .float32)
            #expect(DataType(Double.self) == .float64)
            #expect(DataType(Float.self) == .float32)
            #expect(DataType(Float64.self) == .float64)
            #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
                #expect(DataType(Float16.self) == .float16)
            #endif
            #expect(DataType(Int8.self) == .int8)
            #expect(DataType(Int16.self) == .int16)
            #expect(DataType(Int32.self) == .int32)
            #expect(DataType(Int64.self) == .int64)
            #expect(DataType(UInt8.self) == .uint8)
            #expect(DataType(UInt16.self) == .uint16)
            #expect(DataType(UInt32.self) == .uint32)
            #expect(DataType(UInt64.self) == .uint64)
            #expect(DataType(Bool.self) == .bool)
        }

        @Test func builderEncode() throws {
            let builder = SafetensorsBuilder()
                .addTensor(
                    MLMultiArray(MLShapedArray<Int32>(scalars: [1, 2, 3, 4], shape: [2, 2])),
                    forKey: "test1"
                )
                .addTensor(
                    MLMultiArray(MLShapedArray<Float32>(repeating: 2, shape: [5])), forKey: "test2"
                )
                .withMetadata(["key1": "value1", "key2": "value2"])

            let encoded = try builder.encode()
            let data = try #require(encoded.singleData)
            let decoded = try Safetensors.decode(data)

            #expect(decoded.metadata == ["key1": "value1", "key2": "value2"])

            // Keys should only include tensor names, not __metadata__
            let tensorKeys = decoded.keys.filter { $0 != "__metadata__" }
            #expect(tensorKeys.sorted() == ["test1", "test2"])

            let tensor1 = try decoded.tensorData(forKey: "test1")
            #expect(tensor1.dtype == .int32)
            #expect(tensor1.shape == [2, 2])

            let tensor2 = try decoded.tensorData(forKey: "test2")
            #expect(tensor2.dtype == .float32)
            #expect(tensor2.shape == [5])
        }

        @Test func builderEncodeWithSharding() throws {
            let builder = SafetensorsBuilder()
                .addTensor(
                    MLMultiArray(MLShapedArray<Int32>(repeating: 1, shape: [2, 2])),
                    forKey: "small1"
                )
                .addTensor(
                    MLMultiArray(MLShapedArray<Int32>(repeating: 4, shape: [20, 20])),
                    forKey: "large1"
                )
                .addTensor(
                    MLMultiArray(MLShapedArray<Float>(repeating: 3, shape: [10, 10])),
                    forKey: "medium"
                )
                .withMetadata(["test_key": "test_value"])
                .withMaxShardingSize(2_000)

            let encoded = try builder.encode()

            // Should be sharded
            guard case .sharded(let shardedData) = encoded else {
                Issue.record("Expected sharded encoding")
                return
            }

            #expect(shardedData.shards.count > 1)
            #expect(shardedData.totalSize == 2_016)  // (4*4) + (400*4) + (100*4)

            // Verify all tensors are accounted for
            let allTensorNames = shardedData.tensorNames.flatMap { $0 }
            #expect(Set(allTensorNames) == Set(["small1", "large1", "medium"]))
        }
    }
#endif
