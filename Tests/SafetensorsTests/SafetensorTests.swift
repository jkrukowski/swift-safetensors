#if canImport(Testing)
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

            #expect(testTensor.dtype == "I32")
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

            #expect(testTensor.dtype == "I32")
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

            #expect(testTensor.dtype == "I32")
            #expect(testTensor.shape == [2, 2])
            #expect(testTensor.dataOffsets == OffsetRange(start: 0, end: 16))
            #expect(safeTensors.metadata == nil)
        }

        @Test func readFromFile() async throws {
            let fileUrl = try #require(
                Bundle.module.url(forResource: "data", withExtension: "safetensors"))
            let safeTensors = try Safetensors.read(at: fileUrl)
            let testTensor: TensorData = try safeTensors.tensorData(forKey: "test")

            #expect(testTensor.dtype == "I32")
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
            let temporaryDirectoryURL = URL(
                fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
            let temporaryFileURL = temporaryDirectoryURL.appendingPathComponent("data.safetensors")
            try Safetensors.write(data, to: temporaryFileURL)
            defer {
                try? FileManager.default.removeItem(at: temporaryFileURL)
            }

            #expect(FileManager.default.fileExists(atPath: temporaryFileURL.path))
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

            #expect(testTensor.dtype == "I32")
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

            #expect(testTensor.dtype == "I32")
            #expect(testTensor.shape == [])
            #expect(testTensor.dataOffsets == OffsetRange(start: 0, end: 0))
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
    }
#endif
