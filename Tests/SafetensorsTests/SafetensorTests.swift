import CoreML
import Foundation
import Testing

@testable import Safetensors

@Suite struct SafeTensorTests {
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
                #"{"test":{"dtype":"I32","shape":[2,2],"data_offsets":[0,16]},"__metadata__":{"key1":"value1"}}"#,
            tensorData: Data([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        )
        let safeTensors = try Safetensors.decode(data)
        let testTensor = try safeTensors.tensorData(forKey: "test")

        #expect(testTensor.dtype == "I32")
        #expect(testTensor.shape == [2, 2])
        #expect(testTensor.dataOffsets == OffsetRange(start: 0, end: 16))
        #expect(safeTensors.metadata == ["key1": "value1"])
    }

    @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
    @Test func decodeMLTensor() throws {
        let data = createRawSafetensors(
            headerString: #"{"test":{"dtype":"I32","shape":[2,2],"data_offsets":[0,16]}}"#,
            tensorData: Data([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        )
        let safeTensors = try Safetensors.decode(data)
        let tensor = try safeTensors.mlTensor(forKey: "test")
        #expect(tensor.shape == [2, 2])
        #expect(tensor.scalarType is Int32.Type == true)
    }

    @Test func decodeMLMultiArray() throws {
        let data = createRawSafetensors(
            headerString: #"{"test":{"dtype":"I32","shape":[2,2],"data_offsets":[0,16]}}"#,
            tensorData: Data([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        )
        let safeTensors = try Safetensors.decode(data)
        let tensor = try safeTensors.mlMultiArray(forKey: "test")
        #expect(tensor.shape == [2, 2])
        #expect(tensor.dataType == .int32)
        #expect(tensor.strides == [2, 1])
    }

    @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
    @Test func readFromFile() async throws {
        let fileUrl = try #require(
            Bundle.module.url(forResource: "data", withExtension: "safetensors"))
        let safeTensors = try Safetensors.read(at: fileUrl)
        let testTensor: TensorData = try safeTensors.tensorData(forKey: "test")

        #expect(testTensor.dtype == "I32")
        #expect(testTensor.shape == [2, 2])
        #expect(testTensor.dataOffsets == OffsetRange(start: 0, end: 16))
        let tensor = try safeTensors.mlTensor(forKey: "test")
        let array = await tensor.shapedArray(of: Int32.self)
        #expect(array.scalars == [0, 0, 0, 0])
    }

    @Test func writeToFile() throws {
        let data: [String: any SafetensorsEncodable] = [
            "test1": MLMultiArray(MLShapedArray<Int32>(repeating: 1, shape: [2, 2])),
            "test2": MLMultiArray(MLShapedArray<Int32>(repeating: 2, shape: [9])),
        ]
        let temporaryDirectoryURL = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
        let temporaryFileURL = temporaryDirectoryURL.appendingPathComponent("data.safetensors")
        try Safetensors.write(data, to: temporaryFileURL)
        defer {
            try? FileManager.default.removeItem(at: temporaryFileURL)
        }

        #expect(FileManager.default.fileExists(atPath: temporaryFileURL.path))
    }

    @Test func decodeEmpty() throws {
        #expect(throws: SafetensorsError.self) {
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

    @Test func dataTooShort() throws {
        let data = createRawSafetensors(
            headerSize: 60,
            headerString: #"{"test":{"dtype":"I32","shape":[2,2],"data_offsets":[0,16]}}"#,
            tensorData: Data([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  // missing 2 bytes
        )
        #expect(throws: SafetensorsError.self) {
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
        #expect(throws: SafetensorsError.self) {
            _ = try Safetensors.decode(data)
        }
    }

    @Test func headerTooLong() throws {
        let data = createRawSafetensors(
            headerSize: 1_000_000,
            headerString: #"{"test":{"dtype":"I32","shape":[2,2],"data_offsets":[0,16]}}"#,
            tensorData: Data([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        )
        #expect(throws: SafetensorsError.self) {
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

    @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
    @Test func encodeDecode() async throws {
        let data: [String: any SafetensorsEncodable] = [
            "test1": MLMultiArray(MLShapedArray<Int32>(repeating: 1, shape: [2, 2])),
            "test2": MLMultiArray(MLShapedArray<Int32>(repeating: 2, shape: [9])),
        ]
        let decoded = try Safetensors.decode(Safetensors.encode(data))

        let test1Tensor = try #require(try decoded.mlTensor(forKey: "test1"))
        #expect(test1Tensor.shape == [2, 2])
        #expect(test1Tensor.scalarType is Int32.Type == true)
        let shapedArray1 = await test1Tensor.shapedArray(of: Int32.self)
        #expect(shapedArray1.scalars == [1, 1, 1, 1])

        let test2Tensor = try #require(try decoded.mlTensor(forKey: "test2"))
        #expect(test2Tensor.shape == [9])
        #expect(test2Tensor.scalarType is Int32.Type == true)
        let shapedArray2 = await test2Tensor.shapedArray(of: Int32.self)
        #expect(shapedArray2.scalars == [2, 2, 2, 2, 2, 2, 2, 2, 2])
    }
}
