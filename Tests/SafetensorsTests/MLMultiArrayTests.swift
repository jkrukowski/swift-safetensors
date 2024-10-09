import CoreML
import Foundation
import Testing

@testable import Safetensors

@Suite struct MLMultiArrayTests {
    @Test func encodableConformance() throws {
        let array1 = MLMultiArray(MLShapedArray<Int32>(repeating: 1, shape: [2, 3]))
        #expect(array1.tensorScalarCount == 2 * 3)
        #expect(array1.tensorShape == [2, 3])
        #expect(try array1.scalarSize == 4)
        #expect(try array1.dtype == "I32")

        if #available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *) {
            let array2 = MLMultiArray(MLShapedArray<Float16>(repeating: 1, shape: [2, 3]))
            #expect(array2.tensorScalarCount == 2 * 3)
            #expect(array2.tensorShape == [2, 3])
            #expect(try array2.scalarSize == 2)
            #expect(try array2.dtype == "F16")
        }

        let array3 = MLMultiArray(MLShapedArray<Float32>(repeating: 1, shape: [2, 3]))
        #expect(array3.tensorScalarCount == 2 * 3)
        #expect(array3.tensorShape == [2, 3])
        #expect(try array3.scalarSize == 4)
        #expect(try array3.dtype == "F32")

        let array4 = MLShapedArray<Float64>(repeating: 1, shape: [2, 3])
        #expect(array4.tensorScalarCount == 2 * 3)
        #expect(array4.tensorShape == [2, 3])
        #expect(try array4.scalarSize == 8)
        #expect(try array4.dtype == "F64")
    }

    @Test func decodeRaw() throws {
        let data = createRawSafetensors(
            headerString: #"{"test":{"dtype":"I32","shape":[2,2],"data_offsets":[0,16]}}"#,
            tensorData: Data([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0])
        )
        let safeTensors = try Safetensors.decode(data)
        let tensor = try safeTensors.mlMultiArray(forKey: "test")

        #expect(tensor.shape == [2, 2])
        #expect(tensor.dataType == .int32)
        #expect(tensor.strides == [2, 1])
        #expect(tensor[[0, 0] as [NSNumber]] == 1)
        #expect(tensor[[0, 1] as [NSNumber]] == 1)
        #expect(tensor[[1, 0] as [NSNumber]] == 1)
        #expect(tensor[[1, 1] as [NSNumber]] == 1)
    }

    @Test func encodeDecode() async throws {
        let data: [String: any SafetensorsEncodable] = [
            "test1": MLMultiArray(MLShapedArray<Int32>(scalars: [1, 2, 3, 4], shape: [2, 2])),
            "test2": MLMultiArray(MLShapedArray<Float32>(repeating: 2, shape: [5])),
        ]
        let decoded = try Safetensors.decode(
            Safetensors.encode(data, metadata: ["key1": "value1", "key2": "value2"])
        )

        #expect(decoded.metadata == ["key1": "value1", "key2": "value2"])
        let mlMultiArray1 = try #require(
            try decoded.mlMultiArray(forKey: "test1"))
        #expect(mlMultiArray1.shape == [2, 2])
        #expect(mlMultiArray1.dataType == .int32)
        #expect(mlMultiArray1[[0, 0] as [NSNumber]] == 1)
        #expect(mlMultiArray1[[0, 1] as [NSNumber]] == 2)
        #expect(mlMultiArray1[[1, 0] as [NSNumber]] == 3)
        #expect(mlMultiArray1[[1, 1] as [NSNumber]] == 4)

        let mlMultiArray2 = try #require(try decoded.mlMultiArray(forKey: "test2"))
        #expect(mlMultiArray2.shape == [5])
        #expect(mlMultiArray2.dataType == .float32)
        #expect(mlMultiArray2[[0] as [NSNumber]] == 2)
        #expect(mlMultiArray2[[1] as [NSNumber]] == 2)
        #expect(mlMultiArray2[[2] as [NSNumber]] == 2)
        #expect(mlMultiArray2[[3] as [NSNumber]] == 2)
        #expect(mlMultiArray2[[4] as [NSNumber]] == 2)
        #expect(mlMultiArray2.dataType == .float32)
    }
}
