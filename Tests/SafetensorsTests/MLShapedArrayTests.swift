import CoreML
import Foundation
import Testing

@testable import Safetensors

@Suite struct MLShapedArrayTests {
    @Test func encodableConformance() throws {
        let array1 = MLShapedArray<Int32>(repeating: 1, shape: [2, 3])
        #expect(array1.tensorScalarCount == 2 * 3)
        #expect(array1.tensorShape == [2, 3])
        #expect(try array1.scalarSize == 4)
        #expect(try array1.dtype == "I32")

        if #available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *) {
            let array2 = MLShapedArray<Float16>(repeating: 1, shape: [2, 3])
            #expect(array2.tensorScalarCount == 2 * 3)
            #expect(array2.tensorShape == [2, 3])
            #expect(try array2.scalarSize == 2)
            #expect(try array2.dtype == "F16")
        }

        let array3 = MLShapedArray<Float32>(repeating: 1, shape: [2, 3])
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
        let tensor: MLShapedArray<Int32> = try safeTensors.mlShapedArray(forKey: "test")

        #expect(tensor.shape == [2, 2])
        #expect(tensor.strides == [2, 1])
        #expect(tensor[scalarAt: 0, 0] == 1)
        #expect(tensor[scalarAt: 0, 1] == 1)
        #expect(tensor[scalarAt: 1, 0] == 1)
        #expect(tensor[scalarAt: 1, 1] == 1)
    }

    @Test func encodeDecode() async throws {
        let data: [String: any SafetensorsEncodable] = [
            "test1": MLMultiArray(MLShapedArray<Int32>(scalars: [1, 2, 3, 4], shape: [2, 2])),
            "test2": MLShapedArray<Float32>(repeating: 2, shape: [5]),
        ]
        let decoded = try Safetensors.decode(
            Safetensors.encode(data, metadata: ["key1": "value1", "key2": "value2"])
        )

        #expect(decoded.metadata == ["key1": "value1", "key2": "value2"])
        let mlShapedArray1: MLShapedArray<Int32> = try #require(
            try decoded.mlShapedArray(forKey: "test1"))
        #expect(mlShapedArray1.shape == [2, 2])
        #expect(mlShapedArray1[scalarAt: 0, 0] == 1)
        #expect(mlShapedArray1[scalarAt: 0, 1] == 2)
        #expect(mlShapedArray1[scalarAt: 1, 0] == 3)
        #expect(mlShapedArray1[scalarAt: 1, 1] == 4)

        let mlShapedArray2: MLShapedArray<Float32> = try #require(
            try decoded.mlShapedArray(forKey: "test2"))
        #expect(mlShapedArray2.shape == [5])
        #expect(mlShapedArray2[scalarAt: 0] == 2)
        #expect(mlShapedArray2[scalarAt: 1] == 2)
        #expect(mlShapedArray2[scalarAt: 2] == 2)
        #expect(mlShapedArray2[scalarAt: 3] == 2)
        #expect(mlShapedArray2[scalarAt: 4] == 2)
    }
}
