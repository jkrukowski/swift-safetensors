#if canImport(Testing)
    import CoreML
    import Foundation
    import Testing

    @testable import Safetensors

    @Suite struct MLTensorTests {
        @available(macOS 15.0, macCatalyst 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
        @Test func decodeRaw() async throws {
            let data = createRawSafetensors(
                headerString: #"{"test":{"dtype":"I32","shape":[2,2],"data_offsets":[0,16]}}"#,
                tensorData: Data([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0])
            )
            let safeTensors = try Safetensors.decode(data)

            let tensor = try safeTensors.mlTensor(forKey: "test")
            #expect(tensor.shape == [2, 2])
            #expect(tensor.scalarCount == 4)
            #expect(tensor.scalarType is Int32.Type == true)
            let shapedArray1 = await tensor.shapedArray(of: Int32.self)
            #expect(shapedArray1.scalars == [1, 1, 1, 1])
        }

        @available(macOS 15.0, macCatalyst 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
        @Test func encodeDecode() async throws {
            let data: [String: any SafetensorsEncodable] = [
                "test1": MLMultiArray(MLShapedArray<Int32>(scalars: [1, 2, 3, 4], shape: [2, 2])),
                "test2": MLMultiArray(MLShapedArray<Float32>(repeating: 2, shape: [5])),
            ]
            let decoded = try Safetensors.decode(
                Safetensors.encode(data, metadata: ["key1": "value1", "key2": "value2"])
            )

            #expect(decoded.metadata == ["key1": "value1", "key2": "value2"])
            let mlTensor1 = try #require(
                try decoded.mlTensor(forKey: "test1"))
            #expect(mlTensor1.shape == [2, 2])
            #expect(mlTensor1.scalarCount == 4)
            #expect(mlTensor1.scalarType is Int32.Type == true)
            #expect(await mlTensor1[0, 0].shapedArray(of: Int32.self).scalars == [1])
            #expect(await mlTensor1[0, 1].shapedArray(of: Int32.self).scalars == [2])
            #expect(await mlTensor1[1, 0].shapedArray(of: Int32.self).scalars == [3])
            #expect(await mlTensor1[1, 1].shapedArray(of: Int32.self).scalars == [4])

            let mlTensor2 = try #require(try decoded.mlTensor(forKey: "test2"))
            #expect(mlTensor2.shape == [5])
            #expect(mlTensor2.scalarCount == 5)
            #expect(mlTensor2.scalarType is Float32.Type == true)
            #expect(await mlTensor2[0].shapedArray(of: Float32.self).scalars == [2])
            #expect(await mlTensor2[1].shapedArray(of: Float32.self).scalars == [2])
            #expect(await mlTensor2[2].shapedArray(of: Float32.self).scalars == [2])
            #expect(await mlTensor2[3].shapedArray(of: Float32.self).scalars == [2])
            #expect(await mlTensor2[4].shapedArray(of: Float32.self).scalars == [2])
        }
    }
#endif
