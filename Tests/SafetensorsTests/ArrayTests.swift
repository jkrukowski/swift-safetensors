#if canImport(Testing)
    import Foundation
    import Testing

    @testable import Safetensors

    @Suite struct ArrayTests {
        @Test func decodeFloat() throws {
            let array: [Float] = [1.0, 2.0, 3.0, 4.0]
            let data = Data(
                bytes: array, count: array.count * MemoryLayout<Float>.stride)
            let rawSafetensors = createRawSafetensors(
                headerString: #"{"test":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}}"#,
                tensorData: data
            )
            let safeTensors = try Safetensors.decode(rawSafetensors)
            let decodedArray: [Float] = try safeTensors.array(forKey: "test")

            #expect(decodedArray == array)
        }

        @Test func decodeInt32() throws {
            let array: [Int32] = [1, 2, 3, 4]
            let data = Data(
                bytes: array, count: array.count * MemoryLayout<Int32>.stride)
            let rawSafetensors = createRawSafetensors(
                headerString: #"{"test":{"dtype":"I32","shape":[4],"data_offsets":[0,16]}}"#,
                tensorData: data
            )
            let safeTensors = try Safetensors.decode(rawSafetensors)
            let decodedArray: [Int32] = try safeTensors.array(forKey: "test")

            #expect(decodedArray == array)
        }

        @Test func decodeDouble() throws {
            let array: [Double] = [1.5, 2.5, 3.5, 4.5]
            let data = Data(
                bytes: array, count: array.count * MemoryLayout<Double>.stride)
            let rawSafetensors = createRawSafetensors(
                headerString: #"{"test":{"dtype":"F64","shape":[4],"data_offsets":[0,32]}}"#,
                tensorData: data
            )
            let safeTensors = try Safetensors.decode(rawSafetensors)
            let decodedArray: [Double] = try safeTensors.array(forKey: "test")

            #expect(decodedArray == array)
        }

        @Test func decodeMultidimensionalTensor() throws {
            let array: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
            let data = Data(
                bytes: array, count: array.count * MemoryLayout<Float>.stride)
            let rawSafetensors = createRawSafetensors(
                headerString: #"{"test":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}}"#,
                tensorData: data
            )
            let safeTensors = try Safetensors.decode(rawSafetensors)
            let decodedArray: [Float] = try safeTensors.array(forKey: "test")

            #expect(decodedArray == array)
        }

        @Test func dataTypeMismatch() throws {
            let array: [Int32] = [1, 2, 3, 4]
            let data = Data(
                bytes: array, count: array.count * MemoryLayout<Int32>.stride)
            let rawSafetensors = createRawSafetensors(
                headerString: #"{"test":{"dtype":"I32","shape":[2,2],"data_offsets":[0,16]}}"#,
                tensorData: data
            )
            let safeTensors = try Safetensors.decode(rawSafetensors)

            #expect(throws: Safetensors.Error.self) {
                _ = try safeTensors.array(forKey: "test") as [Float]
            }
        }
    }
#endif
