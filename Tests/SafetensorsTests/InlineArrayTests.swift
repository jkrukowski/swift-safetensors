#if canImport(Testing) && swift(>=6.2)
    import Foundation
    import Testing

    @testable import Safetensors

    @Suite struct InlineArrayTests {
        @available(macOS 26.0, iOS 26.0, tvOS 26.0, watchOS 26.0, visionOS 26.0, *)
        @Test func decodeFloat() throws {
            let array: [Float] = [1.0, 2.0, 3.0, 4.0]
            let data = Data(
                bytes: array, count: array.count * MemoryLayout<Float>.stride)
            let rawSafetensors = createRawSafetensors(
                headerString: #"{"test":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}}"#,
                tensorData: data
            )
            let safeTensors = try Safetensors.decode(rawSafetensors)
            let inlineArray: InlineArray<4, Float> = try safeTensors.inlineArray(forKey: "test")

            #expect(inlineArray[0] == 1.0)
            #expect(inlineArray[1] == 2.0)
            #expect(inlineArray[2] == 3.0)
            #expect(inlineArray[3] == 4.0)
        }

        @available(macOS 26.0, iOS 26.0, tvOS 26.0, watchOS 26.0, visionOS 26.0, *)
        @Test func decodeInt32() throws {
            let array: [Int32] = [10, 20, 30]
            let data = Data(
                bytes: array, count: array.count * MemoryLayout<Int32>.stride)
            let rawSafetensors = createRawSafetensors(
                headerString: #"{"test":{"dtype":"I32","shape":[3],"data_offsets":[0,12]}}"#,
                tensorData: data
            )
            let safeTensors = try Safetensors.decode(rawSafetensors)
            let inlineArray: InlineArray<3, Int32> = try safeTensors.inlineArray(forKey: "test")

            #expect(inlineArray[0] == 10)
            #expect(inlineArray[1] == 20)
            #expect(inlineArray[2] == 30)
        }

        @available(macOS 26.0, iOS 26.0, tvOS 26.0, watchOS 26.0, visionOS 26.0, *)
        @Test func decodeDouble() throws {
            let array: [Double] = [1.5, 2.5]
            let data = Data(
                bytes: array, count: array.count * MemoryLayout<Double>.stride)
            let rawSafetensors = createRawSafetensors(
                headerString: #"{"test":{"dtype":"F64","shape":[2],"data_offsets":[0,16]}}"#,
                tensorData: data
            )
            let safeTensors = try Safetensors.decode(rawSafetensors)
            let inlineArray: InlineArray<2, Double> = try safeTensors.inlineArray(forKey: "test")

            #expect(inlineArray[0] == 1.5)
            #expect(inlineArray[1] == 2.5)
        }

        @available(macOS 26.0, iOS 26.0, tvOS 26.0, watchOS 26.0, visionOS 26.0, *)
        @Test func decodeMultidimensionalTensor() throws {
            let array: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
            let data = Data(
                bytes: array, count: array.count * MemoryLayout<Float>.stride)
            let rawSafetensors = createRawSafetensors(
                headerString: #"{"test":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}}"#,
                tensorData: data
            )
            let safeTensors = try Safetensors.decode(rawSafetensors)
            let inlineArray: InlineArray<6, Float> = try safeTensors.inlineArray(forKey: "test")

            #expect(inlineArray[0] == 1.0)
            #expect(inlineArray[1] == 2.0)
            #expect(inlineArray[2] == 3.0)
            #expect(inlineArray[3] == 4.0)
            #expect(inlineArray[4] == 5.0)
            #expect(inlineArray[5] == 6.0)
        }

        @available(macOS 26.0, iOS 26.0, tvOS 26.0, watchOS 26.0, visionOS 26.0, *)
        @Test func countMismatch() throws {
            let array: [Float] = [1.0, 2.0, 3.0, 4.0]
            let data = Data(
                bytes: array, count: array.count * MemoryLayout<Float>.stride)
            let rawSafetensors = createRawSafetensors(
                headerString: #"{"test":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}}"#,
                tensorData: data
            )
            let safeTensors = try Safetensors.decode(rawSafetensors)

            // Tensor has 4 elements but we request InlineArray<3, Float>
            #expect(throws: Safetensors.Error.self) {
                let _: InlineArray<3, Float> = try safeTensors.inlineArray(forKey: "test")
            }
        }

        @available(macOS 26.0, iOS 26.0, tvOS 26.0, watchOS 26.0, visionOS 26.0, *)
        @Test func dataTypeMismatch() throws {
            let array: [Int32] = [1, 2, 3, 4]
            let data = Data(
                bytes: array, count: array.count * MemoryLayout<Int32>.stride)
            let rawSafetensors = createRawSafetensors(
                headerString: #"{"test":{"dtype":"I32","shape":[4],"data_offsets":[0,16]}}"#,
                tensorData: data
            )
            let safeTensors = try Safetensors.decode(rawSafetensors)

            // Tensor is I32 but we request Float
            #expect(throws: Safetensors.Error.self) {
                let _: InlineArray<4, Float> = try safeTensors.inlineArray(forKey: "test")
            }
        }
    }
#endif
