import CoreML
import Foundation

// NOTE: Right now `MLTensor` does not conform to `SafetensorsEncodable`.

#if swift(>=6)
    @available(macOS 15.0, macCatalyst 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
    extension MLTensor {
        static func toMLTensorScalarType(from dtype: String) throws -> MLTensorScalar.Type {
            switch dtype {
            case "F32":
                return Float32.self
            case "F16":
                #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
                    return Float16.self
                #else
                    throw Safetensors.Error.unsupportedDataType(dtype)
                #endif
            case "I32":
                return Int32.self
            case "U32":
                return UInt32.self
            case "I16":
                return Int16.self
            case "U16":
                return UInt16.self
            case "I8":
                return Int8.self
            case "U8":
                return UInt8.self
            case "BOOL":
                return Bool.self
            default:
                throw Safetensors.Error.unsupportedDataType(dtype)
            }
        }
    }

    extension ParsedSafetensors {
        /// Get the MLTensor for the given key.
        /// - Parameters:
        ///   - key: key for the tensor
        ///   - noCopy: if true, the returned MLTensor will not copy the data from the original buffer
        /// - Returns: the MLTensor for the given key
        @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
        public func mlTensor(forKey key: String, noCopy: Bool = false) throws -> MLTensor {
            let tensorData = try tensorData(forKey: key)
            let scalarType = try MLTensor.toMLTensorScalarType(from: tensorData.dtype)
            let startIndex = tensorData.dataOffsets.start + headerOffset
            let endIndex = tensorData.dataOffsets.end + headerOffset
            let count = endIndex - startIndex
            if noCopy {
                return rawData.withUnsafeBytes { (ptr: UnsafeRawBufferPointer) in
                    let startPtr = ptr.baseAddress!.advanced(by: startIndex)
                    return MLTensor(
                        bytesNoCopy: UnsafeRawBufferPointer(start: startPtr, count: count),
                        shape: tensorData.shape,
                        scalarType: scalarType,
                        deallocator: .none
                    )
                }
            } else {
                return rawData.withUnsafeBytes { (sourcePtr: UnsafeRawBufferPointer) in
                    MLTensor(
                        unsafeUninitializedShape: tensorData.shape,
                        scalarType: scalarType,
                        initializingWith: { ptr in
                            ptr.copyMemory(
                                from: UnsafeRawBufferPointer(
                                    start: sourcePtr.baseAddress!.advanced(by: startIndex),
                                    count: count
                                )
                            )
                        }
                    )
                }
            }
        }
    }
#endif
