import CoreML
import Foundation

extension MLMultiArray: SafetensorsEncodable {
    public var scalarCount: Int {
        shape.reduce(1) { $0 * $1.intValue }
    }
    
    public var tensorShape: [Int] {
        shape.map { $0.intValue }
    }
    
    public var dtype: DType {
        get throws {
            try .init(mlMultiArrayDataType: dataType)
        }
    }
    
    public var scalarSize: Int {
        get throws {
            try dataType.scalarSize
        }
    }
    
    public func toData() throws -> Data {
        switch dataType {
            case .double:
                data(ofType: Double.self)
            case .float32:
                data(ofType: Float32.self)
            case .int32:
                data(ofType: Int32.self)
#if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64)) && os
            case .float16:
                guard #available(macOS 15.0, iOS 16.0, tvOS 16.0, watchOS 9.0, visionOS 1.0, *) else {
                    fallthrough
                }
                
                data(ofType: Float16.self)
#endif
            default:
                throw SafetensorsError.unsupportedDataType(dataType.rawValue.description)
        }
    }
    
    private func data<T>(ofType type: T.Type) -> Data where T: MLShapedArrayScalar {
        withUnsafeBufferPointer(ofType: type) { ptr in
            Data(buffer: ptr)
        }
    }
}
