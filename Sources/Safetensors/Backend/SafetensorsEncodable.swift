import Foundation

/// Protocol for types that can be encoded to a `Data` in `Safetensors` format.
public protocol SafetensorsEncodable {
    var tensorScalarCount: Int { get }
    var tensorShape: [Int] { get }
    var dtype: DataType { get throws }
    var scalarSize: Int { get throws }
    var tensorByteCount: Int { get throws }

    func toData() throws -> Data
}

extension SafetensorsEncodable {
    public var tensorByteCount: Int {
        get throws {
            try scalarSize * tensorScalarCount
        }
    }
}

public struct AnySafetensorsEncodable: SafetensorsEncodable {
    public var tensorScalarCount: Int
    public var tensorShape: [Int]
    public var dtype: DataType
    public var scalarSize: Int
    public var tensorByteCount: Int
    public var data: Data

    public init(
        tensorScalarCount: Int,
        tensorShape: [Int],
        dtype: DataType,
        scalarSize: Int,
        tensorByteCount: Int,
        data: Data
    ) {
        self.tensorScalarCount = tensorScalarCount
        self.tensorShape = tensorShape
        self.dtype = dtype
        self.scalarSize = scalarSize
        self.tensorByteCount = tensorByteCount
        self.data = data
    }

    public init(
        tensorScalarCount: Int,
        tensorShape: [Int],
        type: Any.Type,
        scalarSize: Int,
        tensorByteCount: Int,
        data: Data
    ) throws {
        try self.init(
            tensorScalarCount: tensorScalarCount,
            tensorShape: tensorShape,
            dtype: DataType(type),
            scalarSize: scalarSize,
            tensorByteCount: tensorByteCount,
            data: data
        )
    }

    public func toData() throws -> Data {
        data
    }
}
