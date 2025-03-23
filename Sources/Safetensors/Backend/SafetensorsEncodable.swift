import Foundation

/// Protocol for types that can be encoded to a `Data` in `Safetensors` format.
public protocol SafetensorsEncodable {
    var tensorScalarCount: Int { get }
    var tensorShape: [Int] { get }
    var dtype: String { get throws }
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
