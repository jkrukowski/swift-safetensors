import Foundation

/// Protocol for types that can be encoded to a `Data` in `Safetensors` format.
public protocol SafetensorsEncodable {
    var tensorScalarCount: Int { get }
    var tensorShape: [Int] { get }
    var dtype: String { get throws }
    var scalarSize: Int { get throws }

    func toData() throws -> Data
}
