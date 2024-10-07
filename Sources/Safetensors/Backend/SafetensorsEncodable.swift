import Foundation

/// Protocol for types that can be encoded to a `Data` in `Safetensors` format.
public protocol SafetensorsEncodable {
    var scalarCount: Int { get }
    var tensorShape: [Int] { get }

    func dtype() throws -> String
    func scalarSize() throws -> Int
    func toData() throws -> Data
}
