//
// Created by Tomasz Stachowiak on 1.10.2024.
//

import Foundation

/// Protocol for types that can be encoded to a `Data` object.
public protocol SafetensorsEncodable {
    var scalarCount: Int { get }
    var tensorShape: [Int] { get }
    var dtype: DType { get throws }
    
    var scalarSize: Int { get throws }
    
    func toData() throws -> Data
}

extension SafetensorsEncodable {
    var byteCount: Int {
        get throws{
            try self.scalarSize * self.scalarCount
        }
    }
}

extension SafetensorsEncodable {
    func tensorData(at offset: Int = 0) throws -> TensorData {
        try TensorData(
            dtype: dtype,
            shape: tensorShape,
            dataOffsets: OffsetRange(
                start: offset,
                end: offset + byteCount 
            )
        )
    }
}
