//
// Created by Tomasz Stachowiak on 1.10.2024.
//

import Foundation

public struct TensorData: Codable {
    public let dtype: DType
    public let shape: [Int]
    public let dataOffsets: OffsetRange

    public init(dtype: DType, shape: [Int], dataOffsets: OffsetRange) {
        self.dtype = dtype
        self.shape = shape
        self.dataOffsets = dataOffsets
    }
}
