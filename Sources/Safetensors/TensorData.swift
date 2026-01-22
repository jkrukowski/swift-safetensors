import Foundation

public struct TensorData: Codable {
    public let dtype: DataType
    public let shape: [Int]
    public let dataOffsets: OffsetRange

    public init(dtype: DataType, shape: [Int], dataOffsets: OffsetRange) {
        self.dtype = dtype
        self.shape = shape
        self.dataOffsets = dataOffsets
    }
}
