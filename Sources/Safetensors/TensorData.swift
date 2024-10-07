import Foundation

public struct TensorData: Codable {
    public let dtype: String
    public let shape: [Int]
    public let dataOffsets: OffsetRange

    public init(dtype: String, shape: [Int], dataOffsets: OffsetRange) {
        self.dtype = dtype
        self.shape = shape
        self.dataOffsets = dataOffsets
    }
}
