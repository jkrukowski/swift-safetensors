public struct OffsetRange: Equatable, Codable {
    public let start: Int
    public let end: Int

    public init(start: Int, end: Int) {
        self.start = start
        self.end = end
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        let array = try container.decode([Int].self)
        precondition(array.count == 2, "Range array needs to have exactly 2 elements")
        self.start = array[0]
        self.end = array[1]
    }

    public func encode(to encoder: Encoder) throws {
        try [start, end].encode(to: encoder)
    }
}
