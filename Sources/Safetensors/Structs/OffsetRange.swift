//
//  OffsetRange.swift
//  swift-safetensors
//
//  Created by Tomasz Stachowiak on 1.10.2024.
//

public struct OffsetRange: Codable {
    let start: Int
    let end: Int

    public init(start: Int = 0, end: Int) {
        self.start = start
        self.end = end
    }

    public init(from decoder: any Decoder) throws {
        var container = try decoder.singleValueContainer()
        let array = try container.decode([Int].self)

        precondition(array.count == 2, "range array needs to have exactly 2 elements")

        self.start = array[0]
        self.end = array[1]
    }

    public func encode(to encoder: any Encoder) throws {
        try [start, end].encode(to: encoder)
    }
}

extension OffsetRange: Equatable {
    public static func ==(lhs: OffsetRange, rhs: OffsetRange) -> Bool {
        lhs.start == rhs.start && lhs.end == rhs.end
    }
}
