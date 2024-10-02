//
//  HeaderEncoder.swift
//  swift-safetensors
//
//  Created by Tomasz Stachowiak on 1.10.2024.
//

import Foundation

struct HeaderEncoder {
    static func encode(_ headerData: ParsedSafetensors.HeaderData) throws -> Data {
        var jsonEncoder = JSONEncoder()
        jsonEncoder.keyEncodingStrategy = .convertToSnakeCase
        let encodedHeader = try jsonEncoder.encode(headerData)
        return withUnsafeBytes(of: encodedHeader.count) {
            Data($0)
        } + encodedHeader
    }
}
