//
//  HeaderEncoder.swift
//  swift-safetensors
//
//  Created by Tomasz Stachowiak on 1.10.2024.
//

import Foundation

struct HeaderDecoder {
    private var jsonDecoder: JSONDecoder
    
    init() {
        jsonDecoder = JSONDecoder()
        jsonDecoder.keyDecodingStrategy = .convertFromSnakeCase
    }
    
    func decode(_ data: Data) throws -> (size: Int, header: ParsedSafetensors.HeaderData) {
        guard data.count >= MemoryLayout<Int>.size else {
            throw SafetensorsError.invalidHeaderData
        }
        
        let headerOffset = MemoryLayout<Int>.size
        let headerSize = data.withUnsafeBytes {
            $0.load(as: Int.self)
        }
        
        guard data.count >= headerOffset + headerSize else {
            throw SafetensorsError.invalidHeaderData
        }
        
        let headerData = data[headerOffset..<headerOffset + headerSize]
        
        let header = try jsonDecoder.decode(ParsedSafetensors.HeaderData.self, from: headerData)
        
        return (size: headerOffset + headerSize, header: header)
    }
}
