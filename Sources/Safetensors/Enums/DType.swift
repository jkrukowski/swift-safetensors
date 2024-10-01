//
// Created by Tomasz Stachowiak on 1.10.2024.
//

import Foundation
import CoreML


public enum DType: String, Codable {
    case float64 = "F64"
    case float32 = "F32"
    case float16 = "F16"
    case int32 = "I32"
    case uint32 = "U32"
    case int16 = "I16"
    case uint16 = "U16"
    case int8 = "I8"
    case uint8 = "U8"
    case bool = "BOOL"
}
